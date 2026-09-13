#!/usr/bin/env python3
"""Terminal tool: run shell commands in the configured backend.

Backends (``TERMINAL_ENV``): local (default), docker, singularity, modal
(direct or managed gateway), daytona, vercel_sandbox, ssh, plus
plugin-registered backends. Handles background processes, sandbox lifecycle
(per-task cache, idle reaper, atexit teardown) and sudo password plumbing.
Cloud-sandbox persistent filesystems preserve working state across sandbox
recreation but do NOT guarantee the same live sandbox or long-running
processes survive cleanup, idle reaping, or Hermes exit.

Companion modules (re-exported here, so ``tools.terminal_tool.<name>`` stays the
import/patch target): ``terminal_tool_config`` (TERMINAL_* reads, ``_quiet``),
``terminal_tool_backends`` (env builders + requirement checkers),
``terminal_tool_lifecycle`` (reaper/teardown/ensure_task_env),
``terminal_tool_sudo`` (sudo password + shell rewrites), ``terminal_tool_guards``
(pre-exec blocks), ``terminal_tool_background`` (background spawn),
``terminal_tool_result`` (foreground result post-processing).
"""

import json
import logging
import os
import sys
import time
import threading
import atexit
import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional

logger = logging.getLogger(__name__)


def _redact_terminal_error_text(value: Any) -> str:
    """Force-redact text before serializing a terminal error envelope."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text("" if value is None else str(value), force=True)


from tools.registry import tool_error
from tools.terminal_tool_lifecycle import (
    _check_disk_usage_warning, _cleanup_inactive_envs, _create_configured_env,
    _evict_environment_for_task, cleanup_all_environments, ensure_task_env,
)
from tools.terminal_tool_config import (
    _is_container_backend, _is_host_cwd, _is_unusable_container_cwd, _parse_env_var,
    _plugin_env_flag, _quiet, _safe_getcwd, _tenv, _tenv_bool,
)
from tools.terminal_tool_config import _CONTAINER_BACKENDS  # re-export for file_tools
from tools.terminal_tool_sudo import (_handle_sudo_failure, _invalidate_cached_sudo_on_auth_failure,
    _rewrite_compound_background, _sudo_wrong_password_failure, _transform_sudo_command,
    _count_real_sudo_invocations)
from tools.environments.base import get_sandbox_dir
from tools.environments.singularity import _get_scratch_dir
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment

from tools.terminal_tool_backends import (
    _REQUIREMENT_CHECKERS, _VERCEL_SANDBOX_DEFAULT_CWD, _check_plugin_requirements,
)
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)
from tools.tool_backend_helpers import coerce_modal_mode, managed_nous_tools_enabled


def _safe_parse_import_env(name: str, default: Any, converter, type_label: str):
    """Parse a module-level numeric env var; a malformed value must never make
    the module unloadable at import time (CLI, ACP, tests, tool discovery)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name, raw, type_label, default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env("TERMINAL_MAX_FOREGROUND_TIMEOUT", 600, int, "integer")

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env("TERMINAL_DISK_WARNING_GB", 500.0, float, "number")


# Approval / sudo-prompt UI callbacks (CLI registers prompt_toolkit-aware
# ones). Thread-local so overlapping ACP sessions, each on its own executor
# thread, can't stomp on each other (GHSA-qg5c-hvr5-hjgr). Gateway mode
# resolves approvals via the per-session queue in tools.approval instead.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register the CLI's sudo password prompt callback (per-thread slot)."""
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """Register the dangerous-command approval prompt callback (per-thread slot)."""
    _callback_tls.approval = cb


def _current_session_key() -> str:
    """Active gateway/WebUI session key, or "" outside sessions (ContextVar with
    the ``get_session_env`` os.environ fallback for CLI/cron/tests)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "")


def _current_session_profile() -> str:
    """Active session's Hermes profile name, or "" (same lookup discipline as
    :func:`_current_session_key`)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_PROFILE", "")


from tools.approval import (
    check_all_command_guards as _check_all_guards_impl,
)


def _docker_volume_uses_host_path(volume_spec: str) -> bool:
    """Return True when a docker volume spec bind-mounts a host path."""
    if not isinstance(volume_spec, str):
        return False
    vol = volume_spec.strip()
    return bool(vol) and (
        vol.startswith(("/", "~", "./", "../")) or
        (len(vol) >= 3 and vol[1] == ":" and vol[2] in ("/", "\\"))
    )


def _docker_has_host_access(config: Dict[str, Any]) -> bool:
    """Return True when a Docker sandbox exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(_docker_volume_uses_host_path(vol) for vol in config.get("docker_volumes", []))


def _check_all_guards(command: str, env_type: str,
                      has_host_access: bool = False,
                      execution_target: str = "default",
                      execution_backend: Optional[str] = None,
                      execution_target_named: bool = False,
                      execution_target_scope: str = "") -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access,
                                  execution_target=execution_target,
                                  execution_backend=execution_backend or env_type,
                                  execution_target_named=execution_target_named,
                                  execution_target_scope=execution_target_scope)


from tools.environments.base import EnvironmentConnectionError


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on the selected execution target (bash/Linux shell semantics). The host OS, shell, and terminal backend are stated in your environment section — write commands for THAT platform. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers — anything that needs a shell. Output is auto-truncated with the full text saved to a file — never pipe through tail/head to shorten it.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id); add notify=true for bounded tasks, leave silent only for servers/daemons that never exit. After starting a server, verify readiness with a health check in a separate call (no blind sleep loops); manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd; when a command changes the session cwd (cd, pushd), trust the result's "cwd" field instead of prefixing every command with 'cd'.
PTY: pty=true + background=true for interactive CLIs (they hang without a terminal); drive them with process(action="write"/"submit"). Local backend only.
"""

# Environment lifecycle state.
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_retired_environments: list[tuple[Hashable, Any, float]] = []
_retired_environments_lock = threading.Lock()
_creation_locks: Dict[Hashable, threading.Lock] = {}  # Per-target locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """Run the docker orphan reaper once per process, if enabled.

    Sweeps Exited containers labeled ``hermes-agent=1`` for the current
    profile — leftovers of Hermes processes that died without firing
    ``atexit`` (SIGKILL, OOM, closed terminal). Conservative: only containers
    older than ``2 × lifetime_seconds``, profile-scoped. Gates:
    ``terminal.docker_orphan_reaper: false`` (operator opt-out, e.g. several
    Hermes processes sharing a profile) and the once-per-interpreter flag so
    parallel subagent / RL-rollout calls don't re-sweep.
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    if _docker_orphan_reaper_ran:  # double-checked locking
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × the longest configured Docker-target lifetime: the process-wide reaper
    # runs once, so the first-created target's value must cover every named sibling.
    try:
        lifetime = int(container_config.get(
            "lifetime_seconds", os.getenv("TERMINAL_LIFETIME_SECONDS", "300"),
        ))
    except (TypeError, ValueError):
        lifetime = 300
    try:
        from tools.execution_targets import list_execution_targets

        targets = list_execution_targets()
        if targets and targets[0].named:
            for target in targets:
                if target.backend != "docker":
                    continue
                try:
                    lifetime = max(
                        lifetime, int(target.config.get("lifetime_seconds", 300)),
                    )
                except (TypeError, ValueError):
                    continue
    except Exception:
        logger.debug("Could not resolve Docker target lifetimes", exc_info=True)
    lifetime = max(60, lifetime)
    max_age = lifetime * 2

    try:
        from tools.environments.docker import reap_orphan_containers, _container_identity
    except ImportError:
        return
    # Never fail the env-creation path because of a janitor problem.
    with _quiet("Docker orphan reaper raised"):
        profile = _container_identity(container_config.get("docker_shared_container_key", ""))
        removed = reap_orphan_containers(max_age_seconds=max_age, profile_filter=profile)
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )


# Per-task environment overrides (never exposed to the model). RL/benchmark
# envs and ACP register a custom image / cwd for a task_id BEFORE the agent
# loop; sandbox creation consults this first, then the TERMINAL_* env vars.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}

# Per-session cwd records: the durable source of truth for "which directory
# is THIS session in". Keyed by the raw session/task key, NOT the collapsed
# container id — the env is shared across sessions, so cwd state stored on
# it is a global mutable timeshared between sessions (the wrong-worktree bug
# class). Written after every completed command and on cwd-override
# registration; readers resolve against it before any env-side cwd.
_session_cwd: Dict[str, str] = {}
_session_cwd_lock = threading.Lock()

# Subagent → parent container aliasing. delegate_task children have their own
# task_id but must share the PARENT's container; under per-session isolation
# the collapse-to-"default" shortcut no longer provides that, so the spawn
# site registers an explicit alias.
_container_aliases: Dict[str, str] = {}
_container_alias_lock = threading.Lock()

# Readers still use the legacy env.cwd ladder; later steps flip file_tools and
# _resolve_command_cwd to this store, then delete env-side tracking + guards.
_session_cwd_specs: Dict[Hashable, str] = {}
class _EnvironmentTurnLease:
    def __init__(
        self,
        task_id: Hashable,
        *,
        environment_key: Hashable | None = None,
    ):
        self._key = (
            _register_environment_turn_key(environment_key)
            if environment_key is not None
            else register_environment_turn(task_id)
        )
        self._released = False
        self._lock = threading.Lock()

    @property
    def key(self) -> Hashable:
        return self._key

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._released

    def release(self) -> int:
        with self._lock:
            if self._released:
                return _active_turns_for_environment_key(self._key)
            self._released = True
        return _release_environment_turn_key(self._key)


_active_turn_counts: Dict[Hashable, int] = {}
_active_turn_counts_lock = threading.RLock()
_deferred_environment_cleanups: Dict[Hashable, Hashable] = {}
_logical_environment_lease: contextvars.ContextVar[Optional[_EnvironmentTurnLease]] = contextvars.ContextVar(
    "hermes_logical_environment_lease", default=None)
_tool_environment_lease: contextvars.ContextVar[Optional[_EnvironmentTurnLease]] = contextvars.ContextVar(
    "hermes_tool_environment_lease", default=None)


def _target_resolution(target=None):
    from tools.execution_targets import resolve_execution_target

    return resolve_execution_target(target)


def _environment_scope_key(task_key: Hashable, resolution) -> Hashable:
    return resolution.environment_key(task_key)


def _profile_scoped_task_key(task_key: Hashable) -> Hashable:
    try:
        return _target_resolution(None).scope_task_key(task_key)
    except Exception:
        return task_key


def _turn_scope_key(task_id: Hashable) -> Hashable:
    collapsed = _resolve_container_task_id(str(task_id))
    return _profile_scoped_task_key(collapsed)


def _run_deferred_environment_cleanup(task_id: Hashable) -> None:
    from tools import terminal_tool_lifecycle

    try:
        terminal_tool_lifecycle.cleanup_vm(
            task_id,
            preserve_persistent=True,
            include_collapsed=True,
        )
    except Exception:
        logger.warning(
            "Deferred environment cleanup failed for task %s",
            task_id,
            exc_info=True,
        )


def _turn_keys_overlap(left: Hashable, right: Hashable) -> bool:
    if left == right:
        return True
    if isinstance(left, tuple) and left and left[0] == right:
        return True
    if isinstance(right, tuple) and right and right[0] == left:
        return True
    return False


def _related_active_turns_unlocked(environment_key: Hashable) -> int:
    return sum(
        count for key, count in _active_turn_counts.items()
        if _turn_keys_overlap(key, environment_key)
    )


def _register_environment_turn_key(key: Hashable) -> Hashable:
    with _active_turn_counts_lock:
        _active_turn_counts[key] = _active_turn_counts.get(key, 0) + 1
    return key


def register_environment_turn(task_id: Hashable) -> Hashable:
    return _register_environment_turn_key(_turn_scope_key(task_id))


def _release_environment_turn_key(key: Hashable) -> int:
    deferred_task_ids = []
    with _active_turn_counts_lock:
        current = _active_turn_counts.get(key, 0)
        if current <= 1:
            _active_turn_counts.pop(key, None)
        else:
            _active_turn_counts[key] = current - 1
        remaining = _related_active_turns_unlocked(key)
        for deferred_key, deferred_task_id in list(
            _deferred_environment_cleanups.items()
        ):
            if _related_active_turns_unlocked(deferred_key) == 0:
                deferred_task_ids.append(deferred_task_id)
                _deferred_environment_cleanups.pop(deferred_key, None)
    for deferred_task_id in deferred_task_ids:
        _run_deferred_environment_cleanup(deferred_task_id)
    return remaining


def release_environment_turn(task_id: Hashable) -> int:
    return _release_environment_turn_key(_turn_scope_key(task_id))


def defer_environment_turn_cleanup(task_id: Hashable) -> None:
    # Run collapsed cleanup when the final overlapping lease releases.
    key = _turn_scope_key(task_id)
    run_now = False
    with _active_turn_counts_lock:
        if _related_active_turns_unlocked(key) > 0:
            _deferred_environment_cleanups.setdefault(key, task_id)
        else:
            run_now = True
    if run_now:
        _run_deferred_environment_cleanup(task_id)


def active_environment_turns(task_id: Hashable) -> int:
    return _active_turns_for_environment_key(_turn_scope_key(task_id))


def _active_turns_for_environment_key(environment_key: Hashable) -> int:
    with _active_turn_counts_lock:
        return _related_active_turns_unlocked(environment_key)


class _EnvironmentTurnLease:
    def __init__(
        self,
        task_id: Hashable,
        *,
        environment_key: Hashable | None = None,
    ):
        self._key = (
            _register_environment_turn_key(environment_key)
            if environment_key is not None
            else register_environment_turn(task_id)
        )
        self._released = False
        self._lock = threading.Lock()

    @property
    def key(self) -> Hashable:
        return self._key

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._released

    def release(self) -> int:
        with self._lock:
            if self._released:
                return _active_turns_for_environment_key(self._key)
            self._released = True
        return _release_environment_turn_key(self._key)


_logical_environment_lease: contextvars.ContextVar[Optional[_EnvironmentTurnLease]] = contextvars.ContextVar(
    "logical_environment_lease", default=None,
)
_tool_environment_lease: contextvars.ContextVar[Optional[_EnvironmentTurnLease]] = contextvars.ContextVar(
    "tool_environment_lease", default=None,
)


@contextmanager
def logical_environment_turn(task_id: Hashable):
    # Hold the shared environment for one complete logical conversation turn.
    lease = _EnvironmentTurnLease(task_id)
    token = _logical_environment_lease.set(lease)
    try:
        yield lease
    finally:
        lease.release()
        _logical_environment_lease.reset(token)


def release_logical_environment_turn(task_id: Hashable) -> int:
    # Release this context's logical lease before final cleanup.
    lease = _logical_environment_lease.get()
    key = _turn_scope_key(task_id)
    if lease is not None and lease.key == key:
        return lease.release()
    return active_environment_turns(task_id)


def release_logical_environment_turn_for_cleanup(task_id: Hashable) -> bool:
    # Preserve the established boolean cleanup-hook contract.
    lease = _logical_environment_lease.get()
    if lease is None or lease.key != _turn_scope_key(task_id):
        return False
    lease.release()
    return True


def execution_environment_turn_key(
    function_name: str,
    arguments: Mapping[str, Any],
    *,
    task_id: Hashable | None = None,
) -> Hashable | None:
    if function_name not in {
        "terminal", "read_file", "write_file", "patch", "search_files",
        "execute_code", "process",
    }:
        return None
    task_id = arguments.get("task_id") or task_id
    if not task_id:
        return None
    if function_name == "process":
        # Follow-up calls select a persisted session_id rather than a target.
        # A parent-scope lease safely covers whichever named runtime owns it.
        return _turn_scope_key(task_id)
    target = arguments.get("target")
    if function_name == "search_files":
        target = arguments.get("execution_target")
    try:
        from tools.execution_targets import resolve_execution_target

        resolution = resolve_execution_target(target)
        base_task_id = _resolve_container_task_id(str(task_id))
        return resolution.session_key(base_task_id)
    except Exception:
        # Invalid-target tools still execute to return their normal user-visible
        # validation error; the raw logical lease remains the safe fallback.
        return None


@contextmanager
def environment_turn_usage(
    task_id: Hashable,
    *,
    environment_key: Hashable | None = None,
):
    # Protect one terminal, file, or code invocation from idle cleanup.
    lease = _EnvironmentTurnLease(task_id, environment_key=environment_key)
    token = _tool_environment_lease.set(lease)
    try:
        yield
    finally:
        lease.release()
        _tool_environment_lease.reset(token)


def _current_owned_environment_turns(environment_key: Hashable) -> int:
    # Count this call's own logical/tool leases for replacement checks.
    owned = 0
    for lease in (
        _logical_environment_lease.get(),
        _tool_environment_lease.get(),
    ):
        if (
            lease is not None
            and lease.active
            and _turn_keys_overlap(lease.key, environment_key)
        ):
            owned += 1
    return owned



def record_session_cwd(
    session_key: Optional[str],
    cwd: Optional[str],
    target: Optional[str] = None,
    *,
    _resolution=None,
) -> None:
    """Record *cwd* as the working directory of *session_key*.

    Called wherever a session's live cwd becomes known: after a terminal
    command completes (the env's post-command tracking has just parsed the
    resulting cwd) and when a surface registers a workspace cwd override.
    Empty/None session keys collapse to ``"default"`` (single-session CLI).
    Non-string / empty cwds are ignored.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return
    resolution = _resolution or _target_resolution(target)
    key = resolution.session_key(session_key)
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd
        if resolution.named:
            _session_cwd_specs[key] = resolution.spec_fingerprint
        else:
            _session_cwd_specs.pop(key, None)


def get_session_cwd(
    session_key: Optional[str],
    target: Optional[str] = None,
    *,
    _resolution=None,
) -> Optional[str]:
    """Return the recorded working directory for *session_key*, if any.

    No fallback chain here on purpose: callers decide what an absent record
    means (config default, TERMINAL_CWD seed, process cwd). ``None``/empty
    keys read the ``"default"`` record.
    """
    resolution = _resolution or _target_resolution(target)
    key = resolution.session_key(session_key)
    with _session_cwd_lock:
        recorded_spec = _session_cwd_specs.get(key)
        if (
            resolution.named
            and recorded_spec is not None
            and recorded_spec != resolution.spec_fingerprint
        ):
            return None
        return _session_cwd.get(key)


def inherit_session_cwds(parent_task_id: str, child_task_id: str) -> int:
    """Seed a child with every cwd scope currently owned by its parent."""
    if not parent_task_id or not child_task_id:
        return 0
    parent_key = _profile_scoped_task_key(parent_task_id)
    child_key = _profile_scoped_task_key(child_task_id)
    inherited: Dict[Hashable, str] = {}
    inherited_specs: Dict[Hashable, str] = {}
    with _session_cwd_lock:
        for key, cwd in _session_cwd.items():
            if key == parent_key:
                inherited[child_key] = cwd
                if key in _session_cwd_specs:
                    inherited_specs[child_key] = _session_cwd_specs[key]
            elif (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] == parent_key
            ):
                child_target_key = (child_key, key[1])
                inherited[child_target_key] = cwd
                if key in _session_cwd_specs:
                    inherited_specs[child_target_key] = _session_cwd_specs[key]
        _session_cwd.update(inherited)
        _session_cwd_specs.update(inherited_specs)
    return len(inherited)


def clear_session_cwd(session_key: str) -> None:
    """Drop all legacy and named-target cwd records for a raw session."""
    raw = str(session_key or "default")
    scoped = _profile_scoped_task_key(raw)
    with _session_cwd_lock:
        _session_cwd.pop(raw, None)
        _session_cwd.pop(scoped, None)
        _session_cwd_specs.pop(raw, None)
        _session_cwd_specs.pop(scoped, None)
        for key in list(_session_cwd):
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] in {raw, scoped}
            ):
                _session_cwd.pop(key, None)
                _session_cwd_specs.pop(key, None)
def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """
    Register environment overrides for a specific task/rollout.

    Called by Atropos environments before the agent loop to configure
    per-task sandbox settings (e.g., a custom Dockerfile for the Modal image).

    Supported override keys:
        - modal_image: str -- Path to Dockerfile or Docker Hub image name
        - docker_image: str -- Docker image name
        - cwd: str -- Working directory inside the sandbox

    Args:
        task_id: The rollout's unique task identifier
        overrides: Dict of config keys to override
    """
    _task_env_overrides[_profile_scoped_task_key(task_id)] = overrides

    # A live env for this task must pick up a freshly registered ``cwd`` override
    # (e.g. ACP session/load switching project root) immediately: update the session
    # record (what commands resolve against) and the live env, keeping seeding consistent.
    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        # A registered workspace cwd IS the session's cwd until a `cd` changes it; with
        # named targets it applies to the default target only, never explicit targets.
        try:
            default_resolution = _target_resolution(None)
        except Exception:
            default_resolution = None
        if (
            default_resolution is not None
            and not (
                default_resolution.named
                and default_resolution.backend == "ssh"
            )
        ):
            record_session_cwd(
                task_id, new_cwd, _resolution=default_resolution,
            )
        # Live envs are cached under the raw task_id (per-session surfaces) or the
        # collapsed container id (RL rollouts): try raw first, then container, so a
        # CWD-only override (→ "default") still updates the originating session's env.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            if (
                default_resolution is not None
                and default_resolution.named
                and default_resolution.backend != "ssh"
            ):
                candidate_keys = {
                    default_resolution.environment_key(task_id),
                    default_resolution.environment_key(container_id),
                }
            else:
                candidate_keys = {task_id, container_id}
            envs = [
                env for key, env in _active_environments.items()
                if key in candidate_keys
            ]
        for env in envs:
            if getattr(env, "cwd", None) is not None:
                env.cwd = new_cwd
def clear_task_env_overrides(task_id: str):
    """Drop a task's overrides, cwd record and container alias (rollout cleanup)."""
    _task_env_overrides.pop(task_id, None)
    clear_session_cwd(task_id)
    with _container_alias_lock:
        _container_aliases.pop(task_id, None)


def register_container_alias(child_task_id: str, parent_task_id: Optional[str]) -> None:
    """Make *child_task_id* resolve to *parent_task_id*'s container (called at
    delegate_task spawn). A missing parent id aliases to ``"default"``."""
    if not child_task_id:
        return
    with _container_alias_lock:
        _container_aliases[child_task_id] = str(parent_task_id or "default")


def _resolve_container_alias(task_id: str) -> str:
    """Follow the child→parent alias chain (cycle-safe) for *task_id*."""
    seen = set()
    key = task_id
    with _container_alias_lock:
        while key in _container_aliases and key not in seen:
            seen.add(key)
            key = _container_aliases[key]
    return key


_ISOLATION_OVERRIDE_KEYS = frozenset({
    "docker_image", "modal_image", "singularity_image",
    "daytona_image", "env_type",
})


def _has_isolation_overrides(task_id: Optional[str]) -> bool:
    """True when *task_id* registered image/env_type overrides — the single
    "isolated RL/benchmark rollout" predicate shared by key resolution and
    container creation so the two can't drift."""
    if not task_id or task_id not in _task_env_overrides:
        return False
    return bool(set(_task_env_overrides[task_id].keys()) & _ISOLATION_OVERRIDE_KEYS)


@dataclass(frozen=True)
class _SessionScope:
    """Backend identity + scoping predicates for one call, read once.

    ``env_type`` is the scope-aware TERMINAL_ENV; ``persistent`` is
    ``TERMINAL_CONTAINER_PERSISTENT``. Derived predicates:

    * ``session_isolated`` — non-persistent sandboxes get per-session identities:
      ``container_persistent: false`` means state must not survive or be shared
      across sessions, so one shared sandbox contradicts it. Docker, plus plugin
      backends declaring ``session_isolated_when_nonpersistent`` (sandboxes resumed
      by name, where a shared deterministic name would let two ephemeral runs
      attach one VM and delete it under each other).
    * ``docker_session_isolated`` — docker-only view: the workspace mount and
      session-scoped teardown paths must not fire for other backends.
    * ``docker_profile_scoped`` — docker + ``container_persistent: true``: ONE
      long-lived container per profile shared by every session (CLI, gateway,
      WebUI). The session-key fallback in :func:`_resolve_container_task_id` stops
      cross-profile SSH reuse; ungated it fragmented persistent Docker into one
      container per gateway session, so this restores profile scoping for exactly
      this backend/mode.
    """
    env_type: str
    persistent: bool

    @property
    def session_isolated(self) -> bool:
        if self.env_type != "docker" and not _plugin_env_flag(
            self.env_type, "session_isolated_when_nonpersistent"
        ):
            return False
        return not self.persistent

    @property
    def docker_session_isolated(self) -> bool:
        return self.env_type == "docker" and self.session_isolated

    @property
    def docker_profile_scoped(self) -> bool:
        return self.env_type == "docker" and self.persistent


def _session_scope() -> _SessionScope:
    """Bridge config → env once, then snapshot the backend scope for this call."""
    _ensure_terminal_env_bridged()
    return _SessionScope(
        env_type=_tenv("TERMINAL_ENV", "local"),
        persistent=_tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true"),
    )


def _docker_session_isolation_enabled() -> bool:
    """See :attr:`_SessionScope.docker_session_isolated` (used by the docker builder)."""
    return _session_scope().docker_session_isolated


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """Map a tool-call ``task_id`` to the ``_active_environments`` key. Order matters —
    earlier branches are authoritative where they apply:

    1. Image/``env_type`` overrides (RL/benchmark rollouts) key their own sandbox;
       CWD-only overrides (ACP workspace tracking) are NOT isolation signals.
    2. Per-session isolation (docker + ``container_persistent: false``): each
       session's task_id is its own key (a fresh chat gets a fresh sandbox with only
       ITS mounts); delegate_task children follow the alias registry to the parent.
    3. Session key present (WebUI per-session, gateway per-message): persistent
       Docker is PROFILE-scoped — ``shared:<key>`` opt-in, else ``profile:<name>``,
       with the default profile staying literally ``"default"`` so CLI and
       default-profile gateway sessions share ONE container; other backends key
       ``session:<key>`` so switching profiles can't reuse another profile's
       SSHEnvironment on the wrong host.
    4. No session key (CLI): ``shared:<key>`` when opted in (else a CLI run of a
       keyed profile would split from its gateway sessions), else ``"default"``,
       which subagent ids collapse onto to share the parent's container.
    """
    if task_id and _has_isolation_overrides(task_id):
        return task_id
    scope = _session_scope()
    if task_id and scope.session_isolated:
        return _resolve_container_alias(task_id)
    # Per-session isolation: when a session key is present (the WebUI streaming layer sets it per-session,
    # the gateway per-message via contextvars), scope the container to it so switching profiles can't reuse
    # a previous profile's SSHEnvironment and silently run commands on the wrong remote host. Subagents
    # inherit the same session key, so they still collapse onto the parent's container (the #16177
    # shared-container intent). CLI mode has no session key and falls through to "default", behaviour
    # unchanged. See commit e00f940a9. This runs *after* the isolation-override and
    # docker/container_persistent branches above: those paths already key containers per task_id, so they
    # stay authoritative where they apply and this only covers the cases that would otherwise collapse to
    # the shared "default" key (notably SSH).
    session_key = _current_session_key()
    shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip() if scope.docker_profile_scoped else ""
    if shared:
        # Explicit opt-in: trusted profiles configuring the same terminal.docker_shared_container_key share
        # ONE container/cache slot (and sandbox dir) regardless of profile name (#84671).
        return f"shared:{shared}"
    if not session_key:
        return "default"
    if not scope.docker_profile_scoped:
        return f"session:{session_key}"
    profile = _current_session_profile() or "default"
    return "default" if profile == "default" else f"profile:{profile}"


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return the env overrides for *task_id*, raw key first then collapsed.

    ``register_task_env_overrides`` writes under the *raw* task/session id, but
    a CWD-only override collapses (:func:`_resolve_container_task_id`) to the
    shared ``"default"`` container. Callers must therefore read the raw id
    FIRST and only fall back to the collapsed container id, or the originating
    session's override is silently dropped. Single source of that lookup so
    the terminal and file layers can't drift apart.
    """
    raw = task_id or "default"
    scoped_raw = _profile_scoped_task_key(raw)
    scoped_collapsed = _profile_scoped_task_key(_resolve_container_task_id(raw))
    return (
        _task_env_overrides.get(scoped_raw)
        or _task_env_overrides.get(scoped_collapsed)
        or {}
    )


# Backends that take an image, keyed to the override/config key carrying it.
_IMAGE_KEY_BY_BACKEND = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}


def _select_image(env_type: str, overrides: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Image for *env_type*: per-task override first, then config; "" for imageless backends."""
    key = _IMAGE_KEY_BY_BACKEND.get(env_type)
    if key is None:
        return ""
    return overrides.get(key) or config[key]


def _lookup_active_env(effective_task_id: str, task_id: Optional[str]):
    """Return the cached env for the collapsed id, else for the raw task_id, else None.

    Caller holds ``_env_lock``. Per-session surfaces (ACP/gateway/dashboard)
    with a CWD-only override collapse to ``"default"`` for container sharing,
    yet an env may already be cached under the originating task_id; honor it
    instead of spawning a duplicate. Refreshes ``_last_activity`` on a hit.
    """
    for key in (effective_task_id, task_id):
        if key and key in _active_environments:
            _last_activity[key] = time.time()
            return _active_environments[key]
    return None


def _resolve_task_host_cwd(config: Dict[str, Any], task_id: Optional[str]) -> Optional[str]:
    """Host directory to bind-mount at ``/workspace`` for *task_id*'s container.

    Single owner of the cwd-mount policy for every creation site. Shared-
    container mode: the ``TERMINAL_CWD``-derived ``config["host_cwd"]``.
    Per-session isolation (docker + ``container_persistent: false``): only
    the SESSION's own registered workspace may mount — the process env var is
    a launch artifact that outlives the session that set it, so deriving a
    fresh session's mount from it would leak the previous session's directory.
    Overrides tagged ``cwd_source: "process"`` are refused for the same reason;
    ``cwd_source: "session"`` or untagged (ACP/RL) overrides mount.
    """
    if config.get("env_type") != "docker" or not config.get("docker_mount_cwd_to_workspace"):
        return None
    # Top-level CLI parent ("default") is a single-session process — legacy behavior.
    if not _docker_session_isolation_enabled() or _resolve_container_task_id(task_id) == "default":
        return config.get("host_cwd")
    overrides = resolve_task_overrides(task_id)
    candidate = overrides.get("cwd")
    if overrides.get("cwd_source") == "process" or not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate = os.path.abspath(os.path.expanduser(candidate))
    # Must exist on the host and not already be an in-container path.
    if not os.path.isdir(candidate) or candidate.startswith(("/workspace", "/root")):
        return None
    return candidate


# One-shot guard for the config-fallback bridge: after the first attempt
# either TERMINAL_ENV is set or the import failed, so retrying is wasted work.
_terminal_config_bridge_attempted = False


def _ensure_terminal_env_bridged() -> None:
    """Backfill TERMINAL_* env vars from config.yaml when no launcher did.

    CLI, gateway and TUI/dashboard PTY launches bridge ``terminal.*`` into env vars
    at startup; processes that skip those paths (``hermes serve``, Desktop
    in-process agents, desktop cron ticker, ACP) would otherwise fall back to the
    local backend even when config selects docker — running on the host the user
    meant to sandbox. Explicit keys in the ``terminal`` section override matching
    env values (possibly stale from ``hermes setup``); env values for omitted keys
    are preserved. Without a terminal section an existing TERMINAL_ENV is kept and
    defaults are backfilled only when none is set. A per-turn terminal scope
    suppresses the bridge entirely: writing scope values into the process-global
    env would re-create the first-writer-wins cross-profile leak the scope fixes.

    Ambient ``os.environ`` is the *launch* profile's authority only. Under a
    context-local ``HERMES_HOME`` override (multiplexed dashboard / gateway
    secondary profile), this bridge is a no-op — otherwise the first unscoped
    call under that override would latch the secondary profile's ``terminal.*``
    into process-global env and poison later unscoped launch-profile turns
    (#107422 residual of #68559). Routed profiles must bind a terminal scope
    instead (same rule as ``env_loader._reapply_terminal_config_bridge``).

    terminal_tool reads ALL terminal settings from os.environ (TERMINAL_*). See #61115, #65696.
    """
    from tools.terminal_scope import get_terminal_scope

    if get_terminal_scope() is not None:
        return
    # Never write a secondary profile's terminal.* into process-global env.
    from hermes_constants import get_hermes_home_override

    if get_hermes_home_override() is not None:
        return
    global _terminal_config_bridge_attempted
    if _terminal_config_bridge_attempted:
        return
    _terminal_config_bridge_attempted = True
    # Never let a config problem take the terminal tool down.
    with _quiet("terminal config → env fallback bridge failed"):
        from hermes_cli.config import apply_terminal_config_to_env, read_raw_config

        raw_config = read_raw_config()
        if isinstance(raw_config.get("terminal"), dict):
            apply_terminal_config_to_env(env=None, override=True)
        elif "TERMINAL_ENV" not in os.environ:
            apply_terminal_config_to_env(env=None, override=False)


# Default cwd per backend; anything else (container backends, plugins) is "/root".
_DEFAULT_CWD_BY_BACKEND = {"ssh": "~", "vercel_sandbox": _VERCEL_SANDBOX_DEFAULT_CWD}


def _resolve_config_cwd(env_type: str, mount_docker_cwd: bool) -> tuple:
    """``(cwd, host_cwd)`` from TERMINAL_CWD for *env_type*.

    Container backends are sanity-checked: with Docker cwd passthrough the host
    path is remapped to /workspace and tracked as host_cwd; otherwise host paths
    are discarded in favor of the backend default.
    """
    default_cwd = _safe_getcwd() if env_type == "local" else _DEFAULT_CWD_BY_BACKEND.get(env_type, "/root")
    cwd = _tenv("TERMINAL_CWD", default_cwd)
    from hermes_cli.config import _is_ssh_remote_tilde_cwd
    if env_type == "local" and cwd in {".", "./", "auto", "cwd"}:
        cwd = _safe_getcwd()
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        candidate = os.path.abspath(os.path.expanduser(_tenv("TERMINAL_CWD") or _safe_getcwd()))
        if (
            _is_host_cwd(candidate)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif _is_container_backend(env_type) and cwd and _is_unusable_container_cwd(cwd) and cwd != default_cwd:
        logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                    "(host/relative path won't work in sandbox). Using %r instead.",
                    cwd, env_type, default_cwd)
        cwd = default_cwd
    return cwd, host_cwd


def _get_env_config(terminal_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return canonical terminal config for legacy env vars or a target mapping.

    ``terminal_config is None`` is the historical flat/env-driven path.  A
    selected named target passes its inherited mapping here directly; values
    are parsed without mutating ``os.environ`` so concurrent target calls
    cannot affect each other.
    """
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    if terminal_config is None:
        _ensure_terminal_env_bridged()

    def _get(key: str, env_name: str, default: Any) -> Any:
        if terminal_config is None:
            # Scope-aware read (``_tenv``): a raw ``os.getenv`` here re-created the
            # first-writer-wins cross-profile leak (#68559) — the launch profile's
            # TERMINAL_* env stays pinned in os.environ. No scope bound = os.getenv.
            return _tenv(env_name, str(default) if not isinstance(default, (list, dict)) else json.dumps(default))
        return terminal_config.get(key, default)

    def _coerce(value: Any, converter: Any, label: str, key: str) -> Any:
        if converter is json.loads and isinstance(value, (list, dict)):
            return value
        try:
            return converter(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            source = (
                f"terminal target setting {key}"
                if terminal_config is not None
                else f"TERMINAL_{key.upper()}"
            )
            raise ValueError(f"Invalid value for {source}: {value!r} (expected {label}).")

    def _bool(key: str, env_name: str, default: bool) -> bool:
        value = _get(key, env_name, default)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if terminal_config is not None:
            raise ValueError(
                f"Invalid value for terminal target setting {key}: {value!r} "
                "(expected boolean)."
            )
        # Preserve the legacy env-var behavior for unknown strings.
        return False

    def _json_shape(
        key: str,
        env_name: str,
        default: Any,
        expected_type: type,
        label: str,
    ) -> Any:
        value = _coerce(_get(key, env_name, default), json.loads, "valid JSON", key)
        if not isinstance(value, expected_type):
            source = (
                f"terminal target setting {key}"
                if terminal_config is not None
                else env_name
            )
            raise ValueError(
                f"Invalid value for {source}: {value!r} (expected {label})."
            )
        return value

    env_type = str(
        _get(
            "backend", "TERMINAL_ENV",
            terminal_config.get("env_type", "local") if terminal_config else "local",
        )
    ).strip().lower() or "local"
    
    mount_docker_cwd = _bool(
        "docker_mount_cwd_to_workspace", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", False,
    )
    container_backend = _is_container_backend(env_type)
    docker_backend = env_type == "docker"

    # Docker-only env vars may be bridged from config.yaml even on local/ssh — don't
    # parse them until a container backend is selected, or a stale value breaks local.
    if container_backend:
        container_cpu = _coerce(_get("container_cpu", "TERMINAL_CONTAINER_CPU", 1), float, "number", "container_cpu")
        container_memory = _coerce(_get("container_memory", "TERMINAL_CONTAINER_MEMORY", 5120), int, "integer", "container_memory")
        container_disk = _coerce(_get("container_disk", "TERMINAL_CONTAINER_DISK", 51200), int, "integer", "container_disk")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _json_shape(
            "docker_forward_env", "TERMINAL_DOCKER_FORWARD_ENV", [], list, "list",
        )
        docker_volumes = _json_shape(
            "docker_volumes", "TERMINAL_DOCKER_VOLUMES", [], list, "list",
        )
        docker_env = _json_shape(
            "docker_env", "TERMINAL_DOCKER_ENV", {}, dict, "mapping",
        )
        docker_extra_args = _json_shape(
            "docker_extra_args", "TERMINAL_DOCKER_EXTRA_ARGS", [], list, "list",
        )
        docker_shm_size = str(_get("docker_shm_size", "TERMINAL_DOCKER_SHM_SIZE", "1g") or "")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []
        docker_shm_size = "1g"

    # Default cwd: host cwd for local, remote home for ssh, the documented
    # workspace root for Vercel, the backend's default root-like cwd otherwise.
    if env_type == "local":
        default_cwd = _safe_getcwd()
    elif env_type == "ssh":
        default_cwd = "~"
    elif env_type == "vercel_sandbox":
        default_cwd = _VERCEL_SANDBOX_DEFAULT_CWD
    else:
        default_cwd = "/root"

    # Read TERMINAL_CWD but sanity-check for container backends: with Docker cwd
    # passthrough enabled, remap host paths to /workspace and track host_cwd; else discard.
    cwd = str(_get("cwd", "TERMINAL_CWD", default_cwd) or default_cwd)
    from hermes_cli.config import _is_ssh_remote_tilde_cwd
    if env_type == "local" and cwd in {".", "./", "auto", "cwd"}:
        cwd = _safe_getcwd()
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = (
            (_tenv("TERMINAL_CWD") or _safe_getcwd())
            if terminal_config is None
            else (cwd or _safe_getcwd())
        )
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            _is_host_cwd(candidate)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif _is_container_backend(env_type) and cwd:
        # Host paths and relative paths that won't work inside containers
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(_get("modal_mode", "TERMINAL_MODAL_MODE", "auto")),
        "docker_image": str(_get("docker_image", "TERMINAL_DOCKER_IMAGE", default_image)),
        "docker_forward_env": docker_forward_env,
        "singularity_image": str(_get("singularity_image", "TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}")),
        "modal_image": str(_get("modal_image", "TERMINAL_MODAL_IMAGE", default_image)),
        "daytona_image": str(_get("daytona_image", "TERMINAL_DAYTONA_IMAGE", default_image)),
        "vercel_runtime": str(_get("vercel_runtime", "TERMINAL_VERCEL_RUNTIME", "")).strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _coerce(_get("timeout", "TERMINAL_TIMEOUT", 180), int, "integer", "timeout"),
        "lifetime_seconds": _coerce(_get("lifetime_seconds", "TERMINAL_LIFETIME_SECONDS", 300), int, "integer", "lifetime_seconds"),
        # SSH-specific config
        "ssh_host": str(_get("ssh_host", "TERMINAL_SSH_HOST", "")),
        "ssh_user": str(_get("ssh_user", "TERMINAL_SSH_USER", "")),
        "ssh_port": _coerce(_get("ssh_port", "TERMINAL_SSH_PORT", 22), int, "integer", "ssh_port"),
        "ssh_key": str(_get("ssh_key", "TERMINAL_SSH_KEY", "")),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local is always opt-in.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": _bool(
            "ssh_persistent", "TERMINAL_SSH_PERSISTENT",
            _bool("persistent_shell", "TERMINAL_PERSISTENT_SHELL", True),
        ),
        "local_persistent": _bool("local_persistent", "TERMINAL_LOCAL_PERSISTENT", False),
        # Container resource config (applies to docker, singularity, modal,
        # daytona, and vercel_sandbox -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": _bool("container_persistent", "TERMINAL_CONTAINER_PERSISTENT", True),
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": _bool("docker_run_as_host_user", "TERMINAL_DOCKER_RUN_AS_HOST_USER", False),
        "docker_snap_compat": _bool("docker_snap_compat", "TERMINAL_DOCKER_SNAP_COMPAT", False),
        "docker_network": _bool("docker_network", "TERMINAL_DOCKER_NETWORK", True),
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process container reuse (#20561): probe for a labeled container at
        # startup and attach instead of starting fresh — makes the docs' "ONE
        # long-lived container shared across sessions" real. ``false`` = isolation.
        "docker_persist_across_processes": _bool(
            "docker_persist_across_processes", "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", True,
        ),
        "docker_shared_container_key": _tenv(
            "TERMINAL_DOCKER_SHARED_CONTAINER_KEY", ""
        ),
        # Startup orphan reaper for hermes-tagged containers left by crashed/SIGKILL'd
        # processes: Exited only, older than 2× idle-reap, current profile. Issue #20561.
        "docker_orphan_reaper": _bool(
            "docker_orphan_reaper", "TERMINAL_DOCKER_ORPHAN_REAPER", True,
        ),
    }
def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    while _cleanup_running:
        with _quiet("Error in cleanup thread", level=logging.WARNING):
            _cleanup_inactive_envs(_get_env_config()["lifetime_seconds"])
        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def _atexit_cleanup():
    """Stop the cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        logger.info("Shutting down %d remaining sandbox(es)...", len(_active_environments))
        # Snapshot BEFORE cleanup_all_environments empties the dict, then
        # block briefly so docker stop/rm completes before the interpreter
        # exits — otherwise daemon cleanup threads die mid-`docker stop` and
        # Exited containers pile up on the host.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is not None:
                with _quiet("wait_for_cleanup raised on exit"):  # never block shutdown on a bad backend
                    wait_fn(timeout=15.0)
    # Workers of envs the idle reaper already detached are not in the registry (#86317).
    if "tools.environments.docker" in sys.modules:
        with _quiet("teardown drain raised on exit"):
            sys.modules["tools.environments.docker"].DockerEnvironment.wait_for_all_teardowns(timeout=15.0)

atexit.register(_atexit_cleanup)


def _command_requires_pipe_stdin(command: str) -> bool:
    """True when PTY mode would break a stdin-driven command: `gh auth login
    --with-token` waits for EOF on piped stdin, and under a PTY
    `process.submit()` only sends a newline, so it hangs forever."""
    normalized = " ".join(command.lower().split())
    return normalized.startswith("gh auth login") and "--with-token" in normalized


from tools.terminal_tool_guards import (
    _foreground_background_guidance, _safe_command_preview, _validate_workdir,
    gateway_lifecycle_block, self_repo_block,
)
from tools.terminal_tool_background import _YIELDED_NOTE, spawn_background_process, yield_to_background_handler
from tools.terminal_tool_result import finalize_foreground_result


def _resolve_notification_flag_conflict(*, notify_on_complete: bool, watch_patterns, background: bool) -> tuple:
    """Resolve notify_on_complete + watch_patterns both set: drop watch_patterns
    (combined they produce duplicate async notifications — one per match plus
    one on exit — that can spam the user long after the process ends).
    Returns ``(watch_patterns_to_use, conflict_note)``; note is "" without conflict."""
    if background and notify_on_complete and watch_patterns:
        return None, (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
    return watch_patterns, ""


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    default_cwd: str,
    session_key: Optional[str] = None,
    env_type: Optional[str] = None,
    target: Optional[str] = None,
    _resolution=None,
) -> str:
    """cwd for a command: explicit ``workdir`` > the session's own cwd record >
    ``default_cwd``.

    The record is written after every completed command of THIS session, so
    it is the session's ``cd`` state with no shared-env ambiguity. On
    container backends a recorded HOST path (a desktop/TUI surface registering
    its workspace) is unusable in the sandbox — ``cd <host path>`` fails with
    exit 126 — so it is discarded in favor of ``default_cwd``.

    Same guard class as the env-creation sanitizers (#50636, #54447); this is the per-command sibling site.
    """
    if workdir:
        return workdir
    recorded = get_session_cwd(
        session_key, target=target, _resolution=_resolution,
    )
    if recorded and _is_container_backend(env_type) and _is_unusable_container_cwd(recorded):
        logger.info(
            "Ignoring recorded session cwd %r for %s backend "
            "(host/relative path won't work in sandbox). Using %r instead.",
            recorded, env_type, default_cwd,
        )
        return default_cwd
    return recorded or default_cwd


def _error_json(error: str, *, exit_code: int = -1, status: Optional[str] = None, **extra) -> str:
    """The terminal error envelope: ``output``/``exit_code``/``error`` (+ ``status``, extras)."""
    body: Dict[str, Any] = {"output": "", "exit_code": exit_code, "error": error}
    if status is not None:
        body["status"] = status
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def _fatal_error_json(e: BaseException) -> str:
    """Log the traceback and return the redacted error+traceback envelope.

    Exception text can embed the failing command line (and any secrets inline
    in it), so both fields are force-redacted before reaching the model.
    """
    import traceback
    tb_str = traceback.format_exc()
    logger.error("terminal_tool exception:\n%s", tb_str)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
        "traceback": _redact_terminal_error_text(tb_str),
        "status": "error"
    }, ensure_ascii=False)


class _Rejected(Exception):
    """Carries a finished tool-result JSON out of the planning/guard helpers, so
    each early-return site is one ``raise`` instead of an isinstance-checked
    ``str | plan`` union at the caller."""

    def __init__(self, result_json: str):
        super().__init__(result_json)
        self.result_json = result_json


@dataclass
class _ApprovalVerdict:
    """Outcome of the pre-exec guard pass.

    ``note`` is the audit note attached to the result. ``approved_run`` is True
    when the user explicitly approved (or pre-confirmed via ``force``); it drives
    the clean-interrupt-slate clear before ``env.execute`` so an approved command
    can't be SIGINT-killed by a bit that landed during the approval-wait.
    """
    note: Optional[str] = None
    approved_run: bool = False


def _run_approval_guards(command: str, env_type: str, config: Dict[str, Any], *, force: bool) -> _ApprovalVerdict:
    """Run tirith + dangerous-command guards; ``force`` skips them entirely.
    Raises :class:`_Rejected` when the command may not run (denied, or pending
    gateway approval)."""
    if force:
        return _ApprovalVerdict(approved_run=True)
    approval = _check_all_guards(command, env_type, has_host_access=_docker_has_host_access(config))
    if not approval["approved"]:
        if approval.get("status") == "pending_approval":  # gateway ask mode
            raise _Rejected(_error_json(
                "", status="pending_approval",
                approval_pending=True,
                command=approval.get("command", command),
                description=approval.get("description", "command flagged"),
                pattern_key=approval.get("pattern_key", ""),
                smart_denied=approval.get("smart_denied", False),
                allow_permanent=approval.get("allow_permanent", True),
            ))
        desc = approval.get("description", "command flagged")
        fallback_msg = (
            f"Command denied: {desc}. "
            "Use the approval prompt to allow it, or rephrase the command."
        )
        raise _Rejected(_error_json(approval.get("message", fallback_msg), status="blocked"))
    desc = approval.get("description", "flagged as dangerous")
    if approval.get("user_approved"):
        return _ApprovalVerdict(
            note=f"Command required approval ({desc}) and was approved by the user.",
            approved_run=True,
        )
    if approval.get("smart_approved"):
        return _ApprovalVerdict(note=f"Command was flagged ({desc}) and auto-approved by smart approval.")
    return _ApprovalVerdict()


@dataclass
class _ExecPlan:
    """Per-call execution parameters resolved before any environment is touched."""
    config: Dict[str, Any]
    env_type: str
    effective_task_id: str
    image: str
    cwd: str
    host_cwd: Optional[str]
    effective_timeout: int
    # Set when a foreground call asked for more than FOREGROUND_MAX_TIMEOUT and was promoted to a
    # tracked background process instead of being refused (the requested seconds, for the note).
    promoted_from_foreground_timeout: Optional[int] = None


_PROMOTED_NOTE = (
    "Requested foreground timeout {requested}s exceeds the {cap}s cap, so this command was started as a "
    "tracked background process with notify_on_complete=true instead of being refused. Do NOT re-run it. "
    "Its completion (exit code + output tail) arrives as a notification; poll with "
    "process(action=\"poll\", session_id=...) if you need it sooner."
)


def _plan_execution(
    command: Any, *, task_id: Optional[str], timeout: Optional[int],
    background: bool, _host_local: bool,
) -> _ExecPlan:
    """Resolve backend, env-cache key, image, cwd and timeout for one call.

    Raises :class:`_Rejected` when the call is rejected up front (non-string
    command, non-positive or over-cap timeout, a foreground command that must
    run in the background).
    """
    if not isinstance(command, str):
        logger.warning("Rejected invalid terminal command value: %s", type(command).__name__)
        raise _Rejected(_error_json(
            f"Invalid command: expected string, got {type(command).__name__}", status="error",
        ))

    config = _get_env_config()
    env_type = "local" if _host_local else config["env_type"]

    # Fail closed under a refusal scope: the routed profile's terminal
    # policy could not be resolved, so running with the launch process's
    # ambient policy is forbidden.
    # See #68559.
    if not _host_local:
        from tools.terminal_scope import enforce_no_refusal

        enforce_no_refusal()

    effective_task_id = _resolve_container_task_id(task_id)
    if _host_local:
        # Control-plane children run beside this interpreter, never inside
        # the configured Docker/SSH backend; keep their env cache separate.
        effective_task_id = f"host-local-{effective_task_id}"

    # Per-task overrides (RL/benchmark envs, ACP workspace cwd) win over
    # the global env-var config; ``resolve_task_overrides`` reads the raw
    # task id first, then the collapsed container id.
    overrides = resolve_task_overrides(task_id)
    image = _select_image(env_type, overrides, config)

    cwd = overrides.get("cwd") or get_session_cwd(task_id) or config["cwd"]
    host_cwd = _resolve_task_host_cwd(config, task_id)
    # config["cwd"] was sanitized for container backends in _get_env_config
    # but an override / session record is raw: a host path would reach
    # `docker run -w` and fail with exit 125. Re-apply the guard to the
    # resolved cwd; when the host path IS this session's mounted workspace,
    # remap to /workspace instead of discarding it.
    if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
        remapped = "/workspace" if host_cwd else config["cwd"]
        if cwd != remapped:
            logger.info(
                "Remapping host/relative cwd override %r for %s backend "
                "(won't exist in sandbox). Using %r instead.",
                cwd, env_type, remapped,
            )
        cwd = remapped
    # Reject non-positive timeouts before deadline math: ``timeout or
    # default`` would silently turn 0 into the default, and a negative
    # value is truthy and would fire an immediate "-Ns" timeout.
    if timeout is not None and timeout <= 0:
        raise _Rejected(tool_error(f"timeout must be a positive number of seconds (got {timeout})."))
    promoted = None
    if not background:
        # An over-cap foreground timeout is a bounded job the caller wants to wait for (test suites,
        # builds). Refusing it only bought a mechanical retry: 454 refusals in one run, every one
        # re-sent lower/split/background. Promote to a tracked background process instead; the
        # caller is told in the result. The `&`/nohup/server guidance below stays a refusal: those
        # need the command itself rewritten, which the tool cannot do safely.
        # The detachment guidance applies whether or not the call is promoted: a promoted `cmd &`
        # would start a tracked shell that exits at once while its payload runs untracked.
        guidance = _foreground_background_guidance(command)
        if guidance:
            raise _Rejected(_error_json(guidance, status="error"))
        if timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            promoted = timeout

    return _ExecPlan(
        config=config, env_type=env_type, effective_task_id=effective_task_id,
        image=image, cwd=cwd, host_cwd=host_cwd, effective_timeout=timeout or config["timeout"],
        promoted_from_foreground_timeout=promoted,
    )


_PROMOTED_NOTE_POLL_ONLY = (
    "Requested foreground timeout {requested}s exceeds the {cap}s cap, so this command was started as a "
    "tracked background process instead of being refused. Do NOT re-run it. This session cannot receive "
    "completion notifications, so poll it with process(action=\"poll\", session_id=...) until it exits."
)


def _with_promoted_note(result_json: str, requested_timeout: int) -> str:
    """Attach the foreground->background promotion note to a spawn result (unchanged on error). The
    note only promises a notification when the spawn actually kept notify_on_complete (finite sessions
    such as one-shot runners cannot route one back; the spawn already said so and cleared the flag)."""
    try:
        data = json.loads(result_json)
    except (TypeError, ValueError):
        return result_json
    if not isinstance(data, dict) or data.get("error"):
        return result_json
    template = _PROMOTED_NOTE if data.get("notify_on_complete") else _PROMOTED_NOTE_POLL_ONLY
    data["promoted_from_foreground"] = template.format(requested=requested_timeout, cap=FOREGROUND_MAX_TIMEOUT)
    return json.dumps(data, ensure_ascii=False)


def _acquire_env(plan: _ExecPlan, task_id: Optional[str]) -> Any:
    """Cached env for the task, else create it under the per-task creation lock.

    Concurrent calls for the same task_id wait for the first sandbox instead
    of each creating their own; the cache is re-checked under that lock.
    Raises :class:`_Rejected` with the ``"disabled"`` envelope when creation
    raises ImportError.
    """
    _start_cleanup_thread()
    env_type, eff = plan.env_type, plan.effective_task_id

    with _env_lock:
        env: Any = _lookup_active_env(eff, task_id)
    if env is not None:
        return env

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(eff, threading.Lock())

    with task_lock:
        with _env_lock:
            env = _lookup_active_env(eff, task_id)
        if env is not None:
            return env

        if env_type == "singularity":
            _check_disk_usage_warning()
        logger.info("Creating new %s environment for task %s...", env_type, eff[:8])
        try:
            new_env = _create_configured_env(
                plan.config, env_type, image=plan.image, cwd=plan.cwd,
                timeout=plan.effective_timeout, task_id=eff, host_cwd=plan.host_cwd,
                local_config=(
                    {"persistent": plan.config.get("local_persistent", False)}
                    if env_type == "local" else None
                ),
            )
        except ImportError as e:
            raise _Rejected(_error_json(
                _redact_terminal_error_text(f"Terminal tool disabled: environment creation failed ({e})"),
                status="disabled",
            ))

        with _env_lock:
            _active_environments[eff] = new_env
            _last_activity[eff] = time.time()
        logger.info("%s environment ready for task %s", env_type, eff[:8])
        return new_env


def _yield_kwargs(command: str, **ctx) -> dict:
    """``env.execute`` kwargs enabling yield-to-background (local backend only)."""
    handler = yield_to_background_handler(command=command, **ctx)
    return {"yield_handler": handler} if handler is not None else {}


def _run_foreground(
    command: str, env: Any, plan: _ExecPlan, *,
    task_id: Optional[str], session_id: Optional[str], session_key: str,
    workdir: Optional[str], approval_note: Optional[str], clear_interrupt: bool,
) -> str:
    """Execute in the foreground with retry on transient errors, then finalize."""
    max_retries = 3
    env_type, eff, effective_timeout = plan.env_type, plan.effective_task_id, plan.effective_timeout

    # Clean interrupt slate for an approved command, ONCE before the retry
    # loop: drop a stale bit that landed during the approval-wait so it
    # can't SIGINT the just-approved run. Do NOT re-clear inside the loop —
    # a genuine interrupt during the backoff sleep must survive and abort
    # the next attempt (rc 130).
    if clear_interrupt:
        from tools.interrupt import clear_current_thread_interrupt
        clear_current_thread_interrupt()

    for retry_count in range(max_retries + 1):
        try:
            command_cwd = _resolve_command_cwd(
                workdir=workdir, default_cwd=plan.cwd, session_key=session_key, env_type=env_type,
            )
            # bounded_capture: model-facing output keeps a head/tail window
            # while streaming so a verbose command can't OOM the gateway;
            # internal env.execute() consumers stay unbounded.
            result = env.execute(
                command, timeout=effective_timeout, cwd=command_cwd, bounded_capture=True,
                **_yield_kwargs(command, env_type=env_type, cwd=command_cwd, effective_task_id=eff,
                                task_id=task_id, session_key=session_key),
            )
            break
        except Exception as e:
            if "timeout" in str(e).lower():
                return _error_json(f"Command timed out after {effective_timeout} seconds", exit_code=124)
            # Retry on transient errors
            if retry_count < max_retries:
                wait_time = 2 ** (retry_count + 1)
                logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                               wait_time, retry_count + 1, max_retries, _safe_command_preview(command), type(e).__name__, e, eff, env_type)
                time.sleep(wait_time)
                continue
            logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                         max_retries, _safe_command_preview(command), type(e).__name__, e, eff, env_type)
            return _error_json(_redact_terminal_error_text(f"Command execution failed: {type(e).__name__}: {e}"))

    if result.get("yielded_session_id"):
        return json.dumps({
            "output": result.get("output", ""), "exit_code": None, "error": None,
            "status": "yielded_to_background", "session_id": result["yielded_session_id"],
            "pid": result.get("pid"), "notify_on_complete": True, "note": _YIELDED_NOTE,
        }, ensure_ascii=False)
    return finalize_foreground_result(
        command=command, result=result, env=env, env_type=env_type, effective_task_id=eff,
        task_id=task_id, session_id=session_id, session_key=session_key, workdir=workdir,
        command_cwd=command_cwd, approval_note=approval_note,
    )


def _pre_exec_block(
    command: str, *, env: Any, env_type: str, cwd: str,
    workdir: Optional[str], session_key: str,
) -> None:
    """Raise :class:`_Rejected` with the blocked-result JSON when the command must not run.

    Order matters: gateway lifecycle first (protects the running gateway),
    then the dangerous-workdir check, then the self-repo guard (local only).
    """
    blocked = gateway_lifecycle_block(
        command=command, env=env, env_type=env_type, cwd=cwd, workdir=workdir, session_key=session_key,
    )
    if blocked:
        raise _Rejected(blocked)
    if workdir:
        workdir_error = _validate_workdir(workdir)
        if workdir_error:
            logger.warning("Blocked dangerous workdir: %s (command: %s)",
                           workdir[:200], _safe_command_preview(command))
            raise _Rejected(_error_json(workdir_error, status="blocked"))
    if env_type == "local":
        blocked = self_repo_block(command=command, cwd=cwd, workdir=workdir, session_key=session_key)
        if blocked:
            raise _Rejected(blocked)


_PTY_DISABLED_REASON = (
    "PTY disabled for this command because it expects piped stdin/EOF "
    "(for example gh auth login --with-token). For local background "
    "processes, call process(action='close') after writing so it receives "
    "EOF."
)


def _degraded_result(e: EnvironmentConnectionError, task_id: Optional[str]) -> str:
    """Infrastructure failure (SSH host down, Docker daemon unreachable), distinct
    from a nonzero exit. ``terminal.degraded_mode``: warn (default) returns a
    structured degraded result with a retry hint; fail preserves the historical
    error+traceback result."""
    if _tenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower() == "fail":
        return _fatal_error_json(e)
    logger.warning("terminal backend degraded: %s", e.reason)
    # Evict the possibly-broken backend so the next call re-creates it.
    with _quiet("degraded-env eviction failed"):
        _evict_environment_for_task(task_id)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "status": "degraded",
        "reason": e.reason,
        "retry_hint": e.retry_hint,
        "error": f"Terminal backend degraded: {e.reason}",
    }, ensure_ascii=False)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
    _host_local: bool = False,
    target: Optional[str] = None,
) -> str:
    """
    Execute a command in the configured terminal environment.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)
        session_id: Conversation/session identifier for durable observability
        force: If True, skip dangerous command check (use after user confirms)
        workdir: Working directory for this command (optional, uses session cwd if not set)
        pty: If True, use pseudo-terminal for interactive CLI tools (local backend only)
        notify_on_complete: If True and background=True, you'll be notified exactly once when the process exits. The right choice for almost every long task. MUTUALLY EXCLUSIVE with watch_patterns.
        watch_patterns: List of strings to watch for in background output. HARD rate limit: 1 notification per 15s per process. After 3 strike windows in a row — or after a small lifetime cap of delivered matches, however cleanly spaced — watch_patterns is disabled and the session is auto-promoted to notify_on_complete. Use ONLY for rare, one-shot mid-process signals on long-lived processes (server readiness, migration-done markers). NEVER use in loops/batch jobs — error patterns there will hit the strike limit and get disabled. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both.
        target: Named execution target. Omit to use the configured default.

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = terminal_tool(command="long_task.sh", timeout=300)
        
        # Force run after user confirmation
        # Note: force parameter is internal only, not exposed to model API
    """
    try:
        if not isinstance(command, str):
            logger.warning(
                "Rejected invalid terminal command value: %s",
                type(command).__name__,
            )
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": f"Invalid command: expected string, got {type(command).__name__}",
                "status": "error",
            }, ensure_ascii=False)

        # Resolve configuration per call. Named targets read merged config
        # directly; legacy flat config keeps the existing env-driven path.
        from tools.execution_targets import ExecutionTargetError
        try:
            target_resolution = _target_resolution(target)
        except ExecutionTargetError as exc:
            return json.dumps({
                "output": "", "exit_code": -1, "error": str(exc), "status": "error",
            }, ensure_ascii=False)
        config = (
            _get_env_config(dict(target_resolution.config))
            if target_resolution.named else _get_env_config()
        )
        env_type = "local" if _host_local else config["env_type"]

        # Fail closed under a refusal scope (#68559): the routed profile's terminal
        # policy didn't resolve — refuse with a typed error; never run on ambient policy.
        if not _host_local:
            from tools.terminal_scope import enforce_no_refusal

            enforce_no_refusal()

        # Use task_id for environment isolation: subagent task_ids collapse back to
        # "default" (shared container); only registered RL/benchmark overrides isolate.
        effective_base_task_id = _resolve_container_task_id(task_id)
        if _host_local:
            # Hermes-owned control-plane children run beside this interpreter, never
            # in the model's configured backend — keep their env cache separate.
            effective_base_task_id = f"host-local-{effective_base_task_id}"
        effective_task_id = _environment_scope_key(
            effective_base_task_id, target_resolution,
        )
        raw_environment_key = _environment_scope_key(
            task_id, target_resolution,
        ) if task_id else None
        backend_task_id = target_resolution.backend_task_id(effective_base_task_id)

        # Per-task overrides (TerminalBench2Env, ...) beat global env var config:
        # ``resolve_task_overrides`` reads the raw task id, then the collapsed id.
        overrides = resolve_task_overrides(task_id)
        
        # Select image based on env type, with per-task override support
        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        elif env_type == "daytona":
            image = overrides.get("daytona_image") or config["daytona_image"]
        else:
            image = ""

        cwd_override = (
            overrides.get("cwd")
            if (
                not target_resolution.named
                or (
                    target_resolution.is_default
                    and target_resolution.backend != "ssh"
                )
            )
            else None
        )
        cwd = cwd_override or get_session_cwd(
            task_id, _resolution=target_resolution,
        ) or config["cwd"]
        cwd = _apply_task_cwd_override(config, cwd, cwd_override)
        # Session-scoped mount resolution (single owner: _resolve_task_host_cwd):
        # a fresh session must not inherit a previous session's TERMINAL_CWD mount.
        host_cwd = _resolve_task_host_cwd(config, task_id)
        # A raw per-task cwd override bypasses the sanitizing applied to config["cwd"];
        # re-apply the host-path guard (raw host path reaches `docker run -w`, exit 125).
        # Remap this session's mounted workspace to /workspace; valid paths pass through.
        if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
            remapped = "/workspace" if host_cwd else config["cwd"]
            if cwd != remapped:
                logger.info(
                    "Remapping host/relative cwd override %r for %s backend "
                    "(won't exist in sandbox). Using %r instead.",
                    cwd, env_type, remapped,
                )
            cwd = remapped
        default_timeout = config["timeout"]

        # Validate an explicit timeout before it flows into deadline math: ``timeout or
        # default`` turns 0 into the default (0 can't mean "no timeout"), and a negative
        # value is truthy — it would fire an immediate, nonsensical "-Ns" timeout.
        if timeout is not None and timeout <= 0:
            return tool_error(
                f"timeout must be a positive number of seconds (got {timeout})."
            )
        effective_timeout = timeout or default_timeout

        # An over-cap foreground timeout is a bounded job the caller wants to wait
        # for (tests, builds): promote to a tracked background process instead of
        # refusing — 454 refusals in one run were all mechanically re-sent. The
        # `&`/nohup/server guidance still rejects; it applies even when promoted.
        promoted_from_foreground_timeout = None

        # Guardrail: long-lived server/watch commands belong in managed background sessions.
        if not background:
            guidance = _foreground_background_guidance(command)
            if guidance:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": guidance,
                    "status": "error",
                }, ensure_ascii=False)
            if timeout and timeout > FOREGROUND_MAX_TIMEOUT:
                promoted_from_foreground_timeout = timeout

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment. Per-task creation lock: concurrent calls for the
        # same task_id wait for the first sandbox instead of each creating one.
        env: Any = None
        with _env_lock:
            # Prefer the collapsed container id, then an env cached under the raw
            # task_id: a per-session surface (ACP/gateway/dashboard) with a CWD-only
            # override collapses to "default", so an existing env must be honored.
            _existing_key = (
                effective_task_id if effective_task_id in _active_environments
                else (raw_environment_key if raw_environment_key in _active_environments else None)
            )
            if (
                _existing_key is not None
                and _environment_matches_target(
                    _active_environments[_existing_key], target_resolution,
                )
            ):
                _last_activity[_existing_key] = time.time()
                env = _active_environments[_existing_key]
                needs_creation = False
            else:
                needs_creation = True

        if needs_creation:
            # Per-task lock: only one thread creates the sandbox, others wait
            with _creation_locks_lock:
                if effective_task_id not in _creation_locks:
                    _creation_locks[effective_task_id] = threading.Lock()
                task_lock = _creation_locks[effective_task_id]

            with task_lock:
                # Double-check after acquiring the per-task lock
                existing_env = None
                existing_key = effective_task_id
                with _env_lock:
                    _existing_key = (
                        effective_task_id if effective_task_id in _active_environments
                        else (raw_environment_key if raw_environment_key in _active_environments else None)
                    )
                    if (
                        _existing_key is not None
                        and _environment_matches_target(
                            _active_environments[_existing_key], target_resolution,
                        )
                    ):
                        _last_activity[_existing_key] = time.time()
                        env = _active_environments[_existing_key]
                        needs_creation = False
                    elif _existing_key is not None:
                        existing_env = _active_environments[_existing_key]
                        existing_key = _existing_key

                try:
                    _prepare_environment_replacement(
                        existing_env,
                        existing_key,
                        target_name=target_resolution.target,
                    )
                except _EnvironmentReplacementError as exc:
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": str(exc),
                        "status": "error",
                    }, ensure_ascii=False)

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id)
                    try:
                        container_config, ssh_config, local_config = (
                            _build_environment_constructor_configs(
                                config, target_resolution, effective_base_task_id,
                            )
                        )

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config,
                            container_config=container_config,
                            local_config=local_config,
                            task_id=backend_task_id,
                            host_cwd=host_cwd,
                        )
                        _record_environment_lifetime(new_env, config)
                        _record_environment_target(new_env, target_resolution)
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": _redact_terminal_error_text(
                                f"Terminal tool disabled: environment creation failed ({e})"
                            ),
                            "status": "disabled"
                        }, ensure_ascii=False)

                    publish_error = None
                    with _env_lock:
                        if target_resolution.named:
                            try:
                                from tools.execution_targets import (
                                    execution_target_config_is_frozen,
                                    resolve_live_execution_target,
                                )

                                live_resolution = (
                                    target_resolution
                                    if execution_target_config_is_frozen()
                                    else resolve_live_execution_target(target)
                                )
                            except Exception as exc:
                                publish_error = str(exc)
                            else:
                                if (
                                    live_resolution.security_scope
                                    != target_resolution.security_scope
                                ):
                                    publish_error = (
                                        f"Execution target {target_resolution.target!r} "
                                        "changed while its environment was being created."
                                    )
                        replaced_envs = []
                        if publish_error is None:
                            current = _active_environments.get(effective_task_id)
                            if current is not None and current is not new_env:
                                replaced_envs.append((effective_task_id, current))
                            if (
                                raw_environment_key is not None
                                and raw_environment_key != effective_task_id
                            ):
                                raw_env = _active_environments.get(raw_environment_key)
                                if (
                                    raw_env is not None
                                    and not _environment_matches_target(
                                        raw_env, target_resolution,
                                    )
                                ):
                                    _active_environments.pop(raw_environment_key, None)
                                    _last_activity.pop(raw_environment_key, None)
                                    replaced_envs.append((raw_environment_key, raw_env))
                            _active_environments[effective_task_id] = new_env
                            _last_activity[effective_task_id] = time.time()
                            env = new_env
                    if publish_error is not None:
                        _cleanup_environment_resource(
                            new_env,
                            force_remove=True,
                            preserve_storage=_environment_has_stable_storage(new_env),
                        )
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": publish_error + " Retry the command.",
                            "status": "error",
                        }, ensure_ascii=False)
                    for replaced_key, replaced_env in replaced_envs:
                        if replaced_env is not new_env:
                            _retire_replaced_environment(replaced_env, replaced_key)
                    logger.info("%s environment ready for task %s", env_type, effective_task_id)

        assert env is not None  # all creation failure paths return above

        # Session key for cwd records: get_current_session_key()'s contextvar doesn't
        # cross tool-worker threads, so fall back to raw task_id (thread-safe anchor).
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")

        # Hard-block: gateway lifecycle commands (systemctl/launchctl/hermes
        # restart|stop|uninstall targeting hermes-gateway) must never run inside
        # the gateway process — the restart SIGTERMs this subprocess before it can
        # finish; force=True can't help. Gate on the SUPERVISED-gateway probe: the
        # raw _HERMES_GATEWAY marker leaks into serve/CLI/web-server importers.
        from tools.process_registry import _is_supervised_gateway_process

        if _is_supervised_gateway_process():
            from cron.lifecycle_guard import (
                _MAX_REFERENCED_SCRIPT_BYTES,
                contains_gateway_lifecycle_command_or_referenced_script,
                contains_launchctl_submit_command,
                lifecycle_scan_root_within_budget,
            )
            # Keep the specific launchctl diagnostic when this optional pre-scan
            # fits the budget; the full fail-closed guard below still runs otherwise.
            if (
                lifecycle_scan_root_within_budget(command)
                and contains_launchctl_submit_command(command)
            ):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: launchctl submit/bootstrap registers a persistent "
                        "KeepAlive job and is unsafe from inside the gateway process. "
                        "Use Hermes cron for one-shot delayed work, or install an "
                        "explicit LaunchAgent from a separate shell."
                    ),
                    "status": "error",
                }, ensure_ascii=False)
            selected_target = (
                target_resolution.target if target_resolution.named else None
            )
            guard_cwd_base = get_session_cwd(
                session_key, selected_target, _resolution=target_resolution,
            )
            if guard_cwd_base is None:
                guard_cwd_base = getattr(env, "cwd", None) or cwd
            guard_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=guard_cwd_base,
                session_key=session_key,
                env_type=env_type,
                _resolution=target_resolution,
            )

            def _read_script_in_env(
                script_path: str,
            ) -> Optional[str] | tuple[Optional[str], bool]:
                """Read a script without crossing the selected target boundary.

                Host filesystem reads are allowed only for a local target. Other
                targets, and local-read misses, use the selected environment at
                ``guard_cwd``. All reads are bounded and NUL-bearing binary content
                is skipped before it can re-enter the lifecycle-command scanner.
                """
                if env is None:
                    return None
                if target_resolution.backend == "local":
                    try:
                        from cron.lifecycle_guard import _read_referenced_script

                        local_path = Path(script_path).expanduser()
                        if not local_path.is_absolute():
                            local_path = Path(guard_cwd) / local_path
                        local_result = _read_referenced_script(local_path)
                        if local_result[0] is not None or local_result[1]:
                            return local_result
                    except Exception:
                        return None
                # Remote / sandboxed backend: read via the environment's shell,
                # bounded at the source with `head -c` so an oversized file (a 166MB
                # ELF pinned the tool thread 30+ min on a superlinear shlex scan)
                # never crosses the wire; one byte over budget fails closed in
                # lifecycle_guard. `< path` keeps leading-dash paths out of argv.
                try:
                    result = env.execute(
                        f"head -c {_MAX_REFERENCED_SCRIPT_BYTES + 1} "
                        f"< {shlex.quote(script_path)}",
                        cwd=guard_cwd,
                    )
                    if isinstance(result, dict):
                        returncode = result.get(
                            "returncode", result.get("exit_code", -1)
                        )
                        output = result.get("output", "")
                    else:
                        returncode = getattr(
                            result,
                            "returncode",
                            getattr(result, "exit_code", -1),
                        )
                        output = getattr(result, "output", "")
                    if returncode == 0:
                        if output and "\x00" in output:
                            # Binary content from a remote read: skip for the
                            # same reason as the local branch above (#77703).
                            return None
                        return output
                except Exception:
                    pass
                return None

            if contains_gateway_lifecycle_command_or_referenced_script(
                command,
                cwd=guard_cwd,
                read_remote_script=_read_script_in_env,
            ):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: command or referenced script cannot restart, stop, or "
                        "uninstall the gateway from inside the gateway process. The gateway would "
                        "kill this command before it could complete (SIGTERM propagates "
                        "to child processes). Run `hermes gateway restart` from a "
                        "separate shell outside the running gateway."
                    ),
                    "status": "error",
                }, ensure_ascii=False)

        # Validate before the source guard resolves an explicit workdir.
        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning("Blocked dangerous workdir: %s (command: %s)",
                               workdir[:200], _safe_command_preview(command))
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": workdir_error,
                    "status": "blocked"
                }, ensure_ascii=False)

        # Windows-only: NTFS locks loaded module files — rewriting the checkout
        # backing this interpreter can corrupt the process. POSIX keeps old inodes.
        if env_type == "local":
            from tools.self_repo_guard import (
                detect_self_repo_git_mutation,
                guard_active,
            )

            guard_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=cwd,
                session_key=session_key,
            )
            _self_repo_hit, _self_repo_msg = (
                detect_self_repo_git_mutation(command, guard_cwd)
                if guard_active()
                else (False, None)
            )
            if _self_repo_hit:
                logger.warning(
                    "Blocked self-repo git mutation (command: %s)",
                    _safe_command_preview(command),
                )
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": _self_repo_msg,
                    "status": "blocked",
                }, ensure_ascii=False)

        # Pre-exec security checks (tirith + dangerous command detection)
        # Skip check if force=True (user has confirmed they want to run it)
        approval_note = None
        # User-approved run: drives the interrupt-slate clear so a stale bit from
        # the approval-wait can't SIGINT it (see clear_current_thread_interrupt).
        _approved_run = bool(force)
        if not force:
            approval = _check_all_guards(
                command, env_type,
                has_host_access=_docker_has_host_access(config),
                execution_target=target_resolution.target,
                execution_backend=target_resolution.backend,
                execution_target_named=target_resolution.named,
                execution_target_scope=(
                    target_resolution.security_scope
                    if target_resolution.named else ""
                ),
            )
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "pending_approval":
                    pending_result = {
                        "output": "",
                        "exit_code": -1,
                        "error": "",
                        "status": "pending_approval",
                        "approval_pending": True,
                        "command": approval.get("command", command),
                        "description": approval.get("description", "command flagged"),
                        "pattern_key": approval.get("pattern_key", ""),
                        "smart_denied": approval.get("smart_denied", False),
                        "allow_permanent": approval.get("allow_permanent", True),
                    }
                    pending_result.update(target_resolution.metadata(
                        cwd=cwd if target_resolution.named else None,
                    ))
                    return json.dumps(pending_result, ensure_ascii=False)
                # Command was blocked
                desc = approval.get("description", "command flagged")
                fallback_msg = (
                    f"Command denied: {desc}. "
                    "Use the approval prompt to allow it, or rephrase the command."
                )
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", fallback_msg),
                    "status": "blocked"
                }, ensure_ascii=False)
            # Track whether approval was explicitly granted by the user
            if approval.get("user_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command required approval ({desc}) and was approved by the user."
                _approved_run = True
            elif approval.get("smart_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command was flagged ({desc}) and auto-approved by smart approval."

        # Prepare command for execution
        pty_disabled_reason = None
        effective_pty = pty
        if pty and _command_requires_pipe_stdin(command):
            effective_pty = False
            pty_disabled_reason = (
                "PTY disabled for this command because it expects piped stdin/EOF "
                "(for example gh auth login --with-token). For local background "
                "processes, call process(action='close') after writing so it receives "
                "EOF."
            )

        # The session key is already computed above the gateway guard.
        if promoted_from_foreground_timeout is not None:
            # Promotion implies notify_on_complete; watch_patterns is a background-only flag the
            # caller could not have meant for a foreground call, and the two are exclusive anyway.
            background, notify_on_complete, watch_patterns = True, True, None
        if background:
            # Spawn a tracked background process via the process registry: local =
            # Popen with buffered output, non-local = env.execute() in the sandbox.
            from tools.process_registry import process_registry

            effective_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=cwd,
                session_key=session_key,
                env_type=env_type,
                _resolution=target_resolution,
            )
            try:
                spawn_metadata = {}
                environment_task_key = str(
                    target_resolution.scope_task_key(effective_base_task_id)
                )
                if target_resolution.named:
                    spawn_metadata = {
                        "target": target_resolution.target,
                        "backend": target_resolution.backend,
                        "timeout_seconds": effective_timeout,
                        "environment_task_key": environment_task_key,
                        "runtime_scope": target_resolution.security_scope,
                    }
                    if env_type == "local":
                        spawn_metadata["env_ref"] = env
                with _scoped_sudo_execution(
                    target_resolution.target,
                    target_resolution.backend,
                    named=target_resolution.named,
                    sudo_password=target_resolution.config.get("sudo_password"),
                    target_scope=(
                        target_resolution.security_scope
                        if target_resolution.named else ""
                    ),
                ):
                    if env_type == "local":
                        proc_session = process_registry.spawn_local(
                            command=command,
                            cwd=effective_cwd,
                            task_id=effective_base_task_id,
                            owner_task_id=task_id or effective_base_task_id,
                            session_key=session_key,
                            env_vars=env.env if hasattr(env, 'env') else None,
                            use_pty=effective_pty,
                            **spawn_metadata,
                        )
                    else:
                        proc_session = process_registry.spawn_via_env(
                            env=env,
                            command=command,
                            cwd=effective_cwd,
                            task_id=effective_base_task_id,
                            owner_task_id=task_id or effective_base_task_id,
                            session_key=session_key,
                            timeout=effective_timeout,
                            **spawn_metadata,
                        )


                # Preserve the exact legacy spawn call signature while still
                # attaching additive metadata to subsequent process results.
                if not target_resolution.named:
                    proc_session.target = target_resolution.target
                    proc_session.backend = target_resolution.backend
                    proc_session.timeout_seconds = effective_timeout
                    proc_session.environment_task_key = environment_task_key
                    checkpoint = getattr(process_registry, "_write_checkpoint", None)
                    if callable(checkpoint):
                        checkpoint()

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }
                result_data.update(target_resolution.metadata(
                    cwd=effective_cwd if target_resolution.named else None,
                ))
                # Background spawns detach (exit_code 0 immediately) and never poll
                # is_interrupted(), so this note never co-occurs with rc=130.
                if approval_note:
                    result_data["approval"] = approval_note
                if pty_disabled_reason:
                    result_data["pty_note"] = pty_disabled_reason

                # Nudge: background=True without notify_on_complete/watch_patterns is a
                # silent process — correct for never-exiting servers, almost never for
                # bounded tasks (tests/builds/CI pollers). May 2026 PR #31231: the agent
                # never noticed a green exit. Cheap nudge + explicit flag beats silent
                # blindness.
                if background and not notify_on_complete and not watch_patterns:
                    result_data["hint"] = (
                        "background=true without notify_on_complete=true means "
                        "this process runs SILENTLY — you will not be told when "
                        "it exits. If this is a bounded task (test suite, build, "
                        "CI poller, deploy, anything with a defined end), you "
                        "almost certainly wanted notify_on_complete=true so the "
                        "system pings you on exit. Re-launch with "
                        "notify_on_complete=true, or call process(action='poll') "
                        "/ process(action='wait') yourself to learn the outcome. "
                        "Only ignore this hint for genuine long-lived processes "
                        "that never exit (servers, watchers, daemons)."
                    )

                # Nudge: homebrewed CI watchers (`gh pr view --json statusCheckRollup`,
                # `gh pr checks | jq`) are the #1 cause of silent CI-watcher failures
                # (May 2026 PRs #31329…#33131): jq null-conclusion loops, buffered
                # stdout lost in bg capture, conclusion-vs-status confusion, TTY-only
                # banner greps. Deliberately narrow: never the column-2 awk poller.
                if background and command:
                    _gh = ("gh pr view" in command or "gh pr checks" in command)
                    _has_jq = (
                        " jq " in command or "| jq" in command or "$(jq" in command
                    )
                    _bad_shape = (
                        # JSON-API anti-pattern: `--json statusCheckRollup` + parsing is
                        # conclusion-vs-status field hell, jq or not.
                        "statusCheckRollup" in command
                        # `gh pr checks` doesn't emit JSON, so any `| jq` here is confused
                        # intent — the canonical column-2 poller uses awk-on-tabs, not jq.
                        or (_gh and _has_jq)
                    )
                    if _bad_shape:
                        existing = result_data.get("hint", "")
                        canonical_hint = (
                            "This looks like a homebrewed CI poller built from "
                            "`gh pr view --json statusCheckRollup` and/or "
                            "`gh pr checks | jq`. That shape has burned us "
                            "repeatedly in hermes-agent dev work (PRs #31329, "
                            "#31448, #31695, #31709, #31745, #32264, #33131) — "
                            "stdout buffering kills output capture, jq null-key "
                            "edge cases silently exit the loop, conclusion-vs-"
                            "status field confusion exits early with bogus "
                            "all-green verdicts, TTY-only summary banners "
                            "never appear when piped. Use the canonical "
                            "snippets in the green-ci-policy skill instead: "
                            "the exit-code-driven `gh pr checks $PR >/dev/null` "
                            "(rc 0 = green, 8 = pending, else fail) for "
                            "exit-on-first-fail behavior, or the column-2 "
                            "awk-on-tabs poller "
                            "(`awk -F\"\\t\" \"$2==\\\"pending\\\"\"`) for "
                            "sharded matrices. Load skill_view("
                            "name='github/hermes-agent-dev', "
                            "file_path='references/green-ci-policy.md') for "
                            "the verbatim snippets. If you must roll a custom "
                            "loop with rich structured output, write each tick "
                            "to a known file (`tee -a /tmp/ci.log`) and rely "
                            "on `process(action='log')` to read THAT file — "
                            "do not rely on background-process stdout capture "
                            "for line-buffered shell loops."
                        )
                        result_data["hint"] = (
                            existing + "\n\n" + canonical_hint if existing
                            else canonical_hint
                        )

                # Routing metadata so watch-pattern/completion notifications reach the right chat/thread.
                if background and (notify_on_complete or watch_patterns):
                    from gateway.session_context import (
                        async_delivery_supported as _async_ok,
                        get_session_env as _gse,
                    )

                    # Finite sessions (stateless HTTP, one-shot Kanban workers) can't
                    # route a completion after the turn ends: drop the flags, say poll.
                    if not _async_ok():
                        notify_on_complete = False
                        watch_patterns = None
                        result_data["notify_on_complete"] = False
                        result_data["notify_unsupported"] = (
                            "notify_on_complete / watch_patterns are not available in "
                            "this session — it cannot receive an async completion after "
                            "the turn ends (a one-shot runner such as `hermes -z`, a "
                            "cron job, a Kanban worker, or a stateless HTTP endpoint). "
                            "The process is "
                            "running in the background; retrieve its result with "
                            "process(action='poll') or process(action='wait')."
                        )
                        logger.info(
                            "background proc %s: async delivery unsupported on this "
                            "session; notify_on_complete/watch_patterns disabled",
                            proc_session.id,
                        )
                    else:
                        _gw_platform = _gse("HERMES_SESSION_PLATFORM", "")
                        if _gw_platform:
                            _gw_chat_id = _gse("HERMES_SESSION_CHAT_ID", "")
                            _gw_thread_id = _gse("HERMES_SESSION_THREAD_ID", "")
                            _gw_user_id = _gse("HERMES_SESSION_USER_ID", "")
                            _gw_user_name = _gse("HERMES_SESSION_USER_NAME", "")
                            _gw_message_id = _gse("HERMES_SESSION_MESSAGE_ID", "")
                            proc_session.watcher_platform = _gw_platform
                            proc_session.watcher_chat_id = _gw_chat_id
                            proc_session.watcher_user_id = _gw_user_id
                            proc_session.watcher_user_name = _gw_user_name
                            proc_session.watcher_thread_id = _gw_thread_id
                            proc_session.watcher_message_id = _gw_message_id
                            # Stamp the spawning session's db id so the completion
                            # pre-flight (_classify_completion_target) drops the
                            # notification after /new instead of hitting the new chat.
                            proc_session.parent_session_id = _gse(
                                "HERMES_SESSION_ID", ""
                            )

                # Mutual exclusion: both set → drop watch_patterns — the pair yields
                # duplicate async notifications (per match + on exit) that can spam long
                # after exit. notify_on_complete is the better finish signal.
                watch_patterns, conflict_note = _resolve_notification_flag_conflict(
                    notify_on_complete=bool(notify_on_complete),
                    watch_patterns=watch_patterns,
                    background=bool(background),
                )
                if conflict_note:
                    logger.warning("background proc %s: %s", proc_session.id, conflict_note)
                    result_data["watch_patterns_ignored"] = conflict_note

                # Mark for agent notification on completion
                if notify_on_complete and background:
                    proc_session.notify_on_complete = True
                    result_data["notify_on_complete"] = True

                    # Gateway mode: auto-register a fast watcher so completion can trigger
                    # a new agent turn; CLI mode uses the completion_queue directly.
                    if proc_session.watcher_platform:
                        proc_session.watcher_interval = 5
                        process_registry.pending_watchers.append({
                            "session_id": proc_session.id,
                            "check_interval": 5,
                            "session_key": session_key,
                            "platform": proc_session.watcher_platform,
                            "chat_id": proc_session.watcher_chat_id,
                            "user_id": proc_session.watcher_user_id,
                            "user_name": proc_session.watcher_user_name,
                            "thread_id": proc_session.watcher_thread_id,
                            "message_id": proc_session.watcher_message_id,
                            "notify_on_complete": True,
                            "parent_session_id": proc_session.parent_session_id,
                        })

                # Set watch patterns for output monitoring
                if watch_patterns and background:
                    proc_session.watch_patterns = list(watch_patterns)
                    result_data["watch_patterns"] = proc_session.watch_patterns

                if promoted_from_foreground_timeout is not None:
                    return _with_promoted_note(
                        json.dumps(result_data, ensure_ascii=False),
                        promoted_from_foreground_timeout,
                    )
                return json.dumps(result_data, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": _redact_terminal_error_text(
                        f"Failed to start background process: {e}"
                    )
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            command_cwd = None

            # Clean interrupt slate for an approved command, ONCE before the retry
            # loop: drop the stale approval-wait bit so it can't SIGINT the run. Do NOT
            # re-clear in the loop — a genuine interrupt must survive to abort (-> 130).
            if _approved_run:
                from tools.interrupt import clear_current_thread_interrupt
                clear_current_thread_interrupt()

            while retry_count <= max_retries:
                try:
                    command_cwd = _resolve_command_cwd(
                        workdir=workdir,
                        default_cwd=cwd,
                        session_key=session_key,
                        env_type=env_type,
                        _resolution=target_resolution,
                    )
                    execute_kwargs = {
                        "timeout": effective_timeout,
                        "cwd": command_cwd,
                        # Foreground model-facing output: cap retention while streaming
                        # (head/tail window) so a verbose command can't OOM the gateway
                        # before truncation (#64435). Internal env.execute() stays unbounded.
                        "bounded_capture": True,
                        **_yield_kwargs(
                            command, env_type=env_type, cwd=command_cwd,
                            effective_task_id=effective_task_id, task_id=task_id,
                            session_key=session_key,
                        ),
                    }
                    with _scoped_sudo_execution(
                        target_resolution.target,
                        target_resolution.backend,
                        named=target_resolution.named,
                        sudo_password=target_resolution.config.get("sudo_password"),
                        target_scope=(
                            target_resolution.security_scope
                            if target_resolution.named else ""
                        ),
                    ):
                        result = env.execute(command, **execute_kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        return json.dumps({
                            "output": "",
                            "exit_code": 124,
                            "error": f"Command timed out after {effective_timeout} seconds"
                        }, ensure_ascii=False)
                    
                    # Retry on transient errors
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                       wait_time, retry_count, max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                 max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": _redact_terminal_error_text(
                            f"Command execution failed: {type(e).__name__}: {e}"
                        )
                    }, ensure_ascii=False)
                
                # Got a result
                break

            if (result or {}).get("yielded_session_id"):
                # Redirected mid-command: the process now runs as a registry-tracked
                # notify-on-complete session — return without killing it or post-processing.
                return json.dumps({
                    "output": result.get("output", ""), "exit_code": None, "error": None,
                    "status": "yielded_to_background", "session_id": result["yielded_session_id"],
                    "pid": result.get("pid"), "notify_on_complete": True, "note": _YIELDED_NOTE,
                }, ensure_ascii=False)

            # Dual-write (cwd rearch step 1): record the env's post-command cwd
            # under the session key so the durable record never depends on the
            # shared env surviving. Skip when a transient per-command ``workdir``
            # override is set, and when the command reported no cwd (interrupted/
            # killed: env.cwd may hold another session's dir — silent re-homing).
            observed_cwd = None
            if (result or {}).get("cwd_observed"):
                # New/current environments return the CWD observed by THIS command;
                # env.cwd is shared mutable compat state that may belong to a concurrent
                # command, so keep the fallback for providers on the older contract.
                observed_cwd = (result or {}).get("cwd") or getattr(env, "cwd", None)
            if not workdir and observed_cwd:
                record_session_cwd(
                    session_key, observed_cwd,
                    _resolution=target_resolution,
                )

            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)
            # Spill metadata from the bounded collector: present only when
            # output overflowed the capture window (see _wait_for_process).
            spill_total_chars = result.get("output_total_chars")
            spill_file_path = result.get("full_output_path")

            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)

            sudo_auth_failed = _sudo_wrong_password_failure(output)
            sudo_cache_cleared = _invalidate_cached_sudo_on_auth_failure(
                command,
                output,
                target_resolution.target,
                target_resolution.backend,
                (
                    target_resolution.security_scope
                    if target_resolution.named else ""
                ),
            )
            if sudo_cache_cleared:
                has_sudo_prompt_callback = _get_sudo_password_callback() is not None
                can_reprompt = (
                    has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE")
                ) and not _in_delegated_child_context()
                if can_reprompt:
                    output += (
                        "\n\n⚠️ Sudo authentication failed — cached password "
                        "cleared. You will be prompted again on the next sudo "
                        "command."
                    )

            # Foreground output canonicalization seam: BaseEnvironment already bounded
            # the capture; plugins may replace that string (fail-open, first valid
            # return wins), still subject to the final output limit below.
            try:
                from hermes_cli.lifecycle import invoke_hook
                hook_kwargs = {
                    "command": command,
                    "output": output,
                    "returncode": returncode,
                    "task_id": effective_base_task_id or "",
                    "env_type": env_type,
                }
                if target_resolution.named:
                    hook_kwargs.update({
                        "execution_target": target_resolution.target,
                        "execution_backend": target_resolution.backend,
                    })
                hook_results = invoke_hook(
                    "transform_terminal_output",
                    **hook_kwargs,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        output = hook_result
                        break
            except Exception:
                pass
            
            # Truncate output if too long, keeping both head and tail
            from tools.tool_output_limits import get_max_bytes
            MAX_OUTPUT_CHARS = get_max_bytes()
            if len(output) > MAX_OUTPUT_CHARS:
                head_chars = int(MAX_OUTPUT_CHARS * 0.4)  # 40% head (error messages often appear early)
                tail_chars = MAX_OUTPUT_CHARS - head_chars  # 60% tail (most recent/relevant output)
                omitted = len(output) - head_chars - tail_chars
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
                    f"out of {len(output)} total] ...\n\n"
                )
                output = output[:head_chars] + truncated_notice + output[-tail_chars:]

            # Strip ANSI escape sequences so the model never sees terminal
            # formatting — prevents it from copying escapes into file writes.
            from tools.ansi_strip import strip_ansi
            output = strip_ansi(output)

            # Redact secrets from command output: source/config dumps (MAX_TOKENS=100,
            # "apiKey" fixtures, postgresql:// f-strings) skip the ENV/JSON/template
            # passes (code_file=True) to avoid false positives; env-dump commands
            # (env/printenv/set/export/declare) DO run the ENV pass (code_file=False) —
            # a KEY=value credential dump. See issue #43025; real prefixes mask both.
            from agent.redact import redact_terminal_output
            output = redact_terminal_output(output.strip(), command) if output else ""

            # Interpret non-zero exit codes that aren't real errors
            # (e.g. grep=1 means "no matches", diff=1 means "files differ")
            exit_note = _interpret_exit_code(command, returncode)

            # Output-pattern failure hints: map well-known shapes (module-not-found,
            # gh field drift, merge conflicts) to one hint. See tools/terminal_hints.py.
            failure_hint = None
            if returncode != 0 and not exit_note:
                try:
                    from tools.terminal_hints import annotate_failure
                    failure_hint = annotate_failure(command, returncode, output)
                except Exception:
                    failure_hint = None
            elif returncode == 0:
                # Masked-success backstop: pipelines (`cargo build | tail -20`) return
                # the last command's exit 0 even when the build failed. If the shape can
                # mask an upstream failure and output shows it, warn — advisory only.
                try:
                    from tools.terminal_hints import annotate_masked_success
                    failure_hint = annotate_masked_success(command, output)
                except Exception:
                    failure_hint = None

            result_dict = {
                "output": output,
                "exit_code": returncode,
                "error": None,
            }
            # cwd echo: when the command changed the session's working directory
            # (cd, pushd, ...), tell the model where it ended up — 60% of terminal
            # calls carry defensive 'cd X && ' because cwd is invisible. Gated on
            # the observation flag: without it an interrupted command echoes the
            # shared env's leftover cwd (possibly another session's).
            result_dict.update(target_resolution.metadata(
                cwd=command_cwd if target_resolution.named else None,
            ))
            try:
                post_cwd = observed_cwd
                if post_cwd and command_cwd and os.path.realpath(str(post_cwd)) != os.path.realpath(str(command_cwd)):
                    result_dict["cwd"] = str(post_cwd)
            except Exception:
                pass
            if spill_file_path:
                try:
                    _sp = Path(spill_file_path)
                    raw_spill = _sp.read_text(encoding="utf-8", errors="replace")
                    from tools.spill_safety import write_text_exclusive

                    # Rewrite in place via lstat-checked unlink + exclusive create so the
                    # redacted copy can't be diverted through a planted symlink.
                    write_text_exclusive(
                        _sp,
                        redact_terminal_output(strip_ansi(raw_spill), command),
                        private=True,
                        overwrite=True,
                        errors="replace",
                    )
                    result_dict["output_total_chars"] = spill_total_chars
                    result_dict["full_output_path"] = spill_file_path
                    result_dict["truncation_note"] = (
                        "Output exceeded the capture window (head+tail shown). "
                        f"Full output ({spill_total_chars:,} chars) saved to "
                        f"{spill_file_path} — search it with search_files or page it "
                        "with read_file instead of re-running the command."
                    )
                except Exception:
                    logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
                    try:
                        Path(spill_file_path).unlink()
                    except OSError:
                        pass
            if target_resolution.backend == "local":
                try:
                    from agent.verification_evidence import record_terminal_result

                    evidence = record_terminal_result(
                        command=command,
                        cwd=command_cwd,
                        session_id=(
                            session_id or task_id or backend_task_id or "default"
                        ),
                        exit_code=returncode,
                        output=output,
                    )
                    if evidence:
                        result_dict["verification_evidence"] = {
                            "status": evidence.get("status"),
                            "kind": evidence.get("kind"),
                            "scope": evidence.get("scope"),
                            "canonical_command": evidence.get("canonical_command"),
                        }
                except Exception:
                    logger.debug(
                        "verification evidence recording failed", exc_info=True,
                    )
            if approval_note:
                # Treat rc=130 as an interrupt only when the executor's marker is
                # present — `bash -c 'exit 130'` exits 130 with no marker and must
                # not be relabelled a user interrupt in the audit note.
                if returncode == 130 and "[Command interrupted]" in output:
                    # Interrupted by a genuine Stop: keep the audit trail but never imply
                    # success — "...approved by the user." must not co-occur with rc=130.
                    result_dict["approval"] = approval_note.rstrip(".") + ", then interrupted."
                else:
                    result_dict["approval"] = approval_note
            if exit_note:
                result_dict["exit_code_meaning"] = exit_note
            if failure_hint:
                result_dict["hint"] = failure_hint
            if sudo_auth_failed:
                result_dict["sudo_auth_failed"] = True
            if sudo_cache_cleared:
                result_dict["sudo_cache_cleared"] = True

            return json.dumps(result_dict, ensure_ascii=False)

    except EnvironmentConnectionError as e:
        # Infrastructure/connection-class failure (SSH host down, Docker daemon
        # unreachable), distinct from a command's nonzero exit. ``terminal.degraded_mode``:
        # warn (default) = structured reason+retry hint; fail = error+traceback.
        degraded_mode = _tenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower()
        if degraded_mode == "fail":
            import traceback
            tb_str = traceback.format_exc()
            logger.error("terminal_tool exception:\n%s", tb_str)
            # Exception text can embed the failing command line (and any
            # secrets inline in it) — redact before returning to the model.
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
                "traceback": _redact_terminal_error_text(tb_str),
                "status": "error"
            }, ensure_ascii=False)

        logger.warning("terminal backend degraded: %s", e.reason)
        # Evict a possibly-broken backend so the next call re-creates it (works once reachable).
        try:
            _evict_environment_for_task(task_id)
        except Exception:
            logger.debug("degraded-env eviction failed", exc_info=True)
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "status": "degraded",
            "reason": e.reason,
            "retry_hint": e.retry_hint,
            "error": f"Terminal backend degraded: {e.reason}",
        }, ensure_ascii=False)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error("terminal_tool exception:\n%s", tb_str)
        # Exception text can embed the failing command line (and any
        # secrets inline in it) — redact before returning to the model.
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
            "traceback": _redact_terminal_error_text(tb_str),
            "status": "error"
        }, ensure_ascii=False)
def _check_terminal_config_requirements(config: Dict[str, Any]) -> bool:
    """Check one already-resolved backend configuration."""
    try:
        config = _get_env_config()
        checker = _REQUIREMENT_CHECKERS.get(config["env_type"], _check_plugin_requirements)
        return checker(config)
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        return False


def check_terminal_requirements() -> bool:
    """Keep tools available when any configured execution target is usable."""
    try:
        from tools.execution_targets import list_execution_targets

        inventory = list_execution_targets()
    except Exception as exc:
        # Keep the fail-closed registration contract for invalid config; tool handlers
        # still produce actionable target errors when invoked directly.
        logger.error("Invalid execution target config: %s", exc)
        return False

    if inventory and inventory[0].named:
        # Local targets first so one healthy target avoids a scary startup error from
        # an unavailable Docker daemon or cloud credential elsewhere.
        ordered = sorted(
            inventory,
            key=lambda item: (item.backend != "local", not item.is_default, item.target),
        )
        for resolution in ordered:
            try:
                config = _get_env_config(dict(resolution.config))
            except Exception:
                continue
            if _check_terminal_config_requirements(config):
                return True
        return False

    try:
        return _check_terminal_config_requirements(_get_env_config())
    except Exception as exc:
        logger.error("Invalid terminal configuration: %s", exc)
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("Terminal Tool Module")
    print("=" * 50)
    
    config = _get_env_config()
    print("\nCurrent Configuration:")
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")

    if not check_terminal_requirements():
        print("\n❌ Requirements not met. Please check the messages above.")
        sys.exit(1)

    print("\n✅ All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands in sandboxed environments")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    default_img = "nikolaik/python-nodejs:python3.11-nodejs20"
    print(
        "  TERMINAL_ENV: "
        f"{_tenv('TERMINAL_ENV', 'local')} "
        "(local/docker/singularity/modal/daytona/vercel_sandbox/ssh)"
    )
    print(f"  TERMINAL_DOCKER_IMAGE: {_tenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {_tenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {_tenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_DAYTONA_IMAGE: {_tenv('TERMINAL_DAYTONA_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {_tenv('TERMINAL_CWD', _safe_getcwd())}")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  TERMINAL_SANDBOX_DIR: {_tenv('TERMINAL_SANDBOX_DIR', f'{_dhh()}/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {_tenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {_tenv('TERMINAL_LIFETIME_SECONDS', '300')}")


# ---- Registry ----


from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background, returning a session_id. Pair with notify=true for anything with a defined end (tests, builds, deploys) — without it the process runs silently. Only servers/watchers/daemons that never exit should stay silent. Short commands: prefer foreground with a generous timeout.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. A foreground timeout above {FOREGROUND_MAX_TIMEOUT}s runs the command as a tracked background process with notify_on_complete=true instead (the result says so; do not re-run it).",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "target": {
                "type": "string",
                "description": "Named execution target from terminal.targets (for example 'local' or 'devbox'). Omit to use terminal.default_target; legacy flat config accepts only 'default'.",
            },
            "pty": {
                "type": "boolean",
                "description": "With background=true: run in a pseudo-terminal for interactive CLI tools (Codex, Claude Code, Python REPL). Local backend only. Default: false.",
                "default": False
            },
            "notify": {
                "description": "With background=true: notify=true fires exactly one notification when the process exits (the right choice for nearly every bounded task — builds, tests, deploys). notify=['pattern', ...] instead notifies when a line matches a pattern — ONLY for one-shot readiness signals on processes that never exit (e.g. ['Application startup complete']); rate-limited and auto-disabled if it over-fires. Omit for silent daemons.",
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}}
                ]
            }
            # Legacy aliases (unadvertised, still accepted): notify_on_complete
            # (bool) and watch_patterns (list). notify=true|[...] maps onto
            # them in the dispatch wrapper; explicit notify wins on conflict.
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    # Models sometimes send execute_code's ``code`` here; name the stray
    # argument and the right tool instead of failing on command=None.
    if "command" not in args and "code" in args:
        return tool_error(
            "terminal received a 'code' parameter, but it requires a shell "
            "command in 'command'. Use execute_code(code=...) for Python; "
            "for shell, retry as terminal(command=...)."
        )
    # `notify` is the advertised interface (true → notify_on_complete,
    # [...] → watch_patterns); the legacy args stay accepted, explicit
    # `notify` wins. Background-only modifiers on a foreground call fail
    # with the corrected call instead of being silently ignored.
    notify = args.get("notify")
    notify_on_complete = args.get("notify_on_complete", False)
    watch_patterns = args.get("watch_patterns")
    if not args.get("background", False):
        if notify or watch_patterns or notify_on_complete:
            return tool_error(
                "notify only applies to background commands (foreground "
                "results return directly). Either drop notify, or run as "
                "terminal(command=..., background=true, notify=...)."
            )
        if args.get("pty", False):
            return tool_error(
                "pty requires background=true (a PTY session is interacted "
                "with via process(action='write'/'submit'), which needs a "
                "tracked background process). Retry as terminal(command=..., "
                "background=true, pty=true)."
            )
    if notify is not None:
        if isinstance(notify, bool):
            notify_on_complete = notify
            watch_patterns = None
        elif isinstance(notify, list):
            watch_patterns = notify
            notify_on_complete = False
        else:
            return tool_error(
                "notify must be true/false (notify on exit) or a list of "
                "strings (notify on output pattern match)."
            )
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=notify_on_complete,
        watch_patterns=watch_patterns,
        target=args.get("target"),
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import importlib.util  # noqa: F401,E402
import platform  # noqa: F401,E402
import re  # noqa: F401,E402
import shlex  # noqa: F401,E402
import shutil  # noqa: F401,E402
import stat  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import sys  # noqa: F401,E402


def _environment_is_persistent(env: Any) -> bool:
    return bool(
        getattr(env, "_persistent", False)
        or getattr(env, "persistent_filesystem", False)
    )


from tools.terminal_tool_lifecycle import is_persistent_env  # noqa: E402,F401

def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    from tools import terminal_tool_lifecycle

    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            terminal_tool_lifecycle.cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)

    cleaned += _cleanup_retired_environments(
        min_age_seconds=0.0, require_idle=False,
    )

    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError as e:
            logger.debug("Failed to remove orphaned path %s: %s", path, e)
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def _cleanup_env(env, *, force_remove: bool = False) -> None:
    """Tear down one environment, passing ``force_remove`` only when accepted.

    ``DockerEnvironment.cleanup(force_remove=...)`` (issue #20561) diverges
    from the base ``cleanup(self)``; other backends expose ``stop`` /
    ``terminate`` instead. Shared by ``cleanup_vm`` and the prompt-time
    backend probe so the signature check lives in one place.
    """
    if hasattr(env, 'cleanup'):
        import inspect
        if "force_remove" in inspect.signature(env.cleanup).parameters:
            env.cleanup(force_remove=force_remove)
        else:
            env.cleanup()
    elif hasattr(env, 'stop'):
        env.stop()
    elif hasattr(env, 'terminate'):
        env.terminate()


def _build_environment_constructor_configs(
    config: Dict[str, Any],
    resolution: 'ExecutionTargetResolution',
    base_task_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build backend constructor inputs from one canonical normalized config."""
    env_type = config["env_type"]
    container_config: Optional[Dict[str, Any]] = None
    if _is_container_backend(env_type):
        container_config = {
            "container_cpu": config.get("container_cpu", 1),
            "container_memory": config.get("container_memory", 5120),
            "container_disk": config.get("container_disk", 51200),
            "container_persistent": config.get("container_persistent", True),
            "vercel_runtime": config.get("vercel_runtime", ""),
            "modal_mode": config.get("modal_mode", "auto"),
            "docker_volumes": config.get("docker_volumes", []),
            "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
            "docker_forward_env": config.get("docker_forward_env", []),
            "docker_env": config.get("docker_env", {}),
            "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
            "docker_extra_args": config.get("docker_extra_args", []),
            "docker_network": config.get("docker_network", True),
            "docker_shm_size": config.get("docker_shm_size", "1g"),
            "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
            "docker_shared_container_key": config.get("docker_shared_container_key", ""),
            "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
            "lifetime_seconds": config.get("lifetime_seconds", 300),
            "storage_task_id": resolution.storage_task_id(base_task_id),
            "legacy_storage_task_id": resolution.legacy_backend_task_id(base_task_id),
        }

    ssh_config: Optional[Dict[str, Any]] = None
    if env_type == "ssh":
        ssh_config = {
            "host": config.get("ssh_host", ""),
            "user": config.get("ssh_user", ""),
            "port": config.get("ssh_port", 22),
            "key": config.get("ssh_key", ""),
            "persistent": config.get("ssh_persistent", False),
            "file_sync": not resolution.named,
            "runtime_scope": resolution.security_scope if resolution.named else "",
        }

    local_config: Optional[Dict[str, Any]] = None
    if env_type == "local":
        local_config = {"persistent": config.get("local_persistent", False)}
    return container_config, ssh_config, local_config


def _record_environment_lifetime(env: Any, config: Dict[str, Any]) -> None:
    """Attach the resolved target's idle lifetime to its environment."""
    try:
        env._hermes_lifetime_seconds = int(config["lifetime_seconds"])
    except (AttributeError, KeyError, TypeError, ValueError):
        pass


def _record_environment_target(env: Any, resolution: Any) -> None:
    """Bind a created environment to the exact resolved named-target spec."""
    try:
        setattr(env, "_hermes_target_name", resolution.target)
        setattr(
            env, "_hermes_target_fingerprint",
            resolution.spec_fingerprint if resolution.named else None,
        )
        setattr(env, "_hermes_target_backend", resolution.backend)
        setattr(
            env, "_hermes_target_scope",
            resolution.security_scope if resolution.named else None,
        )
        setattr(env, "_hermes_target_resolution", resolution)
        persistent = resolution.config.get("container_persistent", True)
        if isinstance(persistent, str):
            persistent = persistent.strip().lower() in {"1", "true", "yes", "on"}
        setattr(
            env,
            "_hermes_stable_storage",
            resolution.backend == "docker" and bool(persistent),
        )
    except (AttributeError, TypeError):
        pass


def _environment_matches_target(env: Any, resolution: Any) -> bool:
    """Reject cache reuse after a named target's effective config changes."""
    if env is None or not resolution.named:
        return env is not None
    fingerprint = getattr(env, "_hermes_target_fingerprint", None)
    # Environments predating named targets carry no binding metadata — preserve their
    # registration contract; core-created envs are stamped before entering the cache.
    if fingerprint is None:
        return True
    return (
        fingerprint == resolution.spec_fingerprint
        and getattr(env, "_hermes_target_name", resolution.target) == resolution.target
        and getattr(env, "_hermes_target_backend", resolution.backend) == resolution.backend
    )


def _environment_has_stable_storage(env: Any) -> bool:
    return bool(getattr(env, "_hermes_stable_storage", False))


def _environment_replacement_is_busy(env: Any, environment_key: Hashable) -> bool:
    """Protect shared persistent storage while the old runtime is still active."""
    if not _environment_has_stable_storage(env):
        return False
    active = _active_turns_for_environment_key(environment_key)
    owned = _current_owned_environment_turns(environment_key)
    if active > owned:
        return True
    try:
        from tools.process_registry import process_registry

        return process_registry.has_active_environment(env)
    except Exception:
        return False


def _cleanup_environment_resource(
    env: Any,
    *,
    force_remove: bool = False,
    preserve_storage: bool = False,
) -> None:
    """Stop one environment, optionally preserving its persistent storage."""
    import inspect

    ownership_attrs = {}
    if force_remove:
        # A replaced environment is unreachable by config and must not keep persist-mode
        # lifecycle semantics; storage stays owned by the profile/target identity.
        attrs = ["_persist_across_processes"]
        if not preserve_storage:
            attrs.extend(["_persistent", "persistent_filesystem"])
        for attr in attrs:
            if hasattr(env, attr):
                try:
                    ownership_attrs[attr] = getattr(env, attr)
                    setattr(env, attr, False)
                except (AttributeError, TypeError):
                    pass

    try:
        if hasattr(env, "cleanup"):
            cleanup = env.cleanup
            kwargs = {}
            if force_remove:
                try:
                    if "force_remove" in inspect.signature(cleanup).parameters:
                        kwargs["force_remove"] = True
                except (TypeError, ValueError):
                    pass
            result = cleanup(**kwargs)
        elif hasattr(env, "stop"):
            result = env.stop()
        elif hasattr(env, "terminate"):
            result = env.terminate()
        else:
            return

        if inspect.isawaitable(result):
            import asyncio

            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(result)
                loop.close()
            except Exception:
                try:
                    close = getattr(result, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    pass
                raise

        wait_fn = getattr(env, "wait_for_cleanup", None)
        if wait_fn is not None and not wait_fn(timeout=60.0):
            raise RuntimeError("environment cleanup did not finish within 60 seconds")
    except BaseException:
        # If cleanup fails the caller restores the handle to the active cache, so
        # restore persistence ownership flags too — no live runtime left disowned.
        for attr, value in ownership_attrs.items():
            try:
                setattr(env, attr, value)
            except (AttributeError, TypeError):
                pass
        raise


class _EnvironmentReplacementError(RuntimeError):
    """Base error for a fail-closed named-target runtime replacement."""


class _EnvironmentReplacementBusyError(_EnvironmentReplacementError):
    """The previous stable-storage runtime still has active users."""


class _EnvironmentReplacementCleanupError(_EnvironmentReplacementError):
    """The previous stable-storage runtime could not be retired safely."""


def _prepare_environment_replacement(
    env: Any,
    environment_key: Hashable,
    *,
    target_name: str,
) -> bool:
    """Retire an idle stable-storage runtime before creating its replacement.

    Persistent Docker generations share one storage identity, so the obsolete
    runtime must be gone before the replacement container is created. The
    caller holds the per-environment creation lock; this helper owns the shared
    detach/cleanup/restore handoff used by terminal, file, and execute_code.
    """
    if env is None:
        return False
    if _environment_replacement_is_busy(env, environment_key):
        raise _EnvironmentReplacementBusyError(
            f"Execution target {target_name!r} changed while its persistent "
            "Docker runtime is still active. Wait for its commands/background "
            "processes to finish, then retry."
        )
    if not _environment_has_stable_storage(env):
        return False

    with _env_lock:
        if _active_environments.get(environment_key) is not env:
            raise _EnvironmentReplacementBusyError(
                f"Execution target {target_name!r} changed again while its "
                "previous runtime was being retired. Retry the operation."
            )
        owned_keys = [
            (key, key in _last_activity, _last_activity.get(key, 0.0))
            for key, candidate in list(_active_environments.items())
            if candidate is env
        ]
        for key, _, _ in owned_keys:
            _active_environments.pop(key, None)
            _last_activity.pop(key, None)

    try:
        _cleanup_environment_resource(
            env,
            force_remove=True,
            preserve_storage=True,
        )
    except BaseException as exc:
        with _env_lock:
            for key, had_activity, activity in owned_keys:
                if key not in _active_environments:
                    _active_environments[key] = env
                    if had_activity:
                        _last_activity[key] = activity
        if isinstance(exc, Exception):
            raise _EnvironmentReplacementCleanupError(
                "Could not retire the previous persistent Docker runtime for "
                f"execution target {target_name!r}: {exc}"
            ) from exc
        raise
    return True


def _retire_replaced_environment(env: Any, task_key: Hashable) -> None:
    """Defer teardown until no operation/process can still reference *env*."""
    if env is None:
        return
    with _retired_environments_lock:
        if all(existing_env is not env for _, existing_env, _ in _retired_environments):
            _retired_environments.append((task_key, env, time.time()))


def _collect_retired_environments(
    *,
    task_key: Hashable | None = None,
    min_age_seconds: float = 60.0,
    require_idle: bool = True,
) -> list[tuple[Hashable, Any]]:
    """Detach retired resources that are old enough and no longer in use."""
    now = time.time()
    ready: list[tuple[Hashable, Any]] = []
    keep: list[tuple[Hashable, Any, float]] = []
    try:
        from tools.process_registry import process_registry
    except ImportError:
        process_registry = None

    with _retired_environments_lock:
        candidates = list(_retired_environments)
        _retired_environments.clear()

    for retired_key, env, retired_at in candidates:
        if task_key is not None and retired_key != task_key:
            keep.append((retired_key, env, retired_at))
            continue
        busy = False
        if require_idle:
            busy = _active_turns_for_environment_key(retired_key) > 0
            if not busy and process_registry is not None:
                busy = process_registry.has_active_environment(env)
        if busy or now - retired_at < min_age_seconds:
            keep.append((retired_key, env, retired_at))
        else:
            ready.append((retired_key, env))

    # Merge records retired concurrently while we performed potentially slow
    # process liveness checks. Avoid duplicate records by environment identity.
    ready_ids = {id(env) for _, env in ready}
    with _retired_environments_lock:
        concurrent = [
            record for record in _retired_environments
            if id(record[1]) not in ready_ids
        ]
        seen = {id(record[1]) for record in concurrent}
        concurrent.extend(
            record for record in keep
            if id(record[1]) not in seen
        )
        _retired_environments[:] = concurrent
    return ready


def _cleanup_retired_environments(
    *,
    task_key: Hashable | None = None,
    min_age_seconds: float = 60.0,
    require_idle: bool = True,
) -> int:
    """Force-remove retired environments selected by lifecycle policy."""
    ready = _collect_retired_environments(
        task_key=task_key,
        min_age_seconds=min_age_seconds,
        require_idle=require_idle,
    )
    cleaned = 0
    for retired_key, env in ready:
        try:
            _cleanup_environment_resource(
                env,
                force_remove=True,
                preserve_storage=_environment_has_stable_storage(env),
            )
            cleaned += 1
            logger.info("Cleaned retired environment for task: %s", retired_key)
        except Exception as exc:
            error_str = str(exc)
            if "404" in error_str or "not found" in error_str.lower():
                cleaned += 1
                logger.info("Retired environment for task %s was already gone", retired_key)
            else:
                logger.warning(
                    "Error cleaning retired environment for task %s: %s",
                    retired_key, exc,
                )
                _retire_replaced_environment(env, retired_key)
    return cleaned


from tools.terminal_tool_lifecycle import cleanup_vm  # noqa: E402,F401


def _create_environment(
    env_type: str,
    image: str,
    cwd: str,
    timeout: int,
    ssh_config: Optional[dict] = None,
    container_config: Optional[dict] = None,
    local_config: Optional[dict] = None,
    task_id: str = "default",
    host_cwd: Optional[str] = None,
):
    """Create an execution environment for *env_type* (delegates to the split
    builders in ``tools.terminal_tool_backends``; kept as a module-level
    indirection so tests and task overrides can patch either site)."""
    from tools.terminal_tool_backends import _create_environment as _backends_create

    return _backends_create(
        env_type=env_type, image=image, cwd=cwd, timeout=timeout,
        ssh_config=ssh_config, container_config=container_config,
        local_config=local_config, task_id=task_id, host_cwd=host_cwd,
    )


def _apply_task_cwd_override(
    config: Dict[str, Any], cwd: str, cwd_override: Optional[str],
) -> str:
    """Apply a task workspace cwd without leaking host paths into containers.

    Docker's explicit mount-cwd mode is the exception: a registered host
    workspace should become the bind source and commands should run in
    ``/workspace``. Other container backends fall back to the target's already
    sanitized configured cwd.
    """
    env_type = config.get("env_type")
    if (
        env_type == "docker"
        and config.get("docker_mount_cwd_to_workspace")
        and isinstance(cwd_override, str)
        and cwd_override.strip()
    ):
        candidate = os.path.abspath(os.path.expanduser(cwd_override))
        is_host_path = (
            _is_host_cwd(candidate)
            or (
                os.path.isabs(candidate)
                and os.path.isdir(candidate)
            )
        )
        if is_host_path:
            config["host_cwd"] = candidate
            return "/workspace"
    if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
        return config["cwd"]
    return cwd


# One-shot guard for the config-fallback bridge below: after the first attempt
# TERMINAL_ENV is set (bridge succeeded) or the import failed — retrying is wasted.
_terminal_config_bridge_attempted = False


_sudo_execution_context: contextvars.ContextVar[
    tuple[str, str, bool, Optional[str], str] | None
] = contextvars.ContextVar(
    "hermes_sudo_execution_context", default=None,
)


@contextmanager
def _scoped_sudo_execution(
    target: str,
    backend: str,
    *,
    named: bool = False,
    sudo_password: Optional[str] = None,
    target_scope: str = "",
):
    token = _sudo_execution_context.set(
        (target, backend, named, sudo_password, target_scope),
    )
    try:
        yield
    finally:
        _sudo_execution_context.reset(token)


def _interpret_signal_exit(exit_code: int) -> str | None:
    """Map signal-termination exit codes to a human-readable note.

    Returns None when ``exit_code`` does not look like a signal death.
    Negative codes are Python ``subprocess`` semantics (definite); codes in
    the 128+signum band are the shell convention (very likely but not
    guaranteed, so those notes hedge with "usually").
    """
    from tools.terminal_tool_result import _SIGNAL_EXIT_NOTES

    if exit_code < 0:
        signum = -exit_code
        if signum == 2:  # SIGINT — executor's interrupt-marker path owns it
            return None
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return f"Command terminated by signal {signum}: {note}"
        try:
            import signal as _signal
            name = _signal.Signals(signum).name
        except (ValueError, ImportError):
            name = f"signal {signum}"
        return f"Command terminated by {name} (signal {signum})"

    if exit_code > 128:
        signum = exit_code - 128
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return (
                f"Exit code {exit_code} usually means the command was "
                f"terminated by signal {signum}: {note}"
            )

    return None


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Return a human-readable note when a non-zero exit code is non-erroneous.

    Returns None when the exit code is 0 or genuinely signals an error.
    The note is appended to the tool result so the model doesn't waste
    turns investigating expected exit codes.
    """
    if exit_code == 0:
        return None

    # Signal terminations (ported from Kilo-Org/kilocode#12698): negative codes
    # (``-signum``) are definite signal death; 128+signum is the conventional
    # shell encoding (heuristic — a program can ``exit 139``), so notes say
    # "usually". Without a note the model sees a bare ``exit_code=-9``/``137``
    # (OOM) and burns turns re-diagnosing. rc=130 is absent (bespoke handling).
    signal_note = _interpret_signal_exit(exit_code)
    if signal_note is not None:
        return signal_note

    # Last command in a pipeline/chain determines the exit code (`&&`, `|`, `;`):
    # split on shell operators and take the last piece — deliberately simple.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Get base command name (first word), stripping env var assignments
    # like  VAR=val cmd ...
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue  # skip VAR=val
        base_cmd = w.split("/")[-1]  # handle /usr/bin/grep -> grep
        break

    if not base_cmd:
        return None

    # Command-specific semantics
    semantics: dict[str, dict[int, str]] = {
        # grep/rg/ag/ack: 1=no matches found (normal), 2+=real error
        "grep":  {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg":    {1: "No matches found (not an error)"},
        "ag":    {1: "No matches found (not an error)"},
        "ack":   {1: "No matches found (not an error)"},
        # diff: 1=files differ (expected), 2+=real error
        "diff":  {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        # find: 1=some dirs inaccessible but results may still be valid
        "find":  {1: "Some directories were inaccessible (partial results may still be valid)"},
        # test/[: 1=condition is false (expected)
        "test":  {1: "Condition evaluated to false (expected, not an error)"},
        "[":     {1: "Condition evaluated to false (expected, not an error)"},
        # curl: common non-error codes
        "curl":  {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        # git: 1 is context-dependent but often normal (e.g. git diff with changes)
        "git":   {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
    }

    cmd_semantics = semantics.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]

    return None


def get_environment_for_target_scope(
    task_id: str, target: str, runtime_scope: str,
):
    """Find the active/retired environment that produced a scoped result."""
    raw = task_id or "default"
    collapsed = _resolve_container_task_id(raw)
    bases = {
        _profile_scoped_task_key(raw),
        _profile_scoped_task_key(collapsed),
    }

    def _matches(key: Hashable, env: Any) -> bool:
        belongs = key in bases or (
            isinstance(key, tuple) and len(key) == 2 and key[0] in bases
        )
        return bool(
            belongs
            and getattr(env, "_hermes_target_name", None) == target
            and getattr(env, "_hermes_target_scope", None) == runtime_scope
        )

    with _env_lock:
        for key, env in _active_environments.items():
            if _matches(key, env):
                return env
    with _retired_environments_lock:
        for key, env, _retired_at in _retired_environments:
            if _matches(key, env):
                return env
    return None


_PLUGIN_COMPAT_LAZY = {
    '_scoped_sudo_execution': ('tools.terminal_tool_sudo', '_scoped_sudo_execution'),
    '_reset_cached_sudo_passwords': ('tools.terminal_tool_sudo', '_reset_cached_sudo_passwords'),
    '_get_approval_mode': ('tools.approval_context', '_get_approval_mode'),
    '_handle_sudo_failure': ('tools.terminal_tool_sudo', '_handle_sudo_failure'),
    '_set_cached_sudo_password': ('tools.terminal_tool_sudo', '_set_cached_sudo_password'),
    '_get_cached_sudo_password': ('tools.terminal_tool_sudo', '_get_cached_sudo_password'),
    '_sudo_nopasswd_works': ('tools.terminal_tool_sudo', '_sudo_nopasswd_works'),
    '_reset_cached_sudo_passwords': ('tools.terminal_tool_sudo', '_reset_cached_sudo_passwords'),
    '_get_approval_mode': ('tools.approval_context', '_get_approval_mode'),
    'cleanup_vm': ('tools.terminal_tool_lifecycle', 'cleanup_vm'),
    'env_var_enabled': ('utils', 'env_var_enabled'),
    'get_active_env': ('tools.terminal_tool_lifecycle', 'get_active_env'),
    'has_direct_modal_credentials': ('tools.tool_backend_helpers', 'has_direct_modal_credentials'),
    'is_interrupted': ('tools.interrupt', 'is_interrupted'),
    'is_managed_tool_gateway_ready': ('tools.managed_tool_gateway', 'is_managed_tool_gateway_ready'),
    'is_persistent_env': ('tools.terminal_tool_lifecycle', 'is_persistent_env'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'resolve_modal_backend_state': ('tools.tool_backend_helpers', 'resolve_modal_backend_state'),
    'strip_inert_heredoc_bodies': ('tools.shell_heredoc', 'strip_inert_heredoc_bodies'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
