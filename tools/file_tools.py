#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools.

Companions: ``file_tools_paths`` (task-aware resolution), ``file_tools_write_guards``
(write-side guards), ``file_tools_read_tracking`` (per-task dedup / loop-detection /
staleness state).
"""

import base64
import errno
import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from agent.file_safety import get_read_block_error
from tools.binary_extensions import has_binary_extension, has_opaque_document_extension, is_pdf_path
from tools.file_operations import (
    ShellFileOperations, normalize_read_pagination, normalize_search_pagination)
from tools.file_operations_common import DEFAULT_READ_LIMIT
from tools import file_state
from agent.redact import redact_sensitive_text
from tools.file_tools_paths import (
    _authoritative_workspace_root, _expand_tilde, _path_resolution_warning, _resolve_base_dir,
    _resolve_path_for_task, _terminal_env_type_for_task)
from tools.file_tools_write_guards import (
    _READ_DEDUP_STATUS_MESSAGE, _check_approval_required_write, _check_binary_document_write,
    _check_cross_profile_path, _check_protected_instruction_write, _check_sensitive_path,
    _is_internal_file_tool_content)
from tools.file_tools_read_tracking import (
    _bump_consecutive, _cap_read_tracker_data, _check_file_staleness, _check_not_found_cache,
    _mark_verification_stale, _patch_failure_lock, _patch_failure_tracker, _read_tracker,
    _read_tracker_lock, _record_not_found, _record_patch_failure, _reset_patch_failures,
    _task_data, _update_read_timestamp)

logger = logging.getLogger(__name__)


_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}

# Read-size guard. Model-agnostic, so characters proxy tokens: 100K chars is
# ~25-35K tokens across typical tokenisers. Configurable: file_read_max_chars.
_DEFAULT_MAX_READ_CHARS = 100_000
def _get_max_read_chars() -> int:
    """Return ``file_read_max_chars`` from config.yaml (default on missing/invalid). No module
    cache: ``load_config_readonly`` is already mtime+path cached, and a process-lifetime slot
    would pin the launch profile's value under the multiplexed gateway."""
    try:
        from hermes_cli.config import load_config_readonly
        val = load_config_readonly().get("file_read_max_chars")
    except Exception:
        val = None
    valid = isinstance(val, (int, float)) and val > 0
    return int(val) if valid else _DEFAULT_MAX_READ_CHARS


def _truncate_to_char_budget(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Trim line-numbered ``read_file`` content to the last COMPLETE line within *max_chars*.

    Returns ``(kept_text, lines_kept, truncated)`` so the caller can offer a
    ``next_offset`` instead of rejecting the read. If not even the first line
    fits it is clamped mid-line so the read is never empty and the cursor advances.

    Ported in spirit from nearai/ironclaw#5029 (dual line/byte cap on ``read_file``). Where hermes
    previously hard-rejected an oversized read (forcing the model to guess a smaller ``limit`` and burn a
    round-trip returning nothing), this trims the content to the last *complete line* that fits within
    ``max_chars`` and reports how many lines were kept so the caller can offer a ``next_offset``
    continuation.
    """
    if len(content) <= max_chars:
        return content, (content.count("\n") + 1 if content else 0), False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        addition = len(line) + (1 if kept else 0)  # +1 for the rejoining "\n"
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition
    if not kept:
        kept.append(lines[0][:max_chars])
    return "\n".join(kept), len(kept), True


def _apply_char_budget(result_dict: dict, content: str, offset: int, total_lines, max_chars: int) -> str:
    """Trim *content* to the char budget, annotate *result_dict* with the
    continuation hint, and return the trimmed text."""
    trimmed, lines_kept, _ = _truncate_to_char_budget(content, max_chars)
    next_offset = offset + lines_kept
    result_dict["content"] = trimmed
    result_dict["truncated"] = True
    result_dict["truncated_by"] = "bytes"
    result_dict["next_offset"] = next_offset
    result_dict["hint"] = (
        f"Output truncated at the {max_chars:,}-char read budget after "
        f"{lines_kept} line(s) (showing lines {offset}-{next_offset - 1} of "
        f"{total_lines}). Use offset={next_offset} to continue.")
    if len(trimmed.split("\n", 1)[0]) >= max_chars:
        result_dict["hint"] += (
            " Note: the first line alone exceeded the budget and was "
            "clamped mid-line; its remainder is not retrievable via offset.")
    return trimmed


# Above this size, a wide read (limit > 200) gets a hint toward targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000

# Device/fd paths whose reads hang the process. Checked by path only — no I/O.
_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",     # never reach EOF
    "/dev/stdin", "/dev/tty", "/dev/console",                    # block on input
    "/dev/stdout", "/dev/stderr",                                # nonsensical to read
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",                       # fd aliases
})
# /proc/<pid>/... (and /proc/<pid>/task/<tid>/...) files that leak secrets,
# argv, memory layout (ASLR oracle: maps family, auxv, pagemap) or raw memory.
_BLOCKED_PROC_SUFFIXES = (
    "/fd/0", "/fd/1", "/fd/2",  # stdio aliases
    "/environ", "/cmdline", "/maps", "/smaps", "/smaps_rollup", "/numa_maps",
    "/mem", "/auxv", "/pagemap")


def _file_ops_uses_host_paths(file_ops) -> bool:
    """True when *file_ops* targets the host filesystem (only then may we stat paths
    or rewrite V4A headers to host-absolute paths; sandboxes have their own namespace)."""
    env = getattr(file_ops, "env", None)
    if env is None:
        return True
    try:
        from tools.environments.local import LocalEnvironment
    except ImportError:
        return True
    return isinstance(env, LocalEnvironment)


# V4A file headers: group 1 = header prefix, 2 = op, 3 = path. ``\s*`` after
# ``***`` mirrors patch_parser's leniency (``***Update File:`` applies, so it
# must be checked).
_V4A_SINGLE_HEADER_RE = re.compile(r'^(\*\*\*\s*(Update|Add|Delete)\s+File:\s*)(.+)$', re.MULTILINE)
_V4A_MOVE_HEADER_RE = re.compile(r'^(\*\*\*\s*Move\s+File:\s*)(.+?)\s*->\s*(.+)$', re.MULTILINE)


def _rewrite_v4a_patch_paths_for_host(patch: str, path_to_resolved: dict, file_ops) -> str:
    """Rewrite V4A file headers to the resolved host paths (host backends only).

    The shell layer must patch the SAME files ``patch_tool`` resolved for
    locking/staleness, not re-resolve a relative header against its own cwd
    (which can differ — the git-worktree cwd bug).
    """
    if not _file_ops_uses_host_paths(file_ops):
        return patch

    def _res(raw: str) -> str:
        raw = raw.strip()
        return path_to_resolved.get(raw) or raw

    patch = _V4A_SINGLE_HEADER_RE.sub(lambda m: f"{m.group(1)}{_res(m.group(3))}", patch)
    return _V4A_MOVE_HEADER_RE.sub(lambda m: f"{m.group(1)}{_res(m.group(2))} -> {_res(m.group(3))}", patch)


def _is_blocked_device_path(path: str) -> bool:
    """Return True for concrete device/fd/proc paths that can hang reads or leak process state."""
    normalized = os.path.normpath(_expand_tilde(path))
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    return normalized.startswith("/proc/") and normalized.endswith(_BLOCKED_PROC_SUFFIXES)


def _is_blocked_device(filepath: str, base_dir: str | Path | None = None) -> bool:
    """True if the path (literal, any symlink hop, or final realpath) is a blocked device.

    Literal first so /dev/stdin is caught before resolving to a terminal path;
    every symlink hop is checked so an alias cannot bypass the guard.
    """
    expanded = _expand_tilde(filepath)
    if base_dir is not None and not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(base_dir), expanded)
    normalized = os.path.normpath(expanded)
    if _is_blocked_device_path(normalized):
        return True

    seen: set[str] = set()
    current = normalized
    for _ in range(20):
        try:
            target = os.readlink(current)
        except OSError:
            break
        if not os.path.isabs(target):
            target = os.path.join(os.path.dirname(current), target)
        target = os.path.normpath(target)
        if _is_blocked_device_path(target):
            return True
        if target in seen:
            break
        seen.add(target)
        current = target

    try:
        resolved = os.path.normpath(os.path.realpath(normalized))
    except (OSError, ValueError):
        return False
    return _is_blocked_device_path(resolved)


def _filter_read_blocked_search_results(
    result, task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> int:
    """Remove credential/cache/env paths from a SearchResult in-place."""
    omitted = 0

    if hasattr(result, "matches") and result.matches:
        allowed_matches = []
        for match in result.matches:
            if _search_result_read_block_error(
                match.path, task_id, execution_target, _resolution=_resolution,
            ):
                omitted += 1
                continue
            allowed_matches.append(match)
        result.matches = allowed_matches

    if hasattr(result, "files") and result.files:
        allowed_files = []
        for file_path in result.files:
            if _search_result_read_block_error(
                file_path, task_id, execution_target, _resolution=_resolution,
            ):
                omitted += 1
                continue
            allowed_files.append(file_path)
        result.files = allowed_files

    if hasattr(result, "counts") and result.counts:
        allowed_counts = {}
        for file_path, count in result.counts.items():
            if _search_result_read_block_error(
                file_path, task_id, execution_target, _resolution=_resolution,
            ):
                omitted += 1
                continue
            allowed_counts[file_path] = count
        result.counts = allowed_counts

    return omitted


# Paths that file tools should refuse to write to without going through the
# terminal tool's approval system.  These match prefixes after os.path.realpath.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/",
    # macOS: /private/var mirrors /var — block only the sensitive subtrees. A
    # blanket "/private/var/" refused every legitimate temp-file write ($TMPDIR,
    # /tmp, and /var/folders all realpath() into /private/var/folders/).
    "/private/var/db/", "/private/var/root/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False
def _is_expected_write_exception(exc: Exception) -> bool:
    """Return True for expected write denials that should not hit error logs."""
    return isinstance(exc, PermissionError) or (
        isinstance(exc, OSError) and exc.errno in _EXPECTED_WRITE_ERRNOS)


# ── ShellFileOperations per terminal environment ─────────────────────────
_file_ops_lock = threading.Lock()
_file_ops_cache: dict = {}


def _create_terminal_env_for_file_ops(raw_task_id: str, task_id: str):
    """Build the terminal environment for *task_id* via the shared ``_create_configured_env``,
    so a file tool that runs before any terminal command still gets the configured backend."""
    from tools.terminal_tool_config import _is_container_backend
    from tools.terminal_tool import (
        _create_configured_env, _get_env_config, _is_unusable_container_cwd,
        _resolve_task_host_cwd, _select_image, get_session_cwd, resolve_task_overrides)

    config = _get_env_config()
    env_type = config["env_type"]
    overrides = resolve_task_overrides(raw_task_id)
    try:
        recorded_cwd = get_session_cwd(raw_task_id)
    except Exception:
        recorded_cwd = None
    cwd = overrides.get("cwd") or recorded_cwd or config["cwd"]
    # Re-apply the container cwd guard: a gateway/TUI/ACP override is a raw HOST
    # path and ``docker run -w <host-path>`` makes search_files & co silently
    # return nothing. Valid in-container overrides (/workspace, /root) pass.
    # Re-apply the container cwd guard that _get_env_config() already ran on config["cwd"] (see #50636). A
    # per-task cwd override registered by the gateway/TUI/ACP for workspace tracking is a raw host path
    # (e.g. a Desktop session's /Users/<me>/workspace or C:\\Users\\<me>). On a container backend that
    # reaches ``docker run -w <host-path>`` and the container starts in a directory that doesn't exist
    # inside the sandbox, so search_files and friends silently return empty results (#54447). Sanitize it
    # back to the already-validated config["cwd"] so the override can't bypass the guard.
    if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
        if cwd != config["cwd"]:
            logger.info(
                "Ignoring host/relative cwd override %r for %s backend "
                "(won't exist in sandbox). Using %r instead.",
                cwd, env_type, config["cwd"])
        cwd = config["cwd"]
    logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])
    terminal_env = _create_configured_env(
        config, env_type, image=_select_image(env_type, overrides, config), cwd=cwd,
        timeout=config["timeout"], task_id=task_id,
        host_cwd=_resolve_task_host_cwd(config, raw_task_id),
        local_config={"persistent": config.get("local_persistent", False)} if env_type == "local" else None,
    )
    return env_type, terminal_env


def _get_file_ops(
    task_id: str = "default", target: str | None = None, *, _resolution: Any = None,
) -> ShellFileOperations:
    """Get or create ShellFileOperations for a terminal environment.

    Respects the TERMINAL_ENV setting -- if the task_id doesn't have an
    environment yet, creates one using the configured backend (local, docker,
    modal, etc.) rather than always defaulting to local.

    Thread-safe: uses the same per-task creation locks as terminal_tool to
    prevent duplicate sandbox creation from concurrent tool calls.

    Note: subagent task_ids are collapsed to "default" via
    ``_resolve_container_task_id`` so delegate_task children share the
    parent's container and its cached file_ops. RL/benchmark task_ids with
    a registered env override keep their isolation.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _create_environment,
        _get_env_config, _last_activity, _start_cleanup_thread,
        _creation_locks,
        _creation_locks_lock,
        _resolve_container_task_id,
        _resolve_task_host_cwd,
        _is_unusable_container_cwd,
        _CONTAINER_BACKENDS,
        _record_environment_lifetime,
        _record_environment_target,
        _environment_matches_target,
        _prepare_environment_replacement,
        _retire_replaced_environment,
        _cleanup_environment_resource,
        _environment_has_stable_storage,
        _apply_task_cwd_override,
        _build_environment_constructor_configs,
    )
    from tools.execution_targets import (
        execution_target_config_is_frozen,
        resolve_execution_target,
        resolve_live_execution_target,
    )
    import time

    raw_task_id = task_id or "default"
    resolution = _resolution or resolve_execution_target(target)
    base_task_id = _resolve_container_task_id(raw_task_id)
    task_id = resolution.environment_key(base_task_id)  # type: ignore[assignment]
    backend_task_id = resolution.backend_task_id(base_task_id)

    # Fast path: check cache -- but also verify the underlying environment
    # is still alive (it may have been killed by the cleanup thread).
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
    if cached is not None:
        with _env_lock:
            active_env = _active_environments.get(task_id)
            if (
                _environment_matches_target(active_env, resolution)
                and getattr(cached, "env", None) is active_env
            ):
                _last_activity[task_id] = time.time()
                return cached
            else:
                # Environment was cleaned up -- preserve the old cwd in the
                # session record before invalidating the stale cache entry
                # (fixes #26211: silent file-creation failures in long-running
                # conversations). Fill-only: ``cached.cwd`` snapshots the SHARED
                # env at cache-build time, so it is not attributable to this
                # session (#85658 class); rescue a record-less session only.
                old_cwd = getattr(cached, "cwd", None)
                if active_env is None and old_cwd:
                    try:
                        from tools.terminal_tool import (
                            get_session_cwd,
                            record_session_cwd,
                        )
                        if get_session_cwd(raw_task_id, _resolution=resolution) is None:
                            record_session_cwd(
                                raw_task_id, old_cwd, _resolution=resolution,
                            )
                    except Exception:
                        pass
                with _file_ops_lock:
                    _file_ops_cache.pop(task_id, None)

    # Need to ensure the environment exists before building file_ops.
    # Acquire per-task lock so only one thread creates the sandbox.
    with _creation_locks_lock:
        if task_id not in _creation_locks:
            _creation_locks[task_id] = threading.Lock()
        task_lock = _creation_locks[task_id]

    with task_lock:
        # Double-check: another thread may have created it while we waited
        with _env_lock:
            active_env = _active_environments.get(task_id)
            if _environment_matches_target(active_env, resolution):
                _last_activity[task_id] = time.time()
                terminal_env = active_env
            else:
                terminal_env = None

        if terminal_env is None:
            _prepare_environment_replacement(
                active_env,
                task_id,
                target_name=resolution.target,
            )
            from tools.terminal_tool import resolve_task_overrides

            config = (
                _get_env_config(dict(resolution.config))
                if resolution.named else _get_env_config()
            )
            env_type = config["env_type"]
            overrides = resolve_task_overrides(raw_task_id)

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

            try:
                from tools.terminal_tool import get_session_cwd
                recorded_cwd = get_session_cwd(
                    raw_task_id, _resolution=resolution,
                )
            except Exception:
                recorded_cwd = None
            cwd_override = (
                overrides.get("cwd")
                if (
                    not resolution.named
                    or (resolution.is_default and resolution.backend != "ssh")
                )
                else None
            )
            cwd = cwd_override or recorded_cwd or config["cwd"]
            cwd = _apply_task_cwd_override(config, cwd, cwd_override)
            logger.info("Creating new %s environment for task %s...", env_type, task_id)

            container_config, ssh_config, local_config = (
                _build_environment_constructor_configs(
                    config, resolution, base_task_id,
                )
            )
            if container_config is not None:
                # Explicit module-local copy of the shared builder's map, preserving the
                # cross-call-site Docker option invariant (test_docker_network_config.py).
                container_config = {
                    "container_cpu": container_config["container_cpu"],
                    "container_memory": container_config["container_memory"],
                    "container_disk": container_config["container_disk"],
                    "container_persistent": container_config["container_persistent"],
                    "vercel_runtime": container_config["vercel_runtime"],
                    "modal_mode": container_config["modal_mode"],
                    "docker_volumes": container_config["docker_volumes"],
                    "docker_mount_cwd_to_workspace": container_config["docker_mount_cwd_to_workspace"],
                    "docker_forward_env": container_config["docker_forward_env"],
                    "docker_env": container_config["docker_env"],
                    "docker_run_as_host_user": container_config["docker_run_as_host_user"],
                    "docker_extra_args": container_config["docker_extra_args"],
                    "docker_network": container_config["docker_network"],
                    "docker_shm_size": container_config["docker_shm_size"],
                    "docker_persist_across_processes": container_config["docker_persist_across_processes"],
                    "docker_shared_container_key": container_config.get("docker_shared_container_key", ""),
                    "docker_orphan_reaper": container_config["docker_orphan_reaper"],
                    "lifetime_seconds": container_config["lifetime_seconds"],
                    "storage_task_id": container_config["storage_task_id"],
                    "legacy_storage_task_id": container_config["legacy_storage_task_id"],
                }

            terminal_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=cwd,
                timeout=config["timeout"],
                ssh_config=ssh_config,
                container_config=container_config,
                local_config=local_config,
                task_id=backend_task_id,
                host_cwd=_resolve_task_host_cwd(config, raw_task_id),
            )
            _record_environment_lifetime(terminal_env, config)
            _record_environment_target(terminal_env, resolution)

            publish_error = None
            with _env_lock:
                if resolution.named:
                    try:
                        live_resolution = (
                            resolution
                            if execution_target_config_is_frozen()
                            else resolve_live_execution_target(target)
                        )
                    except Exception as exc:
                        publish_error = str(exc)
                    else:
                        if live_resolution.security_scope != resolution.security_scope:
                            publish_error = (
                                f"Execution target {resolution.target!r} changed "
                                "while its environment was being created."
                            )
                replaced_env = None
                if publish_error is None:
                    replaced_env = _active_environments.get(task_id)
                    _active_environments[task_id] = terminal_env
                    _last_activity[task_id] = time.time()
            if publish_error is not None:
                _cleanup_environment_resource(
                    terminal_env,
                    force_remove=True,
                    preserve_storage=_environment_has_stable_storage(terminal_env),
                )
                raise RuntimeError(publish_error + " Retry the file operation.")
            if replaced_env is not None and replaced_env is not terminal_env:
                _retire_replaced_environment(replaced_env, task_id)

            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id)

    # Build file_ops from the (guaranteed live) environment and cache it
    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
    return file_ops



def _file_state_namespace(
    task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> str | None:
    """Namespace remote/container paths, but share locks for local aliases."""
    try:
        from tools.execution_targets import resolve_execution_target

        resolution = _resolution or resolve_execution_target(execution_target)
        runtime_prefix = (
            f"runtime-{resolution.security_scope}:"
            if resolution.provider is not None
            else ""
        )
        if resolution.backend == "local":
            return f"{profile}{runtime_prefix}local" if runtime_prefix else None
        # Preserve the pre-target single-profile legacy state key.
        if not resolution.named:
            return None
        profile = f"profile-{resolution.profile_scope}:" if resolution.profile_scope else ""
        if resolution.backend == "ssh":
            # Two names for the same SSH account/root address the same physical
            # filesystem and must share stale-write locks/read state.
            cfg = resolution.config
            physical = json.dumps({
                "host": str(cfg.get("ssh_host") or ""),
                "user": str(cfg.get("ssh_user") or ""),
                "port": int(cfg.get("ssh_port") or 22),
            }, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(physical.encode("utf-8")).hexdigest()[:20]
            return f"{profile}{runtime_prefix}ssh:{digest}"
        if resolution.backend == "docker":
            persistent = resolution.config.get("container_persistent", True)
            if isinstance(persistent, str):
                persistent = persistent.strip().lower() in {"true", "1", "yes"}
            if persistent:
                from tools.terminal_tool import _resolve_container_task_id

                owner = resolution.storage_task_id(
                    _resolve_container_task_id(str(task_id)),
                )
                return f"{profile}{runtime_prefix}docker-storage:{owner}"
        # Other remote/container targets create distinct backend resources even
        # when their visible path strings happen to match.
        return (
            f"{profile}{runtime_prefix}{resolution.backend}:"
            f"{resolution.security_scope}"
        )
    except Exception:
        # Unknown targets are reported by the tool resolver before state access.
        return execution_target


def _write_precheck_error(paths: list[str], content_paths: list[str], task_id: str,
                          cross_profile: bool) -> str | None:
    """Run the shared write/patch guards in order; return the first error string.

    Order matters: hard denies (sensitive path, mirror) and the corruption
    guard run before anything that could prompt the user, and ONE approval
    prompt covers every path of a multi-file patch.
    """
    for p in paths:
        err = _check_sensitive_path(p, task_id) or (
            None if cross_profile else _check_cross_profile_path(p, task_id))
        if err:
            return err
    for p in content_paths:
        err = _check_binary_document_write(p, task_id)
        if err:
            return err
    return (_check_protected_instruction_write(paths, task_id)
            or _check_approval_required_write(paths, task_id))
def _edit_warnings(paths: list[str], path_to_resolved: dict, task_id: str) -> list[str]:
    """One pre-edit warning per path, in priority order: cross-agent registry
    (names the sibling subagent) > per-task staleness > workspace divergence
    (relative path resolving outside the terminal's cwd — the worktree-cwd bug)."""
    warnings: list[str] = []
    for p in paths:
        r = path_to_resolved.get(p)
        w = (file_state.check_stale(task_id, r) if r else None) or _check_file_staleness(p, task_id)
        if not w and r:
            w = _path_resolution_warning(p, Path(r), task_id)
        if w:
            warnings.append(w)
    return warnings


def _note_edited(task_id: str, paths: list[str], path_to_resolved: dict, session_id: str | None) -> None:
    """Post-success bookkeeping: verification-stale marker, then per path refresh
    the read stamp (no false staleness on the next edit) and record the write."""
    _mark_verification_stale(task_id, [path_to_resolved.get(p) or p for p in paths], session_id=session_id)
    for p in paths:
        _update_read_timestamp(p, task_id)
        if path_to_resolved.get(p):
            file_state.note_write(task_id, path_to_resolved[p])


def _resolve_or_none(filepath: str, task_id: str) -> str | None:
    """Task-resolved path string, or None when resolution fails for any reason."""
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except Exception:
        return None


def _search_result_read_block_error(
    path: str, task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> str | None:
    """Return the read-safety error for a search result path.

    Search backends may return paths relative to the task cwd, while
    ``get_read_block_error`` expects an already-resolved path when the task cwd
    can differ from the Python process cwd. Mirror ``read_file_tool``'s path
    resolution before applying the shared read guard.
    """
    try:
        resolved = _resolve_path_for_task(
            path, task_id, execution_target, _resolution=_resolution,
        )
    except (OSError, ValueError, RuntimeError):
        return get_read_block_error(path)
    return get_read_block_error(str(resolved))


def _backend_v4a_patch(
    patch_text: str,
    task_id: str,
    execution_target: str | None,
    *,
    _resolution: Any = None,
) -> str:
    """Anchor relative V4A headers to the calling SSH session's cwd."""
    if _terminal_env_type_for_task(
        task_id, execution_target, _resolution=_resolution,
    ) != "ssh":
        return patch_text
    import re

    def _single(match):
        path = match.group(2).strip()
        routed = _backend_operation_path(
            path, path, task_id, execution_target, _resolution=_resolution,
        )
        return match.group(1) + routed

    rewritten = re.sub(
        r"^(\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*)(.+)$",
        _single,
        patch_text,
        flags=re.MULTILINE,
    )

    def _move(match):
        source = match.group(2).strip()
        destination = match.group(3).strip()
        routed_source = _backend_operation_path(
            source, source, task_id, execution_target, _resolution=_resolution,
        )
        routed_destination = _backend_operation_path(
            destination, destination, task_id, execution_target,
            _resolution=_resolution,
        )
        return f"{match.group(1)}{routed_source} -> {routed_destination}"

    return re.sub(
        r"^(\*\*\*\s*Move\s+File:\s*)(.+?)\s*->\s*(.+)$",
        _move,
        rewritten,
        flags=re.MULTILINE,
    )


def _backend_operation_path(
    original: str,
    resolved: str | Path | PurePosixPath,
    task_id: str = "default",
    execution_target: str | None = None,
    *,
    _resolution: Any = None,
) -> str:
    """Return the path that should be sent to the selected backend.

    SSH owns relative and ``~`` expansion. Resolving those paths on the Hermes
    host first creates a host-absolute path that may name a different file (or
    no file) on the remote machine. Container/local backends retain the existing
    exact resolved-path contract.
    """
    if _terminal_env_type_for_task(
        task_id, execution_target, _resolution=_resolution,
    ) == "ssh":
        original = str(original)
        if posixpath.isabs(original) or original.startswith("~"):
            return original
        try:
            from tools.terminal_tool import get_session_cwd

            recorded_cwd = get_session_cwd(
                task_id, target=execution_target, _resolution=_resolution,
            )
        except Exception:
            recorded_cwd = None
        if recorded_cwd and posixpath.isabs(str(recorded_cwd)):
            return posixpath.normpath(posixpath.join(str(recorded_cwd), original))
        try:
            from tools.execution_targets import resolve_execution_target

            resolution = _resolution or resolve_execution_target(execution_target)
        except Exception:
            resolution = None
        # Legacy SSH file ops followed the live env cwd; only a named target has a
        # stable configured root to anchor before a session cwd is recorded.
        if resolution is not None and resolution.named:
            session_cwd = _authoritative_workspace_root(
                task_id, execution_target, _resolution=resolution,
            )
            if session_cwd and posixpath.isabs(str(session_cwd)):
                return posixpath.normpath(posixpath.join(str(session_cwd), original))
        return original
    return str(resolved)


def _file_ops_for_resolution(task_id: str, resolution: Any):
    # Preserve the established one-argument seam in legacy mode.
    if resolution.named:
        file_ops = _get_file_ops(
            task_id, target=resolution.target, _resolution=resolution,
        )
        if resolution.backend == "ssh":
            # Named SSH environments are shared across conversations, so pin every
            # operation to this session's cwd record (falling back to the target's
            # configured remote root); "." / "~" stay remote shell paths.
            operation_cwd = _authoritative_workspace_root(
                task_id, resolution.target, _resolution=resolution,
            )
            environment = getattr(file_ops, "env", None)
            if operation_cwd and environment is not None:
                scoped_ops = ShellFileOperations(
                    environment,
                    cwd=operation_cwd,
                    fixed_cwd=operation_cwd,
                )
                scoped_ops._command_cache = file_ops._command_cache
                return scoped_ops
        return file_ops
    return _get_file_ops(task_id)


def clear_file_ops_cache(task_id=None):
    """Clear file-operation state for a finished task, or all tasks."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()

    with _read_tracker_lock:
        if task_id:
            _read_tracker.pop(task_id, None)
        else:
            _read_tracker.clear()

    with _patch_failure_lock:
        if task_id:
            _patch_failure_tracker.pop(task_id, None)
        else:
            _patch_failure_tracker.clear()

    if task_id:
        file_state.get_registry().forget_task(task_id)
    else:
        file_state.get_registry().clear()


def _special_file_kind(path) -> str | None:
    """Return a human name for non-regular file types that block reads.

    Stat-based sibling of the name-based ``_is_blocked_device`` guard: a
    FIFO at ``logs/live.pipe`` or a socket in a workspace hangs ``read_file``
    just as hard as ``/dev/zero``, but carries no recognizable name. Only
    called for host-visible filesystems (see ``_file_ops_uses_host_paths``);
    remote backends cannot be statted from here.

    Returns None for regular files, missing paths, and anything unstattable
    (those flow to the normal read path and its own error handling).
    """
    import stat as _stat

    try:
        st = os.stat(os.fspath(path))  # follows symlinks, matching a real read
    except OSError:
        return None
    mode = st.st_mode
    if _stat.S_ISREG(mode) or _stat.S_ISDIR(mode):
        return None
    if _stat.S_ISFIFO(mode):
        return "a FIFO (named pipe)"
    if _stat.S_ISSOCK(mode):
        return "a socket"
    if _stat.S_ISCHR(mode):
        return "a character device"
    if _stat.S_ISBLK(mode):
        return "a block device"
    return "a special (non-regular) file"


def read_file_tool(
    path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT,
    task_id: str = "default", target: str | None = None,
    runtime_scope: str | None = None,
) -> str:
    """Read a file with pagination and line numbers.

    Guard order: device-path blocklist (no I/O) → stat-based special-file
    guard (host only) → document extraction → binary-extension guard → Hermes
    internal denylist → negative-result cache → dedup stub → real read.
    """
    try:
        from tools.execution_targets import resolve_execution_target

        pinned_file_ops = None
        if runtime_scope:
            from tools.terminal_tool import get_environment_for_target_scope

            selected_name = str(target or "")
            if not selected_name:
                return tool_error(
                    "read_file runtime_scope requires the target from the saved-output hint."
                )
            pinned_env = get_environment_for_target_scope(
                task_id, selected_name, runtime_scope,
            )
            if pinned_env is None:
                return tool_error(
                    "The producing execution environment for this saved output "
                    "is no longer available. Re-run the originating tool."
                )
            resolution = getattr(pinned_env, "_hermes_target_resolution", None)
            if resolution is None:
                return tool_error(
                    "The producing environment lacks immutable target metadata. "
                    "Re-run the originating tool."
                )
            pinned_file_ops = ShellFileOperations(
                pinned_env, str(getattr(pinned_env, "cwd", ".")),
            )
        else:
            resolution = resolve_execution_target(target)
        selected_target = resolution.target if resolution.named else None
        host_mtime_tracking = resolution.backend == "local"
        state_task_id = resolution.file_coordination_key(task_id)
        state_namespace = _file_state_namespace(
            task_id, selected_target, _resolution=resolution,
        )
        offset, limit = normalize_read_pagination(offset, limit)

        # ── Device path guard ─────────────────────────────────────────
        # Block paths that hang the process (infinite output/blocking input); pure path check.
        device_base = None if Path(path).expanduser().is_absolute() else _resolve_base_dir(
            task_id, execution_target=selected_target,
            _resolution=resolution,
        )
        if _is_blocked_device(path, base_dir=device_base):
            return tool_error(
                f"Cannot read '{path}': this is a device file that would "
                "block or produce infinite output."
            )

        _resolved = _resolve_path_for_task(
                path, task_id, selected_target, _resolution=resolution,
            )

        # ── Special-file type guard (stat-based) ──────────────────────
        # Catches FIFO/socket/device wherever it lives; a FIFO read blocks until timeout (self-inflicted DoS).
        if _file_ops_uses_host_paths(_get_file_ops(task_id)):
            kind = _special_file_kind(_resolved)
            if kind is not None:
                return json.dumps({
                    "success": False,
                    "note": (
                        f"'{path}' is {kind}, not a regular file — reading "
                        "it would block indefinitely, so no read was "
                        "attempted. Use terminal utilities if you need to "
                        "interact with it."
                    ),
                })

        # ── Structured-document extraction ────────────────────────────
        # Runs before the binary-extension guard so .docx/.xlsx render as text; malformed docs fall through.
        from tools.read_extract import (
            ANYDOC_EXTENSIONS,
            EXTRACTABLE_EXTENSIONS,
            MAX_DOCUMENT_BYTES,
            ExtractionError,
            extract_document_bytes,
            is_extractable_document,
        )

        if (
            _terminal_env_type_for_task(
                task_id, selected_target, _resolution=resolution,
            ) == "local"
            and is_extractable_document(str(_resolved))
        ):
            file_ops = _get_file_ops(task_id)
            try:
                binary = file_ops.read_file_bytes(
                    str(_resolved), max_bytes=MAX_DOCUMENT_BYTES
                )
                if binary.error or binary.base64_content is None:
                    raise ExtractionError(binary.error or "Document bytes unavailable")
                document_bytes = base64.b64decode(
                    binary.base64_content, validate=True
                )
                extracted_text = extract_document_bytes(
                    document_bytes, str(_resolved)
                )
            except (ExtractionError, ValueError, base64.binascii.Error) as exc:
                logger.debug("document extraction failed for %s", path, exc_info=True)
                # Binary document formats surface the specific failure (size cap,
                # encrypted, malformed…) — the fallthrough only gives a generic
                # binary error or raw bytes. .ipynb and byte-transport errors
                # (ValueError/binascii) stay on the fallthrough; only a specific
                # ExtractionError carries an actionable reason.
                _doc_ext = _resolved.suffix.lower()
                _binary_doc = _doc_ext in ANYDOC_EXTENSIONS or (
                    _doc_ext in EXTRACTABLE_EXTENSIONS and _doc_ext != ".ipynb"
                )
                if (
                    _binary_doc
                    and isinstance(exc, ExtractionError)
                    and not str(exc).startswith("Unsupported document type")
                ):
                    return tool_error(
                        f"Cannot read '{path}' ({_doc_ext}): document "
                        f"extraction failed — {exc}. Use terminal utilities "
                        "to inspect or convert the file."
                    )
            else:
                file_ops = pinned_file_ops or _file_ops_for_resolution(task_id, resolution)
                lines = extracted_text.splitlines()
                total_lines = len(lines)
                end_line = offset + limit - 1
                page_text = "\n".join(lines[offset - 1:end_line])
                result_dict = {
                    "content": file_ops._add_line_numbers(page_text, offset) if page_text else "",
                    "total_lines": total_lines,
                    "file_size": binary.file_size,
                    "truncated": total_lines > end_line,
                    "extracted_document": True,
                }
                if result_dict["truncated"]:
                    result_dict["hint"] = (
                        f"Use offset={end_line + 1} to continue reading "
                        f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)"
                    )
                content_len = len(result_dict["content"])
                max_chars = _get_max_read_chars()
                if content_len > max_chars:
                    # Graceful char-budget truncation (nearai/ironclaw#5029): trim to
                    # the last complete line that fits and offer next_offset.
                    trimmed, lines_kept, _ = _truncate_to_char_budget(
                        result_dict["content"], max_chars
                    )
                    next_offset = offset + lines_kept
                    shown_end = offset + lines_kept - 1
                    result_dict["content"] = trimmed
                    result_dict["truncated"] = True
                    result_dict["truncated_by"] = "bytes"
                    result_dict["next_offset"] = next_offset
                    result_dict["hint"] = (
                        f"Output truncated at the {max_chars:,}-char read budget "
                        f"after {lines_kept} line(s) (showing lines {offset}-"
                        f"{shown_end} of {total_lines}). Use offset={next_offset} "
                        "to continue."
                    )
                    if len(trimmed.split("\n", 1)[0]) >= max_chars:
                        result_dict["hint"] += (
                            " Note: the first line alone exceeded the budget and "
                            "was clamped mid-line; its remainder is not "
                            "retrievable via offset."
                        )
                if result_dict["content"]:
                    result_dict["content"] = redact_sensitive_text(result_dict["content"], file_read=True)
                result_dict.update(resolution.metadata(
                    cwd=_authoritative_workspace_root(task_id, selected_target),
                ))
                return json.dumps(result_dict, ensure_ascii=False)

        # ── Binary file guard ─────────────────────────────────────────
        # Block by extension (no I/O); the content-sniffing path below names the actual magic-byte type.
        if has_binary_extension(str(_resolved)):
            _ext = _resolved.suffix.lower()
            return tool_error(
                f"Cannot read binary file '{path}' ({_ext}). "
                "Use vision_analyze for images, or terminal to inspect binary files."
            )

        # ── Hermes internal path guard ────────────────────────────────
        # Block credential stores under HERMES_HOME and prompt-injection vectors (catalog/
        # hub metadata); pass the resolved path (its resolve() uses the process cwd).
        block_error = get_read_block_error(str(_resolved))
        if block_error:
            return tool_error(block_error)

        # ── Negative-result cache ─────────────────────────────────────
        # Cached "not found" (within TTL) skips the subprocess + similar-files walk; cleared by write/patch.
        resolved_str_for_neg = str(_resolved)
        cached_not_found = _check_not_found_cache(
            "read", resolved_str_for_neg, state_task_id,
            check_host_filesystem=resolution.backend == "local",
        )
        if cached_not_found is not None:
            return cached_not_found

        # ── Dedup check ───────────────────────────────────────────────
        # Same (path, offset, limit) + unmodified file → lightweight stub instead of re-sending content.
        resolved_str = str(_resolved)
        dedup_key = (resolved_str, offset, limit)
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(state_task_id, {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "dedup": {},
                "dedup_hits": {}, "dedup_generation_reads": set(),
                "read_timestamps": {},
            })
            # Backward-compat for tracker entries predating dedup_hits/read_timestamps
            # (long-lived task or upgrade boundary).
            if "dedup_hits" not in task_data:
                task_data["dedup_hits"] = {}
            if "read_timestamps" not in task_data:
                task_data["read_timestamps"] = {}
            generation_reads = task_data.setdefault("dedup_generation_reads", set())
            cached_mtime = (
                task_data.get("dedup", {}).get(dedup_key)
                if host_mtime_tracking
                else None
            )
            content_served_in_generation = dedup_key in generation_reads

        if cached_mtime is not None:
            try:
                current_mtime = os.path.getmtime(resolved_str)
                if current_mtime == cached_mtime and content_served_in_generation:
                    # Count repeated stubs so weak tool-followers ignoring the "refer
                    # to earlier result" hint don't loop forever; after 2 stubs for
                    # one key escalate to a hard block (mirrors count>=4 on reads).
                    with _read_tracker_lock:
                        hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
                        task_data["dedup_hits"][dedup_key] = hits
                        _cap_read_tracker_data(task_data)

                    if hits >= 2:
                        return tool_error(
                            f"BLOCKED: You have called read_file on this "
                            f"exact region {hits + 1} times and the file "
                            "has NOT changed. STOP calling read_file for "
                            "this path — the content from your earlier "
                            "read_file result in this conversation is "
                            "still current. Proceed with your task using "
                            "the information you already have.",
                            path=path,
                            already_read=hits + 1,
                        )

                    unchanged = {
                        "status": "unchanged",
                        "message": _READ_DEDUP_STATUS_MESSAGE,
                        "path": path,
                        "dedup": True,
                        "content_returned": False,
                    }
                    unchanged.update(resolution.metadata(
                        cwd=_authoritative_workspace_root(
                            task_id, selected_target, _resolution=resolution,
                        ),
                    ))
                    return json.dumps(unchanged, ensure_ascii=False)
            except OSError:
                pass  # stat failed — fall through to full read

        # ── Perform the read ──────────────────────────────────────────
        file_ops = pinned_file_ops or _file_ops_for_resolution(task_id, resolution)
        operation_path = _backend_operation_path(
            path, _resolved, task_id, selected_target,
            _resolution=resolution,
        )
        result = file_ops.read_file(operation_path, offset, limit)
        result_dict = result.to_dict()
        result_dict.setdefault("resolved_path", operation_path)

        # ── Populate negative-result cache on not-found ───────────────
        # Cache the JSON so a retry skips the parent-dir walk. Deliberately NO
        # early return — error results must keep flowing through the tracking
        # block below and the normal exit; short-circuiting changed that behavior
        # and broke a real test. Serving from cache is the optimization.
        _err = result_dict.get("error") or ""
        if isinstance(_err, str) and _err.startswith("File not found:"):
            _not_found_json = json.dumps(result_dict, ensure_ascii=False)
            _record_not_found(
                "read", resolved_str_for_neg, state_task_id, _not_found_json,
            )

        # ── Character-count guard ─────────────────────────────────────
        # Characters proxy for tokens (model-agnostic): check the formatted
        # content (line-number prefixes — what enters context) before redaction.
        content_len = len(result.content or "")
        file_size = result_dict.get("file_size", 0)
        max_chars = _get_max_read_chars()
        if content_len > max_chars:
            # Graceful char-budget truncation (nearai/ironclaw#5029): trim to the
            # last complete line and offer `next_offset` instead of rejecting the
            # read — rescues "few but very long lines" that blow the char budget.
            total_lines = result_dict.get("total_lines", "unknown")
            trimmed, lines_kept, _ = _truncate_to_char_budget(
                result.content or "", max_chars
            )
            next_offset = offset + lines_kept
            shown_end = offset + lines_kept - 1
            result.content = trimmed
            result_dict["content"] = trimmed
            result_dict["truncated"] = True
            result_dict["truncated_by"] = "bytes"
            result_dict["next_offset"] = next_offset
            result_dict["hint"] = (
                f"Output truncated at the {max_chars:,}-char read budget after "
                f"{lines_kept} line(s) (showing lines {offset}-{shown_end} of "
                f"{total_lines}). Use offset={next_offset} to continue."
            )
            if len(trimmed.split("\n", 1)[0]) >= max_chars:
                result_dict["hint"] += (
                    " Note: the first line alone exceeded the budget and was "
                    "clamped mid-line; its remainder is not retrievable via "
                    "offset."
                )
            content_len = len(trimmed)

        # ── Redact secrets (after guard check to skip oversized content) ──
        if result.content:
            result.content = redact_sensitive_text(result.content, file_read=True)
            result_dict["content"] = result.content

        # Large-file hint: if the file is big and the caller didn't ask
        # for a narrow window, nudge toward targeted reads.
        if (file_size and file_size > _LARGE_FILE_HINT_BYTES
                and limit > 200
                and result_dict.get("truncated")):
            result_dict.setdefault("_hint", (
                f"This file is large ({file_size:,} bytes). "
                "Consider reading only the section you need with offset and limit "
                "to keep context usage efficient."
            ))

        # ── Track for consecutive-loop detection ──────────────────────
        read_key = ("read", path, offset, limit)
        with _read_tracker_lock:
            # Ensure "dedup" / "dedup_hits" keys exist (backward compat with
            # old tracker state from pre-dedup-guard sessions).
            if "dedup" not in task_data:
                task_data["dedup"] = {}
            if "dedup_hits" not in task_data:
                task_data["dedup_hits"] = {}
            # Real read succeeded — reset the stub-loop hit counter for this key.
            task_data["dedup_hits"].pop(dedup_key, None)
            task_data.setdefault("dedup_generation_reads", set()).add(dedup_key)
            task_data["read_history"].add((path, offset, limit))
            if task_data["last_key"] == read_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = read_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

            # Store mtime at read time: dedup skips unchanged re-reads; staleness warns
            # on write/patch when the file changed since the last read.
            if host_mtime_tracking:
                try:
                    _mtime_now = os.path.getmtime(resolved_str)
                    task_data["dedup"][dedup_key] = _mtime_now
                    task_data.setdefault("read_timestamps", {})[resolved_str] = _mtime_now
                except OSError:
                    pass  # Can't stat — skip tracking for this entry

            # Bound the per-task containers so a long CLI session doesn't
            # accumulate megabytes of dict/set state.  See _cap_read_tracker_data.
            _cap_read_tracker_data(task_data)

        # Cross-agent file-state registry (separate from the per-task tracker):
        # records reads so write/patch can detect sibling-subagent writes after
        # ours. Partial read when offset>1 or truncated. Outside _read_tracker_lock.
        _partial = (offset > 1) or bool(result_dict.get("truncated"))
        try:
            file_state.record_read(
                state_task_id, resolved_str, partial=_partial,
                namespace=state_namespace,
                stat_path=host_mtime_tracking,
            )
        except Exception:
            logger.debug("file_state.record_read failed", exc_info=True)

        # Background-review read-before-write guard (#61521): a review-fork
        # read_file on a skill file registers the read like skill_view does, so a
        # follow-up skill_manage(action='patch') is accepted; a partial read
        # doesn't count. No-op outside review forks (mark_background_review_skill_read
        # gates on is_background_review).
        if not _partial:
            try:
                from tools.skill_manager_guards import mark_background_review_skill_read

                mark_background_review_skill_read(Path(resolved_str))
            except Exception:
                logger.debug(
                    "background-review read-mark failed", exc_info=True
                )

        if count >= 4:
            # Hard block: stop returning content to break the loop
            return tool_error(
                f"BLOCKED: You have read this exact file region {count} times in a row. "
                "The content has NOT changed. You already have this information. "
                "STOP re-reading and proceed with your task.",
                path=path,
                already_read=count,
            )
        elif count >= 3:
            result_dict["_warning"] = (
                f"You have read this exact file region {count} times consecutively. "
                "The content has not changed since your last read. Use the information you already have. "
                "If you are stuck in a loop, stop reading and proceed with writing or responding."
            )

        result_dict.update(resolution.metadata(
            cwd=_authoritative_workspace_root(task_id, selected_target),
        ))

        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))




from tools.file_tools_read_tracking import reset_file_dedup  # noqa: E402,F401
from tools.file_tools_read_tracking import notify_other_tool_call  # noqa: E402,F401
def _invalidate_dedup_for_path(
    filepath: str,
    task_id: str,
    execution_target: str | None = None,
    *,
    _resolution: Any = None,
) -> None:
    """Remove all dedup cache entries whose resolved path matches *filepath*.

    Called after write_file and patch so that a subsequent read_file on
    the same path always returns fresh content instead of a stale
    "File unchanged" stub.  The dedup cache keys are tuples of
    ``(resolved_path, offset, limit)``; we must evict **all** offset/limit
    combinations for the written path because any cached range could now
    be stale.

    Must be called with ``_read_tracker_lock`` **not** held — acquires it
    internally.
    """
    from tools.execution_targets import resolve_execution_target
    resolution = _resolution or resolve_execution_target(execution_target)
    selected_target = resolution.target if resolution.named else None
    state_task_id = resolution.file_coordination_key(task_id)
    try:
        resolved = str(_resolve_path_for_task(
            filepath, task_id, selected_target, _resolution=resolution,
        ))
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(state_task_id)
        if task_data is None:
            return
        dedup = task_data.get("dedup")
        if dedup:
            # Collect keys to remove (can't mutate dict during iteration).
            stale_keys = [k for k in dedup if k[0] == resolved]
            for k in stale_keys:
                del dedup[k]
        # Also evict the negative-result cache: a write that creates the path means
        # subsequent reads/searches must hit disk.
        nf = task_data.get("not_found")
        if nf:
            nf.pop(("read", resolved), None)
            nf.pop(("search", resolved), None)


def _update_read_timestamp(
    filepath: str,
    task_id: str,
    execution_target: str | None = None,
    *,
    _resolution: Any = None,
) -> None:
    """Record the file's current modification time after a successful write.

    Called after write_file and patch so that consecutive edits by the
    same task don't trigger false staleness warnings — each write
    refreshes the stored timestamp to match the file's new state.

    Also invalidates the dedup cache for the written path so that
    subsequent reads return fresh content (fixes #13144).
    """
    # Invalidate dedup first (before acquiring lock for timestamp update).
    from tools.execution_targets import resolve_execution_target
    resolution = _resolution or resolve_execution_target(execution_target)
    selected_target = resolution.target if resolution.named else None
    state_task_id = resolution.file_coordination_key(task_id)
    _invalidate_dedup_for_path(
        filepath, task_id, selected_target, _resolution=resolution,
    )
    try:
        resolved = str(_resolve_path_for_task(
            filepath, task_id, selected_target, _resolution=resolution,
        ))
        current_mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(state_task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime
            _cap_read_tracker_data(task_data)


def _check_file_staleness(
    filepath: str,
    task_id: str,
    execution_target: str | None = None,
    *,
    _resolution: Any = None,
) -> str | None:
    """Check whether a file was modified since the agent last read it.

    Returns a warning string if the file is stale (mtime changed since
    the last read_file call for this task), or None if the file is fresh
    or was never read.  Does not block — the write still proceeds.
    """
    from tools.execution_targets import resolve_execution_target
    resolution = _resolution or resolve_execution_target(execution_target)
    selected_target = resolution.target if resolution.named else None
    state_task_id = resolution.file_coordination_key(task_id)
    try:
        resolved = str(_resolve_path_for_task(
            filepath, task_id, selected_target, _resolution=resolution,
        ))
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        task_data = _read_tracker.get(state_task_id)
        if not task_data:
            return None
        read_mtime = task_data.get("read_timestamps", {}).get(resolved)
    if read_mtime is None:
        return None  # File was never read — nothing to compare against
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None  # Can't stat — file may have been deleted, let write handle it
    if current_mtime != read_mtime:
        return (
            f"Warning: {filepath} was modified since you last read it "
            "(external edit or concurrent agent). The content you read may be "
            "stale. Consider re-reading the file to verify before writing."
        )
    return None


def _mark_verification_stale(
    task_id: str,
    resolved_paths: list[str],
    session_id: str | None = None,
    execution_target: str | None = None,
    *,
    _resolution: Any = None,
) -> None:
    """Best-effort note that successful edits made prior verification stale."""
    paths = [p for p in resolved_paths if p]
    if not paths:
        return
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import mark_workspace_edited

        cwd = None
        for path in paths:
            try:
                candidate = str(Path(path).parent)
            except Exception:
                continue
            if project_facts_for(candidate):
                cwd = candidate
                break
        if cwd is None:
            cwd = _authoritative_workspace_root(
                task_id, execution_target, _resolution=_resolution,
            )
        if cwd is None:
            try:
                cwd = str(Path(paths[0]).parent)
            except Exception:
                cwd = None
        mark_workspace_edited(session_id=session_id or task_id, cwd=cwd, paths=paths)
    except Exception:
        logger.debug("verification stale marker failed", exc_info=True)


def _check_binary_document_write(filepath: str, task_id: str = "default") -> str | None:
    """Reject text-tool writes that would corrupt a binary document.

    ``read_file`` auto-extracts .docx/.xlsx/.pptx (and PDF, via anydoc) to
    readable text, so the model plausibly believes it holds the file's
    contents and tries to write the edited text back with write_file/patch.
    A plain-text write can never produce a valid OOXML/OLE/ODF container, so
    that write silently destroys the document (port of nearai/ironclaw#7109).

    Rules:
    - Opaque container formats (.doc/.docx/.xls/.xlsx/.ppt/.pptx/.odt/.ods/
      .odp): always rejected — text bytes are never a valid document, whether
      creating or overwriting.
    - .pdf: rejected only when OVERWRITING an existing regular file. Raw PDF
      syntax is text-authorable, so new-file creation stays allowed.
    """
    if has_opaque_document_extension(filepath):
        ext = filepath[filepath.rfind("."):].lower()
        return (
            f"Refusing to write plain text to binary document '{filepath}' ({ext}). "
            "A text write cannot produce a valid document container and would "
            "corrupt the file (read_file showed you EXTRACTED text, not the real "
            "bytes). Use the docx/xlsx/powerpoint skills or a library like "
            "python-docx/openpyxl/python-pptx via the terminal to create or edit "
            "this document."
        )
    if is_pdf_path(filepath):
        try:
            resolved = Path(_resolve_path_for_task(filepath, task_id))
        except Exception:
            resolved = Path(_expand_tilde(filepath))
        try:
            if resolved.is_file():
                return (
                    f"Refusing to overwrite existing PDF '{filepath}' with plain text. "
                    "read_file showed you EXTRACTED text, not the real bytes — writing "
                    "text back would destroy the document. Use the pdf skill or a PDF "
                    "library via the terminal to modify it. (Creating a NEW .pdf file "
                    "is allowed.)"
                )
        except OSError:
            pass
    return None


# Whole-file rewrite hint: an overwrite of an existing file this large whose new content keeps at least
# this fraction of the old lines is a patch written the expensive way. In one 1,393-agent run 661 such
# rewrites of >20k-char files cost ~25M output chars (~$155) where `patch` averaged 1.3k chars/call.
_REWRITE_HINT_MIN_CHARS = 20_000
_REWRITE_HINT_MIN_UNCHANGED = 0.80


# Above this the line diff is skipped: SequenceMatcher on pathological repeated-line files is
# quadratic (a 460 KB same-line file took ~22 s under the write lock).
_REWRITE_HINT_MAX_CHARS = 400_000


def _whole_file_rewrite_hint(task_id: str, resolved: str | None, new_content: str, *, file_ops=None) -> str | None:
    """Return a hint when ``new_content`` mostly re-sends what is already at ``resolved``.

    Reads the OLD content through the change site's own file ops (``read_file_raw``, the
    sandbox/remote backend the write targets — pass ``file_ops`` for a named target, else the
    task's default ops), never the host path: on a remote backend the host file is a
    different file, and a host FIFO at that path would block the write lock forever. Bounded size
    and a line multiset comparison (linear) instead of a sequence diff (quadratic on repeated lines)."""
    if not resolved or not (_REWRITE_HINT_MIN_CHARS <= len(new_content) <= _REWRITE_HINT_MAX_CHARS):
        return None
    try:
        ops = file_ops if file_ops is not None else _get_file_ops(task_id)
        result = ops.read_file_raw(resolved)
        old = getattr(result, "content", None)
        if getattr(result, "error", None) or not isinstance(old, str):
            return None
    except Exception:
        return None
    if not (_REWRITE_HINT_MIN_CHARS <= len(old) <= _REWRITE_HINT_MAX_CHARS):
        return None
    old_lines, new_lines = old.splitlines(), new_content.splitlines()
    if not old_lines:
        return None
    from collections import Counter
    unchanged = sum((Counter(old_lines) & Counter(new_lines)).values())
    ratio = unchanged / max(len(old_lines), len(new_lines))
    if ratio < _REWRITE_HINT_MIN_UNCHANGED:
        return None
    changed = max(len(old_lines), len(new_lines)) - unchanged
    return (
        f"{unchanged:,} of {len(new_lines):,} lines were already on disk ({ratio:.0%} unchanged); ~{changed:,} "
        f"line(s) actually changed. Re-sending a {len(new_content):,}-char file costs output tokens for every "
        "unchanged line; for edits like this use patch (old_string/new_string), which sends only the changed region."
    )


def write_file_tool(path: str, content: str, task_id: str = "default",
                    cross_profile: bool = False,
                    session_id: str | None = None,
                    target: str = None) -> str:
    """Write content to a file.

    ``cross_profile`` bypasses the #32049 sandbox-mirror lost-write
    guards (writes the host process would never read). Unadvertised in
    the schema — the mirror rejection error teaches it. The cross-PROFILE
    guard this flag was named for is removed (profiles are not isolated).
    """
    try:
        from tools.execution_targets import resolve_execution_target
        resolution = resolve_execution_target(target)
    except Exception as exc:
        return tool_error(str(exc))
    selected_target = resolution.target if resolution.named else None
    host_mtime_tracking = resolution.backend == "local"
    state_task_id = resolution.file_coordination_key(task_id)
    state_namespace = _file_state_namespace(
            task_id, selected_target, _resolution=resolution,
        )
    sensitive_err = _check_sensitive_path(
        path, task_id, selected_target, _resolution=resolution,
    )
    if sensitive_err:
        return tool_error(sensitive_err)
    binary_doc_err = _check_binary_document_write(path, task_id)
    if binary_doc_err:
        return tool_error(binary_doc_err)
    protected_err = _check_protected_instruction_write([path], task_id)
    if protected_err:
        return tool_error(protected_err)
    approval_err = _check_approval_required_write([path], task_id)
    if approval_err:
        return tool_error(approval_err)
    if not cross_profile:
        cross_warning = _check_cross_profile_path(
            path, task_id, selected_target, _resolution=resolution,
        )
        if cross_warning:
            return tool_error(cross_warning)
    if _is_internal_file_tool_content(content):
        return tool_error(
            "Refusing to write internal read_file display text as file content. "
            "Strip read_file line-number prefixes or reconstruct the intended "
            "file contents before writing."
        )
    try:
        # Resolve once for the registry lock + stale check; on failure fall back to
        # the legacy path (write proceeds, per-task staleness still runs).
        try:
            _resolved = str(_resolve_path_for_task(
                path, task_id, selected_target, _resolution=resolution,
            ))
        except Exception:
            _resolved = None

        if _resolved is None:
            stale_warning = _check_file_staleness(
                path, task_id, selected_target, _resolution=resolution,
            )
            file_ops = _file_ops_for_resolution(task_id, resolution)
            result = file_ops.write_file(path, content)
            result_dict = result.to_dict()
            if stale_warning:
                result_dict["_warning"] = stale_warning
            if not result_dict.get("error"):
                _mark_verification_stale(
                    task_id, [path], session_id=session_id,
                    execution_target=selected_target,
                    _resolution=resolution,
                )
            _update_read_timestamp(
                path, task_id, selected_target, _resolution=resolution,
            )
            result_dict.update(resolution.metadata(
                cwd=_authoritative_workspace_root(task_id, selected_target),
            ))
            return json.dumps(result_dict, ensure_ascii=False)

        # Serialize read→modify→write per-path so concurrent subagents can't
        # interleave on the same file; different paths stay fully parallel.
        with file_state.lock_path(_resolved, namespace=state_namespace):
            # Cross-agent staleness wins over the per-task warning — it names the sibling.
            cross_warning = file_state.check_stale(
                state_task_id, _resolved, namespace=state_namespace,
            )
            stale_warning = _check_file_staleness(
                path, task_id, selected_target, _resolution=resolution,
            )
            # Workspace-divergence warning: relative path resolving outside the
            # terminal's cwd (the worktree-cwd bug). Lowest priority of the three.
            cwd_warning = _path_resolution_warning(
                path, Path(_resolved), task_id, selected_target,
                _resolution=resolution,
            )
            file_ops = _file_ops_for_resolution(task_id, resolution)
            operation_path = _backend_operation_path(
                path, _resolved, task_id, selected_target,
                _resolution=resolution,
            )
            rewrite_hint = _whole_file_rewrite_hint(task_id, operation_path, content, file_ops=file_ops)
            result = file_ops.write_file(operation_path, content)
            result_dict = result.to_dict()
            effective_warning = cross_warning or stale_warning or cwd_warning
            if effective_warning:
                result_dict["_warning"] = effective_warning
            if rewrite_hint and not result_dict.get("error"):
                result_dict["hint"] = rewrite_hint
            # Always report the ABSOLUTE path actually written, so a wrong-cwd
            # mismatch is visible instead of silently routing the edit elsewhere.
            result_dict["resolved_path"] = operation_path
            if not result_dict.get("error"):
                result_dict["files_modified"] = [operation_path]
                _mark_verification_stale(
                    task_id, [operation_path], session_id=session_id,
                    execution_target=selected_target,
                    _resolution=resolution,
                )
            # Refresh stamps after the successful write so consecutive
            # writes by this task don't trigger false staleness warnings.
            _update_read_timestamp(
                path, task_id, selected_target, _resolution=resolution,
            )
            if not result_dict.get("error"):
                file_state.note_write(
                    state_task_id, _resolved, namespace=state_namespace,
                    stat_path=host_mtime_tracking,
                )
        result_dict.update(resolution.metadata(
            cwd=_authoritative_workspace_root(task_id, selected_target),
        ))
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s: %s", type(e).__name__, e)
        else:
            logger.error("write_file error: %s: %s", type(e).__name__, e, exc_info=True)
        return tool_error(str(e))
def _collect_v4a_header_paths(patch: str) -> tuple[list[str], list[str]] | str:
    """Extract every path named in V4A headers, rejecting ``..`` traversal.

    Returns ``(all_paths, content_write_paths)`` or a tool_error string. Header
    paths come from patch CONTENT (more attacker-influenceable than ``path=``,
    which keeps its legitimate ``..`` use). Move headers check BOTH endpoints;
    only Update/Add write text and feed the binary-document guard.
    """
    from tools.path_security import has_traversal_component

    headers = [(m.group(3), m.group(2) in ("Update", "Add")) for m in _V4A_SINGLE_HEADER_RE.finditer(patch)]
    headers += [(g, False) for m in _V4A_MOVE_HEADER_RE.finditer(patch) for g in (m.group(2), m.group(3))]
    paths: list[str] = []
    content_paths: list[str] = []
    for raw, writes_text in headers:
        v4a_path = raw.strip()
        if has_traversal_component(v4a_path):
            return tool_error(
                f"V4A patch header contains '..' traversal: {v4a_path!r}. "
                "Use the agent's cwd-relative path (no '..') or an absolute "
                "path in '*** Update File:' / '*** Add File:' / "
                "'*** Delete File:' / '*** Move File:' headers.")
        paths.append(v4a_path)
        if writes_text:
            content_paths.append(v4a_path)
    return paths, content_paths


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default", cross_profile: bool = False,
               session_id: str | None = None, target: str = None) -> str:
    """Patch a file using replace mode or V4A patch format.

    ``cross_profile``: same semantics as ``write_file``'s flag (mirror-guard
    bypass only; unadvertised).
    """
    try:
        from tools.execution_targets import resolve_execution_target
        resolution = resolve_execution_target(target)
    except Exception as exc:
        return tool_error(str(exc))
    selected_target = resolution.target if resolution.named else None
    host_mtime_tracking = resolution.backend == "local"
    state_task_id = resolution.file_coordination_key(task_id)
    state_namespace = _file_state_namespace(
            task_id, selected_target, _resolution=resolution,
        )

    # Check sensitive paths for both replace (explicit path) and V4A patch (extract paths)
    _paths_to_check = []
    # Paths whose CONTENT will be text-written (Update/Add + explicit path); Delete/Move skip the binary-document guard.
    _content_write_paths = []
    if path:
        _paths_to_check.append(path)
        _content_write_paths.append(path)
    if mode == "patch" and patch:
        import re as _re
        from tools.path_security import has_traversal_component
        def _reject_v4a_traversal(v4a_path: str) -> str | None:
            # V4A path headers come from patch CONTENT (attacker-influenceable:
            # skill content, web extract, prompt injection), so reject ``..``
            # traversal — a legitimate patch can use absolute or cwd-relative
            # paths. The explicit ``path=`` arg keeps ``..`` (legitimate from a
            # worktree, e.g. ``patch path="../other_module/x.py"``).
            if has_traversal_component(v4a_path):
                return tool_error(
                    f"V4A patch header contains '..' traversal: {v4a_path!r}. "
                    "Use the agent's cwd-relative path (no '..') or an absolute "
                    "path in '*** Update File:' / '*** Add File:' / "
                    "'*** Delete File:' / '*** Move File:' headers."
                )
            return None

        # ``\s*`` (not ``\s+``) after ``***``: patch_parser accepts no-space headers,
        # which would otherwise parse + apply while skipping this check.
        for _m in _re.finditer(r'^\*\*\*\s*(Update|Add|Delete)\s+File:\s*(.+)$', patch, _re.MULTILINE):
            _op = _m.group(1)
            v4a_path = _m.group(2).strip()
            _err = _reject_v4a_traversal(v4a_path)
            if _err:
                return _err
            _paths_to_check.append(v4a_path)
            if _op in ("Update", "Add"):
                _content_write_paths.append(v4a_path)
        # ``*** Move File:`` (patch_parser.py:114) was never extracted, so a Move to
        # /etc/crontab skipped the sensitive-path pre-check; check BOTH endpoints.
        for _m in _re.finditer(r'^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$', patch, _re.MULTILINE):
            for v4a_path in (_m.group(1).strip(), _m.group(2).strip()):
                _err = _reject_v4a_traversal(v4a_path)
                if _err:
                    return _err
                _paths_to_check.append(v4a_path)
    for _p in _paths_to_check:
        sensitive_err = _check_sensitive_path(
            _p, task_id, selected_target, _resolution=resolution,
        )
        if sensitive_err:
            return tool_error(sensitive_err)
        if not cross_profile:
            cross_warning = _check_cross_profile_path(
                _p, task_id, selected_target, _resolution=resolution,
            )
            if cross_warning:
                return tool_error(cross_warning)
    for _p in _content_write_paths:
        binary_doc_err = _check_binary_document_write(_p, task_id)
        if binary_doc_err:
            return tool_error(binary_doc_err)
    # One approval prompt for the whole patch: a single protected file gates
    # the ENTIRE patch (deny applies nothing — see the helper's docstring).
    protected_err = _check_protected_instruction_write(_paths_to_check, task_id)
    if protected_err:
        return tool_error(protected_err)
    approval_err = _check_approval_required_write(_paths_to_check, task_id)
    if approval_err:
        return tool_error(approval_err)
    try:
        # Resolve paths for locking, ordered + deduplicated so concurrent callers lock
        # in the same order — prevents deadlock on overlapping multi-file V4A patches.
        _resolved_paths: list[str] = []
        _seen: set[str] = set()
        for _p in _paths_to_check:
            try:
                _r = str(_resolve_path_for_task(
                    _p, task_id, selected_target, _resolution=resolution,
                ))
            except Exception:
                _r = None
            if _r and _r not in _seen:
                _resolved_paths.append(_r)
                _seen.add(_r)
        _resolved_paths.sort()

        # Acquire per-path locks in sorted order via ExitStack; a single path
        # degenerates to one lock, an empty list to a no-op.
        from contextlib import ExitStack
        with ExitStack() as _locks:
            for _r in _resolved_paths:
                _locks.enter_context(
                    file_state.lock_path(_r, namespace=state_namespace)
                )

            # Collect warnings — cross-agent registry first (names sibling),
            # then per-task tracker as a fallback.
            stale_warnings: list[str] = []
            _path_to_resolved: dict[str, str] = {}
            for _p in _paths_to_check:
                try:
                    _r = str(_resolve_path_for_task(
                    _p, task_id, selected_target, _resolution=resolution,
                ))
                except Exception:
                    _r = None
                _path_to_resolved[_p] = _r
                _cross = file_state.check_stale(
                    state_task_id, _r, namespace=state_namespace,
                ) if _r else None
                _sw = _cross or _check_file_staleness(
                    _p, task_id, selected_target, _resolution=resolution,
                )
                if not _sw and _r:
                    # Workspace-divergence warning (worktree-cwd bug): relative
                    # path resolving outside the terminal's cwd.
                    _sw = _path_resolution_warning(
                        _p, Path(_r), task_id, selected_target,
                        _resolution=resolution,
                    )
                if _sw:
                    stale_warnings.append(_sw)

            file_ops = _file_ops_for_resolution(task_id, resolution)

            if mode == "replace":
                if not path:
                    return tool_error("path required")
                if old_string is None or new_string is None:
                    return tool_error("old_string and new_string required")
                # Pass the resolved ABSOLUTE path to the shell layer: its own cwd may
                # differ (worktree-cwd bug), and a relative path would let the layers
                # disagree about which file is edited.
                _replace_target = _backend_operation_path(
                    path, _path_to_resolved.get(path) or path,
                    task_id, selected_target,
                    _resolution=resolution,
                )
                result = file_ops.patch_replace(_replace_target, old_string, new_string, replace_all)
            elif mode == "patch":
                if not patch:
                    return tool_error("patch content required")
                if resolution.backend == "ssh":
                    patch_for_ops = _backend_v4a_patch(
                        patch, task_id, selected_target, _resolution=resolution,
                    )
                else:
                    # Preserve current-upstream absolute-path reconciliation for
                    # local/container backends; SSH paths must stay remote.
                    patch_for_ops = _rewrite_v4a_patch_paths_for_host(
                        patch, _path_to_resolved, file_ops
                    )
                result = file_ops.patch_v4a(patch_for_ops)
            else:
                return tool_error(f"Unknown mode: {mode}")

            result_dict = result.to_dict()
            if stale_warnings:
                result_dict["_warning"] = stale_warnings[0] if len(stale_warnings) == 1 else " | ".join(stale_warnings)
            # Report the ABSOLUTE path(s) actually patched so a wrong-cwd mismatch
            # (worktree session editing the main checkout) is visible, not silent.
            _resolved_modified = [
                _backend_operation_path(
                    _p, _path_to_resolved.get(_p) or _p,
                    task_id, selected_target,
                    _resolution=resolution,
                )
                for _p in _paths_to_check
            ]
            # Refresh stored timestamps for all successfully-patched paths so
            # consecutive edits by this task don't trigger false warnings.
            if not result_dict.get("error"):
                result_dict["files_modified"] = _resolved_modified
                if len(_resolved_modified) == 1:
                    result_dict["resolved_path"] = _resolved_modified[0]
                _mark_verification_stale(
                    task_id, _resolved_modified, session_id=session_id,
                    execution_target=selected_target,
                    _resolution=resolution,
                )
                for _p in _paths_to_check:
                    _update_read_timestamp(
                        _p, task_id, selected_target, _resolution=resolution,
                    )
                    _r = _path_to_resolved.get(_p)
                    if _r:
                        file_state.note_write(
                            state_task_id, _r, namespace=state_namespace,
                            stat_path=host_mtime_tracking,
                        )
                # Successful patch: clear consecutive-failure counters so a future
                # failure on the same path restarts the escalation cycle.
                _reset_patch_failures(state_task_id, [
                    _r for _r in (_path_to_resolved.get(_p) for _p in _paths_to_check) if _r
                ])
        # Hint when old_string not found (stale-content retries waste iterations);
        # suppressed when patch_replace attached a richer "Did you mean?" snippet.
        if result_dict.get("error") and "Could not find" in str(result_dict["error"]):
            # Track per-file consecutive failures for replace mode only — V4A
            # failures are rarer and the existing _hint covers them.
            failure_count = 0
            if mode == "replace" and path:
                resolved = _path_to_resolved.get(path) or path
                failure_count = _record_patch_failure(state_task_id, resolved)

            if failure_count >= 3:
                # Escalating hint after repeated failures on one path — usually a
                # stale view: same old_string against changed content. The count
                # tells the model it's looping and to re-read or use write_file.
                result_dict["_hint"] = (
                    f"This is failure #{failure_count} patching {path!r}. "
                    "Stop retrying with variations of the same old_string. "
                    "Either: (1) re-read the file fresh to verify current "
                    "content, (2) use a longer / more unique old_string with "
                    "surrounding context lines, or (3) use write_file to "
                    "replace the entire file if the targeted region is hard "
                    "to anchor."
                )
            elif "Did you mean one of these sections?" not in str(result_dict["error"]):
                result_dict["_hint"] = (
                    "old_string not found. Use read_file to verify the current "
                    "content, or search_files to locate the text."
                )
        result_dict.update(resolution.metadata(
            cwd=_authoritative_workspace_root(task_id, selected_target),
        ))
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))
def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                order: str = "discovery",
                task_id: str = "default", execution_target: str = None) -> str:
    """Search for content or files."""
    try:
        from tools.execution_targets import resolve_execution_target

        resolution = resolve_execution_target(execution_target)
        selected_target = resolution.target if resolution.named else None
        state_task_id = resolution.file_coordination_key(task_id)
        offset, limit = normalize_search_pagination(offset, limit)

        # Track searches to detect *consecutive* repeated search loops; include
        # pagination args so paging through truncated results doesn't trip the guard.
        search_key = (
            "search",
            pattern,
            target,
            str(path),
            file_glob or "",
            limit,
            offset,
            order,
        )
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(state_task_id, {
                "last_key": None, "consecutive": 0, "read_history": set(),
            })
            if task_data["last_key"] == search_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = search_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

        if count >= 4:
            return tool_error(
                f"BLOCKED: You have run this exact search {count} times in a row. "
                "The results have NOT changed. You already have this information. "
                "STOP re-searching and proceed with your task.",
                pattern=pattern,
                already_searched=count,
            )

        try:
            resolved_path = _resolve_path_for_task(
                path, task_id, selected_target, _resolution=resolution,
            )
        except (OSError, ValueError, RuntimeError):
            resolved_path = None
        block_error = get_read_block_error(str(resolved_path) if resolved_path else path)
        if block_error:
            return tool_error(block_error)

        # ── Negative-result cache ─────────────────────────────────────
        # Cache "Path not found: <path>" for missing search roots so a retry skips
        # the expensive parent-directory listing (file_operations.py:1402).
        resolved_search_path = str(resolved_path) if resolved_path is not None else path
        cached_search_nf = _check_not_found_cache(
            "search", resolved_search_path, state_task_id,
            check_host_filesystem=resolution.backend == "local",
        )
        if cached_search_nf is not None:
            return cached_search_nf

        file_ops = _file_ops_for_resolution(task_id, resolution)
        operation_path = _backend_operation_path(
            path, resolved_path or path, task_id, selected_target,
            _resolution=resolution,
        )
        result = file_ops.search(
            pattern=pattern, path=operation_path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context,
            order=order,
        )
        omitted = _filter_read_blocked_search_results(
            result, task_id, selected_target, _resolution=resolution,
        )
        if hasattr(result, 'matches'):
            for m in result.matches:
                if hasattr(m, 'content') and m.content:
                    m.content = redact_sensitive_text(m.content, file_read=True)
        result_dict = result.to_dict(densify=True)

        if omitted:
            result_dict["_omitted"] = (
                f"{omitted} result(s) omitted because they target credential, "
                "token, cache, or secret-bearing environment files."
            )

        # Populate negative cache when the search root was missing. No early return —
        # same rationale as the read path: error results keep flowing through bookkeeping.
        _search_err = result_dict.get("error") or ""
        if isinstance(_search_err, str) and _search_err.startswith("Path not found:"):
            _search_nf_json = json.dumps(result_dict, ensure_ascii=False)
            _record_not_found(
                "search", resolved_search_path, state_task_id, _search_nf_json,
            )

        if count >= 3:
            result_dict["_warning"] = (
                f"You have run this exact search {count} times consecutively. "
                "The results have not changed. Use the information you already have."
            )

        result_dict.update(resolution.metadata(
            cwd=_authoritative_workspace_root(task_id, selected_target),
        ))

        # Structured like ``_warning`` above: text appended after the JSON
        # breaks every json.loads consumer (execute_code RPC, strict tool-message
        # providers) — #90322.
        if result_dict.get("truncated"):
            result_dict["_hint"] = (
                f"Results truncated. Use offset={offset + limit} to see more, "
                "or narrow with a more specific pattern or file_glob."
            )
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))




# ---------------------------------------------------------------------------
# Schemas + Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error
def _check_file_reqs():
    """Lazy wrapper to avoid circular import with tools/__init__.py."""
    from tools import check_file_requirements
    return check_file_requirements()

READ_FILE_SCHEMA = {
    "name": "read_file",
    # Document formats are stated unconditionally: firecrawl-anydoc is a
    # core dependency (bundled), so its absence is a broken install, not a
    # configuration — the teaching error in read_extract handles that rare
    # case with the pip-install fix. The ONE dynamic word: "PDF (text
    # layer)" upgrades to "PDF (scanned or text)" when hosted OCR has a
    # route we trust (_read_file_schema_overrides). Scanned-page coverage
    # teaching lives in the response-time NEEDS-OCR warning
    # (read_extract.py); the schema doesn't pre-teach it.
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are truncated on a line boundary and return a next_offset; continue with offset to read the rest. Documents auto-extract to readable text: .ipynb, Office (.docx/.xlsx/.pptx and legacy .doc/.ppt/.xls), PDF (text layer), OpenDocument, RTF, EPUB. Cannot read images/binary — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read (default: 2000, max: 2000). Reads are additionally capped at a ~100K-character budget with a next_offset continuation.", "default": DEFAULT_READ_LIMIT, "maximum": 2000},
            "target": {"type": "string", "description": "Optional named execution target, for example 'local' or 'devbox'. Uses terminal.default_target when omitted."},
            "runtime_scope": {"type": "string", "description": "Immutable producing-runtime scope from a saved-output hint. Pass only when the hint supplies it."},
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out). The result's verified:true means the on-disk content hash was confirmed — do NOT re-read the file to check the write landed.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"},
            "target": {"type": "string", "description": "Optional named execution target, for example 'local' or 'devbox'. Uses terminal.default_target when omitted."},
            # NOTE: the handler still accepts `cross_profile` (bool) — it now
            # bypasses only the #32049 sandbox-mirror lost-write guards, whose
            # rejection error teaches it. Unadvertised: the cross-PROFILE
            # guard it was named for was removed (profiles are not isolated,
            # maintainer decision), and mirror hits are rare + self-teaching.
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    # BASE = replace-only (what nearly every model family was trained on).
    # The V4A patch mode (mode + patch params, dual-mode description) is
    # LAYERED ON dynamically for OpenAI-family mains only — V4A is the
    # OpenAI apply_patch dialect their models emit natively; advertising
    # it to everyone cost every other session ~148 tok/call
    # (_patch_schema_overrides below). The handler accepts BOTH shapes
    # from any model regardless (replay compat + strong models that know
    # V4A anyway): mode defaults to 'replace' when omitted.
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing. "
        "Finds a unique string and replaces it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "Changed replacement text; it must differ from old_string. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            "target": {"type": "string", "description": "Optional named execution target, for example 'local' or 'devbox'. Uses terminal.default_target when omitted."},
            # NOTE: handler still accepts `cross_profile` — see write_file's
            # NOTE (mirror-guard bypass only; unadvertised by design).
            # NOTE: handler still accepts `mode` + `patch` (V4A) from ANY
            # model — the schema just doesn't advertise them off-family.
        },
        "required": ["path", "old_string", "new_string"],
    },
}


# V4A layer, rendered only for OpenAI-family main models (see PATCH_SCHEMA
# comment). Kept as data so the override composes it deterministically.
_PATCH_V4A_DESCRIPTION = (
    "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
    "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
    "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
    "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
    "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
    "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
    "REQUIRED PARAMETERS: mode, patch."
)

_PATCH_V4A_PARAMS = {
    "mode": {
        "type": "string",
        "enum": ["replace", "patch"],
        "description": "Edit mode. 'replace' (default): requires path + old_string + new_string. 'patch': requires patch content only.",
        "default": "replace",
    },
    "patch": {
        "type": "string",
        "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
    },
}


def _is_openai_family_main() -> bool:
    """Whether the active main provider/model is the OpenAI/codex family —
    the population trained on the V4A apply_patch dialect.

    Provider-family-coarse on purpose (no per-model training-diet table to
    go stale): direct OpenAI providers always qualify; on aggregators
    (openrouter/nous/azure...) the MODEL slug decides (gpt-*/o-series/
    codex). Fail-closed to the universal replace-only schema.
    """
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider

        provider = (_read_main_provider() or "").strip().lower()
        model = (_read_main_model() or "").strip().lower()
    except Exception:  # noqa: BLE001
        return False
    if provider in {"openai", "openai-chat", "openai-codex", "azure-openai", "codex"}:
        return True
    # Aggregators: the model slug carries the family.
    slug = model.split("/", 1)[-1]
    if slug.startswith(("gpt-", "gpt.", "chatgpt", "codex", "o1", "o3", "o4", "o5")):
        return True
    return "openai/" in model


SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents. On macOS, broad searches above the user home automatically skip TCC-protected folders (Desktop, Documents, Downloads, Library, Movies, Music, Pictures); target one directly when access is intentional.\n\nCompatibility exception: target still selects search mode ('content' or 'files'); execution_target selects the named terminal execution target.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls. Discovery order is the fast bounded default; exact global newest-first order is an explicit opt-in and may scan the full tree.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "order": {"type": "string", "enum": ["discovery", "modified"], "description": "File-search order: 'discovery' is fast bounded traversal order; 'modified' is exact global newest-first and may scan the full tree; ignored for content", "default": "discovery"},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0},
            "execution_target": {"type": "string", "description": "Optional named execution target, for example 'local' or 'devbox'. This is separate from target, which selects content/files search mode."},
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(
        path=args.get("path", ""), offset=args.get("offset", 1),
        limit=args.get("limit", DEFAULT_READ_LIMIT), task_id=tid, target=args.get("target"),
        runtime_scope=args.get("runtime_scope"),
    )


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    if not args.get("path") or not isinstance(args.get("path"), str):
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with "
            "both 'path' and 'content' set."
        )
    if "content" not in args:
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a "
            "path but no content argument — this is almost always a dropped-arg bug "
            "under context pressure. Re-emit the tool call with the full content "
            "payload, or use execute_code with hermes_tools.write_file() for very "
            "large files."
        )
    if not isinstance(args["content"], str):
        return tool_error(
            f"write_file: 'content' must be a string, got "
            f"{type(args['content']).__name__}."
        )
    return write_file_tool(
        path=args["path"], content=args["content"], task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
        target=args.get("target"),
    )


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid,
        cross_profile=bool(args.get("cross_profile", False)),
        session_id=kw.get("session_id"),
        target=args.get("target"),
    )


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0),
        order=args.get("order", "discovery"), task_id=tid,
        execution_target=args.get("execution_target"),
    )


def _read_file_schema_overrides():
    """One-word capability upgrade: "PDF (text layer)" → "PDF (scanned or
    text)" when hosted OCR has a trusted route (see
    read_extract.hosted_ocr_available). Config/env probe only — no
    network at schema-build time. Compaction's tool refresh (#97073)
    picks up a key added mid-session.
    """
    try:
        from tools.read_extract import hosted_ocr_available

        if hosted_ocr_available():
            return {
                "description": READ_FILE_SCHEMA["description"].replace(
                    "PDF (text layer)", "PDF (scanned or text)"
                )
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read_file, check_fn=_check_file_reqs, emoji="📖", max_result_size_chars=100_000, dynamic_schema_overrides=_read_file_schema_overrides)
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA, handler=_handle_write_file, check_fn=_check_file_reqs, emoji="✍️", max_result_size_chars=100_000)
def _patch_schema_overrides():
    """Layer the V4A patch mode onto the base replace-only schema for
    OpenAI-family mains (see PATCH_SCHEMA comment). Config/context probe
    only — no I/O at schema-build time; compaction's tool refresh
    (#97073) re-evaluates on model switches."""
    try:
        if not _is_openai_family_main():
            return {}
        params = {
            "type": "object",
            "properties": {
                "mode": _PATCH_V4A_PARAMS["mode"],
                **PATCH_SCHEMA["parameters"]["properties"],
                "patch": _PATCH_V4A_PARAMS["patch"],
            },
            "required": ["mode"],
        }
        return {"description": _PATCH_V4A_DESCRIPTION, "parameters": params}
    except Exception:  # noqa: BLE001
        return {}


registry.register(name="patch", toolset="file", schema=PATCH_SCHEMA, handler=_handle_patch, check_fn=_check_file_reqs, emoji="🔧", max_result_size_chars=100_000, dynamic_schema_overrides=_patch_schema_overrides)
registry.register(name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA, handler=_handle_search_files, check_fn=_check_file_reqs, emoji="🔎", max_result_size_chars=100_000)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import PurePosixPath  # noqa: F401,E402
import posixpath  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'has_opaque_document_extension': ('tools.binary_extensions', 'has_opaque_document_extension'),
    'is_pdf_path': ('tools.binary_extensions', 'is_pdf_path'),
    'notify_other_tool_call': ('tools.file_tools_read_tracking', 'notify_other_tool_call'),
    'reset_file_dedup': ('tools.file_tools_read_tracking', 'reset_file_dedup'),
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
