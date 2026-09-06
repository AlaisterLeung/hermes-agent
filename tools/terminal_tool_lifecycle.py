"""Sandbox lifecycle for the terminal tool: idle reaping, teardown, manual/atexit
cleanup, and the lazy ensure_task_env bring-up. The env cache dicts and locks
stay in tools.terminal_tool (tests patch them there) and are read through it
at call time.

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""

import glob
import logging
import inspect
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Hashable, Optional
from tools.environments.singularity import _get_scratch_dir
from tools.terminal_tool_backends import (
    _container_config_from_config,
    _ssh_config_from_config,
)
from tools.terminal_tool_config import _quiet

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


# Advisory disk-usage check; cached so the recursive scan doesn't run on
# every command (a result up to 5 minutes stale is harmless).
_disk_usage_cache: dict = {"timestamp": 0.0, "result": False}

_DISK_USAGE_CACHE_TTL = 300.0  # seconds


def _scratch_paths():
    return glob.glob(str(_get_scratch_dir() / "hermes-*"))


def _check_disk_usage_warning():
    """True when hermes scratch dirs exceed the warning threshold (cached, advisory)."""
    from tools.terminal_tool import DISK_USAGE_WARNING_THRESHOLD_GB
    if time.monotonic() - _disk_usage_cache["timestamp"] < _DISK_USAGE_CACHE_TTL:
        return _disk_usage_cache["result"]
    try:
        total_bytes = 0
        for path in _scratch_paths():
            for f in Path(path).rglob('*'):
                if f.is_file():
                    with _quiet("Could not stat file %s", f, exc=OSError):
                        total_bytes += f.stat().st_size
        total_gb = total_bytes / (1024 ** 3)
        exceeded = total_gb > DISK_USAGE_WARNING_THRESHOLD_GB
        if exceeded:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
        _disk_usage_cache["timestamp"] = time.monotonic()
        _disk_usage_cache["result"] = exceeded
        return exceeded
    except Exception:
        # Don't update cache on error so the next call retries.
        logger.debug("Disk usage warning check failed", exc_info=True)
        return False


def _create_configured_env(
    config: Dict[str, Any], env_type: str, *, image: str, cwd: str, timeout: int,
    task_id: str, host_cwd: Optional[str], local_config: Optional[dict] = None,
):
    """``_create_environment`` with the ssh/container kwargs shaped from *config*
    (shared by the terminal tool and the lazy :func:`ensure_task_env` bring-up)."""
    from tools.terminal_tool_backends import _create_environment
    from tools.terminal_tool_config import _is_container_backend
    return _create_environment(
        env_type=env_type, image=image, cwd=cwd, timeout=timeout,
        ssh_config=_ssh_config_from_config(config) if env_type == "ssh" else None,
        container_config=(
            _container_config_from_config(config) if _is_container_backend(env_type) else None
        ),
        local_config=local_config, task_id=task_id, host_cwd=host_cwd,
    )


def _cleanup_env(env: Any, *, force_remove: Optional[bool] = None) -> None:
    """Tear down one environment via cleanup()/stop()/terminate(), whichever it has.

    ``force_remove`` is forwarded to ``cleanup()`` only when given and the backend's
    signature accepts it (``DockerEnvironment``, issue #20561; other backends don't).
    Shared by ``cleanup_vm``, the idle reaper and the prompt-time backend probe so
    the signature check lives in one place.
    """
    if hasattr(env, 'cleanup'):
        if force_remove is not None and "force_remove" in inspect.signature(env.cleanup).parameters:
            env.cleanup(force_remove=force_remove)
        else:
            env.cleanup()
    elif hasattr(env, 'stop'):
        env.stop()
    elif hasattr(env, 'terminate'):
        env.terminate()


def _teardown_env(env: Any, task_id: str, *, force_remove: Optional[bool] = None, done_msg: str = "Cleaned up inactive environment for task: %s") -> None:
    """``_cleanup_env`` plus outcome logging. A 404/"not found" error means the
    sandbox is already gone — logged at info."""
    try:
        _cleanup_env(env, force_remove=force_remove)
        logger.info(done_msg, task_id)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _clear_file_ops_cache(task_id: str) -> None:
    """Invalidate the file_ops cache entry so ShellFileOperations can't reference a dead sandbox."""
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass


def _unregister_env(task_id: str):
    """Pop *task_id* from the env cache, activity map and creation locks; return
    the env (or None). Callers run the (slow) teardown OUTSIDE the lock —
    Modal/Docker teardown can block 10-15s and would stall every concurrent
    terminal/file tool call."""
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _last_activity,
    )
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)
    return env


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _last_activity, _active_turns_for_environment_key,
    )
    current_time = time.time()

    # Check the process registry -- skip cleanup for sandboxes with active
    # background processes (their _last_activity gets refreshed to keep them alive).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # Phase 1: collect stale entries and remove them from tracking dicts while
    # holding the lock.  Do NOT call env.cleanup() inside the lock -- Modal and
    # Docker teardown can block for 10-15s, which would stall every concurrent
    # terminal/file tool call waiting on _env_lock.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if _active_turns_for_environment_key(task_id) > 0:
                # An active tool or overlapping logical turn owns this runtime.
                # Refresh activity so it gets a complete idle window afterward.
                _last_activity[task_id] = current_time
                continue
            tracked_env = _active_environments.get(task_id)
            effective_lifetime = getattr(
                tracked_env, "_hermes_lifetime_seconds", lifetime_seconds,
            )
            if current_time - last_time > effective_lifetime:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the actual sandboxes OUTSIDE the lock so other tool calls
    # are not blocked while Modal/Docker sandboxes shut down.
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)

    # Replaced environments are no longer selectable. Give concurrent foreground
    # calls a one-minute grace period, then force-remove them once no tool turn or
    # background process still references their task scope.
    from tools.terminal_tool import _cleanup_retired_environments
    _cleanup_retired_environments(min_age_seconds=60.0, require_idle=True)
def get_active_env(task_id: str, target: Optional[str] = None):
    """Return the active BaseEnvironment for *task_id*, or None."""
    from tools.terminal_tool import _active_environments, _env_lock, _resolve_container_task_id
    lookup = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(task_id)


def ensure_task_env(task_id: Optional[str] = None):
    """Lazily create and cache the sandbox env for *task_id* if none is active.

    Lets non-terminal callers (``tools.image_source`` reading container-only
    paths) bring the sandbox up on demand with the same machinery as the
    terminal tool. No-op on local. Returns the env, or ``None`` when local or
    when creation fails (best-effort; the caller's fail-closed path stays intact).

    :func:`terminal_tool` creates the environment on the first terminal command, but nothing else did — so
    under a non-local backend (ssh, docker, …) a session whose first action is ``vision_analyze`` on a
    container-only path hit "no active sandbox session" because the SSH/Docker handshake never ran (issue
    #62825). vision reads such paths inside the sandbox (see ``tools.image_source``), so it calls this to
    bring the env up on demand, reusing the same creation machinery as the terminal tool.
    """
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _get_env_config, _last_activity, _resolve_container_task_id,
        _resolve_task_host_cwd, _select_image, _start_cleanup_thread, resolve_task_overrides,
    )
    config = _get_env_config()
    env_type = config["env_type"]
    if env_type == "local":
        return None

    effective_task_id = _resolve_container_task_id(task_id)

    existing = get_active_env(effective_task_id)
    if existing is not None:
        with _env_lock:
            _last_activity[effective_task_id] = time.time()
        return existing

    image = _select_image(env_type, resolve_task_overrides(task_id), config)

    _start_cleanup_thread()

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(effective_task_id, threading.Lock())

    with task_lock:
        existing = get_active_env(effective_task_id)
        if existing is not None:
            return existing
        try:
            new_env = _create_configured_env(
                config, env_type, image=image, cwd=config["cwd"],
                timeout=config["timeout"], task_id=effective_task_id,
                host_cwd=_resolve_task_host_cwd(config, task_id),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort bring-up
            logger.warning(
                "Lazy %s environment init failed for task %s: %s",
                env_type, effective_task_id[:8], exc,
            )
            return None

        with _env_lock:
            _active_environments[effective_task_id] = new_env
            _last_activity[effective_task_id] = time.time()
        logger.info(
            "%s environment lazily initialized for task %s",
            env_type, effective_task_id[:8],
        )
        return new_env


def is_persistent_env(task_id: str, target: Optional[str] = None) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.

    Session-scoped docker containers (per-session isolation mode) also count
    as persistent HERE: their lifetime is the SESSION, not the turn — they
    are removed by ``AIAgent.close()`` → ``cleanup_vm`` at session teardown
    and by the idle reaper, not per-turn.
    """
    from tools.terminal_tool import _environment_is_persistent

    env = get_active_env(task_id, target=target)
    if env is None:
        return False
    if getattr(env, "_session_scoped", False):
        return True
    return _environment_is_persistent(env)


def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    from tools.terminal_tool import _active_environments
    cleaned = 0
    for task_id in list(_active_environments.keys()):
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)

    # Also clean any orphaned directories
    for path in _scratch_paths():
        with _quiet("Failed to remove orphaned path %s", path, exc=OSError):
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)

    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(
    task_id: Hashable,
    *,
    force_remove: bool = False,
    preserve_persistent: bool = False,
    target: Optional[str] = None,
    include_collapsed: bool = False,
):
    """Manually clean up a specific environment by task_id.

    *force_remove* is forwarded to backends that accept it (currently only
    ``DockerEnvironment``). Default False matches session-lifecycle semantics:
    callers must honor the user's persist-mode preference — stopping the
    container here would break the "ONE long-lived container shared across
    sessions" contract. Pass ``force_remove=True`` only for user-initiated
    teardown. The idle reaper calls ``env.cleanup()`` directly, so
    persist-mode idle envs are likewise no-op'd; only the orphan reaper at
    next startup reclaims them.

    ``preserve_persistent`` (per-turn cleanup) keeps persistent named sibling
    environments live while removing only non-persistent targets.
    ``include_collapsed`` extends key matching to the collapsed container id;
    the caller must first release its logical turn lease.
    """
    from tools.terminal_tool import (
        _active_environments, _env_lock, _creation_locks, _creation_locks_lock,
        _last_activity, _resolve_container_task_id, _environment_is_persistent,
        _cleanup_retired_environments,
    )

    if isinstance(task_id, tuple):
        keys = [task_id]
    elif target is None:
        try:
            from tools.terminal_tool import _target_resolution
            resolution = _target_resolution(None)
            scoped_task_id = resolution.scope_task_key(task_id)
            collapsed_task_id = _resolve_container_task_id(str(task_id))
            scoped_collapsed_task_id = resolution.scope_task_key(collapsed_task_id)
        except Exception:
            scoped_task_id = task_id
            collapsed_task_id = task_id
            scoped_collapsed_task_id = task_id
        matching_task_ids = {task_id, scoped_task_id}
        if include_collapsed:
            matching_task_ids.update({
                collapsed_task_id, scoped_collapsed_task_id,
            })
        with _env_lock:
            keys = [
                key for key in _active_environments
                if key in matching_task_ids
                or (
                    isinstance(key, tuple) and len(key) == 2
                    and key[0] in matching_task_ids
                )
            ]
        if not keys:
            keys = [task_id]
    else:
        from tools.terminal_tool import _target_resolution
        resolution = _target_resolution(target)
        if resolution.named:
            keys = [resolution.environment_key(task_id)]
        else:
            keys = [task_id]

    active_process_keys = set()
    if preserve_persistent:
        try:
            from tools.process_registry import process_registry

            active_process_keys = {
                key for key in keys
                if process_registry.has_active_processes(key)
            }
        except Exception:
            logger.debug(
                "Failed to inspect active processes before cleanup",
                exc_info=True,
            )

    envs = []
    removed_keys = []
    with _env_lock:
        for key in keys:
            existing = _active_environments.get(key)
            if key in active_process_keys:
                continue
            if (
                preserve_persistent
                and existing is not None
                and _environment_is_persistent(existing)
            ):
                continue
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            removed_keys.append(key)
            if env is not None:
                envs.append((key, env))

    # Clean up per-task creation lock
    with _creation_locks_lock:
        for key in removed_keys:
            _creation_locks.pop(key, None)

    # Invalidate stale file_ops cache entry
    for key in removed_keys:
        _clear_file_ops_cache(key)

    for key in keys:
        _cleanup_retired_environments(
            task_key=key,
            min_age_seconds=0.0,
            require_idle=preserve_persistent,
        )

    if not envs:
        return

    for key, env in envs:
        try:
            _cleanup_env(env, force_remove=force_remove)

            logger.info("Manually cleaned up environment for task: %s", key)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", key)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", key, e)


def _evict_environment_for_task(task_id: Optional[str]) -> None:
    """Drop any cached env for *task_id* (and its collapsed key) after an
    infrastructure failure, so later calls don't reuse a dead connection."""
    from tools.terminal_tool import (
        _active_environments, _env_lock, _last_activity, _resolve_container_task_id,
    )
    keys = {_resolve_container_task_id(task_id)}
    if task_id:
        keys.add(task_id)
    evicted = []
    with _env_lock:
        for key in keys:
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            if env is not None:
                evicted.append(env)
    for env in evicted:
        with _quiet("cleanup of degraded environment failed"):
            env.cleanup()
