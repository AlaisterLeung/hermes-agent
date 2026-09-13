"""Path resolution for the file tools: task-aware base dir, ``~`` expansion, workspace-divergence warning.

Core invariant: the base directory anchoring relative paths is ALWAYS absolute
and derived from the task's terminal cwd, never from the process cwd unless no
other anchor exists (a relative/sentinel ``TERMINAL_CWD`` would silently anchor
edits to the agent process cwd, e.g. the main repo during a worktree session).
"""

import os
import posixpath
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Optional

# ``TERMINAL_CWD`` values that mean "not configured" ("." from a stale config;
# "auto"/"cwd" are wizard placeholders). gateway/run.py sanitizes the same set.
_TERMINAL_CWD_SENTINELS = frozenset({"", ".", "./", "auto", "cwd"})
_CONTAINER_PATH_BACKENDS_FALLBACK = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})
# Backend name inferred from the live environment's class name (first match wins).
_ENV_CLASS_NAME_HINTS = ("local", "ssh", "docker", "singularity", "modal", "daytona")


def _expand_tilde(path: str) -> str:
    """Expand ``~`` using the effective profile home (``get_subprocess_home``) so
    gateway/cron runs, whose process HOME may differ, agree with interactive CLI sessions.

    This mirrors ``hermes_constants.get_subprocess_home()`` so that ``~`` resolves consistently regardless
    of whether the tool runs interactively or inside a gateway-driven cron job (#48552).
    """
    if not path or "~" not in path:
        return path
    try:
        from hermes_constants import get_subprocess_home

        home = get_subprocess_home()
    except Exception:
        home = None
    if home and (path == "~" or path.startswith("~/")):
        return home if path == "~" else os.path.join(home, path[2:])
    return os.path.expanduser(path)


def _terminal_env_type_for_task(
    task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> str:
    """Best-effort terminal backend type for path-resolution decisions."""
    try:
        from tools.terminal_tool import (
            _active_environments, _env_lock, _get_env_config, _resolve_container_task_id)
        from tools.execution_targets import resolve_execution_target

        resolution = _resolution or resolve_execution_target(execution_target)
        try:
            container_key = resolution.environment_key(
                _resolve_container_task_id(task_id)
            )
            raw_key = resolution.environment_key(task_id)
        except Exception:
            container_key = task_id
            raw_key = task_id
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(raw_key)
        if env is not None:
            name = env.__class__.__name__.lower()
            stamped = getattr(env, "_hermes_backend_name", None)
            hint = next((h for h in _ENV_CLASS_NAME_HINTS if h in name), None)
            if hint or (isinstance(stamped, str) and stamped):
                return hint or stamped
        cfg = (
            _get_env_config(dict(resolution.config))
            if resolution.named else _get_env_config()
        )
        return str(cfg.get("env_type") or os.getenv("TERMINAL_ENV") or "local").lower()
    except Exception:
        return str(os.getenv("TERMINAL_ENV") or "local").lower()


def _uses_container_paths(
    task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> bool:
    env_type = _terminal_env_type_for_task(
        task_id, execution_target, _resolution=_resolution,
    )
    try:
        from tools.terminal_tool import _is_container_backend

        return _is_container_backend(env_type)
    except Exception:
        return env_type in _CONTAINER_PATH_BACKENDS_FALLBACK


def _normalize_without_host_deref(path: str | Path | PurePosixPath) -> PurePosixPath:
    """Normalize path syntax without following host symlinks: container paths are
    meaningful inside the sandbox, and a host-side ``/workspace`` symlink must not rewrite them."""
    return PurePosixPath(posixpath.normpath(str(path)))


def _sentinel_free_abs_cwd(raw: str | None) -> str | None:
    """Return *raw* expanded when it is a non-sentinel ABSOLUTE anchor, else ``None``
    (a relative anchor is exactly the ambiguity that misroutes worktree edits)."""
    raw = str(raw or "").strip()
    if raw.lower() in _TERMINAL_CWD_SENTINELS:
        return None
    expanded = _expand_tilde(raw)
    return expanded if os.path.isabs(expanded) else None


def _configured_terminal_cwd() -> str | None:
    """Return ``$TERMINAL_CWD`` only when it names a real (absolute, non-sentinel) anchor.
    Scope-aware: under gateway multiplexing the routed profile's cwd lives in the per-turn scope."""
    # See #68559.
    from agent.runtime_cwd import scope_terminal_cwd

    return _sentinel_free_abs_cwd(scope_terminal_cwd() or None)


def _registered_task_cwd_override(task_id: str = "default") -> str | None:
    """Return a registered cwd override keyed by the RAW task id, when available.

    ``terminal_tool`` collapses CWD-only overrides to the shared ``"default"``
    env, but the cwd value stays keyed by the raw session id.
    """
    try:
        from tools.terminal_tool import resolve_task_overrides

        overrides = resolve_task_overrides(task_id)
    except Exception:
        return None

    return _sentinel_free_abs_cwd(overrides.get("cwd"))


def _host_text(text: str, container_paths: bool) -> str:
    """Expand ``~``; on host backends also translate Git Bash ``/c/Users/...`` drive
    paths before Path sees them. Container/WSL Linux paths are never rewritten."""
    if not container_paths:
        from tools.environments.local import _msys_to_windows_path

        text = _msys_to_windows_path(text)
    return _expand_tilde(text)


def _anchor(text: str, base, container_paths: bool) -> Path | PurePosixPath:
    """Return *text* as an absolute, normalized path, joining it onto ``base()`` when
    relative. Container: pure-posix, no host deref. Host: resolve() (win32: ntpath normpath)."""
    if container_paths:
        if not posixpath.isabs(text):
            text = posixpath.join(str(base()), text)
        return _normalize_without_host_deref(text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(text):
            text = ntpath.join(str(base()), text)
        return Path(ntpath.normpath(text))
    p = Path(text)
    if not p.is_absolute():
        p = Path(base()) / p
    return p.resolve()


def _authoritative_workspace_root(
    task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> str | None:
    """Best-effort absolute workspace root for divergence checks.

    Resolution:

      1. The session's own cwd RECORD (``terminal_tool.get_session_cwd``) —
         written on every completed terminal command and seeded by workspace
         registration, keyed by the raw session id. Because the record is
         per-session, one session's ``cd`` can never leak into another
         session's resolution.
      2. For an explicit non-default named target, that target's configured
         cwd. Host workspace overrides must not leak into remote targets.
      3. A registered task/session cwd override (TUI/Desktop/ACP sessions
         register a raw-keyed cwd before any tool runs). Normally already
         mirrored into the default target's record; kept as a fallback.
      4. A sentinel-free absolute ``$TERMINAL_CWD`` (the worktree path set by
         ``cli.py``/``main.py`` for ``-w`` sessions).

    Returns ``None`` only when there is genuinely no reliable anchor, in which
    case callers fall back to the process cwd.
    """
    try:
        from tools.terminal_tool import get_session_cwd

        recorded = get_session_cwd(
            task_id, target=execution_target, _resolution=_resolution,
        )
    except Exception:
        recorded = None
    if recorded:
        return recorded

    target_resolution = _resolution
    configured_target_cwd = None
    if target_resolution is None and execution_target is not None:
        try:
            from tools.execution_targets import resolve_execution_target

            target_resolution = resolve_execution_target(execution_target)
        except Exception:
            target_resolution = None
    if target_resolution is not None and target_resolution.named:
        try:
            from tools.terminal_tool import _get_env_config

            configured = _get_env_config(
                dict(target_resolution.config)
            ).get("cwd")
            if isinstance(configured, str) and configured.strip():
                if (
                    target_resolution.backend == "ssh"
                    and (
                        configured in {".", "./", "auto", "cwd", "~"}
                        or configured.startswith("~/")
                    )
                ):
                    # Keep relative/tilde SSH paths remote: before the SSH env
                    # exists there is no truthful host-side anchor; a later terminal
                    # call records the resolved remote cwd for this session.
                    configured_target_cwd = configured
                elif configured in {".", "./", "auto", "cwd"}:
                    configured_target_cwd = os.getcwd()
                elif configured == "~" or configured.startswith("~/"):
                    # SSH expands this remotely; local targets expand here.
                    configured_target_cwd = (
                        configured
                        if target_resolution.backend == "ssh"
                        else _expand_tilde(configured)
                    )
                else:
                    configured_target_cwd = configured
        except Exception:
            target_resolution = None

    if (
        target_resolution is not None
        and target_resolution.named
        and not target_resolution.is_default
        and configured_target_cwd
    ):
        return configured_target_cwd

    registered = (
        None
        if (
            target_resolution is not None
            and target_resolution.named
            and target_resolution.backend == "ssh"
        )
        else _registered_task_cwd_override(task_id)
    )
    if registered:
        return registered
    if configured_target_cwd:
        return configured_target_cwd
    return _configured_terminal_cwd()


def _resolve_base_dir(
    task_id: str = "default",
    *,
    container_paths: bool | None = None,
    execution_target: str | None = None,
    _resolution: Any = None,
) -> Path | PurePosixPath:
    """Return the ABSOLUTE base directory for resolving relative paths:
    ``_authoritative_workspace_root``, else the process cwd as a last resort."""
    root = _authoritative_workspace_root(
        task_id, execution_target, _resolution=_resolution,
    )
    if container_paths is None:
        container_paths = _uses_container_paths(
            task_id, execution_target, _resolution=_resolution,
        )
    if root:
        base_text = _expand_tilde(root)
    else:
        base_text = os.getcwd()
    if container_paths:
        if not posixpath.isabs(base_text):
            base_text = posixpath.join(os.getcwd(), base_text)
        return _normalize_without_host_deref(base_text)
    # Git Bash ``pwd -P`` reports ``/c/Users/...``; translate before Path so
    # relative file-tool paths don't anchor under a nonexistent ``\c\Users``.
    from tools.environments.local import _msys_to_windows_path

    base_text = _msys_to_windows_path(base_text)
    if sys.platform == "win32":
        import ntpath

        if not ntpath.isabs(base_text):
            base_text = ntpath.join(os.getcwd(), base_text)
        return Path(ntpath.normpath(base_text))
    base = Path(base_text)
    if not base.is_absolute():
        # A backend's relative cwd is anchored to the process cwd once, here,
        # so the result no longer depends on cwd at resolve().
        base = Path(os.getcwd()) / base
    return base.resolve()


def _resolve_path_for_task(
    filepath: str, task_id: str = "default", execution_target: str | None = None,
    *, _resolution: Any = None,
) -> Path | PurePosixPath:
    """Resolve *filepath* against the task's absolute base directory.

    See :func:`_resolve_base_dir` for how the base is chosen. Absolute input
    paths are returned resolved-but-unanchored.

    On native Windows, Git Bash / MSYS drive paths (``/c/Users/...``) are
    translated to ``C:\\Users\\...`` before resolution so file tools don't
    treat them as relative ``\\c\\Users\\...`` under the process cwd.
    """
    container_paths = _uses_container_paths(
        task_id, execution_target, _resolution=_resolution,
    )
    if container_paths:
        expanded = _expand_tilde(filepath)
        if posixpath.isabs(expanded):
            return _normalize_without_host_deref(expanded)
        resolved = _resolve_base_dir(
            task_id, container_paths=True, execution_target=execution_target,
            _resolution=_resolution,
        ) / expanded
        return _normalize_without_host_deref(resolved)

    # Host paths only — never rewrite Linux paths inside a container/WSL env.
    from tools.environments.local import _msys_to_windows_path

    expanded = _expand_tilde(_msys_to_windows_path(filepath))
    if sys.platform == "win32":
        import ntpath

        if ntpath.isabs(expanded):
            return Path(ntpath.normpath(expanded))
        joined = ntpath.join(str(_resolve_base_dir(
            task_id, container_paths=False, execution_target=execution_target,
            _resolution=_resolution,
        )), expanded)
        return Path(ntpath.normpath(joined))

    p = Path(expanded)
    if p.is_absolute():
        return p.resolve()
    resolved = _resolve_base_dir(
        task_id, container_paths=False, execution_target=execution_target,
        _resolution=_resolution,
    ) / p
    return resolved.resolve()


def _path_resolution_warning(filepath: str, resolved: Path, task_id: str = "default",
                             execution_target: str | None = None,
                             *, _resolution: Any = None) -> str | None:
    """Warn when a RELATIVE path resolved OUTSIDE the task's workspace root (the
    edit is about to land in a different checkout than the terminal's cwd).
    ``None`` for absolute paths, an unknown root, or a path under the root."""
    try:
        if Path(_expand_tilde(filepath)).is_absolute():
            return None
        workspace_root = _authoritative_workspace_root(
            task_id, execution_target, _resolution=_resolution,
        )
        if not workspace_root:
            return None
        if _uses_container_paths(task_id, execution_target, _resolution=_resolution):
            root = _normalize_without_host_deref(Path(_expand_tilde(workspace_root)))
        else:
            root = Path(_expand_tilde(workspace_root)).resolve()
        if resolved.is_relative_to(root):
            return None
        return (
            f"Relative path {filepath!r} resolved to {str(resolved)!r}, which is "
            f"OUTSIDE the active workspace ({str(root)!r}). The edit will land in "
            f"a different directory than the terminal's cwd. If this is not "
            f"intended (e.g. a git-worktree session writing into the main "
            f"checkout), pass an absolute path under the workspace instead.")
    except Exception:
        return None
