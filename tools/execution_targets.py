"""Resolve static and profile-scoped runtime terminal execution targets.

The resolver is deliberately small and lazy: importing tool schemas never reads
user configuration, and configured names never enter a schema.  Legacy flat
terminal configuration stays on the existing environment-variable path.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import difflib
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Hashable, Iterable, Mapping, Optional


class ExecutionTargetError(ValueError):
    """Raised when terminal target configuration or selection is invalid."""


_NAMED_TARGET_BACKENDS = frozenset({
    "local", "docker", "ssh", "modal", "daytona", "singularity",
    "vercel_sandbox",
})

_TARGET_COMMON_SETTINGS = frozenset({
    "backend", "cwd", "timeout", "lifetime_seconds", "sudo_password",
})
_TARGET_CONTAINER_SETTINGS = frozenset({
    "container_cpu", "container_memory", "container_disk",
    "container_persistent",
})
_TARGET_BACKEND_SETTINGS = {
    "local": frozenset({"local_persistent"}),
    "ssh": frozenset({
        "ssh_host", "ssh_user", "ssh_port", "ssh_key",
        "ssh_persistent", "persistent_shell", "file_sync",
    }),
    "docker": _TARGET_CONTAINER_SETTINGS | frozenset({
        "docker_image", "docker_forward_env", "docker_env",
        "docker_volumes", "docker_mount_cwd_to_workspace",
        "docker_network", "docker_extra_args", "docker_shm_size",
        "docker_run_as_host_user", "docker_persist_across_processes",
        "docker_orphan_reaper",
    }),
    "singularity": _TARGET_CONTAINER_SETTINGS | frozenset({"singularity_image"}),
    "modal": _TARGET_CONTAINER_SETTINGS | frozenset({"modal_image", "modal_mode"}),
    "daytona": _TARGET_CONTAINER_SETTINGS | frozenset({"daytona_image"}),
    "vercel_sandbox": _TARGET_CONTAINER_SETTINGS | frozenset({"vercel_runtime"}),
}
_TARGET_ENTRY_SETTINGS = frozenset().union(
    _TARGET_COMMON_SETTINGS, *_TARGET_BACKEND_SETTINGS.values()
)
_TARGET_BOOLEAN_SETTINGS = frozenset({
    "container_persistent", "local_persistent", "ssh_persistent",
    "persistent_shell", "file_sync", "docker_mount_cwd_to_workspace",
    "docker_network", "docker_run_as_host_user",
    "docker_persist_across_processes", "docker_orphan_reaper",
})
_TARGET_LIST_SETTINGS = frozenset({
    "docker_forward_env", "docker_volumes", "docker_extra_args",
})
_TARGET_STRING_SETTINGS = frozenset({
    "cwd", "docker_image", "singularity_image", "modal_image",
    "daytona_image", "vercel_runtime", "ssh_host", "ssh_user", "ssh_key",
    "docker_shm_size", "modal_mode", "sudo_password",
})
_REGISTRY_METADATA_KEY = "__execution_target_registry_v1__"


_effective_config_override: ContextVar[dict[str, Any] | None] = ContextVar(
    "hermes_execution_target_config", default=None,
)
_frozen_config_active: ContextVar[bool] = ContextVar(
    "hermes_execution_target_frozen", default=False,
)
_classic_config_lock = threading.RLock()
_classic_config_override: dict[str, Any] | None = None
_fingerprint_keys: dict[Path, bytes] = {}
_fingerprint_key_lock = threading.Lock()


def _target_fingerprint_key() -> bytes:
    """Return a stable, private machine/profile key for exposed target digests."""
    from hermes_constants import get_hermes_home

    path = Path(get_hermes_home()) / ".execution-target-fingerprint-key"
    with _fingerprint_key_lock:
        cached = _fingerprint_keys.get(path)
        if cached is not None:
            return cached

        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = os.urandom(32)
        temp = path.with_name(
            f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        )
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(candidate)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                pass
            except OSError:
                # Filesystems without hard-link support still get O_EXCL
                # publication; readers below wait for the winning writer.
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(candidate)
                        handle.flush()
                        os.fsync(handle.fileno())
        finally:
            try:
                temp.unlink()
            except OSError:
                pass

        key = b""
        for _ in range(20):
            try:
                key = path.read_bytes()
            except OSError:
                key = b""
            if len(key) >= 32:
                break
            time.sleep(0.01)
        if len(key) < 32:
            raise ExecutionTargetError(
                f"Could not initialize private execution-target key at {path}."
            )
        try:
            path.chmod(0o600)
        except OSError:
            pass
        key = key[:32]
        _fingerprint_keys[path] = key
        return key


def set_execution_target_config_source(config: Mapping[str, Any] | None) -> None:
    """Use an entry point's already-effective config for target resolution.

    The classic CLI has project-fallback and ``--ignore-user-config`` rules
    that differ from the shared loader. Registering its merged result here
    keeps tool routing on the same authority boundary. A live reload updates
    the process-wide source, but never replaces an in-flight frozen snapshot.
    """
    global _classic_config_override
    value = deepcopy(dict(config)) if isinstance(config, Mapping) else None
    if not _frozen_config_active.get():
        _effective_config_override.set(value)
    with _classic_config_lock:
        _classic_config_override = deepcopy(value)


@contextmanager
def execution_target_config_scope(config: Mapping[str, Any]):
    """Temporarily pin target resolution in the current context only."""
    prior_override = deepcopy(_effective_config_override.get())
    token = _effective_config_override.set(deepcopy(dict(config)))
    frozen_token = _frozen_config_active.set(True)
    try:
        yield
    finally:
        _frozen_config_active.reset(frozen_token)
        _effective_config_override.reset(token)
        # A reload that occurred during the frozen dispatch is the authority
        # for the next call. Adopt it only when leaving the outermost scope;
        # nested execute-code scopes remain pinned to their outer generation.
        if not _frozen_config_active.get():
            with _classic_config_lock:
                live_override = deepcopy(_classic_config_override)
            if live_override != prior_override:
                _effective_config_override.set(live_override)


def execution_target_config_is_frozen() -> bool:
    """Return whether this call is pinned to one dispatch generation."""
    return _frozen_config_active.get()


@contextmanager
def frozen_execution_target_config():
    """Freeze the current merged target config for one complete tool dispatch."""
    with execution_target_config_scope(_load_merged_config()):
        yield


def _active_profile_scope() -> str:
    """Return a stable multiplex-profile scope, empty in legacy mode."""
    try:
        from agent.secret_scope import is_multiplex_active
        if not is_multiplex_active():
            return ""
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home().resolve())
        return hashlib.sha256(home.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


@dataclass(frozen=True)
class ExecutionTargetResolution:
    """A resolved execution target and its inherited terminal configuration."""

    target: str
    backend: str
    config: Mapping[str, Any]
    named: bool
    is_default: bool = True
    profile_scope: str = ""
    provider: str | None = None
    owner_id: str | None = None
    generation: str | None = None

    @property
    def spec_fingerprint(self) -> str:
        """Hash the fully inherited target spec used to create resources.

        Target names are mutable configuration aliases, not security identities.
        Including the complete resolved spec prevents a hot config edit from
        reusing an environment created for a different host/backend/image.
        """
        canonical = json.dumps(
            {
                "config": dict(self.config),
                "is_default": self.is_default,
                "runtime_registry": {
                    "provider": self.provider,
                    "owner_id": self.owner_id,
                    "generation": self.generation,
                } if self.provider is not None else None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=repr,
        )
        return hmac.new(
            _target_fingerprint_key(),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:20]

    @property
    def security_scope(self) -> str:
        """Return the profile/name/spec identity for approvals and credentials."""
        value = f"{self.profile_scope}:{self.target}:{self.spec_fingerprint}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def scope_task_key(self, task_key: Hashable) -> Hashable:
        if not self.profile_scope:
            return task_key
        return f"profile-{self.profile_scope}:{task_key}"

    def environment_key(self, task_key: Hashable) -> Hashable:
        """Scope an already-collapsed environment key to this target."""
        scoped = self.scope_task_key(task_key)
        return (scoped, self.target) if self.named else scoped

    def session_key(self, raw_session_key: Optional[str]) -> Hashable:
        """Scope per-session state without changing legacy string keys."""
        key = str(self.scope_task_key(str(raw_session_key or "default")))
        return (key, self.target) if self.named else key

    def file_coordination_key(self, raw_session_key: Optional[str]) -> Hashable:
        """Return file-operation state identity, generation-scoped at runtime.

        Static named targets retain their historical per-alias session key.
        Runtime aliases add their immutable security scope so a provider
        repoint cannot inherit repeated-read, stale-write, or patch state.
        """
        key = self.session_key(raw_session_key)
        if self.provider is None or not isinstance(key, tuple):
            return key
        return (key[0], f"{key[1]}@runtime-{self.security_scope}")

    def backend_task_id(self, task_key: Hashable) -> str:
        """Return a bounded backend-safe isolation id.

        Docker labels are limited to 63 characters. Keep fixed hash suffixes of
        both the complete task identity and target spec so long task IDs cannot
        truncate away the only target-distinguishing bytes.
        """
        base = str(self.scope_task_key(task_key))
        if not self.named:
            return base
        base_digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
        return f"task-{base_digest}-target-{self.security_scope}"

    def storage_task_id(self, base_task_id: str) -> str:
        """Stable, bounded identity for persistent storage owned by a target.

        Runtime identity includes the full effective config so a repointed
        target cannot reuse a stale container. Persistent filesystem ownership
        intentionally depends only on profile + target name, keeping data stable
        across timeout, lifetime, image, and other policy edits.
        """
        if not self.named:
            return base_task_id
        base_digest = hashlib.sha256(
            base_task_id.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:16]
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home().resolve())
        owner = f"{self.profile_scope}\0{home}\0{self.target}"
        owner_digest = hashlib.sha256(owner.encode("utf-8")).hexdigest()[:20]
        return f"task-{base_digest}-storage-{owner_digest}"

    def legacy_backend_task_id(self, base_task_id: str) -> str:
        """Return the pre-fingerprint named-target backend ID for migration."""
        base = str(self.scope_task_key(base_task_id))
        if not self.named:
            return base
        target_digest = hashlib.sha256(
            self.target.encode("utf-8")
        ).hexdigest()[:12]
        return f"{base}-target-{target_digest}"

    def metadata(self, *, cwd: Optional[str] = None) -> dict[str, Any]:
        data: dict[str, Any] = {"target": self.target, "backend": self.backend}
        if self.named:
            data["runtime_scope"] = self.security_scope
        if cwd:
            data["cwd"] = cwd
        return data


def _load_static_unscoped_config() -> dict[str, Any]:
    """Load static effective config without runtime target fragments."""
    try:
        from agent.secret_scope import is_multiplex_active

        multiplex = is_multiplex_active()
    except Exception:
        multiplex = False
    if not multiplex:
        with _classic_config_lock:
            override = deepcopy(_classic_config_override)
        if override is not None:
            return override
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    return config if isinstance(config, dict) else {}


def load_static_execution_target_config() -> dict[str, Any]:
    """Return current effective static config for registry management."""
    return deepcopy(_load_static_unscoped_config())


def _load_unscoped_config() -> dict[str, Any]:
    """Load live static config plus the current profile registry."""
    from tools.execution_target_overlay import overlay_runtime_execution_targets

    return overlay_runtime_execution_targets(_load_static_unscoped_config())


def _load_merged_config() -> dict[str, Any]:
    """Load merged config lazily so importing fixed tool schemas stays cheap."""
    override = _effective_config_override.get()
    if override is not None:
        if _frozen_config_active.get():
            return deepcopy(override)
        from tools.execution_target_overlay import overlay_runtime_execution_targets

        return overlay_runtime_execution_targets(override)
    return _load_unscoped_config()


def _available(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(none)"



def _validate_named_target_entry(
    name: str,
    entry: Mapping[str, Any],
    effective: Mapping[str, Any],
    backend: str,
) -> None:
    """Fail fast on misspelled, inapplicable, or malformed target settings."""
    allowed = _TARGET_COMMON_SETTINGS | _TARGET_BACKEND_SETTINGS[backend]
    for key in entry:
        if not isinstance(key, str):
            raise ExecutionTargetError(
                f"Execution target {name!r} setting names must be strings; got {key!r}."
            )
        if key not in allowed:
            if key in _TARGET_ENTRY_SETTINGS:
                raise ExecutionTargetError(
                    f"Execution target {name!r} setting {key!r} does not apply "
                    f"to backend {backend!r}."
                )
            match = difflib.get_close_matches(
                key, sorted(_TARGET_ENTRY_SETTINGS), n=1, cutoff=0.7,
            )
            hint = f" Did you mean {match[0]!r}?" if match else ""
            raise ExecutionTargetError(
                f"Execution target {name!r} has unknown setting {key!r}.{hint}"
            )

    def invalid(key: str, expected: str) -> None:
        raise ExecutionTargetError(
            f"Execution target {name!r} setting {key!r} has an invalid shape; "
            f"expected {expected}."
        )

    for key in allowed & set(effective):
        value = effective[key]
        if key in _TARGET_BOOLEAN_SETTINGS:
            if isinstance(value, bool):
                continue
            if str(value).strip().lower() not in {
                "true", "false", "1", "0", "yes", "no",
            }:
                invalid(key, "boolean")
        elif key in _TARGET_LIST_SETTINGS:
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                invalid(key, "list of strings")
        elif key == "docker_env":
            if not isinstance(value, Mapping) or any(
                not isinstance(item_key, str) or not isinstance(item_value, str)
                for item_key, item_value in value.items()
            ):
                invalid(key, "mapping of string keys to string values")
        elif key in _TARGET_STRING_SETTINGS:
            if not isinstance(value, str):
                invalid(key, "a string")
        elif key == "container_cpu":
            if isinstance(value, bool):
                invalid(key, "a non-negative number")
            try:
                number = float(value)
            except (TypeError, ValueError):
                invalid(key, "a finite non-negative number")
            if not math.isfinite(number) or number < 0:
                invalid(key, "a finite non-negative number")
        elif key in {"container_memory", "container_disk"}:
            if isinstance(value, bool):
                invalid(key, "a non-negative integer")
            if isinstance(value, float) and not value.is_integer():
                invalid(key, "a non-negative integer")
            try:
                number = int(value)
            except (TypeError, ValueError):
                invalid(key, "a non-negative integer")
            if number < 0:
                invalid(key, "a non-negative integer")
        elif key in {"timeout", "lifetime_seconds"}:
            if isinstance(value, bool):
                invalid(key, "a positive integer")
            if isinstance(value, float) and not value.is_integer():
                invalid(key, "a positive integer")
            try:
                number = int(value)
            except (TypeError, ValueError):
                invalid(key, "a positive integer")
            if number <= 0:
                invalid(key, "a positive integer")
        elif key == "ssh_port":
            if isinstance(value, bool):
                invalid(key, "an integer from 1 to 65535")
            if isinstance(value, float) and not value.is_integer():
                invalid(key, "an integer from 1 to 65535")
            try:
                number = int(value)
            except (TypeError, ValueError):
                invalid(key, "an integer from 1 to 65535")
            if not 1 <= number <= 65535:
                invalid(key, "an integer from 1 to 65535")

    if backend == "ssh":
        for key in ("ssh_host", "ssh_user"):
            value = effective.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ExecutionTargetError(
                    f"Execution target {name!r} with backend 'ssh' requires "
                    f"non-empty {key!r}."
                )
    if backend == "modal" and "modal_mode" in effective:
        if str(effective["modal_mode"]).strip().lower() not in {
            "auto", "direct", "managed",
        }:
            invalid("modal_mode", "one of 'auto', 'direct', or 'managed'")


def _explicit_legacy_terminal_keys(
    flat: Mapping[str, Any],
    canonical_defaults: Mapping[str, Any],
) -> set[str]:
    """Identify flat terminal settings with real legacy authority.

    Raw profile config and managed scope retain exact presence information.
    The classic CLI's project-fallback source does not, so remaining values are
    considered explicit only when they differ from both shared config defaults
    and the terminal tool's canonical defaults.  This deliberately prefers an
    existing environment value when project provenance is ambiguous.
    """
    from hermes_cli.config import effective_terminal_config, read_raw_config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    authority: set[str] = set()

    def _collect_authority(source: Any) -> None:
        if not isinstance(source, Mapping):
            return
        terminal = source.get("terminal")
        if isinstance(terminal, Mapping):
            authority.update(effective_terminal_config(dict(terminal)))

    if os.environ.get("HERMES_IGNORE_USER_CONFIG") != "1":
        _collect_authority(read_raw_config())
    try:
        from hermes_cli import managed_scope

        _collect_authority(managed_scope.load_managed_config())
    except Exception:
        # Provenance discovery must not make terminal activation unavailable
        # when the optional managed scope cannot be read.
        pass
    if "env_type" in authority:
        authority.add("backend")

    values: dict[str, Any] = {
        key: flat[key] for key in _TARGET_ENTRY_SETTINGS if key in flat
    }
    if "backend" not in values and "env_type" in flat:
        values["backend"] = flat["env_type"]

    shared = DEFAULT_CONFIG.get("terminal", {})
    shared = shared if isinstance(shared, Mapping) else {}
    explicit: set[str] = set()
    missing = object()
    for key, value in values.items():
        if key in authority:
            explicit.add(key)
            continue
        candidates: list[Any] = []
        shared_value = shared.get(key, missing)
        if shared_value is not missing:
            candidates.append(shared_value)
        canonical_key = "env_type" if key == "backend" else key
        canonical_value = canonical_defaults.get(canonical_key, missing)
        if canonical_value is not missing:
            candidates.append(canonical_value)
        if not candidates or all(value != candidate for candidate in candidates):
            explicit.add(key)
    return explicit


def synthesize_legacy_default_target(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the live named ``default`` spec for activated flat config.

    Multiplexed dispatches derive exclusively from the selected profile mapping.
    Single-profile dispatches retain the legacy process-environment backend
    selection, while current static/managed values refresh the effective spec.
    Nothing returned here is persisted by the runtime registry.
    """
    legacy = resolve_execution_target(config=config)
    if legacy.named:
        return None
    from tools.terminal_tool import _get_env_config

    try:
        from agent.secret_scope import is_multiplex_active

        multiplex = is_multiplex_active()
    except Exception:
        multiplex = False

    flat = dict(legacy.config)
    if multiplex:
        # Process-global TERMINAL_* belongs to no profile in multiplex mode.
        effective = _get_env_config(flat)
        explicit_keys = set(_TARGET_ENTRY_SETTINGS) & set(flat)
    else:
        # Start with the canonical environment that legacy terminal execution
        # actually uses.  Only raw/managed authority (or a conservative
        # non-default project fallback) may replace an ambient value.
        ambient = _get_env_config()
        ambient_backend = (
            str(ambient.get("env_type") or legacy.backend or "local").strip().lower()
        )
        canonical_defaults = _get_env_config({})
        explicit_keys = _explicit_legacy_terminal_keys(flat, canonical_defaults)
        seeded = {
            "backend": ambient_backend,
            **{
                key: deepcopy(value)
                for key, value in ambient.items()
                if key in _TARGET_ENTRY_SETTINGS
            },
        }
        for key in explicit_keys:
            source_key = (
                "backend"
                if key == "backend" and "backend" in flat
                else "env_type"
                if key == "backend" and "env_type" in flat
                else key
            )
            if source_key not in flat:
                continue
            value = flat[source_key]
            if key == "cwd" and str(value or "").strip() in {".", "auto", "cwd"}:
                # Match apply_terminal_config_to_env(): placeholders do not
                # erase a concrete legacy TERMINAL_CWD value.
                continue
            seeded[key] = deepcopy(value)
        effective = _get_env_config(seeded)

    backend = (
        str(effective.get("env_type") or legacy.backend or "local").strip().lower()
    )
    if backend not in _NAMED_TARGET_BACKENDS:
        raise ExecutionTargetError(
            f"Cannot activate runtime targets for unknown legacy backend {backend!r}."
        )
    allowed = _TARGET_COMMON_SETTINGS | _TARGET_BACKEND_SETTINGS[backend]
    target_config: dict[str, Any] = {"backend": backend}
    for key in allowed - {"backend"}:
        if key in effective:
            target_config[key] = deepcopy(effective[key])
        elif (
            key == "sudo_password"
            and key in legacy.config
            and (multiplex or key in explicit_keys)
        ):
            target_config[key] = deepcopy(legacy.config[key])
    candidate = {
        "terminal": {
            "default_target": "default",
            "targets": {"default": target_config},
        },
    }
    resolve_execution_target("default", config=candidate)
    return target_config


def resolve_execution_target(
    target: Optional[str] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> ExecutionTargetResolution:
    """Resolve *target* against merged ``terminal`` configuration.

    With no non-empty ``terminal.targets`` mapping, only omitted target and
    explicit ``"default"`` are valid and the returned configuration marks the
    legacy env-driven path.  With named targets, top-level terminal settings
    are inherited and the selected target mapping overrides them.
    """
    if target is not None and (not isinstance(target, str) or not target):
        raise ExecutionTargetError("Execution target must be a non-empty string.")

    root = config if config is not None else _load_merged_config()
    profile_scope = _active_profile_scope()
    if not isinstance(root, Mapping):
        raise ExecutionTargetError("Terminal configuration must be a mapping.")
    terminal = root.get("terminal", {})
    if terminal is None:
        terminal = {}
    if not isinstance(terminal, Mapping):
        raise ExecutionTargetError("terminal must be a mapping.")

    raw_targets = terminal.get("targets")
    if raw_targets is None or raw_targets == {}:
        if target not in (None, "default"):
            from tools.execution_target_lifecycle import (
                unavailable_runtime_target_message,
            )

            unavailable = unavailable_runtime_target_message(root, target)
            if unavailable:
                raise ExecutionTargetError(unavailable)
            raise ExecutionTargetError(
                f"Unknown execution target {target!r}. Named targets are not configured. "
                f"Available targets: {_available(['default'])}."
            )
        flat = dict(terminal)
        flat.pop("targets", None)
        flat.pop("default_target", None)
        # Legacy flat mode preserves the historical TERMINAL_ENV precedence.
        # Metadata must describe the backend that will actually execute, not a
        # stale config.yaml value hidden by an explicit launcher/.env override.
        backend = str(
            os.getenv("TERMINAL_ENV")
            or flat.get("backend")
            or flat.get("env_type")
            or "local"
        ).strip().lower() or "local"
        return ExecutionTargetResolution(
            target="default", backend=backend, config=flat, named=False,
            is_default=True, profile_scope=profile_scope,
        )

    if not isinstance(raw_targets, Mapping):
        raise ExecutionTargetError("terminal.targets must be a mapping.")

    invalid_names = [
        name for name in raw_targets
        if not isinstance(name, str) or not name
    ]
    if invalid_names:
        raise ExecutionTargetError("terminal.targets names must be non-empty strings.")

    names = sorted(raw_targets)
    for name in names:
        if not isinstance(raw_targets[name], Mapping):
            raise ExecutionTargetError(
                f"terminal.targets[{name!r}] must be a mapping. "
                f"Available targets: {_available(names)}."
            )

    selected = target if target is not None else terminal.get("default_target")
    if not isinstance(selected, str) or not selected or selected not in raw_targets:
        if target is not None:
            from tools.execution_target_lifecycle import (
                unavailable_runtime_target_message,
            )

            unavailable = unavailable_runtime_target_message(root, target)
            if unavailable:
                raise ExecutionTargetError(unavailable)
            prefix = f"Unknown execution target {target!r}."
        else:
            prefix = (
                "terminal.default_target must be a non-empty name present in "
                "terminal.targets."
            )
        raise ExecutionTargetError(
            f"{prefix} Available targets: {_available(names)}."
        )

    merged = {
        key: value
        for key, value in terminal.items()
        if key not in {"targets", "default_target"}
    }
    merged.update(dict(raw_targets[selected]))
    backend_setting = (
        "backend" if "backend" in merged
        else "env_type" if "env_type" in merged
        else None
    )
    raw_backend = merged[backend_setting] if backend_setting else "local"
    if not isinstance(raw_backend, str) or not raw_backend.strip():
        setting = backend_setting or "backend"
        raise ExecutionTargetError(
            f"Execution target {selected!r} setting {setting!r} has an invalid "
            "shape; expected a non-empty string."
        )
    backend = raw_backend.strip().lower()
    if backend not in _NAMED_TARGET_BACKENDS:
        available_backends = ", ".join(sorted(_NAMED_TARGET_BACKENDS))
        raise ExecutionTargetError(
            f"Execution target {selected!r} has unknown backend {backend!r}. "
            f"Available backends: {available_backends}."
        )
    _validate_named_target_entry(
        selected, raw_targets[selected], merged, backend,
    )
    from tools.execution_target_lifecycle import runtime_record_metadata

    runtime_records = runtime_record_metadata(root, selected, status="active")
    runtime_record = runtime_records[0] if len(runtime_records) == 1 else {}
    return ExecutionTargetResolution(
        target=selected, backend=backend, config=merged, named=True,
        is_default=selected == terminal.get("default_target"),
        profile_scope=profile_scope,
        provider=runtime_record.get("provider"),
        owner_id=runtime_record.get("owner_id"),
        generation=runtime_record.get("generation"),
    )


def resolve_live_execution_target(
    target: Optional[str] = None,
) -> ExecutionTargetResolution:
    """Resolve against the live source when nested RPC routing is frozen."""
    if not _frozen_config_active.get():
        return resolve_execution_target(target)
    return resolve_execution_target(target, config=_load_unscoped_config())


def list_execution_targets(
    *, config: Optional[Mapping[str, Any]] = None,
) -> tuple[ExecutionTargetResolution, ...]:
    """Return configured targets in deterministic order for runtime guidance.

    Tool schemas remain static for prompt-cache stability; callers such as the
    system-prompt environment hint can use this lightweight runtime inventory
    to tell the model which fixed-string target values are available.
    """
    root = config if config is not None else _load_merged_config()
    terminal = root.get("terminal", {}) if isinstance(root, Mapping) else {}
    targets = terminal.get("targets") if isinstance(terminal, Mapping) else None
    if not isinstance(targets, Mapping) or not targets:
        return (resolve_execution_target(config=root),)
    # Resolve the default first so name/default validation uses the same clear
    # ExecutionTargetError shape as an ordinary omitted-selector tool call.
    resolve_execution_target(config=root)
    return tuple(
        resolve_execution_target(name, config=root)
        for name in sorted(targets)
    )
