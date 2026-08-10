"""Profile-scoped runtime registry for named execution targets.

Each provider owns one versioned JSON fragment below
``$HERMES_HOME/runtime/execution-targets.d``.  Readers cache only a parsed
snapshot whose key includes directory membership plus each fragment's
device/inode/nanosecond-mtime/size tuple.  Writers take a bounded
cross-process lock and publish same-directory atomic replacements.

This module owns storage and schema validation only.  Merging with static
Hermes configuration and backend validation live in :mod:`execution_targets`
so there remains one production target validator.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any, Callable, Iterator, Mapping


REGISTRY_VERSION = 1
MAX_REGISTRY_FILE_BYTES = 1024 * 1024
MAX_TARGETS_PER_PROVIDER = 1024
MAX_TARGET_CONFIG_DEPTH = 16
MAX_TARGET_CONFIG_NODES = 8192
MAX_TARGET_CONFIG_BYTES = 256 * 1024
_PROVIDER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_RECORD_FIELDS = frozenset({"config", "owner_id", "generation", "state"})
_STATES = frozenset({"ready", "draining"})
_STRUCTURAL_CONFIG_KEYS = frozenset({
    "targets",
    "default_target",
    "provider",
    "owner_id",
    "generation",
    "state",
})
_STATE_FILENAME = ".registry-state.json"
_LOCK_FILENAME = ".registry.lock"


class RuntimeRegistryError(ValueError):
    """Raised for an invalid or conflicting runtime-registry operation."""


@dataclass(frozen=True)
class RegistryDiagnostic:
    code: str
    message: str
    provider: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.provider is not None:
            result["provider"] = self.provider
        return result


@dataclass(frozen=True)
class RuntimeTargetRecord:
    execution_target: str
    provider: str
    config: Mapping[str, Any]
    owner_id: str
    generation: str
    state: str

    def as_fragment_record(self) -> dict[str, Any]:
        return {
            "config": deepcopy(dict(self.config)),
            "owner_id": self.owner_id,
            "generation": self.generation,
            "state": self.state,
        }


@dataclass(frozen=True)
class RuntimeRegistrySnapshot:
    records: tuple[RuntimeTargetRecord, ...] = ()
    diagnostics: tuple[RegistryDiagnostic, ...] = ()
    legacy_activated: bool = False


_cache_lock = threading.RLock()
_snapshot_cache: dict[Path, tuple[tuple[Any, ...], RuntimeRegistrySnapshot]] = {}


def registry_directory() -> Path:
    """Return the active profile's runtime target-fragment directory."""
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "runtime" / "execution-targets.d"


def validate_provider_name(provider: str) -> str:
    """Validate a provider name before it can become a filename."""
    if not isinstance(provider, str) or not _PROVIDER_RE.fullmatch(provider):
        raise RuntimeRegistryError(
            "Provider must be 1-64 characters using only letters, digits, '.', "
            "'_', or '-', and must start with a letter or digit."
        )
    return provider


def _validate_target_name(name: Any) -> str:
    if not isinstance(name, str) or not name or len(name) > 256:
        raise RuntimeRegistryError(
            "Execution target names must be non-empty strings of at most 256 characters."
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise RuntimeRegistryError(
            "Execution target names cannot contain control characters."
        )
    return name


def _validate_record_identifier(label: str, value: Any, target: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise RuntimeRegistryError(
            f"Target {target!r} {label} must be a non-empty string of at most 512 characters."
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise RuntimeRegistryError(
            f"Target {target!r} {label} cannot contain control characters."
        )
    return value


def _looks_like_private_key_contents(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().upper()
    return (
        "\n" in value or "-----BEGIN " in normalized or "PRIVATE KEY-----" in normalized
    )


def _validate_config_complexity(config: Mapping[str, Any], target: str) -> None:
    """Bound provider-controlled config before copying or backend use."""
    stack: list[tuple[Any, int]] = [(config, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    encoded_bytes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_TARGET_CONFIG_NODES:
            raise RuntimeRegistryError(f"Target {target!r} config is too complex.")
        if depth > MAX_TARGET_CONFIG_DEPTH:
            raise RuntimeRegistryError(
                f"Target {target!r} config is nested too deeply."
            )
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen_containers:
                raise RuntimeRegistryError(
                    f"Target {target!r} config contains a recursive container."
                )
            seen_containers.add(identity)
            for key, item in value.items():
                if not isinstance(key, str):
                    raise RuntimeRegistryError(
                        f"Target {target!r} config keys must be strings."
                    )
                encoded_bytes += len(key.encode("utf-8"))
                stack.append((item, depth + 1))
        elif isinstance(value, list):
            identity = id(value)
            if identity in seen_containers:
                raise RuntimeRegistryError(
                    f"Target {target!r} config contains a recursive container."
                )
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in value)
        elif isinstance(value, str):
            encoded_bytes += len(value.encode("utf-8"))
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise RuntimeRegistryError(
                f"Target {target!r} config contains a non-JSON value."
            )
        if encoded_bytes > MAX_TARGET_CONFIG_BYTES:
            raise RuntimeRegistryError(f"Target {target!r} config is too large.")


def validate_fragment_targets(
    provider: str,
    targets: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return a normalized provider target mapping or raise safely."""
    validate_provider_name(provider)
    if not isinstance(targets, Mapping):
        raise RuntimeRegistryError("Provider fragment 'targets' must be an object.")
    if len(targets) > MAX_TARGETS_PER_PROVIDER:
        raise RuntimeRegistryError(
            f"Provider fragment exceeds the {MAX_TARGETS_PER_PROVIDER}-target limit."
        )

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in targets.items():
        name = _validate_target_name(raw_name)
        if not isinstance(raw_record, Mapping):
            raise RuntimeRegistryError(f"Target {name!r} record must be an object.")
        extra = set(raw_record) - _RECORD_FIELDS
        missing = _RECORD_FIELDS - set(raw_record)
        if extra or missing:
            raise RuntimeRegistryError(
                f"Target {name!r} record must contain exactly: "
                "config, generation, owner_id, state."
            )
        config = raw_record.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeRegistryError(f"Target {name!r} config must be an object.")
        _validate_config_complexity(config, name)
        for key in config:
            if key in _STRUCTURAL_CONFIG_KEYS:
                raise RuntimeRegistryError(
                    f"Target {name!r} cannot set structural field {key!r}."
                )
        if _looks_like_private_key_contents(config.get("ssh_key")):
            raise RuntimeRegistryError(
                f"Target {name!r} ssh_key must be a path or SSH alias setting, "
                "not private-key contents."
            )
        owner_id = raw_record.get("owner_id")
        generation = raw_record.get("generation")
        state = raw_record.get("state")
        owner_id = _validate_record_identifier("owner_id", owner_id, name)
        generation = _validate_record_identifier("generation", generation, name)
        if state not in _STATES:
            raise RuntimeRegistryError(
                f"Target {name!r} state must be 'ready' or 'draining'."
            )
        normalized[name] = {
            "config": deepcopy(dict(config)),
            "owner_id": owner_id,
            "generation": generation,
            "state": state,
        }
    return normalized


def _fragment_payload(provider: str, targets: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "provider": provider,
        "targets": validate_fragment_targets(provider, targets),
    }


def _user_only_error(path: Path, mode: int, owner: int | None) -> str | None:
    if os.name == "nt":
        return None
    if mode & 0o077:
        return f"{path.name} must not be accessible by group or other users"
    if not mode & 0o400:
        return f"{path.name} is not readable by its owner"
    geteuid = getattr(os, "geteuid", None)
    if owner is not None and callable(geteuid) and owner != geteuid():
        return f"{path.name} is not owned by the current user"
    return None


def _lstat_signature(path: Path) -> tuple[Any, ...]:
    st = path.lstat()
    return (
        path.name,
        st.st_dev,
        st.st_ino,
        st.st_mtime_ns,
        st.st_size,
        stat.S_IFMT(st.st_mode),
        stat.S_IMODE(st.st_mode),
        getattr(st, "st_uid", None),
    )


def _registry_signature(directory: Path) -> tuple[Any, ...]:
    try:
        directory_sig = _lstat_signature(directory)
    except FileNotFoundError:
        return ("missing",)
    except OSError as exc:
        return ("unreadable", type(exc).__name__, getattr(exc, "errno", None))
    if directory_sig[5] != stat.S_IFDIR:
        return (directory_sig,)

    try:
        names = sorted(
            entry.name
            for entry in directory.iterdir()
            if entry.name == _STATE_FILENAME
            or (entry.suffix == ".json" and not entry.name.startswith("."))
        )
    except OSError as exc:
        return (
            *directory_sig,
            "unreadable",
            type(exc).__name__,
            getattr(exc, "errno", None),
        )

    entries: list[tuple[Any, ...]] = []
    for name in names:
        try:
            entries.append(_lstat_signature(directory / name))
        except FileNotFoundError:
            entries.append((name, "vanished"))
        except OSError as exc:
            entries.append((
                name,
                "unreadable",
                type(exc).__name__,
                getattr(exc, "errno", None),
            ))
    return (directory_sig, *entries)


def _read_secure_json(path: Path) -> Any:
    """Read one owner-only regular JSON file without a symlink race."""
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeRegistryError(f"{path.name} is unreadable or malformed") from exc
    if stat.S_ISLNK(before.st_mode):
        raise RuntimeRegistryError(f"{path.name} is a symlink; refusing to follow it")
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeRegistryError(f"{path.name} is not a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = -1
    try:
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeRegistryError(f"{path.name} changed while being opened")
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeRegistryError(f"{path.name} is not a regular file")
        if opened.st_size > MAX_REGISTRY_FILE_BYTES:
            raise RuntimeRegistryError(f"{path.name} exceeds the size limit")
        permission_error = _user_only_error(
            path,
            stat.S_IMODE(opened.st_mode),
            getattr(opened, "st_uid", None),
        )
        if permission_error:
            raise RuntimeRegistryError(permission_error)
        handle = os.fdopen(fd, "rb")
        fd = -1
        with handle:
            raw = handle.read(MAX_REGISTRY_FILE_BYTES + 1)
            if len(raw) > MAX_REGISTRY_FILE_BYTES:
                raise RuntimeRegistryError(f"{path.name} exceeds the size limit")
            return json.loads(raw)
    except RuntimeRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeRegistryError(f"{path.name} is unreadable or malformed") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _parse_state(path: Path) -> bool:
    data = _read_secure_json(path)
    if (
        not isinstance(data, Mapping)
        or type(data.get("version")) is not int
        or data.get("version") != REGISTRY_VERSION
    ):
        raise RuntimeRegistryError(
            "registry activation state has an unsupported schema"
        )
    if set(data) != {"version", "legacy_activated"}:
        raise RuntimeRegistryError("registry activation state has unexpected fields")
    if data.get("legacy_activated") is not True:
        raise RuntimeRegistryError("registry activation marker must be true")
    return True


def _parse_fragment(path: Path) -> tuple[RuntimeTargetRecord, ...]:
    provider_from_name = path.stem
    validate_provider_name(provider_from_name)
    data = _read_secure_json(path)
    if not isinstance(data, Mapping):
        raise RuntimeRegistryError("provider fragment root must be an object")
    if set(data) != {"version", "provider", "targets"}:
        raise RuntimeRegistryError("provider fragment has unexpected or missing fields")
    if type(data.get("version")) is not int or data.get("version") != REGISTRY_VERSION:
        raise RuntimeRegistryError("provider fragment has an unsupported version")
    if data.get("provider") != provider_from_name:
        raise RuntimeRegistryError(
            "provider fragment name does not match its provider field"
        )
    targets = validate_fragment_targets(provider_from_name, data.get("targets"))
    return tuple(
        RuntimeTargetRecord(
            execution_target=name,
            provider=provider_from_name,
            config=record["config"],
            owner_id=record["owner_id"],
            generation=record["generation"],
            state=record["state"],
        )
        for name, record in sorted(targets.items())
    )


def _load_snapshot_uncached(directory: Path) -> RuntimeRegistrySnapshot:
    diagnostics: list[RegistryDiagnostic] = []
    records: list[RuntimeTargetRecord] = []
    legacy_activated = False

    try:
        st = directory.lstat()
    except FileNotFoundError:
        return RuntimeRegistrySnapshot()
    except OSError:
        return RuntimeRegistrySnapshot(
            diagnostics=(
                RegistryDiagnostic(
                    code="registry_unreadable",
                    message="Runtime execution-target registry is unreadable.",
                ),
            )
        )
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return RuntimeRegistrySnapshot(
            diagnostics=(
                RegistryDiagnostic(
                    code="registry_unsafe",
                    message="Runtime execution-target registry is not a secure directory.",
                ),
            )
        )
    permission_error = _user_only_error(
        directory,
        stat.S_IMODE(st.st_mode),
        getattr(st, "st_uid", None),
    )
    if permission_error:
        return RuntimeRegistrySnapshot(
            diagnostics=(
                RegistryDiagnostic(
                    code="registry_permissions",
                    message="Runtime execution-target registry permissions are not user-only.",
                ),
            )
        )

    state_path = directory / _STATE_FILENAME
    try:
        if os.path.lexists(state_path):
            legacy_activated = _parse_state(state_path)
    except RuntimeRegistryError:
        diagnostics.append(
            RegistryDiagnostic(
                code="activation_state_invalid",
                message="Runtime registry activation state is malformed or unsafe.",
            )
        )

    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError:
        return RuntimeRegistrySnapshot(
            diagnostics=tuple(diagnostics)
            + (
                RegistryDiagnostic(
                    code="registry_unreadable",
                    message="Runtime execution-target registry is unreadable.",
                ),
            ),
            legacy_activated=legacy_activated,
        )
    for path in entries:
        if path.name.startswith(".") or path.suffix != ".json":
            continue
        provider = path.stem if _PROVIDER_RE.fullmatch(path.stem) else None
        try:
            records.extend(_parse_fragment(path))
        except (OSError, RuntimeRegistryError):
            diagnostics.append(
                RegistryDiagnostic(
                    code="provider_fragment_invalid",
                    provider=provider,
                    message=(
                        f"Runtime target provider {provider!r} is unavailable because "
                        "its fragment is malformed, unreadable, insecure, or symlinked."
                        if provider is not None
                        else "A runtime target provider fragment has an unsafe filename or schema."
                    ),
                )
            )
    return RuntimeRegistrySnapshot(
        records=tuple(records),
        diagnostics=tuple(diagnostics),
        legacy_activated=legacy_activated,
    )


def load_runtime_registry(*, force: bool = False) -> RuntimeRegistrySnapshot:
    """Load the active profile's registry with strong atomic-replace freshness."""
    directory = registry_directory()
    for _attempt in range(3):
        before = _registry_signature(directory)
        with _cache_lock:
            cached = _snapshot_cache.get(directory)
            if not force and cached is not None and cached[0] == before:
                return cached[1]
        snapshot = _load_snapshot_uncached(directory)
        after = _registry_signature(directory)
        if before == after:
            with _cache_lock:
                _snapshot_cache[directory] = (after, snapshot)
            return snapshot
    # A continuously changing provider should not make static targets fail.
    return RuntimeRegistrySnapshot(
        diagnostics=(
            RegistryDiagnostic(
                code="registry_changing",
                message="Runtime execution-target registry changed while being read; retry the command.",
            ),
        )
    )


def invalidate_runtime_registry_cache() -> None:
    directory = registry_directory()
    with _cache_lock:
        _snapshot_cache.pop(directory, None)


def _ensure_registry_directory() -> Path:
    directory = registry_directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = directory.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeRegistryError("Runtime registry path is not a secure directory.")
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    st = directory.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeRegistryError("Runtime registry path changed while being secured.")
    permission_error = _user_only_error(
        directory,
        stat.S_IMODE(st.st_mode),
        getattr(st, "st_uid", None),
    )
    if permission_error:
        raise RuntimeRegistryError("Runtime registry directory is not user-only.")
    return directory


def _try_lock(handle: Any) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def registry_write_lock(*, timeout_seconds: float = 5.0) -> Iterator[Path]:
    """Acquire the bounded global registry lock for a read-modify-write."""
    directory = _ensure_registry_directory()
    lock_path = directory / _LOCK_FILENAME
    before = None
    if os.path.lexists(lock_path):
        before = lock_path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeRegistryError("Runtime registry lock is not a regular file.")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeRegistryError(
            "Could not safely open the runtime registry lock."
        ) from exc
    try:
        opened = os.fstat(fd)
        if before is not None and (
            before.st_dev,
            before.st_ino,
        ) != (opened.st_dev, opened.st_ino):
            raise RuntimeRegistryError("Runtime registry lock changed while opening.")
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeRegistryError("Runtime registry lock is not a regular file.")
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        opened = os.fstat(fd)
        if _user_only_error(
            lock_path,
            stat.S_IMODE(opened.st_mode),
            getattr(opened, "st_uid", None),
        ):
            raise RuntimeRegistryError("Runtime registry lock is not user-only.")
    except Exception:
        os.close(fd)
        raise
    handle = os.fdopen(fd, "a+b", buffering=0)
    try:
        if os.name == "nt" and opened.st_size == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while not _try_lock(handle):
            if time.monotonic() >= deadline:
                raise RuntimeRegistryError(
                    "Timed out waiting for another execution-target registry writer."
                )
            time.sleep(0.05)
        try:
            yield directory
        finally:
            _unlock(handle)
    finally:
        handle.close()


def _serialized_json_bytes(payload: Mapping[str, Any]) -> bytes:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    if len(data) > MAX_REGISTRY_FILE_BYTES:
        raise RuntimeRegistryError(
            "Runtime registry data exceeds the total fragment size limit."
        )
    return data


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _serialized_json_bytes(payload)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(fd)
        fd = -1
        os.replace(temp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


RegistryMutator = Callable[
    [RuntimeRegistrySnapshot, dict[str, dict[str, Any]]],
    Mapping[str, Any],
]


def update_provider_fragment(
    provider: str,
    mutator: RegistryMutator,
    *,
    activate_legacy: bool = False,
) -> RuntimeRegistrySnapshot:
    """Atomically read/validate/update one provider-owned fragment.

    The callback runs while the global cross-process lock is held.  If legacy
    activation is requested, the callback sees the proposed activation in its
    snapshot; only after it succeeds is activation state durably written,
    followed by the provider fragment.  A crash between those writes therefore
    leaves a valid live synthetic default and never a half-published target.
    """
    provider = validate_provider_name(provider)

    with registry_write_lock() as directory:
        snapshot = load_runtime_registry(force=True)
        if any(diag.provider == provider for diag in snapshot.diagnostics):
            raise RuntimeRegistryError(
                f"Provider {provider!r} is unavailable; repair or remove its invalid fragment first."
            )
        current = {
            record.execution_target: record.as_fragment_record()
            for record in snapshot.records
            if record.provider == provider
        }
        callback_snapshot = snapshot
        if not snapshot.legacy_activated and activate_legacy:
            callback_snapshot = RuntimeRegistrySnapshot(
                records=snapshot.records,
                diagnostics=snapshot.diagnostics,
                legacy_activated=True,
            )
        new_targets = validate_fragment_targets(
            provider,
            mutator(callback_snapshot, deepcopy(current)),
        )
        fragment_payload = _fragment_payload(provider, new_targets)

        # Validate the complete publication before the marker write below.  The
        # atomic writer repeats this check for its own safety, but this preflight
        # keeps legacy activation marker-only and preserves the complete visible
        # target set when a provider's individually-valid configs exceed the
        # fragment-wide byte limit.
        _serialized_json_bytes(fragment_payload)

        if not snapshot.legacy_activated and activate_legacy:
            _atomic_write_json(
                directory / _STATE_FILENAME,
                {
                    "version": REGISTRY_VERSION,
                    "legacy_activated": True,
                },
            )
        _atomic_write_json(
            directory / f"{provider}.json",
            fragment_payload,
        )
        invalidate_runtime_registry_cache()
        return load_runtime_registry(force=True)


def write_provider_fragment_for_tests(
    provider: str,
    targets: Mapping[str, Any],
) -> Path:
    """Publish a valid fragment through the production atomic writer.

    This deliberately remains useful to provider integrations and acceptance
    tests without exposing an unsafe partial-record mutation API.
    """
    with registry_write_lock() as directory:
        path = directory / f"{validate_provider_name(provider)}.json"
        _atomic_write_json(path, _fragment_payload(provider, targets))
        invalidate_runtime_registry_cache()
        return path
