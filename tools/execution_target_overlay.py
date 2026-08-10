"""Merge runtime execution-target records with effective static config."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from tools.execution_target_registry import (
    RegistryDiagnostic,
    RuntimeRegistrySnapshot,
    load_runtime_registry,
)


REGISTRY_METADATA_KEY = "__execution_target_registry_v1__"


def _available(names: list[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "(none)"


def overlay_runtime_execution_targets(
    config: Mapping[str, Any],
    *,
    snapshot: RuntimeRegistrySnapshot | None = None,
    raise_for_target: str | None = None,
) -> dict[str, Any]:
    """Return static config plus one consistent profile-registry snapshot.

    Static names reserve their aliases.  Provider/provider collisions,
    draining records, and invalid provider configs are represented in the
    sidecar metadata but withheld from ``terminal.targets``.
    """
    root = deepcopy(dict(config))
    root.pop(REGISTRY_METADATA_KEY, None)
    if snapshot is None:
        snapshot = load_runtime_registry()
    if (
        not snapshot.records
        and not snapshot.legacy_activated
        and not snapshot.diagnostics
    ):
        return root

    diagnostics = list(snapshot.diagnostics)
    records_metadata: list[dict[str, Any]] = []
    raw_terminal = root.get("terminal", {})
    if raw_terminal is None:
        raw_terminal = {}
    if not isinstance(raw_terminal, Mapping):
        root[REGISTRY_METADATA_KEY] = {
            "records": records_metadata,
            "diagnostics": [diag.as_dict() for diag in diagnostics],
        }
        return root

    terminal = deepcopy(dict(raw_terminal))
    raw_static_targets = terminal.get("targets")
    static_named = isinstance(raw_static_targets, Mapping) and bool(raw_static_targets)
    if raw_static_targets not in (None, {}) and not isinstance(
        raw_static_targets, Mapping
    ):
        diagnostics.append(
            RegistryDiagnostic(
                code="static_targets_invalid",
                message="Runtime targets are unavailable because terminal.targets is not a mapping.",
            )
        )
        root[REGISTRY_METADATA_KEY] = {
            "records": records_metadata,
            "diagnostics": [diag.as_dict() for diag in diagnostics],
        }
        return root

    if static_named:
        targets = deepcopy(dict(raw_static_targets))
    elif snapshot.legacy_activated:
        # Durable state is marker-only. Build the named default live from this
        # dispatch's static/profile authority while retaining shared policy.
        from tools.execution_targets import (
            ExecutionTargetError,
            resolve_execution_target,
            synthesize_legacy_default_target,
        )

        try:
            synthetic = synthesize_legacy_default_target(root)
            if synthetic is None:
                raise ExecutionTargetError("legacy terminal config is already named")
            targets = {"default": synthetic}
            terminal["default_target"] = "default"
            terminal["targets"] = targets
            candidate = deepcopy(root)
            candidate["terminal"] = terminal
            resolve_execution_target("default", config=candidate)
        except (ExecutionTargetError, ValueError):
            if raise_for_target is not None:
                raise
            for record in snapshot.records:
                records_metadata.append({
                    "execution_target": record.execution_target,
                    "provider": record.provider,
                    "owner_id": record.owner_id,
                    "generation": record.generation,
                    "state": record.state,
                    "status": "inactive_legacy",
                })
            diagnostics.append(
                RegistryDiagnostic(
                    code="activation_state_invalid",
                    message=(
                        "Runtime targets are unavailable because legacy activation "
                        "state has invalid backend configuration."
                    ),
                )
            )
            root[REGISTRY_METADATA_KEY] = {
                "records": records_metadata,
                "diagnostics": [diag.as_dict() for diag in diagnostics],
            }
            return root
        static_named = True
    else:
        for record in snapshot.records:
            records_metadata.append({
                "execution_target": record.execution_target,
                "provider": record.provider,
                "owner_id": record.owner_id,
                "generation": record.generation,
                "state": record.state,
                "status": "inactive_legacy",
            })
        if snapshot.records:
            diagnostics.append(
                RegistryDiagnostic(
                    code="legacy_not_activated",
                    message=(
                        "Runtime targets are unavailable in legacy flat terminal mode; "
                        "register one with 'hermes targets register' to activate a stable default."
                    ),
                )
            )
        root[REGISTRY_METADATA_KEY] = {
            "records": records_metadata,
            "diagnostics": [diag.as_dict() for diag in diagnostics],
        }
        return root

    static_names = set(targets)
    by_name: dict[str, list[Any]] = {}
    for record in snapshot.records:
        by_name.setdefault(record.execution_target, []).append(record)

    runtime_records: dict[str, Any] = {}
    for name, records in sorted(by_name.items()):
        if name in static_names:
            for record in records:
                records_metadata.append({
                    "execution_target": name,
                    "provider": record.provider,
                    "owner_id": record.owner_id,
                    "generation": record.generation,
                    "state": record.state,
                    "status": "shadowed_static",
                })
            diagnostics.append(
                RegistryDiagnostic(
                    code="static_name_reserved",
                    message=f"Static execution target {name!r} reserves its name over runtime providers.",
                )
            )
            continue
        if len(records) > 1:
            providers = sorted(record.provider for record in records)
            for record in records:
                records_metadata.append({
                    "execution_target": name,
                    "provider": record.provider,
                    "owner_id": record.owner_id,
                    "generation": record.generation,
                    "state": record.state,
                    "status": "provider_collision",
                })
            diagnostics.append(
                RegistryDiagnostic(
                    code="provider_name_collision",
                    message=(
                        f"Runtime execution target {name!r} is unavailable because multiple "
                        f"providers claim it: {_available(providers)}."
                    ),
                )
            )
            continue
        record = records[0]
        status = "active" if record.state == "ready" else "draining"
        records_metadata.append({
            "execution_target": name,
            "provider": record.provider,
            "owner_id": record.owner_id,
            "generation": record.generation,
            "state": record.state,
            "status": status,
        })
        if record.state == "ready":
            targets[name] = deepcopy(dict(record.config))
            runtime_records[name] = record

    terminal["targets"] = targets
    root["terminal"] = terminal
    root[REGISTRY_METADATA_KEY] = {
        "records": records_metadata,
        "diagnostics": [diag.as_dict() for diag in diagnostics],
    }

    # Provider JSON is an external control-plane boundary. Validate through
    # the production resolver and isolate only the bad provider record.
    from tools.execution_targets import ExecutionTargetError, resolve_execution_target

    for name, record in runtime_records.items():
        try:
            resolve_execution_target(name, config=root)
        except ExecutionTargetError:
            if name == raise_for_target:
                raise
            targets.pop(name, None)
            for metadata in records_metadata:
                if (
                    metadata["execution_target"] == name
                    and metadata["provider"] == record.provider
                ):
                    metadata["status"] = "invalid_config"
            diagnostics.append(
                RegistryDiagnostic(
                    code="runtime_target_invalid",
                    provider=record.provider,
                    message=(
                        f"Runtime execution target {name!r} from provider "
                        f"{record.provider!r} has invalid backend configuration."
                    ),
                )
            )
    root[REGISTRY_METADATA_KEY]["diagnostics"] = [
        diag.as_dict() for diag in diagnostics
    ]
    return root
