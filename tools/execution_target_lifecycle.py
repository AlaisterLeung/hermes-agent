"""Runtime lifecycle metadata helpers for execution-target resolution."""

from __future__ import annotations

from typing import Any, Mapping


_REGISTRY_METADATA_KEY = "__execution_target_registry_v1__"


def runtime_record_metadata(
    root: Mapping[str, Any],
    target: str,
    *,
    status: str | None = None,
) -> list[Mapping[str, Any]]:
    registry = root.get(_REGISTRY_METADATA_KEY)
    records = registry.get("records") if isinstance(registry, Mapping) else None
    if not isinstance(records, list):
        return []
    return [
        item
        for item in records
        if isinstance(item, Mapping)
        and item.get("execution_target") == target
        and (status is None or item.get("status") == status)
    ]


def unavailable_runtime_target_message(
    root: Mapping[str, Any],
    target: str,
) -> str | None:
    statuses = {
        str(item.get("status")) for item in runtime_record_metadata(root, target)
    }
    if "draining" in statuses:
        return (
            f"Execution target {target!r} is draining and rejects new calls. "
            "Use hermes targets show or register a ready replacement."
        )
    if "provider_collision" in statuses:
        return (
            f"Execution target {target!r} is unavailable because multiple runtime "
            "providers claim it. Run hermes targets list --all to reconcile ownership."
        )
    if "invalid_config" in statuses:
        return (
            f"Execution target {target!r} is unavailable because its runtime provider "
            "published invalid backend configuration. Run hermes targets list --all."
        )
    if "inactive_legacy" in statuses:
        return (
            f"Execution target {target!r} is not active because legacy flat terminal "
            "mode has not been initialized. Use hermes targets register."
        )
    return None
