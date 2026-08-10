"""CLI management for profile-scoped runtime execution targets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from typing import Any, Mapping

from tools.execution_target_registry import (
    REGISTRY_VERSION,
    RuntimeRegistryError,
    RuntimeRegistrySnapshot,
    RuntimeTargetRecord,
    load_runtime_registry,
    update_provider_fragment,
    validate_provider_name,
)
from tools.execution_targets import ExecutionTargetError


_STRUCTURAL_KEYS = frozenset({
    "targets",
    "default_target",
    "provider",
    "owner_id",
    "generation",
    "state",
})


def _parse_set_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise RuntimeRegistryError("--set requires KEY=VALUE.")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise RuntimeRegistryError("--set requires a non-empty key.")
    if key in _STRUCTURAL_KEYS:
        raise RuntimeRegistryError(f"--set cannot modify structural field {key!r}.")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def _build_register_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {"backend": args.backend}
    optional = {
        "ssh_host": args.host,
        "ssh_user": args.user,
        "ssh_port": args.port,
        "ssh_key": args.key,
        "cwd": args.cwd,
        "timeout": args.timeout,
    }
    for key, value in optional.items():
        if value is not None:
            config[key] = value
    for raw in args.settings:
        key, value = _parse_set_value(raw)
        if key in config:
            raise RuntimeRegistryError(
                f"Target setting {key!r} was supplied more than once."
            )
        config[key] = value
    return config


def _static_target_names(config: Mapping[str, Any]) -> set[str]:
    terminal = config.get("terminal")
    targets = terminal.get("targets") if isinstance(terminal, Mapping) else None
    if not isinstance(targets, Mapping):
        return set()
    return {name for name in targets if isinstance(name, str)}


def _snapshot_with_provider(
    snapshot: RuntimeRegistrySnapshot,
    provider: str,
    targets: Mapping[str, Mapping[str, Any]],
) -> RuntimeRegistrySnapshot:
    records = [record for record in snapshot.records if record.provider != provider]
    for name, record in sorted(targets.items()):
        records.append(
            RuntimeTargetRecord(
                execution_target=name,
                provider=provider,
                config=deepcopy(dict(record["config"])),
                owner_id=str(record["owner_id"]),
                generation=str(record["generation"]),
                state=str(record["state"]),
            )
        )
    return RuntimeRegistrySnapshot(
        records=tuple(records),
        diagnostics=snapshot.diagnostics,
        legacy_activated=snapshot.legacy_activated,
    )


def _validate_complete_candidate(
    static_config: Mapping[str, Any],
    snapshot: RuntimeRegistrySnapshot,
    target: str,
) -> None:
    from tools.execution_targets import resolve_execution_target

    from tools.execution_target_overlay import overlay_runtime_execution_targets

    candidate = overlay_runtime_execution_targets(
        static_config,
        snapshot=snapshot,
        raise_for_target=target,
    )
    resolve_execution_target(config=candidate)
    resolution = resolve_execution_target(target, config=candidate)
    if resolution.provider is None:
        raise RuntimeRegistryError(
            f"Execution target {target!r} is missing from the candidate."
        )


def _routing_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only allowlisted, non-secret routing fields."""
    summary: dict[str, Any] = {}
    for key in ("backend", "cwd", "ssh_host", "ssh_user", "ssh_port"):
        value = config.get(key)
        if value not in (None, ""):
            summary[key] = value
    if "backend" not in summary:
        backend = config.get("env_type")
        if backend not in (None, ""):
            summary["backend"] = backend
    return summary


def _payload(
    record: RuntimeTargetRecord,
    *,
    status: str | None = None,
    config: Mapping[str, Any] | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "execution_target": record.execution_target,
        "provider": record.provider,
        "owner_id": record.owner_id,
        "generation": record.generation,
        "state": record.state,
        "status": status or ("active" if record.state == "ready" else "draining"),
        "routing": _routing_summary(config or record.config),
    }
    if action:
        result["action"] = action
    return result


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    action = str(payload.get("action") or "target")
    routing = payload.get("routing")
    backend = routing.get("backend") if isinstance(routing, Mapping) else None
    suffix = f", backend={backend}" if backend else ""
    print(
        f"{action.capitalize()} {payload.get('execution_target')!r} "
        f"(provider={payload.get('provider')}, "
        f"generation={payload.get('generation')}, "
        f"state={payload.get('state')}{suffix})"
    )


def _error(args: argparse.Namespace, exc: Exception) -> int:
    from agent.redact import redact_sensitive_text

    message = redact_sensitive_text(str(exc).strip(), force=True)
    if not message:
        message = "Target command failed."
    if args.json:
        print(
            json.dumps(
                {"status": "error", "message": message},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(f"Error: {message}", file=sys.stderr)
    return 2


def _cmd_register(args: argparse.Namespace) -> int:
    try:
        from tools.execution_targets import load_static_execution_target_config

        provider = validate_provider_name(args.provider)
        name = args.name
        config = _build_register_config(args)
        desired = {
            "config": config,
            "owner_id": args.owner_id or name,
            "generation": args.generation or "manual",
            "state": "ready",
        }
        static_config = load_static_execution_target_config()
        initial = load_runtime_registry()
        activate_legacy = False
        terminal = static_config.get("terminal")
        static_targets = (
            terminal.get("targets") if isinstance(terminal, Mapping) else None
        )
        if not isinstance(static_targets, Mapping) or not static_targets:
            if not initial.legacy_activated:
                activate_legacy = True
        outcome = {"action": "registered"}

        def mutate(
            snapshot: RuntimeRegistrySnapshot,
            current: dict[str, dict[str, Any]],
        ) -> Mapping[str, Any]:
            static_names = _static_target_names(static_config)
            if snapshot.legacy_activated and not static_names:
                static_names.add("default")
            if name in static_names:
                raise RuntimeRegistryError(
                    f"Execution target {name!r} is reserved by static configuration."
                )
            other_providers = sorted({
                record.provider
                for record in snapshot.records
                if record.execution_target == name and record.provider != provider
            })
            if other_providers:
                raise RuntimeRegistryError(
                    f"Execution target {name!r} is already owned by runtime "
                    f"provider(s) {', '.join(other_providers)}."
                )
            existing = current.get(name)
            if args.if_generation is not None:
                actual = existing.get("generation") if existing else None
                if actual != args.if_generation:
                    raise RuntimeRegistryError(
                        f"Stale generation for {name!r}: expected "
                        f"{args.if_generation!r}, found {actual!r}."
                    )
            if existing == desired:
                outcome["action"] = "unchanged"
                return current
            if existing is not None and not args.replace:
                raise RuntimeRegistryError(
                    f"Execution target {name!r} already exists for provider "
                    f"{provider!r}; use --replace to repoint it."
                )
            current[name] = deepcopy(desired)
            future = _snapshot_with_provider(snapshot, provider, current)
            _validate_complete_candidate(static_config, future, name)
            outcome["action"] = "replaced" if existing is not None else "registered"
            return current

        updated = update_provider_fragment(
            provider,
            mutate,
            activate_legacy=activate_legacy,
        )
        record = next(
            item
            for item in updated.records
            if item.provider == provider and item.execution_target == name
        )
        from tools.execution_target_overlay import overlay_runtime_execution_targets
        from tools.execution_targets import resolve_execution_target

        merged = overlay_runtime_execution_targets(static_config, snapshot=updated)
        resolution = resolve_execution_target(name, config=merged)
        _print_payload(
            _payload(record, config=resolution.config, action=outcome["action"]),
            as_json=args.json,
        )
        return 0
    except (OSError, RuntimeRegistryError, ExecutionTargetError) as exc:
        return _error(args, exc)


def _metadata_statuses(merged: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    from tools.execution_target_overlay import REGISTRY_METADATA_KEY

    registry = merged.get(REGISTRY_METADATA_KEY)
    records = registry.get("records") if isinstance(registry, Mapping) else None
    result: dict[tuple[str, str], str] = {}
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, Mapping):
                continue
            name = item.get("execution_target")
            provider = item.get("provider")
            status = item.get("status")
            if all(isinstance(value, str) for value in (name, provider, status)):
                result[(name, provider)] = status
    return result


def _collect_rows(
    *,
    include_all: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from tools.execution_target_overlay import (
        REGISTRY_METADATA_KEY,
        overlay_runtime_execution_targets,
    )
    from tools.execution_targets import (
        load_static_execution_target_config,
        resolve_execution_target,
    )

    static_config = load_static_execution_target_config()
    snapshot = load_runtime_registry()
    merged = overlay_runtime_execution_targets(static_config, snapshot=snapshot)
    statuses = _metadata_statuses(merged)
    rows: list[dict[str, Any]] = []
    static_names = _static_target_names(static_config)
    if not static_names:
        static_names.add("default")
    for name in sorted(static_names):
        try:
            resolution = resolve_execution_target(name, config=merged)
        except ExecutionTargetError:
            continue
        rows.append({
            "execution_target": name,
            "provider": "static",
            "owner_id": name,
            "generation": (
                ("legacy-activation-v1" if snapshot.legacy_activated else "legacy-flat")
                if name == "default"
                else "static"
            ),
            "state": "ready",
            "status": "active",
            "routing": _routing_summary(resolution.config),
        })
    for record in sorted(
        snapshot.records,
        key=lambda item: (item.execution_target, item.provider),
    ):
        status = statuses.get(
            (record.execution_target, record.provider),
            "active" if record.state == "ready" else "draining",
        )
        effective: Mapping[str, Any] = record.config
        if status == "active":
            try:
                effective = resolve_execution_target(
                    record.execution_target,
                    config=merged,
                ).config
            except ExecutionTargetError:
                pass
        if include_all or status == "active":
            rows.append(_payload(record, status=status, config=effective))
    registry_meta = merged.get(REGISTRY_METADATA_KEY)
    raw_diagnostics = (
        registry_meta.get("diagnostics") if isinstance(registry_meta, Mapping) else []
    )
    diagnostics = (
        [dict(item) for item in raw_diagnostics if isinstance(item, Mapping)]
        if include_all and isinstance(raw_diagnostics, list)
        else []
    )
    return rows, diagnostics


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        rows, diagnostics = _collect_rows(include_all=args.all)
        if args.json:
            print(
                json.dumps(
                    {
                        "version": REGISTRY_VERSION,
                        "targets": rows,
                        "diagnostics": diagnostics,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not rows:
            print("No execution targets found.")
        for row in rows:
            routing = row.get("routing")
            backend = (
                routing.get("backend", "-") if isinstance(routing, Mapping) else "-"
            )
            print(
                f"{row['execution_target']}  {row['provider']}  "
                f"{row['generation']}  {row['state']}  "
                f"{row['status']}  {backend}"
            )
        for diagnostic in diagnostics:
            print(
                f"Warning: {diagnostic.get('message', 'Runtime provider unavailable.')}",
                file=sys.stderr,
            )
        return 0
    except (OSError, RuntimeRegistryError, ExecutionTargetError) as exc:
        return _error(args, exc)


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        rows, _diagnostics = _collect_rows(include_all=True)
        matches = [
            row
            for row in rows
            if row["execution_target"] == args.name
            and (args.provider is None or row["provider"] == args.provider)
        ]
        if not matches:
            raise RuntimeRegistryError(f"Execution target {args.name!r} was not found.")
        if len(matches) > 1:
            providers = ", ".join(sorted(str(row["provider"]) for row in matches))
            raise RuntimeRegistryError(
                f"Execution target {args.name!r} is claimed by multiple "
                f"providers ({providers}); pass --provider."
            )
        row = matches[0]
        if args.json:
            print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            _print_payload(row, as_json=False)
            print(f"Status: {row['status']}")
        return 0
    except (OSError, RuntimeRegistryError, ExecutionTargetError) as exc:
        return _error(args, exc)


def _find_runtime_record(
    snapshot: RuntimeRegistrySnapshot,
    name: str,
    provider: str | None,
) -> RuntimeTargetRecord:
    matches = [
        record
        for record in snapshot.records
        if record.execution_target == name
        and (provider is None or record.provider == provider)
    ]
    if not matches:
        raise RuntimeRegistryError(f"Runtime execution target {name!r} was not found.")
    if len(matches) > 1:
        providers = ", ".join(sorted(record.provider for record in matches))
        raise RuntimeRegistryError(
            f"Runtime target {name!r} is claimed by {providers}; pass --provider."
        )
    return matches[0]


def _mutate_lifecycle(args: argparse.Namespace, *, remove: bool) -> int:
    try:
        selected = _find_runtime_record(
            load_runtime_registry(),
            args.name,
            args.provider,
        )
        provider = selected.provider
        outcome = {"record": selected}

        def mutate(
            snapshot: RuntimeRegistrySnapshot,
            current: dict[str, dict[str, Any]],
        ) -> Mapping[str, Any]:
            fresh = _find_runtime_record(snapshot, args.name, args.provider)
            if fresh.provider != provider:
                raise RuntimeRegistryError(
                    f"Runtime target {args.name!r} changed provider; retry."
                )
            existing = current.get(args.name)
            if existing is None:
                raise RuntimeRegistryError(
                    f"Runtime target {args.name!r} no longer exists."
                )
            if (
                args.if_generation is not None
                and existing.get("generation") != args.if_generation
            ):
                raise RuntimeRegistryError(
                    f"Stale generation for {args.name!r}: expected "
                    f"{args.if_generation!r}, "
                    f"found {existing.get('generation')!r}."
                )
            outcome["record"] = fresh
            if remove:
                current.pop(args.name)
            else:
                existing["state"] = "draining"
            return current

        update_provider_fragment(provider, mutate)
        record = outcome["record"]
        if not remove:
            record = RuntimeTargetRecord(
                execution_target=record.execution_target,
                provider=record.provider,
                config=record.config,
                owner_id=record.owner_id,
                generation=record.generation,
                state="draining",
            )
        action = "removed" if remove else "drained"
        _print_payload(_payload(record, action=action), as_json=args.json)
        return 0
    except (OSError, RuntimeRegistryError, ExecutionTargetError) as exc:
        return _error(args, exc)


def _cmd_drain(args: argparse.Namespace) -> int:
    return _mutate_lifecycle(args, remove=False)


def _cmd_remove(args: argparse.Namespace) -> int:
    return _mutate_lifecycle(args, remove=True)


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", help="Execution-target name")
    parser.add_argument("--provider", help="Owning provider when ambiguous")
    parser.add_argument(
        "--if-generation",
        metavar="GEN",
        help="Compare-and-swap generation",
    )
    parser.add_argument("--json", action="store_true", help="Emit stable JSON")


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach the built-in targets management command."""
    parent.set_defaults(func=lambda _args: (parent.print_help(), 0)[1])
    sub = parent.add_subparsers(dest="targets_action")
    register = sub.add_parser(
        "register",
        help="Register or replace a ready runtime target",
    )
    register.add_argument("name")
    register.add_argument("--backend", required=True)
    register.add_argument("--host", "--ssh-host", dest="host")
    register.add_argument("--user", "--ssh-user", dest="user")
    register.add_argument("--port", "--ssh-port", dest="port", type=int)
    register.add_argument(
        "--key",
        "--ssh-key",
        dest="key",
        help="SSH private-key path (never contents)",
    )
    register.add_argument("--cwd")
    register.add_argument("--timeout", type=int)
    register.add_argument("--provider", default="cli")
    register.add_argument("--owner-id")
    register.add_argument("--generation")
    register.add_argument("--replace", action="store_true")
    register.add_argument("--if-generation", metavar="OLD")
    register.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Advanced backend setting; JSON values accepted; repeatable",
    )
    register.add_argument("--json", action="store_true")
    register.set_defaults(func=_cmd_register)
    listing = sub.add_parser(
        "list",
        aliases=["ls"],
        help="List callable runtime/static targets",
    )
    listing.add_argument("--all", action="store_true")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_cmd_list)
    show = sub.add_parser("show", help="Show one target without secrets")
    show.add_argument("name")
    show.add_argument("--provider")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=_cmd_show)
    drain = sub.add_parser("drain", help="Reject new calls for one generation")
    _add_identity_args(drain)
    drain.set_defaults(func=_cmd_drain)
    remove = sub.add_parser(
        "remove",
        aliases=["unregister"],
        help="Remove one exact generation",
    )
    _add_identity_args(remove)
    remove.set_defaults(func=_cmd_remove)
