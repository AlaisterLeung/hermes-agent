from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.execution_target_registry import (
    MAX_REGISTRY_FILE_BYTES,
    MAX_TARGET_CONFIG_BYTES,
    MAX_TARGETS_PER_PROVIDER,
    RuntimeRegistryError,
    invalidate_runtime_registry_cache,
    load_runtime_registry,
    registry_directory,
    update_provider_fragment,
    validate_provider_name,
    write_provider_fragment_for_tests,
)
from tools.execution_targets import (
    ExecutionTargetError,
    frozen_execution_target_config,
    list_execution_targets,
    resolve_execution_target,
    set_execution_target_config_source,
)


def _static_config(cwd: str) -> dict:
    return {
        "terminal": {
            "default_target": "local",
            "targets": {"local": {"backend": "local", "cwd": cwd}},
        },
    }


def _record(
    cwd: str,
    *,
    generation: str = "g1",
    state: str = "ready",
    backend: str = "local",
) -> dict:
    config = {"backend": backend, "cwd": cwd}
    if backend == "ssh":
        config.update({"ssh_host": "managed-alias", "ssh_user": "dev"})
    return {
        "config": config,
        "owner_id": "box-17",
        "generation": generation,
        "state": state,
    }


@pytest.fixture
def registry_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    invalidate_runtime_registry_cache()
    set_execution_target_config_source(_static_config(str(tmp_path)))
    try:
        yield home
    finally:
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()


def test_new_target_resolves_and_executes_without_process_restart(
    registry_home,
    tmp_path,
):
    from tools.terminal_tool import terminal_tool

    work = tmp_path / "new-runtime"
    work.mkdir()
    with pytest.raises(ExecutionTargetError):
        resolve_execution_target("box")

    write_provider_fragment_for_tests("hetzner-devbox", {"box": _record(str(work))})
    resolution = resolve_execution_target("box")
    assert resolution.provider == "hetzner-devbox"
    assert resolution.owner_id == "box-17"
    assert resolution.generation == "g1"

    result = json.loads(terminal_tool("pwd", target="box", task_id="registry-e2e"))
    assert result["exit_code"] == 0
    assert Path(result["output"].strip()).resolve() == work.resolve()
    assert result["target"] == "box"
    assert result["runtime_scope"] == resolution.security_scope


def test_separate_cli_process_hot_registers_for_long_lived_reader(
    registry_home,
    tmp_path,
):
    from tools.terminal_tool import TERMINAL_SCHEMA, terminal_tool

    work = tmp_path / "cross-process-target"
    work.mkdir()
    schema_before = deepcopy(TERMINAL_SCHEMA)
    assert [item.target for item in list_execution_targets()] == ["local"]
    with pytest.raises(ExecutionTargetError):
        resolve_execution_target("hot-local")

    env = dict(os.environ)
    env["HERMES_HOME"] = str(registry_home)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "targets",
            "register",
            "hot-local",
            "--backend",
            "local",
            "--cwd",
            str(work),
            "--provider",
            "cross-process",
            "--generation",
            "child-1",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["execution_target"] == "hot-local"

    resolution = resolve_execution_target("hot-local")
    assert resolution.provider == "cross-process"
    assert resolution.generation == "child-1"
    result = json.loads(
        terminal_tool("pwd", target="hot-local", task_id="cross-process-e2e")
    )
    assert result["exit_code"] == 0
    assert Path(result["output"].strip()).resolve() == work.resolve()
    assert TERMINAL_SCHEMA == schema_before
    target_schema = TERMINAL_SCHEMA["parameters"]["properties"]["target"]
    assert target_schema["type"] == "string"
    assert "enum" not in target_schema


def test_schema_and_prebuilt_prompt_hint_are_stable_across_registration(
    registry_home,
    tmp_path,
    monkeypatch,
):
    import agent.prompt_builder as prompt_builder
    from tools.terminal_tool import TERMINAL_SCHEMA

    monkeypatch.setattr(prompt_builder, "is_wsl", lambda: False)
    before_schema = deepcopy(TERMINAL_SCHEMA)
    cached_prompt_input = prompt_builder.build_environment_hints()

    write_provider_fragment_for_tests(
        "provider-a",
        {"ephemeral": _record(str(tmp_path / "ephemeral"))},
    )

    assert TERMINAL_SCHEMA == before_schema
    assert "ephemeral" not in cached_prompt_input
    assert "ephemeral" in prompt_builder.build_environment_hints()


def test_drain_and_remove_reject_new_calls(registry_home, tmp_path):
    fragment = {"box": _record(str(tmp_path))}
    write_provider_fragment_for_tests("provider-a", fragment)
    assert resolve_execution_target("box").generation == "g1"

    fragment["box"]["state"] = "draining"
    write_provider_fragment_for_tests("provider-a", fragment)
    with pytest.raises(ExecutionTargetError, match="draining"):
        resolve_execution_target("box")

    write_provider_fragment_for_tests("provider-a", {})
    with pytest.raises(ExecutionTargetError, match="Unknown execution target"):
        resolve_execution_target("box")


def test_generation_repoint_changes_runtime_identity_even_for_same_alias_config(
    registry_home,
    tmp_path,
):
    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="server-1")},
    )
    first = resolve_execution_target("box")
    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="server-2")},
    )
    second = resolve_execution_target("box")

    assert first.config == second.config
    assert first.spec_fingerprint != second.spec_fingerprint
    assert first.security_scope != second.security_scope
    assert first.session_key("session") == second.session_key("session")
    assert first.backend_task_id("session") != second.backend_task_id("session")


def test_nested_dispatch_preserves_complete_runtime_generation_identity(
    registry_home,
    tmp_path,
):
    from tools import approval as approval_mod
    from tools import code_execution_tool as code_mod

    write_provider_fragment_for_tests(
        "Provider-A",
        {"box": _record(str(tmp_path), generation="server-1")},
    )
    outer = resolve_execution_target("box")
    frozen = code_mod._frozen_target_config(outer)

    def resolve_nested(_name, args, task_id=None):
        nested = resolve_execution_target(args["target"])
        approval_key = approval_mod._execution_scoped_pattern_key(
            "command:deploy",
            nested.target,
            nested.named,
            nested.security_scope,
        )
        return nested, approval_key, task_id

    nested, approval_key, nested_task_id = code_mod._dispatch_rpc_tool(
        resolve_nested,
        "terminal",
        {"target": "box"},
        "nested-session",
        frozen,
    )

    assert nested_task_id == "nested-session"
    assert nested.target == outer.target
    assert nested.backend == outer.backend
    assert nested.config == outer.config
    assert nested.provider == outer.provider == "provider-a"
    assert nested.owner_id == outer.owner_id == "box-17"
    assert nested.generation == outer.generation == "server-1"
    assert nested.spec_fingerprint == outer.spec_fingerprint
    assert nested.security_scope == outer.security_scope
    assert nested.environment_key("default") == outer.environment_key("default")
    assert nested.backend_task_id("default") == outer.backend_task_id("default")
    assert nested.file_coordination_key("session") == outer.file_coordination_key(
        "session"
    )
    assert approval_key == approval_mod._execution_scoped_pattern_key(
        "command:deploy",
        outer.target,
        outer.named,
        outer.security_scope,
    )


def test_nested_approval_identity_distinguishes_same_config_generations(
    registry_home,
    tmp_path,
):
    from tools import approval as approval_mod
    from tools import code_execution_tool as code_mod

    def nested_approval_key(frozen):
        def handler(_name, args, task_id=None):
            del task_id
            nested = resolve_execution_target(args["target"])
            return (
                nested.generation,
                approval_mod._execution_scoped_pattern_key(
                    "command:deploy",
                    nested.target,
                    nested.named,
                    nested.security_scope,
                ),
            )

        return code_mod._dispatch_rpc_tool(
            handler,
            "terminal",
            {"target": "box"},
            "nested-session",
            frozen,
        )

    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="g1")},
    )
    first = resolve_execution_target("box")
    first_frozen = code_mod._frozen_target_config(first)
    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="g2")},
    )
    second = resolve_execution_target("box")
    second_frozen = code_mod._frozen_target_config(second)

    first_generation, first_key = nested_approval_key(first_frozen)
    second_generation, second_key = nested_approval_key(second_frozen)

    assert (first_generation, second_generation) == ("g1", "g2")
    assert first_key != second_key


def test_nested_runtime_docker_dispatch_reuses_outer_hosting_environment(
    registry_home,
    tmp_path,
    monkeypatch,
):
    from tools import code_execution_tool as code_mod
    from tools import file_tools as file_mod
    from tools import terminal_tool as terminal_mod

    record = _record(str(tmp_path), generation="docker-g1", backend="docker")
    record["config"]["container_persistent"] = True
    write_provider_fragment_for_tests("provider-a", {"box": record})
    outer = resolve_execution_target("box")

    class HostingEnvironment:
        cwd = "/workspace"

        def __init__(self):
            self.cleanup_calls = 0

        def cleanup(self, force_remove=False):
            del force_remove
            self.cleanup_calls += 1

    hosting = HostingEnvironment()
    terminal_mod._record_environment_target(hosting, outer)
    environment_key = outer.environment_key("default")
    monkeypatch.setattr(
        terminal_mod,
        "_active_environments",
        {environment_key: hosting},
    )
    monkeypatch.setattr(terminal_mod, "_last_activity", {})
    monkeypatch.setattr(terminal_mod, "_creation_locks", {})
    monkeypatch.setattr(file_mod, "_file_ops_cache", {})
    monkeypatch.setattr(
        terminal_mod,
        "_create_environment",
        lambda **_kwargs: pytest.fail("nested dispatch replaced its hosting runtime"),
    )

    def nested_file_ops(_name, args, task_id=None):
        ops = file_mod._get_file_ops(task_id, target=args["target"])
        return ops.env, resolve_execution_target(args["target"])

    nested_env, nested = code_mod._dispatch_rpc_tool(
        nested_file_ops,
        "read_file",
        {"target": "box"},
        "outer-script",
        code_mod._frozen_target_config(outer),
    )

    assert nested_env is hosting
    assert nested.spec_fingerprint == outer.spec_fingerprint
    assert terminal_mod._active_environments[environment_key] is hosting
    assert hosting.cleanup_calls == 0


def test_runtime_generations_isolate_all_file_coordination_state(
    registry_home,
    tmp_path,
    monkeypatch,
):
    from tools import file_state
    from tools import file_tools as file_mod

    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="g1", backend="ssh")},
    )
    first = resolve_execution_target("box")
    first_key = first.file_coordination_key("reader")
    first_writer_key = first.file_coordination_key("writer")
    first_namespace = file_mod._file_state_namespace("reader", "box", _resolution=first)

    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="g2", backend="ssh")},
    )
    second = resolve_execution_target("box")
    second_key = second.file_coordination_key("reader")
    second_namespace = file_mod._file_state_namespace(
        "reader", "box", _resolution=second
    )

    assert first.session_key("reader") == second.session_key("reader")
    assert first.storage_task_id("default") == second.storage_task_id("default")
    assert first_key != second_key
    assert first_namespace != second_namespace

    registry = file_state.FileStateRegistry()
    path = "/srv/project/shared.txt"
    registry.record_read(
        first_key,
        path,
        namespace=first_namespace,
        stat_path=False,
    )
    registry.note_write(
        first_writer_key,
        path,
        namespace=first_namespace,
        stat_path=False,
    )
    assert registry.check_stale(first_key, path, namespace=first_namespace)
    assert registry.check_stale(second_key, path, namespace=second_namespace) is None
    assert registry._lock_for(path, first_namespace) is not registry._lock_for(
        path, second_namespace
    )

    monkeypatch.setattr(file_mod, "_read_tracker", {})
    monkeypatch.setattr(file_mod, "_patch_failure_tracker", {})
    file_mod._record_not_found("read", path, first_key, "g1-miss")
    assert (
        file_mod._check_not_found_cache(
            "read", path, second_key, check_host_filesystem=False
        )
        is None
    )
    first_data = file_mod._read_tracker[first_key]
    first_data["dedup"][(path, 1, 20)] = 7.0
    first_data["read_history"].add((path, 1, 20))
    assert second_key not in file_mod._read_tracker
    assert file_mod._record_patch_failure(first_key, path) == 1
    assert file_mod._record_patch_failure(first_key, path) == 2
    assert file_mod._record_patch_failure(second_key, path) == 1


def test_frozen_dispatch_does_not_pivot_mid_update(registry_home, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(first_dir), generation="g1")},
    )

    with frozen_execution_target_config():
        first = resolve_execution_target("box")
        write_provider_fragment_for_tests(
            "provider-a",
            {"box": _record(str(second_dir), generation="g2")},
        )
        still_first = resolve_execution_target("box")
        assert still_first.generation == first.generation == "g1"
        assert still_first.config["cwd"] == str(first_dir)

    next_dispatch = resolve_execution_target("box")
    assert next_dispatch.generation == "g2"
    assert next_dispatch.config["cwd"] == str(second_dir)


def test_static_and_provider_collisions_fail_closed_per_name(registry_home, tmp_path):
    write_provider_fragment_for_tests(
        "one",
        {
            "local": _record(str(tmp_path / "shadow")),
            "colliding": _record(str(tmp_path / "one")),
            "healthy": _record(str(tmp_path / "healthy")),
        },
    )
    write_provider_fragment_for_tests(
        "two",
        {
            "colliding": _record(str(tmp_path / "two")),
        },
    )

    assert resolve_execution_target("local").provider is None
    assert resolve_execution_target("healthy").provider == "one"
    with pytest.raises(ExecutionTargetError, match="multiple runtime providers"):
        resolve_execution_target("colliding")


def test_malformed_unreadable_and_symlinked_fragments_are_isolated(
    registry_home,
    tmp_path,
):
    directory = registry_directory()
    write_provider_fragment_for_tests(
        "healthy",
        {"ok": _record(str(tmp_path / "ok"))},
    )
    malformed = directory / "broken.json"
    malformed.write_text("{not json", encoding="utf-8")
    malformed.chmod(0o600)
    unreadable = directory / "unreadable.json"
    unreadable.write_text("{}", encoding="utf-8")
    unreadable.chmod(0o000)
    symlink = directory / "linked.json"
    symlink.symlink_to(directory / "healthy.json")
    invalidate_runtime_registry_cache()

    assert resolve_execution_target("ok").provider == "healthy"
    diagnostics = load_runtime_registry().diagnostics
    providers = {item.provider for item in diagnostics}
    assert {"broken", "unreadable", "linked"} <= providers


def test_oversized_and_overcomplex_providers_are_isolated_secret_safely(
    registry_home,
    tmp_path,
):
    directory = registry_directory()
    write_provider_fragment_for_tests(
        "healthy",
        {"ok": _record(str(tmp_path / "healthy"))},
    )

    secret = "provider-token-must-not-appear"
    oversized = directory / "oversized.json"
    oversized.write_bytes(secret.encode("utf-8") + b"x" * (MAX_REGISTRY_FILE_BYTES + 1))
    oversized.chmod(0o600)

    too_many = directory / "too-many.json"
    too_many.write_text(
        json.dumps({
            "version": 1,
            "provider": "too-many",
            "targets": {
                f"target-{index}": _record(str(tmp_path))
                for index in range(MAX_TARGETS_PER_PROVIDER + 1)
            },
        }),
        encoding="utf-8",
    )
    too_many.chmod(0o600)

    nested: object = "leaf"
    for _ in range(20):
        nested = {"child": nested}
    too_deep = directory / "too-deep.json"
    too_deep.write_text(
        json.dumps({
            "version": 1,
            "provider": "too-deep",
            "targets": {
                "deep": {
                    **_record(str(tmp_path)),
                    "config": {"backend": "local", "payload": nested},
                },
            },
        }),
        encoding="utf-8",
    )
    too_deep.chmod(0o600)

    too_large_config = directory / "too-large-config.json"
    too_large_config.write_text(
        json.dumps({
            "version": 1,
            "provider": "too-large-config",
            "targets": {
                "large": {
                    **_record(str(tmp_path)),
                    "config": {
                        "backend": "local",
                        "payload": "x" * (MAX_TARGET_CONFIG_BYTES + 1),
                    },
                },
            },
        }),
        encoding="utf-8",
    )
    too_large_config.chmod(0o600)
    invalidate_runtime_registry_cache()

    assert resolve_execution_target("local").provider is None
    assert resolve_execution_target("ok").provider == "healthy"
    snapshot = load_runtime_registry()
    providers = {item.provider for item in snapshot.diagnostics}
    assert {"oversized", "too-many", "too-deep", "too-large-config"} <= providers
    assert secret not in json.dumps([item.as_dict() for item in snapshot.diagnostics])


def test_production_update_rejects_oversized_fragment_without_replacing_active_set(
    registry_home,
    tmp_path,
):
    existing = update_provider_fragment(
        "provider-a",
        lambda _snapshot, current: {
            **current,
            "existing": _record(str(tmp_path), generation="still-active"),
        },
    )
    assert {item.execution_target for item in existing.records} == {"existing"}
    assert resolve_execution_target("existing").generation == "still-active"

    path = registry_directory() / "provider-a.json"
    published = path.read_bytes()
    directory_entries = {item.name for item in path.parent.iterdir()}
    secret = "provider-fragment-secret-must-not-appear"
    chunk = secret + "x" * (MAX_TARGET_CONFIG_BYTES - 16 * 1024)
    oversized_targets = {
        f"large-{index}": {
            "config": {
                "backend": "docker",
                "cwd": "/workspace",
                "docker_env": {"PADDING": chunk},
            },
            "owner_id": f"owner-{index}",
            "generation": "oversized-candidate",
            "state": "ready",
        }
        for index in range(5)
    }

    with pytest.raises(RuntimeRegistryError) as error:
        update_provider_fragment(
            "provider-a",
            lambda _snapshot, _current: oversized_targets,
        )

    assert "fragment size limit" in str(error.value)
    assert secret not in str(error.value)
    assert path.read_bytes() == published
    assert {item.name for item in path.parent.iterdir()} == directory_entries
    visible = load_runtime_registry()
    assert {
        (item.execution_target, item.generation)
        for item in visible.records
        if item.provider == "provider-a"
    } == {("existing", "still-active")}
    assert resolve_execution_target("existing").generation == "still-active"


def test_returned_snapshot_nested_mutation_cannot_poison_registry_cache(
    registry_home,
    tmp_path,
):
    victim = _record(str(tmp_path), backend="docker")
    victim["config"].update({
        "docker_env": {"DESTINATION": "trusted"},
        "docker_volumes": ["trusted:/workspace/trusted"],
    })
    write_provider_fragment_for_tests("provider-a", {"victim": victim})

    returned = load_runtime_registry()
    returned_config = returned.records[0].config
    returned_config["docker_env"]["DESTINATION"] = "poisoned"
    returned_config["docker_volumes"].append("poisoned:/workspace/poisoned")

    cached = load_runtime_registry()
    forced = load_runtime_registry(force=True)
    for snapshot in (cached, forced):
        config = snapshot.records[0].config
        assert config["docker_env"] == {"DESTINATION": "trusted"}
        assert config["docker_volumes"] == ["trusted:/workspace/trusted"]


def test_failing_cross_provider_mutator_cannot_poison_cache_or_disk(
    registry_home,
    tmp_path,
):
    victim = _record(str(tmp_path), backend="docker")
    victim["config"].update({
        "docker_env": {"DESTINATION": "trusted"},
        "docker_volumes": ["trusted:/workspace/trusted"],
    })
    write_provider_fragment_for_tests("victim-provider", {"victim": victim})
    victim_path = registry_directory() / "victim-provider.json"
    disk_before = victim_path.read_bytes()

    def poison_then_fail(snapshot, current):
        del current
        record = next(
            item for item in snapshot.records if item.provider == "victim-provider"
        )
        record.config["docker_env"]["DESTINATION"] = "poisoned"
        record.config["docker_volumes"].append("poisoned:/workspace/poisoned")
        raise RuntimeError("provider callback failed")

    with pytest.raises(RuntimeError, match="provider callback failed"):
        update_provider_fragment("attacker", poison_then_fail)

    assert victim_path.read_bytes() == disk_before
    for snapshot in (load_runtime_registry(), load_runtime_registry(force=True)):
        record = next(
            item for item in snapshot.records if item.provider == "victim-provider"
        )
        assert record.config["docker_env"] == {"DESTINATION": "trusted"}
        assert record.config["docker_volumes"] == ["trusted:/workspace/trusted"]


def test_provider_identity_is_lowercase_in_writer_payload_and_parser(
    registry_home,
    tmp_path,
):
    assert validate_provider_name("Foo.Bar") == "foo.bar"
    path = write_provider_fragment_for_tests(
        "Foo.Bar",
        {"box": _record(str(tmp_path))},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot = load_runtime_registry(force=True)

    assert path.name == "foo.bar.json"
    assert payload["provider"] == "foo.bar"
    assert snapshot.records[0].provider == "foo.bar"


def test_noncanonical_casefold_alias_fails_safe_without_overwrite(
    registry_home,
    tmp_path,
):
    directory = registry_directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    alias = directory / "Foo.json"
    alias.write_text(
        json.dumps({
            "version": 1,
            "provider": "Foo",
            "targets": {"first": _record(str(tmp_path / "first"))},
        }),
        encoding="utf-8",
    )
    alias.chmod(0o600)
    original = alias.read_bytes()
    original_entries = {
        entry.name for entry in directory.iterdir() if entry.suffix == ".json"
    }
    invalidate_runtime_registry_cache()

    snapshot = load_runtime_registry(force=True)
    assert snapshot.records == ()
    assert any(diagnostic.provider == "foo" for diagnostic in snapshot.diagnostics)
    with pytest.raises(RuntimeRegistryError, match="Provider 'foo' is unavailable"):
        update_provider_fragment(
            "foo",
            lambda _snapshot, current: {
                **current,
                "second": _record(str(tmp_path / "second")),
            },
        )

    assert alias.read_bytes() == original
    assert {
        entry.name for entry in directory.iterdir() if entry.suffix == ".json"
    } == original_entries


def test_rapid_same_size_atomic_replacement_is_observed(registry_home, tmp_path):
    first = _record(str(tmp_path / "one"), generation="11")
    second = _record(str(tmp_path / "two"), generation="22")
    write_provider_fragment_for_tests("provider-a", {"box": first})
    path = registry_directory() / "provider-a.json"
    before = path.stat()
    assert resolve_execution_target("box").generation == "11"

    write_provider_fragment_for_tests("provider-a", {"box": second})
    after = path.stat()
    assert before.st_size == after.st_size
    assert before.st_ino != after.st_ino
    assert resolve_execution_target("box").generation == "22"


def test_profile_registry_separation(registry_home, tmp_path):
    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path), generation="profile-a")},
    )
    other = tmp_path / "other-profile"
    token = set_hermes_home_override(other)
    try:
        invalidate_runtime_registry_cache()
        with pytest.raises(ExecutionTargetError):
            resolve_execution_target("box")
        write_provider_fragment_for_tests(
            "provider-a",
            {"other": _record(str(tmp_path), generation="profile-b")},
        )
        assert resolve_execution_target("other").generation == "profile-b"
    finally:
        reset_hermes_home_override(token)
        invalidate_runtime_registry_cache()
    assert resolve_execution_target("box").generation == "profile-a"
    with pytest.raises(ExecutionTargetError):
        resolve_execution_target("other")


def test_legacy_activation_state_is_valid_if_fragment_publication_crashes(
    monkeypatch,
    tmp_path,
):
    import tools.execution_target_registry as registry

    home = tmp_path / "legacy-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    set_execution_target_config_source({
        "terminal": {
            "backend": "ssh",
            "ssh_host": "project-fallback-host",
            "ssh_user": "static-user",
        },
    })
    original = registry._atomic_write_json
    calls = []

    def fail_after_activation(path, payload):
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("simulated provider publication crash")
        return original(path, payload)

    monkeypatch.setattr(registry, "_atomic_write_json", fail_after_activation)
    try:
        with pytest.raises(OSError, match="publication crash"):
            update_provider_fragment(
                "provider-a",
                lambda _snapshot, current: {
                    **current,
                    "box": _record(str(tmp_path)),
                },
                activate_legacy=True,
            )
        invalidate_runtime_registry_cache()
        default = resolve_execution_target()
        assert default.named is True
        assert default.target == "default"
        assert default.backend == "ssh"
        assert default.config["ssh_host"] == "project-fallback-host"
        assert calls[0] == ".registry-state.json"
        state = json.loads(
            (registry_directory() / ".registry-state.json").read_text(encoding="utf-8")
        )
        assert state == {"legacy_activated": True, "version": 1}
    finally:
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()


def test_multiplex_activation_uses_selected_profile_not_ambient_environment(
    monkeypatch,
    tmp_path,
):
    from agent.secret_scope import set_multiplex_active
    from tools.execution_target_overlay import overlay_runtime_execution_targets

    home = tmp_path / "selected-profile"
    home.mkdir()
    selected_cwd = tmp_path / "selected-workspace"
    selected_cwd.mkdir()
    ambient_cwd = tmp_path / "wrong-ambient-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "wrong-profile-host")
    monkeypatch.setenv("TERMINAL_SSH_USER", "wrong-profile-user")
    monkeypatch.setenv("TERMINAL_CWD", str(ambient_cwd))
    invalidate_runtime_registry_cache()
    set_multiplex_active(True)
    try:
        snapshot = update_provider_fragment(
            "profile-controller",
            lambda _snapshot, current: current,
            activate_legacy=True,
        )
        selected = {
            "terminal": {
                "backend": "local",
                "cwd": str(selected_cwd),
                "timeout": 91,
            },
        }
        merged = overlay_runtime_execution_targets(selected, snapshot=snapshot)
        resolution = resolve_execution_target(config=merged)
        assert resolution.named is True
        assert resolution.backend == "local"
        assert resolution.config["cwd"] == str(selected_cwd)
        state_text = (registry_directory() / ".registry-state.json").read_text(
            encoding="utf-8"
        )
        assert json.loads(state_text) == {"legacy_activated": True, "version": 1}
        assert "wrong-profile" not in state_text
    finally:
        set_multiplex_active(False)
        invalidate_runtime_registry_cache()


def test_legacy_activation_preserves_env_only_ssh_and_explicit_raw_key_wins(
    monkeypatch,
    tmp_path,
):
    import tools.terminal_tool as terminal_mod
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.execution_target_overlay import overlay_runtime_execution_targets

    home = tmp_path / "legacy-env-home"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for env_name in set(TERMINAL_CONFIG_ENV_MAP.values()) | {
        "TERMINAL_LOCAL_PERSISTENT",
        "TERMINAL_SSH_PERSISTENT",
    }:
        monkeypatch.delenv(env_name, raising=False)
    expected = {
        "backend": "ssh",
        "ssh_host": "env-only.example.test",
        "ssh_user": "env-user",
        "ssh_port": 2207,
        "ssh_key": "/keys/env-only-ed25519",
        "cwd": "/srv/env-only-workspace",
        "timeout": 347,
    }
    monkeypatch.setenv("TERMINAL_ENV", expected["backend"])
    monkeypatch.setenv("TERMINAL_SSH_HOST", expected["ssh_host"])
    monkeypatch.setenv("TERMINAL_SSH_USER", expected["ssh_user"])
    monkeypatch.setenv("TERMINAL_SSH_PORT", str(expected["ssh_port"]))
    monkeypatch.setenv("TERMINAL_SSH_KEY", expected["ssh_key"])
    monkeypatch.setenv("TERMINAL_CWD", expected["cwd"])
    monkeypatch.setenv("TERMINAL_TIMEOUT", str(expected["timeout"]))
    monkeypatch.setattr(terminal_mod, "_terminal_config_bridge_attempted", False)

    merged_defaults = deepcopy(DEFAULT_CONFIG)
    merged_defaults["terminal"].update({
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": 22,
        "ssh_key": "",
    })
    multiplex_before = is_multiplex_active()
    set_multiplex_active(False)
    set_execution_target_config_source(merged_defaults)
    invalidate_runtime_registry_cache()
    try:
        snapshot = update_provider_fragment(
            "profile-controller",
            lambda _snapshot, current: current,
            activate_legacy=True,
        )
        activated = overlay_runtime_execution_targets(
            merged_defaults,
            snapshot=snapshot,
        )
        resolution = resolve_execution_target(config=activated)
        assert resolution.named is True
        assert {key: resolution.config[key] for key in expected} == expected

        # The bridge is one-shot, so reset it while changing the raw authority
        # and environment in-process.  monkeypatch restores the original module
        # state after the test, preventing order-dependent coverage.
        (home / "config.yaml").write_text(
            "terminal:\n  ssh_port: 2022\n",
            encoding="utf-8",
        )
        explicit = deepcopy(merged_defaults)
        explicit["terminal"]["ssh_port"] = 2022
        set_execution_target_config_source(explicit)
        monkeypatch.setenv("TERMINAL_SSH_PORT", "2299")
        monkeypatch.setattr(terminal_mod, "_terminal_config_bridge_attempted", False)

        refreshed = overlay_runtime_execution_targets(explicit, snapshot=snapshot)
        explicit_resolution = resolve_execution_target(config=refreshed)
        assert explicit_resolution.config["ssh_port"] == 2022
        assert os.environ["TERMINAL_SSH_PORT"] == "2022"
    finally:
        set_execution_target_config_source(None)
        set_multiplex_active(multiplex_before)
        invalidate_runtime_registry_cache()


def test_private_key_contents_and_traversal_provider_are_rejected(
    registry_home,
    tmp_path,
):
    with pytest.raises(RuntimeRegistryError):
        write_provider_fragment_for_tests("../escape", {})
    with pytest.raises(RuntimeRegistryError, match="not private-key contents"):
        write_provider_fragment_for_tests(
            "provider-a",
            {
                "box": {
                    **_record(str(tmp_path), backend="ssh"),
                    "config": {
                        "backend": "ssh",
                        "ssh_host": "alias",
                        "ssh_user": "dev",
                        "ssh_key": "-----BEGIN PRIVATE KEY-----\nsecret",
                    },
                },
            },
        )


def test_malformed_replacement_drops_provider_instead_of_using_stale_snapshot(
    registry_home,
    tmp_path,
):
    write_provider_fragment_for_tests(
        "provider-a",
        {"box": _record(str(tmp_path))},
    )
    assert resolve_execution_target("box").provider == "provider-a"
    path = registry_directory() / "provider-a.json"
    replacement = path.with_name(".provider-a.replacement")
    replacement.write_text("{malformed", encoding="utf-8")
    replacement.chmod(0o600)
    os.replace(replacement, path)

    with pytest.raises(ExecutionTargetError):
        resolve_execution_target("box")
    assert resolve_execution_target("local").provider is None
    assert any(
        item.provider == "provider-a" for item in load_runtime_registry().diagnostics
    )


def test_invalid_legacy_activation_state_cannot_break_flat_static_default(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "invalid-activation"
    directory = home / "runtime" / "execution-targets.d"
    directory.mkdir(parents=True, mode=0o700)
    state = directory / ".registry-state.json"
    state.write_text(
        json.dumps({
            "version": 1,
            "legacy_default": {"backend": "telepathy"},
        }),
        encoding="utf-8",
    )
    state.chmod(0o600)
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_execution_target_config_source({"terminal": {"backend": "local"}})
    invalidate_runtime_registry_cache()
    try:
        resolution = resolve_execution_target()
        assert resolution.backend == "local"
        assert resolution.named is False
    finally:
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()
