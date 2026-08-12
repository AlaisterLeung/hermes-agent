from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli.targets import register_cli
from tools.execution_target_registry import (
    invalidate_runtime_registry_cache,
    load_runtime_registry,
    registry_directory,
)
from tools.execution_targets import (
    resolve_execution_target,
    set_execution_target_config_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes targets")
    register_cli(parser)
    return parser


def _run(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    return int(args.func(args) or 0)


def _run_subprocess(
    home: Path,
    argv: list[str],
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_REDACT_SECRETS"] = "true"
    env["TERMINAL_ENV"] = "local"
    env.pop("HERMES_PROFILE", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "targets", *argv],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


@pytest.fixture
def cli_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_execution_target_config_source({
        "terminal": {
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": str(tmp_path)},
            },
        },
    })
    invalidate_runtime_registry_cache()
    try:
        yield home
    finally:
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()


def test_parser_exposes_required_actions_and_aliases(capsys):
    parser = _parser()
    help_text = parser.format_help()
    assert "register" in help_text
    assert "list" in help_text
    assert "show" in help_text
    assert "drain" in help_text
    assert "remove" in help_text

    remove = parser.parse_args(["unregister", "box"])
    assert remove.func.__name__ == "_cmd_remove"
    register = parser.parse_args([
        "register",
        "box",
        "--backend",
        "ssh",
        "--ssh-host",
        "alias",
        "--ssh-user",
        "dev",
    ])
    assert register.host == "alias"
    assert register.user == "dev"
    capsys.readouterr()


def test_register_list_show_and_secret_safe_json(cli_home, capsys):
    secret_path = "/keys/private-build-17"
    rc = _run([
        "register",
        "hetzner-build-17",
        "--backend",
        "ssh",
        "--host",
        "hetzner-dev-build-17",
        "--user",
        "dev",
        "--port",
        "443",
        "--key",
        secret_path,
        "--cwd",
        "/workspace",
        "--provider",
        "hetzner-devbox",
        "--owner-id",
        "build-17",
        "--generation",
        "server-123",
        "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_target"] == "hetzner-build-17"
    assert payload["provider"] == "hetzner-devbox"
    assert payload["owner_id"] == "build-17"
    assert payload["generation"] == "server-123"
    assert payload["state"] == "ready"
    assert payload["routing"]["ssh_host"] == "hetzner-dev-build-17"
    assert secret_path not in json.dumps(payload)

    assert _run(["list", "--all", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["version"] == 1
    assert any(
        row["execution_target"] == "hetzner-build-17" for row in listing["targets"]
    )
    assert (
        _run([
            "show",
            "hetzner-build-17",
            "--provider",
            "hetzner-devbox",
            "--json",
        ])
        == 0
    )
    shown = json.loads(capsys.readouterr().out)
    assert shown["routing"]["ssh_port"] == 443
    assert secret_path not in json.dumps(shown)


def test_provider_aliases_canonicalize_to_one_cli_owner(cli_home, capsys):
    for name, provider in (("first", "Foo"), ("second", "foo")):
        assert (
            _run([
                "register",
                name,
                "--backend",
                "local",
                "--provider",
                provider,
                "--generation",
                f"{name}-g1",
                "--json",
            ])
            == 0
        )
        assert json.loads(capsys.readouterr().out)["provider"] == "foo"

    fragment = registry_directory() / "foo.json"
    payload = json.loads(fragment.read_text(encoding="utf-8"))
    assert payload["provider"] == "foo"
    assert set(payload["targets"]) == {"first", "second"}
    assert not (registry_directory() / "Foo.json").exists() or (
        registry_directory() / "Foo.json"
    ).samefile(fragment)

    assert _run(["show", "second", "--provider", "FOO", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["provider"] == "foo"


def test_register_and_replace_publish_reloadable_0600_files_under_strict_umask(
    monkeypatch,
    tmp_path,
    capsys,
):
    home = tmp_path / "strict-umask-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    set_execution_target_config_source({"terminal": {"backend": "local"}})
    invalidate_runtime_registry_cache()
    original_umask = os.umask(0o777)
    try:
        first = [
            "register",
            "box",
            "--backend",
            "local",
            "--provider",
            "Controller",
            "--generation",
            "g1",
            "--json",
        ]
        assert _run(first) == 0
        assert json.loads(capsys.readouterr().out)["provider"] == "controller"

        replacement = [
            "register",
            "box",
            "--backend",
            "local",
            "--provider",
            "controller",
            "--generation",
            "g2",
            "--replace",
            "--if-generation",
            "g1",
            "--json",
        ]
        assert _run(replacement) == 0
        assert json.loads(capsys.readouterr().out)["generation"] == "g2"

        state_path = registry_directory() / ".registry-state.json"
        provider_path = registry_directory() / "controller.json"
        assert state_path.stat().st_mode & 0o777 == 0o600
        assert provider_path.stat().st_mode & 0o777 == 0o600
        reloaded = load_runtime_registry(force=True)
        record = next(
            item for item in reloaded.records if item.provider == "controller"
        )
        assert record.generation == "g2"
    finally:
        os.umask(original_umask)
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()


def test_post_publication_reload_failure_returns_stable_json_error_under_umask(
    monkeypatch,
    tmp_path,
    capsys,
):
    import tools.execution_target_registry as registry_mod

    home = tmp_path / "reload-failure-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_execution_target_config_source({
        "terminal": {
            "default_target": "local",
            "targets": {"local": {"backend": "local", "cwd": str(tmp_path)}},
        },
    })
    invalidate_runtime_registry_cache()
    original_load = registry_mod.load_runtime_registry

    def unavailable_after_publication(*, force=False):
        snapshot = original_load(force=force)
        if force and (registry_directory() / "controller.json").exists():
            return registry_mod.RuntimeRegistrySnapshot()
        return snapshot

    monkeypatch.setattr(
        registry_mod,
        "load_runtime_registry",
        unavailable_after_publication,
    )
    original_umask = os.umask(0o777)
    try:
        assert (
            _run([
                "register",
                "box",
                "--backend",
                "local",
                "--provider",
                "controller",
                "--generation",
                "g1",
                "--json",
            ])
            == 2
        )
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["status"] == "error"
        assert payload["message"] == ("Runtime execution target 'box' was not found.")
        assert captured.err == ""
        assert "StopIteration" not in captured.out
        provider_path = registry_directory() / "controller.json"
        assert provider_path.stat().st_mode & 0o777 == 0o600
        assert original_load(force=True).records[0].generation == "g1"
    finally:
        os.umask(original_umask)
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()


def test_idempotency_replace_and_generation_cas(cli_home, capsys):
    base = [
        "register",
        "box",
        "--backend",
        "local",
        "--cwd",
        "/one",
        "--provider",
        "controller",
        "--generation",
        "g1",
        "--json",
    ]
    assert _run(base) == 0
    capsys.readouterr()
    assert _run(base) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "unchanged"

    replacement = [
        "register",
        "box",
        "--backend",
        "local",
        "--cwd",
        "/two",
        "--provider",
        "controller",
        "--generation",
        "g2",
        "--json",
    ]
    assert _run(replacement) != 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "--replace" in json.loads(captured.out)["message"]
    assert (
        _run([
            *replacement,
            "--replace",
            "--if-generation",
            "stale",
        ])
        != 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Stale generation" in json.loads(captured.out)["message"]
    assert (
        _run([
            *replacement,
            "--replace",
            "--if-generation",
            "g1",
        ])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["action"] == "replaced"
    assert resolve_execution_target("box").generation == "g2"

    assert (
        _run([
            "drain",
            "box",
            "--provider",
            "controller",
            "--if-generation",
            "g1",
        ])
        != 0
    )
    capsys.readouterr()
    assert (
        _run([
            "drain",
            "box",
            "--provider",
            "controller",
            "--if-generation",
            "g2",
            "--json",
        ])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["state"] == "draining"
    assert (
        _run([
            "remove",
            "box",
            "--provider",
            "controller",
            "--if-generation",
            "g1",
        ])
        != 0
    )
    capsys.readouterr()
    assert (
        _run([
            "remove",
            "box",
            "--provider",
            "controller",
            "--if-generation",
            "g2",
            "--json",
        ])
        == 0
    )


def test_validation_collision_and_structural_injection_fail_nonzero(
    cli_home,
    capsys,
):
    assert (
        _run([
            "register",
            "local",
            "--backend",
            "local",
        ])
        != 0
    )
    assert "reserved by static" in capsys.readouterr().err
    assert (
        _run([
            "register",
            "bad",
            "--backend",
            "ssh",
            "--host",
            "alias",
        ])
        != 0
    )
    assert "requires non-empty 'ssh_user'" in capsys.readouterr().err
    assert (
        _run([
            "register",
            "bad",
            "--backend",
            "local",
            "--set",
            "default_target=oops",
        ])
        != 0
    )
    assert "structural field" in capsys.readouterr().err
    assert (
        _run([
            "register",
            "bad",
            "--backend",
            "ssh",
            "--host",
            "alias",
            "--user",
            "dev",
            "--key",
            "-----BEGIN PRIVATE KEY----- secret",
        ])
        != 0
    )
    assert "not private-key contents" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["register", "bad", "--backend", "ssh", "--host", "alias", "--json"],
        ["show", "missing", "--json"],
        ["drain", "missing", "--json"],
        ["remove", "missing", "--json"],
        ["unregister", "missing", "--json"],
    ],
)
def test_json_handled_errors_use_stdout_for_every_target_action(
    cli_home,
    capsys,
    argv,
):
    assert _run(argv) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert payload["message"]
    assert captured.err == ""
    assert "Error:" not in captured.out


def test_list_json_handled_error_uses_shared_contract(
    cli_home,
    capsys,
    monkeypatch,
):
    from hermes_cli import targets as targets_cli

    def fail_collect(*, include_all):
        del include_all
        raise RuntimeError("programming errors must escape")

    monkeypatch.setattr(targets_cli, "_collect_rows", fail_collect)
    with pytest.raises(RuntimeError, match="programming errors must escape"):
        _run(["list", "--json"])
    assert capsys.readouterr().out == ""

    def fail_domain(*, include_all):
        del include_all
        from tools.execution_target_registry import RuntimeRegistryError

        raise RuntimeRegistryError("registry validation failed")

    monkeypatch.setattr(targets_cli, "_collect_rows", fail_domain)
    assert _run(["list", "--json"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "error",
        "message": "registry validation failed",
    }
    assert captured.err == ""


def test_subprocess_stale_remove_json_contract_preserves_replacement(tmp_path):
    home = tmp_path / "subprocess-home"
    home.mkdir()
    target_cwd = tmp_path / "runtime-target"
    target_cwd.mkdir()
    identity = [
        "box",
        "--backend",
        "local",
        "--cwd",
        str(target_cwd),
        "--provider",
        "controller",
    ]

    registered = _run_subprocess(
        home,
        ["register", *identity, "--generation", "g1", "--json"],
    )
    assert registered.returncode == 0, registered.stderr
    assert json.loads(registered.stdout)["generation"] == "g1"

    replaced = _run_subprocess(
        home,
        [
            "register",
            *identity,
            "--generation",
            "g2",
            "--replace",
            "--if-generation",
            "g1",
            "--json",
        ],
    )
    assert replaced.returncode == 0, replaced.stderr
    assert json.loads(replaced.stdout)["generation"] == "g2"

    stale = _run_subprocess(
        home,
        [
            "remove",
            "box",
            "--provider",
            "controller",
            "--if-generation",
            "stale",
            "--json",
        ],
    )
    assert stale.returncode == 2
    assert stale.stderr == ""
    error = json.loads(stale.stdout)
    assert error["status"] == "error"
    assert "Stale generation" in error["message"]
    assert "Error:" not in stale.stdout + stale.stderr

    shown = _run_subprocess(
        home,
        ["show", "box", "--provider", "controller", "--json"],
    )
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout)["generation"] == "g2"


def test_subprocess_register_validation_error_is_secret_safe_json(tmp_path):
    home = tmp_path / "subprocess-validation-home"
    home.mkdir()
    private_key = "-----BEGIN PRIVATE KEY----- key-material-must-not-appear"
    password = "registration-password-must-not-appear"
    docker_token = "registration-docker-token-must-not-appear"

    invalid = _run_subprocess(
        home,
        [
            "register",
            "bad",
            "--backend",
            "ssh",
            "--host",
            "alias",
            "--user",
            "dev",
            "--key",
            private_key,
            "--set",
            f"sudo_password={password}",
            "--set",
            f'docker_env={{"PRIVATE_TOKEN":"{docker_token}"}}',
            "--json",
        ],
    )

    assert invalid.returncode == 2
    assert invalid.stderr == ""
    payload = json.loads(invalid.stdout)
    assert payload["status"] == "error"
    assert "does not apply to backend 'ssh'" in payload["message"]
    assert "Error:" not in invalid.stdout + invalid.stderr
    for secret in (private_key, password, docker_token):
        assert secret not in invalid.stdout + invalid.stderr


def test_legacy_activation_does_not_edit_yaml_and_survives_last_remove(
    monkeypatch,
    tmp_path,
    capsys,
):
    home = tmp_path / "legacy"
    home.mkdir()
    config_path = home / "config.yaml"
    original = "terminal:\n  backend: local\n  timeout: 77\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "77")
    password = "activation-password-must-not-persist"
    token = "activation-token-must-not-persist"
    key_path = "/keys/activation-private-key"
    sensitive_cwd = "/workspace/activation-sensitive-cwd"
    set_execution_target_config_source({
        "terminal": {
            "backend": "local",
            "timeout": 77,
            "cwd": sensitive_cwd,
            "sudo_password": password,
            "ssh_key": key_path,
            "docker_env": {"PRIVATE_TOKEN": token},
            "env_passthrough": ["FIRST_SHARED_POLICY"],
            "shell_init_files": ["/etc/hermes/first-shell-policy"],
        },
    })
    invalidate_runtime_registry_cache()
    try:
        assert (
            _run([
                "register",
                "box",
                "--backend",
                "local",
                "--provider",
                "controller",
                "--generation",
                "g1",
            ])
            == 0
        )
        capsys.readouterr()
        assert config_path.read_text(encoding="utf-8") == original
        activated = resolve_execution_target()
        assert activated.named is True
        assert activated.target == "default"
        assert activated.config["timeout"] == 77
        assert activated.config["env_passthrough"] == ["FIRST_SHARED_POLICY"]
        assert activated.config["shell_init_files"] == [
            "/etc/hermes/first-shell-policy"
        ]
        state = registry_directory() / ".registry-state.json"
        state_text = state.read_text(encoding="utf-8")
        assert json.loads(state_text) == {"legacy_activated": True, "version": 1}
        for forbidden in (
            password,
            token,
            key_path,
            sensitive_cwd,
            "backend",
            "docker_env",
        ):
            assert forbidden not in state_text

        set_execution_target_config_source({
            "terminal": {
                "backend": "local",
                "timeout": 88,
                "cwd": sensitive_cwd,
                "sudo_password": password,
                "ssh_key": key_path,
                "docker_env": {"PRIVATE_TOKEN": token},
                "env_passthrough": ["UPDATED_SHARED_POLICY"],
                "shell_init_files": ["/etc/hermes/updated-shell-policy"],
            },
        })
        refreshed = resolve_execution_target()
        assert refreshed.named is True
        assert refreshed.config["timeout"] == 88
        assert refreshed.config["env_passthrough"] == ["UPDATED_SHARED_POLICY"]
        assert refreshed.config["shell_init_files"] == [
            "/etc/hermes/updated-shell-policy"
        ]
        assert refreshed.spec_fingerprint != activated.spec_fingerprint
        assert (
            _run([
                "remove",
                "box",
                "--provider",
                "controller",
                "--if-generation",
                "g1",
            ])
            == 0
        )
        capsys.readouterr()
        still_stable = resolve_execution_target()
        assert still_stable.named is True
        assert still_stable.target == "default"
        assert still_stable.spec_fingerprint == refreshed.spec_fingerprint
        assert state.exists()
        assert state.stat().st_mode & 0o777 == 0o600
        assert registry_directory().stat().st_mode & 0o777 == 0o700
    finally:
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()
