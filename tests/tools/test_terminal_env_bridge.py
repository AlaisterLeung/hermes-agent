"""Behavioral regressions for the terminal config → env bridge.

``terminal_tool._get_env_config()`` reads TERMINAL_* variables.  The bridge
must let explicitly configured terminal keys override stale launcher/.env
values while preserving environment values for terminal keys omitted from
config.yaml.
"""

import os

import pytest

import tools.terminal_tool as terminal_tool
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _reset_bridge_state(monkeypatch):
    """Each test starts with an un-attempted bridge and clean mapped env."""
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
    for name in (
        "TERMINAL_ENV",
        "TERMINAL_CWD",
        "TERMINAL_DOCKER_IMAGE",
        "TERMINAL_TIMEOUT",
        "TERMINAL_SSH_HOST",
        "TERMINAL_SSH_USER",
    ):
        monkeypatch.delenv(name, raising=False)
    # The config layer caches by (path, mtime, size); each test writes its own
    # config.yaml and therefore changes the signature.
    yield


def _write_config(text: str) -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text)


def test_unset_terminal_env_backfills_backend_from_config():
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: custom/image:1\n"
    )

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "custom/image:1"
    assert os.environ["TERMINAL_ENV"] == "docker"


def test_explicit_config_backend_overrides_stale_env(monkeypatch):
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert os.environ["TERMINAL_ENV"] == "docker"


def test_partial_terminal_config_preserves_unrelated_env_values(monkeypatch):
    _write_config("terminal:\n  backend: docker\n")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "env/image:2")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "env/image:2"
    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "env/image:2"


def test_bridge_mirrors_named_default_target_for_legacy_consumers():
    _write_config(
        "terminal:\n"
        "  backend: local\n"
        "  timeout: 77\n"
        "  default_target: devbox\n"
        "  targets:\n"
        "    local:\n"
        "      backend: local\n"
        "      cwd: /workspace/local\n"
        "    devbox:\n"
        "      backend: ssh\n"
        "      cwd: /srv/project\n"
        "      ssh_host: devbox.example.com\n"
        "      ssh_user: agent\n"
    )

    from hermes_cli.config import apply_terminal_config_to_env, load_config_readonly

    mirrored = apply_terminal_config_to_env(
        env={}, config=load_config_readonly(), override=True,
    )
    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["cwd"] == "/srv/project"
    assert config["timeout"] == 77
    assert config["ssh_host"] == "devbox.example.com"
    assert mirrored["TERMINAL_ENV"] == "ssh"
    assert mirrored["TERMINAL_CWD"] == "/srv/project"
    assert mirrored["TERMINAL_TIMEOUT"] == "77"
    assert mirrored["TERMINAL_SSH_HOST"] == "devbox.example.com"


def test_bridge_preserves_remote_tilde_for_named_ssh_default():
    from hermes_cli.config import apply_terminal_config_to_env

    mirrored = apply_terminal_config_to_env(
        env={},
        config={
            "terminal": {
                "default_target": "devbox",
                "targets": {
                    "devbox": {
                        "backend": "ssh",
                        "cwd": "~/project",
                        "ssh_host": "devbox.example.com",
                        "ssh_user": "agent",
                    },
                },
            },
        },
        override=True,
    )

    assert mirrored["TERMINAL_ENV"] == "ssh"
    assert mirrored["TERMINAL_CWD"] == "~/project"


def test_invalid_default_target_type_stays_fail_open():
    from hermes_cli.config import effective_terminal_config

    effective = effective_terminal_config({
        "backend": "local",
        "default_target": ["not", "hashable"],
        "targets": {"local": {"backend": "local"}},
    })
    assert effective == {"backend": "local"}


def test_explicit_config_key_overrides_matching_env_value(monkeypatch):
    _write_config(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: config/image:1\n"
    )
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "env/image:2")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "config/image:1"


def test_ssh_config_preserves_remote_tilde_cwd(monkeypatch):
    """SSH ``~`` belongs to the remote user, not the Hermes host/container."""
    _write_config("terminal:\n  backend: ssh\n  cwd: '~'\n")
    monkeypatch.setenv("HOME", "/opt/data/home")
    monkeypatch.setenv("USERPROFILE", r"C:\opt\data\home")

    config = terminal_tool._get_env_config()

    assert os.environ["TERMINAL_CWD"] == "~"
    assert config["cwd"] == "~"


def test_env_is_preserved_when_config_has_no_terminal_section(monkeypatch):
    _write_config("agent:\n  max_turns: 100\n")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == "example.test"


def test_defaults_backfill_when_neither_config_nor_env_selects_backend():
    _write_config("{}\n")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "local"


def test_bridge_only_attempted_once(monkeypatch):
    calls = []

    import hermes_cli.config as config_mod

    real = config_mod.apply_terminal_config_to_env

    def _counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", _counting)
    _write_config("{}\n")

    terminal_tool._get_env_config()
    terminal_tool._get_env_config()

    assert len(calls) == 1


def test_bridge_config_failure_does_not_crash(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: (_ for _ in ()).throw(RuntimeError("config read failed")),
    )
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "example.test")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["ssh_host"] == "example.test"


def test_secondary_home_override_does_not_latch_ambient_env(tmp_path, monkeypatch):
    """#107422: first bridge under a secondary profile must not poison os.environ.

    Multiplexed dashboard sets ``set_hermes_home_override`` for profile B. If
    ``_ensure_terminal_env_bridged`` ran there (no terminal scope yet), the
    one-shot latch used to write B's docker policy into process-global env and
    every later unscoped launch-profile tool call inherited it.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch_home = tmp_path / "launch"
    secondary_home = tmp_path / "profiles" / "docker-bee"
    launch_home.mkdir(parents=True)
    secondary_home.mkdir(parents=True)
    (launch_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )
    (secondary_home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: docker\n"
        "  docker_image: bee/local:1\n"
        '  docker_volumes:\n'
        '    - /bee/vol:/data\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    # Clean ambient — the dashboard process starts without TERMINAL_ENV.
    for name in (
        "TERMINAL_ENV",
        "TERMINAL_DOCKER_IMAGE",
        "TERMINAL_DOCKER_VOLUMES",
    ):
        monkeypatch.delenv(name, raising=False)

    token = set_hermes_home_override(str(secondary_home))
    try:
        # Unscoped call under secondary home (the residual path).
        terminal_tool._ensure_terminal_env_bridged()
    finally:
        reset_hermes_home_override(token)

    assert "TERMINAL_ENV" not in os.environ
    assert "TERMINAL_DOCKER_IMAGE" not in os.environ
    assert "TERMINAL_DOCKER_VOLUMES" not in os.environ
    # Bridge must still be available for the real launch profile afterwards.
    assert terminal_tool._terminal_config_bridge_attempted is False

    config = terminal_tool._get_env_config()
    assert config["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "local"
    assert "bee/local:1" not in os.environ.get("TERMINAL_DOCKER_IMAGE", "")
