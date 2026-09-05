"""Tests for ``hermes acp --target`` named execution target selection.

The flag pins the ACP server's process-wide execution target (terminal, file,
and code-exec dispatch) by overriding ``terminal.default_target`` via
``set_execution_target_config_source`` — the same authority hook the classic
CLI uses. Behavior contracts verified here:

* Unknown/unavailable targets fail closed at launch (exit 2) before the
  JSON-RPC loop starts, with the available-targets message.
* An explicit target overrides ``terminal.default_target`` for default
  (``None``) resolution in this process.
* Omitting the flag leaves resolution byte-for-byte unchanged (legacy local
  path / configured default, no synthetic targets introduced).
"""

import sys
from types import SimpleNamespace

import pytest

import acp_adapter.entry as acp_entry
from tools.execution_targets import (
    ExecutionTargetError,
    resolve_execution_target,
    set_execution_target_config_source,
)


@pytest.fixture(autouse=True)
def _clean_config_source():
    """Ensure each test starts and ends with no registered config source."""
    set_execution_target_config_source(None)
    yield
    set_execution_target_config_source(None)


@pytest.fixture
def acp_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME with a named-target terminal config."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = {
        "terminal": {
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": str(tmp_path)},
                "devbox": {
                    "backend": "ssh",
                    "ssh_host": "203.0.113.10",
                    "ssh_port": 22,
                    "ssh_user": "alaister",
                },
            },
        },
    }
    import yaml

    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))

    # Config caches are keyed by config path, and each test gets a fresh
    # HERMES_HOME/tmp path — but clear defensively in case of key reuse.
    import hermes_cli.config as config_module

    config_module._RAW_CONFIG_CACHE.clear()
    config_module._LOAD_CONFIG_CACHE.clear()

    yield hermes_home


def _write_config(hermes_home, config):
    import yaml

    (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))


# ---------------------------------------------------------------------------
# Argument parsing / forwarding
# ---------------------------------------------------------------------------


def test_parser_accepts_target_long_and_short():
    for argv, expected in (
        (["--target", "devbox"], "devbox"),
        (["-t", "staging"], "staging"),
    ):
        args = acp_entry._parse_args(argv)
        assert args.acp_target == expected


def test_parser_target_defaults_to_none():
    args = acp_entry._parse_args([])
    assert args.acp_target is None


def test_cmd_acp_forwards_target(monkeypatch):
    forwarded = {}

    def fake_acp_main(argv):
        forwarded["argv"] = list(argv)

    import hermes_cli.main as main_module

    # cmd_acp imports acp_adapter.entry.main as acp_main inside the call, so
    # patch the entry module instead.
    monkeypatch.setattr(acp_entry, "main", fake_acp_main)

    args = SimpleNamespace(
        acp_version=False,
        check=False,
        setup=False,
        setup_browser=False,
        assume_yes=False,
        acp_target="devbox",
    )
    main_module.cmd_acp(args)
    assert forwarded["argv"] == ["--target", "devbox"]


def test_cmd_acp_omits_flag_when_unset(monkeypatch):
    forwarded = {}

    def fake_acp_main(argv):
        forwarded["argv"] = list(argv)

    import hermes_cli.main as main_module

    monkeypatch.setattr(acp_entry, "main", fake_acp_main)

    args = SimpleNamespace(
        acp_version=False,
        check=False,
        setup=False,
        setup_browser=False,
        assume_yes=False,
        acp_target=None,
    )
    main_module.cmd_acp(args)
    assert forwarded["argv"] == []


# ---------------------------------------------------------------------------
# _apply_execution_target
# ---------------------------------------------------------------------------


def test_apply_execution_target_noop_when_unset(acp_home):
    # Must not raise and must not register any config source.
    acp_entry._apply_execution_target(None)
    acp_entry._apply_execution_target("")
    # Default resolution stays whatever the config says (local).
    assert resolve_execution_target().target == "local"


def test_apply_execution_target_overrides_default(acp_home):
    acp_entry._apply_execution_target("devbox")

    resolution = resolve_execution_target()
    assert resolution.target == "devbox"
    assert resolution.backend == "ssh"


def test_apply_execution_target_unknown_target_exits(acp_home, capsys):
    with pytest.raises(SystemExit) as excinfo:
        acp_entry._apply_execution_target("does-not-exist")

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "does-not-exist" in err
    assert "Available targets" in err


def test_apply_execution_target_legacy_mode_exits(acp_home):
    # No terminal.targets at all -> named targets unsupported -> fail closed.
    _write_config(
        acp_home,
        {"terminal": {"backend": "local", "cwd": str(acp_home)}},
    )

    with pytest.raises(SystemExit) as excinfo:
        acp_entry._apply_execution_target("devbox")

    assert excinfo.value.code == 2


def test_apply_execution_target_error_message_lists_targets(acp_home, capsys):
    with pytest.raises(SystemExit):
        acp_entry._apply_execution_target("nope")
    err = capsys.readouterr().err
    assert "'devbox'" in err
    assert "'local'" in err


# ---------------------------------------------------------------------------
# main() wiring
# ---------------------------------------------------------------------------


def test_main_applies_target_before_server_start(acp_home, monkeypatch):
    """main() with -t must pin resolution before HermesACPAgent is built."""
    started = {}

    class FakeAgent:
        def __init__(self):
            started["resolution"] = resolve_execution_target().target

    import acp as acp_module
    import acp_adapter.server as server_module

    async def fake_run_agent(agent, use_unstable_protocol=False):
        started["loop"] = True

    monkeypatch.setattr(server_module, "HermesACPAgent", FakeAgent)
    monkeypatch.setattr(acp_module, "run_agent", fake_run_agent)

    # Provide the pieces main() touches after target application but before
    # the loop: skip MCP discovery and keep the loop a no-op.
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    # Logging setup touches agent.redact; allow it but keep output quiet.
    monkeypatch.setattr(acp_entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(acp_entry, "_load_env", lambda: None)

    acp_entry.main(["--target", "devbox"])

    assert started["resolution"] == "devbox"


def test_main_without_target_keeps_local(acp_home, monkeypatch):
    started = {}

    class FakeAgent:
        def __init__(self):
            started["resolution"] = resolve_execution_target().target

    import acp as acp_module
    import acp_adapter.server as server_module

    async def fake_run_agent(agent, use_unstable_protocol=False):
        started["loop"] = True

    monkeypatch.setattr(server_module, "HermesACPAgent", FakeAgent)
    monkeypatch.setattr(acp_module, "run_agent", fake_run_agent)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(acp_entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(acp_entry, "_load_env", lambda: None)

    acp_entry.main([])

    assert started["resolution"] == "local"


def test_main_unknown_target_exits_before_agent(acp_home, monkeypatch, capsys):
    built = []

    class FakeAgent:
        def __init__(self):
            built.append(True)

    import acp_adapter.server as server_module

    monkeypatch.setattr(server_module, "HermesACPAgent", FakeAgent)
    monkeypatch.setattr(acp_entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(acp_entry, "_load_env", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        acp_entry.main(["-t", "ghost"])

    assert excinfo.value.code == 2
    assert built == []  # server never constructed
    assert "ghost" in capsys.readouterr().err
