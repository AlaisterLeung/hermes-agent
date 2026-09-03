"""Regression tests for #100489 — desktop multiplex ticker must not deliver a
secondary profile's cron output through the default profile's identity.

Two halves:

1. ``_deliver_result``'s standalone fallback pool (taken when the caller has a
   RUNNING event loop — the desktop dashboard shape) spawns a fresh thread that
   did not inherit the profile ContextVars; it must run inside a copy of the
   active context so the sender reads THIS profile's home + secrets.
2. The desktop ticker must stand down, per tick, for a profile whose OWN
   gateway is running — that gateway ticks it with live adapters, and racing it
   on the tick lock lets the adapter-less desktop ticker deliver standalone.
"""
import asyncio
import threading
from pathlib import Path
from unittest.mock import patch



def test_standalone_fallback_pool_keeps_profile_scope(tmp_path, monkeypatch):
    from agent.secret_scope import (
        get_secret,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import get_hermes_home, set_hermes_home_override
    import cron.scheduler as sched
    import tools.send_message_tool as smt

    default_home = tmp_path / "default"
    sec_home = tmp_path / "profiles" / "ops"
    for home in (default_home, sec_home):
        (home / "cron").mkdir(parents=True)
        (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "DEFAULT-TOKEN")
    set_multiplex_active(True)

    seen = {}

    async def fake_send(platform, pconfig, chat_id, message, **kwargs):
        seen["home"] = str(get_hermes_home())
        seen["token"] = get_secret("TELEGRAM_BOT_TOKEN", None)
        return {"success": True, "message_id": "1"}

    job = {"id": "j1", "name": "probe", "deliver": "telegram:12345", "schedule": {"kind": "cron"}}

    async def _inside_running_loop():
        # Emulate the multiplex ticker's per-profile scope on the caller.
        set_hermes_home_override(str(sec_home))
        set_secret_scope({"TELEGRAM_BOT_TOKEN": "OPS-TOKEN"})
        return sched._deliver_result(job, "hello", adapters={}, loop=None)

    try:
        with patch.object(smt, "_send_to_platform", fake_send):
            err = asyncio.run(_inside_running_loop())
    finally:
        set_multiplex_active(False)

    assert err is None, err
    assert seen["home"] == str(sec_home.resolve())
    assert seen["token"] == "OPS-TOKEN"


def test_multiplex_ticker_profile_gate_skips_rejected_profile(tmp_path):
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    own_gateway = tmp_path / "own-gateway"
    orphan = tmp_path / "orphan"
    for home in (own_gateway, orphan):
        (home / "cron").mkdir(parents=True)

    stop = threading.Event()
    ticked: list[str] = []

    def _tick(*args, **kwargs):
        ticked.append(str(get_hermes_home()))
        if len(ticked) >= 3:
            stop.set()
        return 0

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_tick):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("own-gateway", own_gateway), ("orphan", orphan)],
                "profile_gate": lambda name, home: name != "own-gateway",
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert set(ticked) == {str(orphan)}
    # The gated profile gets no tick-loop success marker either: its own
    # gateway owns that status surface.
    assert not (own_gateway / "cron" / "ticker_last_success").exists()
    assert (orphan / "cron" / "ticker_last_success").exists()


def test_desktop_ticker_gates_on_profile_gateway_running(monkeypatch):
    """The desktop ticker's per-profile stand-down contract.

    Fork note (b6b03bb643, #7): the desktop ticker is a
    ``GatewayForwardingCronScheduler`` and no longer resolves a provider via
    ``resolve_cron_scheduler`` — the upstream doubles here (resolve_cron_
    scheduler / InProcessCronScheduler) never engage, and the real forwarding
    scheduler blocks in its sweep loop until the per-file watchdog kills the
    subprocess (300s, 3 tests lost). The same reason
    tests/hermes_cli/conftest.py skips test_desktop_cron_ticker_profiles.py.
    Spy the GatewayForwardingCronScheduler class instead so start() records
    its kwargs and returns immediately.

    Contract under the forwarding ticker: profile stand-down is NOT wired on
    the desktop — due fires are forwarded to the owning gateway's
    /api/cron/fire (which claims via store CAS, so a racing gateway tick
    cannot double-run the job). Assert the forwarding sweep contract instead
    of the retired profile_gate kwarg.
    """
    import cron.scheduler_provider as sp
    from hermes_cli import web_server

    captured = {}

    class _SpyForwarding(sp.GatewayForwardingCronScheduler):
        def start(self, stop_event, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        sp, "GatewayForwardingCronScheduler", _SpyForwarding
    )
    monkeypatch.setattr("hermes_logging.enable_profile_log_routing", lambda homes: None)

    web_server._start_desktop_cron_ticker(threading.Event(), interval=7)

    assert captured.get("interval") == 7, (
        "the desktop ticker must pass the sweep interval through to the "
        "forwarding scheduler"
    )
    assert captured.get("profile_homes") is None, (
        "the adapter-less desktop ticker must not tick profile stores locally "
        "— due fires are forwarded to the gateways' /api/cron/fire, which "
        "claims them via store CAS (no profile_gate needed)"
    )
