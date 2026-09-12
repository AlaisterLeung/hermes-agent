from __future__ import annotations

import json

from tools.execution_target_registry import (
    invalidate_runtime_registry_cache,
    write_provider_fragment_for_tests,
)
from tools.execution_targets import (
    set_execution_target_config_source,
)


def test_background_handle_survives_target_drain_and_remove(monkeypatch, tmp_path):
    """Process handles retain their immutable producing runtime after teardown."""
    from tools.process_registry import process_registry
    from tools.terminal_tool import terminal_tool

    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_execution_target_config_source({
        "terminal": {
            "default_target": "local",
            "targets": {"local": {"backend": "local", "cwd": str(tmp_path)}},
        },
    })
    record = {
        "config": {"backend": "local", "cwd": str(work)},
        "owner_id": "box",
        "generation": "server-1",
        "state": "ready",
    }
    write_provider_fragment_for_tests("controller", {"box": record})
    session_id = ""
    try:
        started = json.loads(
            terminal_tool(
                "sleep 30",
                target="box",
                task_id="runtime-background",
                background=True,
            )
        )
        session_id = started["session_id"]
        producing_scope = started["runtime_scope"]

        record["state"] = "draining"
        write_provider_fragment_for_tests("controller", {"box": record})
        drained_poll = process_registry.poll(session_id)
        assert drained_poll["runtime_scope"] == producing_scope

        write_provider_fragment_for_tests("controller", {})
        removed_poll = process_registry.poll(session_id)
        assert removed_poll["runtime_scope"] == producing_scope
        killed = process_registry.kill_process(session_id)
        assert killed["runtime_scope"] == producing_scope
        assert process_registry.poll(session_id)["runtime_scope"] == producing_scope
    finally:
        if session_id:
            process_registry.kill_process(session_id)
        set_execution_target_config_source(None)
        invalidate_runtime_registry_cache()
