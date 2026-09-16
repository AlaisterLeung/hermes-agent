"""Per-fire ``extra_prompt`` must survive the managed-topology worker handoff.

``run_one_job`` receives per-fire context (webhook ``cron_job`` routes, manual
``cronjob(action="run", prompt=...)`` fires) as an in-process argument, but in a managed
systemd gateway the job is serialized — job dict only — to an external worker process
(``_launch_external_cron_worker``). An argument alone never crosses that process boundary,
so without the stamp the run context is silently dropped. The fix stamps it onto the job
dict via the same ``manual_run_prompt`` rail ``trigger_job`` uses for forwarded runs.
"""

import json
import subprocess

import pytest

from cron import scheduler


@pytest.fixture()
def handoff_capture(monkeypatch):
    """Replace the real launcher with a capture stub: return True = handed off."""
    captured = {}

    def fake_launch(job: dict) -> bool:
        captured.update(job)
        return True

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", fake_launch)
    return captured


def test_handoff_carries_extra_prompt(handoff_capture):
    ok = scheduler.run_one_job(
        {"id": "j1", "name": "p", "schedule": "*/5 * * * *"},
        extra_prompt="focus on EU numbers",
    )
    assert ok is True
    assert handoff_capture["manual_run_prompt"] == "focus on EU numbers"
    assert handoff_capture.get("manual_run_at")


def test_handoff_without_extra_prompt_leaves_dict_clean(handoff_capture):
    ok = scheduler.run_one_job({"id": "j1", "name": "p", "schedule": "*/5 * * * *"})
    assert ok is True
    assert "manual_run_prompt" not in handoff_capture
    assert "manual_run_at" not in handoff_capture


def test_existing_stamp_not_overwritten(monkeypatch):
    """A stamp already on the job (trigger_job forwarded-run rail) keeps its original time."""
    captured = {}

    def fake_launch(job: dict) -> bool:
        captured.update(job)
        return True

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", fake_launch)
    job = {"id": "j1", "name": "p", "schedule": "*/5 * * * *",
           "manual_run_prompt": "orphaned", "manual_run_at": "2026-01-01T00:00:00+00:00"}
    ok = scheduler.run_one_job(job, extra_prompt="fresh focus")
    assert ok is True
    assert captured["manual_run_prompt"] == "fresh focus"
    assert captured["manual_run_at"] == "2026-01-01T00:00:00+00:00"


def test_payload_json_carries_the_stamp_end_to_end(tmp_path, monkeypatch):
    """The REAL handoff serializes the job dict to the worker payload file; the stamped
    context must be inside that JSON — this is the exact byte boundary the hotfix guards."""
    import json as _json

    from tools.process_registry import GatewayChildDispatch

    job = {"id": "job-ctx", "execution_id": "exec-1", "prompt": "work"}
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv",
        lambda command, **_kw: GatewayChildDispatch("scoped", ["scope", "--", *command]),
    )
    monkeypatch.setattr(
        scheduler, "mark_execution_handoff_pending",
        lambda _eid: {"id": "exec-1", "handoff_pending": 1},
    )
    payload_path = tmp_path / "cron" / "external-workers" / "exec-1.json"
    captured_payload: dict = {}

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout: float = 0.0):
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)

    def popen(command, **_kwargs):
        # Snapshot the payload bytes at spawn time — the real launcher deletes the
        # payload file once the execution status turns terminal.
        captured_payload.update(
            json.loads(payload_path.read_text(encoding="utf-8")))
        return FakeProcess()

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    monkeypatch.setattr(
        scheduler, "get_execution",
        lambda _eid: {"id": "exec-1", "status": "completed"})

    ok = scheduler.run_one_job(job, extra_prompt="focus on EU numbers")
    assert ok is True
    assert captured_payload["job"]["manual_run_prompt"] == "focus on EU numbers"
    assert captured_payload["job"]["manual_run_at"]
