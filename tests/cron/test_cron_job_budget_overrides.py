"""Per-job budget overrides: ``max_turns`` / ``run_budget_seconds`` (store + scheduler resolution).

A cron job may carry its own tool-calling iteration cap and wall-clock run budget; an absent key
follows the global ``agent.max_turns`` / ``agent.run_budget_seconds`` resolution at fire time. The
store validates strictly (typos must not silently mean "unlimited"); the scheduler resolves
tolerantly so hand-edited jobs can't crash a tick.
"""

import sys

import pytest

from cron.jobs import create_job, update_job
from cron.scheduler import _resolve_job_max_turns, _resolve_job_run_budget


def _make_job(**kw):
    return create_job(prompt="x", schedule="every 5m", name="t", **kw)


class TestStore:
    def test_create_persists_explicit_values(self):
        job = _make_job(max_turns=400, run_budget_seconds=900)
        assert job["max_turns"] == 400
        assert job["run_budget_seconds"] == 900.0

    def test_create_omits_unset_keys(self):
        job = _make_job()
        assert "max_turns" not in job
        assert "run_budget_seconds" not in job

    def test_create_coerces_numeric_strings_and_spellings(self):
        assert _make_job(max_turns="350")["max_turns"] == 350
        assert _make_job(max_turns="inf")["max_turns"] == "unlimited"
        assert _make_job(max_turns=0)["max_turns"] == "unlimited"
        assert _make_job(run_budget_seconds="120")["run_budget_seconds"] == 120.0

    def test_create_rejects_unparseable_max_turns(self):
        with pytest.raises(ValueError):
            _make_job(max_turns="banana")
        with pytest.raises(ValueError):
            _make_job(max_turns=True)

    def test_create_rejects_non_positive_run_budget(self):
        for bad in (0, -5, "banana", True):
            with pytest.raises(ValueError):
                _make_job(run_budget_seconds=bad)

    def test_update_sets_and_clears(self):
        job = _make_job()
        updated = update_job(job["id"], {"max_turns": 250, "run_budget_seconds": 60})
        assert updated is not None
        assert updated["max_turns"] == 250
        assert updated["run_budget_seconds"] == 60.0
        cleared = update_job(job["id"], {"max_turns": "", "run_budget_seconds": ""})
        assert cleared is not None
        assert not cleared.get("max_turns")
        assert not cleared.get("run_budget_seconds")

    def test_update_rejects_invalid(self):
        job = _make_job()
        with pytest.raises(ValueError):
            update_job(job["id"], {"max_turns": "nonsense"})
        with pytest.raises(ValueError):
            update_job(job["id"], {"run_budget_seconds": -1})


class TestResolution:
    def test_job_overrides_config(self):
        cfg = {"agent": {"max_turns": 150, "run_budget_seconds": 60}}
        job = {"max_turns": 400, "run_budget_seconds": 900}
        assert _resolve_job_max_turns(job, cfg) == 400
        assert _resolve_job_run_budget(job, cfg) == 900.0

    def test_falls_back_to_agent_section(self):
        cfg = {"agent": {"max_turns": 150, "run_budget_seconds": 60}}
        assert _resolve_job_max_turns({}, cfg) == 150
        assert _resolve_job_run_budget({}, cfg) == 60.0

    def test_legacy_top_level_max_turns_and_defaults(self):
        assert _resolve_job_max_turns({}, {"max_turns": 99}) == 99
        assert _resolve_job_max_turns({}, {}) == sys.maxsize
        assert _resolve_job_run_budget({}, {}) is None

    def test_unlimited_spelling(self):
        cfg = {"agent": {"max_turns": 150}}
        assert _resolve_job_max_turns({"max_turns": "unlimited"}, cfg) == sys.maxsize

    def test_hand_edited_job_tolerates_junk(self):
        # Hand-edited jobs.json must not crash the tick: junk max_turns resolves like config
        # (unlimited); junk run_budget turns the feature off for that job.
        assert _resolve_job_max_turns({"max_turns": "banana"}, {}) == sys.maxsize
        assert _resolve_job_run_budget({"run_budget_seconds": "banana"}, {}) is None

    def test_created_job_record_resolves(self):
        job = _make_job(max_turns=420, run_budget_seconds=600)
        assert _resolve_job_max_turns(job, {"agent": {"max_turns": 150}}) == 420
        assert _resolve_job_run_budget(job, {"agent": {"run_budget_seconds": 30}}) == 600.0
