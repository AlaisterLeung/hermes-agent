"""Cron digest enrichment for continuable deliveries.

Contract under test: when mirroring is enabled (per-job ``attach_to_session``
or the global ``cron.mirror_delivery`` flag), the seed text the scheduler
appends to the continuation session carries a bounded digest of the run's own
cron session transcript — not just the final response. When mirroring is off,
nothing changes: the plain final-response text flows exactly as before. The
digest is pure enrichment — every failure path falls back to the plain text
and never gates delivery.
"""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from cron.digest import (
    _MAX_DIGEST_CHARS,
    _MAX_DIGEST_EVENTS,
    build_cron_digest,
)


# ---------------------------------------------------------------------------
# Delivery-path wiring: digest flows into the seed only when mirroring is on
# ---------------------------------------------------------------------------

class TestDeliveryPathWiring:
    """Drive _deliver_result end-to-end with a stubbed sender + mirror
    recorder (same pattern as test_mirror_origin_fallback.py): the seed text
    the scheduler appends must carry the digest when mirroring is enabled,
    and the plain final response when it is not."""

    @pytest.fixture()
    def slack_env(self, monkeypatch, tmp_path):
        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        send_calls = []

        async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                              media_files=None, force_document=False,
                              caption=None):
            send_calls.append({"chat_id": chat_id, "thread_id": thread_id})
            return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

        import gateway.platform_registry as reg
        import hermes_cli.plugins as hp

        entry = reg.platform_registry.get("slack")
        if entry is None:
            hp.discover_plugins()
            entry = reg.platform_registry.get("slack")
        if entry is None:
            pytest.skip("slack platform entry not registered")
        monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
        monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

        mirror_calls = []

        def fake_mirror(platform, chat_id, text, source_label="cli",
                        thread_id=None, user_id=None, role="assistant",
                        session_id=None):
            mirror_calls.append({
                "platform": platform, "chat_id": chat_id,
                "thread_id": thread_id, "user_id": user_id, "role": role,
                "text": text,
            })
            return True

        import gateway.mirror as mirror_mod

        monkeypatch.setattr(mirror_mod, "mirror_to_session", fake_mirror)
        return {"send": send_calls, "mirror": mirror_calls}

    def _seed_cron_transcript(self, home, session_id):
        """A real cron session transcript in the test home's state.db."""
        db = home / "state.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE messages ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT, role TEXT, content TEXT,"
            " tool_name TEXT, tool_calls TEXT, active INTEGER DEFAULT 1)"
        )
        tc = json.dumps([{"id": "c1", "type": "function",
                          "function": {"name": "terminal",
                                       "arguments": '{"command": "df -h"}'}}])
        conn.execute(
            "INSERT INTO messages (session_id, role, tool_calls) VALUES"
            " (?, 'assistant', ?)", (session_id, tc))
        conn.commit()
        conn.close()

    def test_mirror_on_seeds_digest(self, slack_env, monkeypatch):
        """attach_to_session=true + a readable cron transcript: the mirrored
        seed text carries the digest header, the tool-call timeline, the
        final response, and the original-session pointer."""
        monkeypatch.setenv(
            "HERMES_SESSION_ID", "cron_jw1_20260901_120000")
        self._seed_cron_transcript(
            Path(os.environ["HERMES_HOME"]), "cron_jw1_20260901_120000")
        job = {
            "id": "jw1", "name": "wired", "deliver": "slack:D0USER1",
            "origin": None, "attach_to_session": True,
        }
        from cron.scheduler import _deliver_result

        err = _deliver_result(job, "the final response",
                              adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 1
        text = slack_env["mirror"][0]["text"]
        assert "[Cron execution digest: wired]" in text
        assert "tool terminal(" in text
        assert "df -h" in text
        assert "the final response" in text
        assert "cron_jw1_20260901_120000" in text

    def test_mirror_off_keeps_plain_text(self, slack_env, monkeypatch):
        """No attach/mirror opt-in: no mirror fires at all, and the digest
        machinery is never invoked (plain text flows unchanged)."""
        monkeypatch.setenv(
            "HERMES_SESSION_ID", "cron_jw2_20260901_120000")
        self._seed_cron_transcript(
            Path(os.environ["HERMES_HOME"]), "cron_jw2_20260901_120000")
        job = {
            "id": "jw2", "name": "unwired", "deliver": "slack:D0USER2",
            "origin": None,
        }
        from cron.scheduler import _deliver_result

        err = _deliver_result(job, "plain final response",
                              adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 0

    def test_global_mirror_flag_also_enriches(self, slack_env, monkeypatch,
                                              tmp_path):
        """cron.mirror_delivery: true (global) must activate the digest the
        same way a per-job attach_to_session does."""
        home = tmp_path / "hermes-home"
        cfg = home / "config.yaml"
        cfg.write_text(cfg.read_text() + "cron:\n  mirror_delivery: true\n")
        monkeypatch.setenv(
            "HERMES_SESSION_ID", "cron_jw3_20260901_120000")
        self._seed_cron_transcript(home, "cron_jw3_20260901_120000")
        job = {
            "id": "jw3", "name": "global-wired", "deliver": "origin",
            "origin": {"platform": "slack", "chat_id": "D0HOME",
                       "chat_type": "dm"},
        }
        monkeypatch.setenv("SLACK_HOME_CHANNEL", "D0HOME")
        from cron.scheduler import _deliver_result

        err = _deliver_result(job, "global final", adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 1
        text = slack_env["mirror"][0]["text"]
        assert "[Cron execution digest: global-wired]" in text
        assert "global final" in text

    def test_digest_failure_falls_back_to_plain_text(self, slack_env,
                                                     monkeypatch):
        """An unreadable transcript must not gate or corrupt the delivery:
        the seed falls back to the plain final response."""
        monkeypatch.setenv(
            "HERMES_SESSION_ID", "cron_jw4_20260901_120000")
        # No state.db at all in this home — the digest read raises.
        job = {
            "id": "jw4", "name": "fallback", "deliver": "slack:D0USER3",
            "origin": None, "attach_to_session": True,
        }
        from cron.scheduler import _deliver_result

        err = _deliver_result(job, "plain final response",
                              adapters=None, loop=None)
        assert err is None
        assert len(slack_env["mirror"]) == 1
        text = slack_env["mirror"][0]["text"]
        assert "plain final response" in text
        assert "Cron execution digest" not in text



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cron_session_db(tmp_path, monkeypatch):
    """A real $HERMES_HOME/state.db with one cron session's transcript.

    Message columns mirror the production schema subset the digest reads
    (role, content, tool_name, tool_calls) with ``active = 1`` rows.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT, role TEXT, content TEXT,"
        " tool_name TEXT, tool_calls TEXT, active INTEGER DEFAULT 1)"
    )

    def _insert(session_id, role, content=None, tool_name=None, tool_calls=None,
                active=1):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name,"
            " tool_calls, active) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, tool_name, tool_calls, active),
        )
        # Commit so the digest's SEPARATE read-only connection sees the rows
        # (SQLite default isolation keeps uncommitted writes invisible to
        # other connections).
        conn.commit()

    yield _insert, conn
    conn.close()


def _tc(name, args):
    """Production tool_calls wire shape."""
    return json.dumps([
        {"id": "call_x", "type": "function",
         "function": {"name": name, "arguments": args}},
    ])


def _seed_transcript(_insert, session_id, final_response="All done."):
    _insert(session_id, "user", content="Daily briefing prompt")
    _insert(session_id, "assistant", tool_calls=_tc("terminal", '{"command": "uname -a"}'))
    _insert(session_id, "tool", content='{"output": "Linux devbox 6.12"}',
            tool_name="terminal")
    _insert(session_id, "assistant", content="Gathering data…")
    _insert(session_id, "assistant", tool_calls=_tc("web_extract", '{"urls": ["https://example.com"]}'))
    _insert(session_id, "tool", content='{"status": "ok"}', tool_name="web_extract")
    _insert(session_id, "assistant", content=final_response)


# ---------------------------------------------------------------------------
# Digest content
# ---------------------------------------------------------------------------

def test_digest_includes_steps_calls_and_final(cron_session_db):
    _insert, _ = cron_session_db
    _seed_transcript(_insert, "cron_j1_20260901_120000")

    digest = build_cron_digest(
        {"name": "Briefing"}, "cron_j1_20260901_120000", "All done."
    )
    assert digest is not None
    # The final response is embedded verbatim.
    assert "All done." in digest
    # Tool calls appear with name and truncated args.
    assert "tool terminal(" in digest
    assert "uname -a" in digest
    assert "tool web_extract(" in digest
    # Assistant text turns appear as steps.
    assert "step: Gathering data…" in digest
    # Original session pointer for full-transcript recall.
    assert "cron_j1_20260901_120000" in digest


def test_digest_wraps_final_response_with_header(cron_session_db):
    _insert, _ = cron_session_db
    _seed_transcript(_insert, "cron_j2_20260901_120000")

    digest = build_cron_digest({"name": "Watch"}, "cron_j2_20260901_120000",
                               "the final answer")
    assert digest.startswith("<!-- cron-digest session=")
    assert "[Cron execution digest: Watch]" in digest
    assert digest.rstrip().endswith("the final answer") or "the final answer" in digest


def test_trailing_assistant_text_not_duplicated(cron_session_db):
    _insert, _ = cron_session_db
    _seed_transcript(_insert, "cron_j3_20260901_120000")

    digest = build_cron_digest({"name": "J"}, "cron_j3_20260901_120000",
                               "All done.")
    # The trailing assistant turn IS the final response — exactly one copy.
    assert digest.count("All done.") == 1


def test_soft_deleted_rows_excluded(cron_session_db):
    _insert, _ = cron_session_db
    _seed_transcript(_insert, "cron_j4_20260901_120000")
    _insert("cron_j4_20260901_120000", "assistant",
            content="SECRET soft-deleted step", active=0)

    digest = build_cron_digest({"name": "J"}, "cron_j4_20260901_120000",
                               "All done.")
    assert "SECRET" not in digest


# ---------------------------------------------------------------------------
# Fallbacks — every failure path returns None / never gates
# ---------------------------------------------------------------------------

def test_no_session_id_returns_none():
    assert build_cron_digest({"name": "J"}, None, "text") is None


def test_missing_db_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no state.db at all
    assert build_cron_digest({"name": "J"}, "cron_x", "text") is None


def test_unknown_session_returns_none(cron_session_db):
    _insert, _ = cron_session_db
    _seed_transcript(_insert, "cron_real")
    assert build_cron_digest({"name": "J"}, "cron_missing", "text") is None


def test_corrupt_db_returns_none(cron_session_db, tmp_path, monkeypatch):
    _insert, conn = cron_session_db
    _seed_transcript(_insert, "cron_j")
    conn.close()
    (tmp_path / "state.db").write_bytes(b"not a database at all")
    assert build_cron_digest({"name": "J"}, "cron_j", "text") is None


def test_final_response_only_run_returns_none(cron_session_db):
    """A run with no tool activity gains nothing from a digest — the plain
    seed already carries everything; return None so no duplication lands."""
    _insert, _ = cron_session_db
    _insert("cron_j5", "user", content="prompt")
    _insert("cron_j5", "assistant", content="All done.")
    assert build_cron_digest({"name": "J"}, "cron_j5", "All done.") is None


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def test_event_cap_enforced(cron_session_db):
    _insert, _ = cron_session_db
    sid = "cron_j6"
    for i in range(_MAX_DIGEST_EVENTS + 10):
        _insert(sid, "assistant", tool_calls=_tc(f"tool_{i}", "{}"))
    digest = build_cron_digest({"name": "J"}, sid, "done")
    assert digest is not None
    assert "[… older activity truncated …]" in digest
    # Oldest-first: the earliest calls survive the cap.
    assert "tool_0" in digest


def test_char_cap_enforced(cron_session_db):
    _insert, _ = cron_session_db
    sid = "cron_j7"
    for i in range(200):
        _insert(sid, "assistant", tool_calls=_tc("big", '{"x": "' + "y" * 300 + '"}'))
    digest = build_cron_digest({"name": "J"}, sid, "done")
    assert digest is not None
    assert len(digest) <= _MAX_DIGEST_CHARS + 40  # cap + truncation marker


def test_tool_result_truncation(cron_session_db):
    _insert, _ = cron_session_db
    sid = "cron_j8"
    _insert(sid, "assistant", tool_calls=_tc("web_extract", "{}"))
    _insert(sid, "tool", content="x" * 5000, tool_name="web_extract")
    digest = build_cron_digest({"name": "J"}, sid, "done")
    assert "x" * 500 not in digest  # heavily truncated
    assert "web_extract →" in digest


def test_ack_results_skipped(cron_session_db):
    _insert, _ = cron_session_db
    sid = "cron_j9"
    _insert(sid, "assistant", tool_calls=_tc("hindsight_retain", "{}"))
    _insert(sid, "tool", content='{"result": "Memory stored successfully."}',
            tool_name="hindsight_retain")
    digest = build_cron_digest({"name": "J"}, sid, "done")
    # The ack result line is noise — only the call line remains.
    assert "hindsight_retain →" not in digest
    assert "tool hindsight_retain(" in digest
