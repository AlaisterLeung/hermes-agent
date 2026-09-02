"""Digest enrichment for continuable cron deliveries.

The shipped cron continuation surface (``attach_to_session`` per job, the
global ``cron.mirror_delivery`` flag, and the ``in_channel`` continuable
surface) seeds the target session with ONLY the job's final response. A user
who replies in-thread gets a continuation session whose entire knowledge of
the run is that final text — every tool call, intermediate result, and
assistant step that produced it is amnesiac.

When a delivery is mirror-eligible anyway, this module enriches the seed with
a bounded digest of the run's own cron session transcript (read-only from the
canonical SQLite session store in ``$HERMES_HOME/state.db``): a timeline of
assistant text turns, tool calls (name + truncated arguments), and tool
results, closing with the final response and a pointer at the original cron
session id so the full transcript stays reachable via session search.

Everything here is best-effort: any failure returns ``None`` and the caller
falls back to the plain (final-response-only) seed. The digest never gates,
blocks, or rewrites a delivery.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Bounds. The seed lands as a single user turn in the continuation session,
# so it must stay a rounding error against the conversation's context budget:
# 24 timeline events and a 6000-character ceiling keep even the busiest run
# to a few thousand tokens, while still covering typical multi-step jobs.
_MAX_DIGEST_EVENTS = 24
_MAX_DIGEST_CHARS = 6000
# Per-item truncation. Tool results can be enormous (web page dumps); the
# digest records what happened, not the payload.
_MAX_TEXT_CHARS = 400
_MAX_TOOL_RESULT_CHARS = 200
_MAX_TOOL_ARGS_CHARS = 160
_MAX_TOOL_NAME_CHARS = 40

# Tool results that are pure acks carry no information — skip the noise.
_ACK_RESULTS = ("{\"result\": \"Memory stored successfully.\"}",)


def build_cron_digest(
    job: Dict[str, Any],
    session_id: Optional[str],
    final_text: str,
) -> Optional[str]:
    """Build the enriched seed text for a mirror-eligible cron delivery.

    Returns the digest (final response embedded), or ``None`` when there is
    nothing to add — no session id, an unreadable/empty transcript, or a run
    with no timeline beyond the final response. The caller then seeds with
    the plain final text exactly as before.

    ``final_text`` is the cleaned, unwrapped delivery text (the same string
    the shipped seeds append); it is always included verbatim so callers can
    treat the return value as a drop-in replacement.
    """
    if not session_id:
        return None
    try:
        messages = _read_cron_session_messages(session_id)
    except Exception as e:
        logger.debug(
            "Cron digest: transcript read failed for session %s: %s",
            session_id, e,
        )
        return None
    if not messages:
        return None

    events = _extract_digest_events(messages)
    # Only the final response and nothing else — the plain seed already
    # carries all of it; a digest would be pure duplication.
    if not events:
        return None

    job_name = str(job.get("name") or job.get("id") or "cron job")
    lines: List[str] = [
        f"<!-- cron-digest session={session_id} -->",
        f"[Cron execution digest: {job_name}]",
        "Context from this scheduled run (tool activity summary, oldest first):",
    ]

    budget = _MAX_DIGEST_CHARS
    included = 0
    for event in events:
        line = _format_event(event)
        cost = len(line) + 1
        if included >= _MAX_DIGEST_EVENTS or cost > budget:
            lines.append("[… older activity truncated …]")
            break
        budget -= cost
        lines.append(line)
        included += 1

    lines.append(
        f"Full transcript of this run: session `{session_id}` "
        "(searchable via session history)."
    )

    # The final response rides at the end, verbatim — a drop-in replacement
    # for the plain final-response seed the caller would otherwise append.
    final = (final_text or "").strip()
    if final:
        lines.append("")
        lines.append(final)

    digest = "\n".join(lines).strip()
    if len(digest) > _MAX_DIGEST_CHARS:
        digest = digest[:_MAX_DIGEST_CHARS].rstrip() + "\n[… digest truncated …]"
    return digest


def _read_cron_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Read a session transcript read-only from the canonical session store.

    Direct SQLite rather than SessionDB: the cron worker runs in a separate
    process from the gateway, its own SessionDB handle is already finalized
    (bounded + closed) by the time delivery runs, and a read-only URI
    connection cannot dirty WAL state the gateway is mid-write on. Raises on
    any failure — the caller catches and falls back.
    """
    db_path = Path(get_hermes_home()) / "state.db"
    if not db_path.exists():
        raise FileNotFoundError(f"session store not found: {db_path}")
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=5,
    )
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_name, tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1 ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _extract_digest_events(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reduce a transcript to digest events, oldest-first.

    Yields one event per assistant text turn, per tool call, and per
    non-ack tool result. The trailing assistant turn (the delivery's final
    response) is dropped — it is already in the seed verbatim. System-prompt
    and tool-result content never leak whole: every item is truncated.
    """
    events: List[Dict[str, Any]] = []
    saw_first_assistant_text = False
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            text = _clean_text(msg.get("content"))
            if text:
                if not saw_first_assistant_text:
                    # Reproduced step from the job prompt — still useful as a
                    # one-line marker of what the run set out to do.
                    saw_first_assistant_text = True
                events.append({"kind": "text", "text": text})
            for call in _parse_tool_calls(msg.get("tool_calls")):
                events.append({
                    "kind": "call",
                    "name": call["name"],
                    "args": call["arguments"],
                })
        elif role == "tool":
            result = _clean_text(msg.get("content"))
            if not result or result in _ACK_RESULTS:
                continue
            events.append({
                "kind": "result",
                "name": str(msg.get("tool_name") or "tool"),
                "result": result,
            })
    # Drop the trailing assistant text turn: it IS the final response the
    # seed already carries. (Mid-run text turns stay.)
    for idx in range(len(events) - 1, -1, -1):
        if events[idx]["kind"] == "text":
            del events[idx]
            break
    return events


def _parse_tool_calls(raw: Any) -> List[Dict[str, str]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    calls: List[Dict[str, str]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = str(fn.get("name") or tc.get("name") or "tool")
        calls.append({
            "name": name,
            "arguments": str(fn.get("arguments") or ""),
        })
    return calls


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _format_event(event: Dict[str, Any]) -> str:
    kind = event["kind"]
    if kind == "text":
        snippet = _ellipsize(event["text"], _MAX_TEXT_CHARS)
        return f"- step: {snippet}"
    if kind == "call":
        name = _ellipsize(event["name"], _MAX_TOOL_NAME_CHARS)
        args = _ellipsize(event["args"], _MAX_TOOL_ARGS_CHARS)
        return f"- tool {name}({args})"
    # result
    name = _ellipsize(event["name"], _MAX_TOOL_NAME_CHARS)
    result = _ellipsize(event["result"], _MAX_TOOL_RESULT_CHARS)
    return f"- {name} → {result}"


def _ellipsize(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
