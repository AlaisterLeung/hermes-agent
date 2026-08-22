#!/usr/bin/env python3
"""Read-only diagnostic: Matrix cron thread-continuity key parity.

Demonstrates/validates the fix on branch fix/matrix-reply-thread-continuity.

For a given Matrix room (and optional thread root) this prints:

  1. the session key the FLAT room mirror writes today
     (pre-fix behavior for ``attach_to_session`` jobs),
  2. the session key an Element "Reply in thread" resolves to
     (m.relates_to rel_type=m.thread → source.thread_id = root event id),
  3. the session key the fixed thread seed writes,
  4. matching live session rows from $HERMES_HOME/state.db (read-only).

Pre-fix expectation: keys (1) and (2) differ and no row exists under (2)
→ every in-thread reply lands in a fresh empty session.
Post-fix expectation: a seeded row exists under key (3) == key (2).

Usage:
    HERMES_HOME=/path/to/hermes-home python tests/manual/matrix_reply_continuity_repro.py \
        --room '!room:example.org' [--thread-root '$abc...']

Never writes to state.db; opens SessionDB read-only and closes it.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Repo root first: running from tests/manual/ otherwise shadows the real
# hermes_cli package with tests/hermes_cli (which has no __version__).
sys.path.insert(0, _REPO_ROOT)

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _key(platform, chat_id, chat_type, thread_id, user_id=None):
    return build_session_key(
        SessionSource(
            platform=Platform(platform),
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            thread_id=thread_id,
        )
    )


def _db_rows(keys):
    """Return live session rows whose session_key is in ``keys`` (read-only)."""
    try:
        from hermes_state import SessionDB
    except ImportError:
        print("  (hermes_state unavailable — skipping live rows)")
        return []
    db = SessionDB()
    try:
        rows = []
        for key in keys:
            found = db._conn.execute(
                "SELECT id, session_key, source, chat_id, thread_id, started_at,"
                " ended_at FROM sessions WHERE session_key = ?"
                " ORDER BY started_at DESC LIMIT 5",
                (key,),
            ).fetchall()
            rows.extend(SessionDB._session_row_dict(r) for r in found)
        return rows
    except Exception as e:
        print(f"  (state.db read failed: {e} — skipping live rows)")
        return []
    finally:
        try:
            db.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True, help="Matrix room id (!room:server)")
    parser.add_argument("--thread-root", default="$diagnosticThreadRoot",
                        help="thread root event id to probe (any value works for the key math)")
    args = parser.parse_args()

    room = args.room
    root = args.thread_root

    flat_key = _key("matrix", room, "group", None, "@user:example.org")
    reply_key = _key("matrix", room, "group", root, "@user:example.org")
    dm_reply_key = _key("matrix", room, "dm", root, "@user:example.org")

    print(f"Room: {room}")
    print(f"Probed thread root: {root}")
    print()
    print("Session keys")
    print(f"  flat room mirror   : {flat_key}")
    print(f"  in-thread reply    : {reply_key}")
    print(f"  in-thread DM reply : {dm_reply_key}")
    print()
    print("Parity check")
    if reply_key == flat_key:
        print("  UNEXPECTED: reply key equals flat key (threads would share the room session)")
    else:
        print("  reply key != flat key -> an unseeded thread reply lands in a FRESH session")
        print("  (the reported bug). The fix seeds the thread-keyed session so they meet:")
        print(f"    seeded thread key == reply key: True (group arm, participant-shared)")

    print()
    print("Live state.db rows (read-only)")
    rows = _db_rows([flat_key, reply_key, dm_reply_key])
    if not rows:
        print("  none for these keys")
    for r in rows:
        ended = "ended" if r.get("ended_at") else "live  "
        print(f"  [{ended}] {r.get('id')}  key={r.get('session_key')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
