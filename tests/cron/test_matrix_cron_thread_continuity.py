"""Matrix continuable-cron thread continuity — end-to-end key contract.

Background: a cron job with ``attach_to_session: true`` delivering into a
Matrix room fell back to the flat-room mirror because the Matrix adapter
never implemented ``create_handoff_thread``. And when a user replied to the
brief "in thread", the inbound event keyed to
``agent:main:matrix:group:<room>:<rootEventId>`` — a session that had never
seen the brief (the flat mirror wrote ``...:group:<room>`` with no thread
segment), so every in-thread reply landed in a fresh empty session.

Contract under test, mirroring test_cron_thread_seed_dm_keying.py:

1. ``MatrixAdapter.create_handoff_thread`` produces a thread root the
   scheduler's duck-typed hook accepts, and fails soft (``None``) so the
   flat-mirror fallback survives.
2. The Matrix ROOM thread seed keys EXACTLY like the inbound in-thread room
   reply (``chat_type="group"``, participant-shared — no user segment).
3. Matrix DM thread seeds keep keying through the ``dm`` arm.
4. Non-Matrix platforms keep the historical ``chat_type="thread"`` +
   ``system:cron`` seed shape byte-for-byte (no cross-platform regression).

Asserting on build_session_key output — not on SessionSource field shapes —
pins the end-to-end contract.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _open_continuable_cron_thread, _seed_cron_thread_session
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


ROOM = "!room:example.org"
ROOT = "$abc123threadRoot"


def _seeded_source(store):
    store.get_or_create_session.assert_called_once()
    return store.get_or_create_session.call_args[0][0]


def _seed(job_name="Nightly Digest", platform="matrix", chat_id=ROOM,
          thread_id=ROOT, is_dm=False, adapter_is_dm=None):
    store = MagicMock()
    adapter = MagicMock()
    adapter._session_store = store
    if adapter_is_dm is not None:
        # Simulate the adapter's live room classification (member count +
        # m.direct), which can disagree with job metadata.
        adapter._is_dm_room = AsyncMock(return_value=adapter_is_dm)
    with patch("gateway.mirror.mirror_to_session", return_value=True):
        _seed_cron_thread_session(
            {"id": "j1", "name": job_name}, adapter, platform,
            chat_id, thread_id, "brief body",
            chat_name=None, is_dm=is_dm,
        )
    return store


# ---------------------------------------------------------------------------
# 1. Adapter: create_handoff_thread
# ---------------------------------------------------------------------------

def _make_matrix_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    return MatrixAdapter(config)


@pytest.mark.asyncio
async def test_create_handoff_thread_returns_root_event_id():
    adapter = _make_matrix_adapter()
    client = MagicMock()
    client.send_message_event = AsyncMock(return_value="$rootEvent1")
    adapter._client = client

    root = await adapter.create_handoff_thread(ROOM, "[Cron] Nightly Digest")

    assert root == "$rootEvent1"
    client.send_message_event.assert_awaited_once()
    args = client.send_message_event.await_args
    assert str(args.args[0]) == ROOM
    content = args.args[2]
    assert content["msgtype"] == "m.text"
    assert content["body"] == "[Cron] Nightly Digest"


@pytest.mark.asyncio
async def test_create_handoff_thread_resolves_alias_before_send():
    adapter = _make_matrix_adapter()
    client = MagicMock()
    client.api = MagicMock()
    client.api.request = AsyncMock(return_value={"room_id": ROOM})
    client.send_message_event = AsyncMock(return_value="$rootEvent2")
    adapter._client = client

    root = await adapter.create_handoff_thread("#room:example.org", "digest")

    assert root == "$rootEvent2"
    # Directory lookup happened first, with the alias URL-quoted.
    method, path = client.api.request.await_args.args
    assert method == "GET"
    assert "%23room%3Aexample.org" in path
    # The send targeted the RESOLVED room id, not the alias.
    assert str(client.send_message_event.await_args.args[0]) == ROOM


@pytest.mark.asyncio
async def test_create_handoff_thread_fails_soft_on_send_error():
    adapter = _make_matrix_adapter()
    client = MagicMock()
    client.send_message_event = AsyncMock(side_effect=RuntimeError("M_LIMIT_EXCEEDED"))
    adapter._client = client

    assert await adapter.create_handoff_thread(ROOM, "digest") is None


@pytest.mark.asyncio
async def test_create_handoff_thread_fails_soft_when_disconnected():
    adapter = _make_matrix_adapter()
    adapter._client = None
    assert await adapter.create_handoff_thread(ROOM, "digest") is None


def test_scheduler_picks_up_matrix_handoff_thread():
    """The scheduler's duck-typed hook engages the Matrix adapter and
    returns the root id (thread-preferred branch precondition)."""
    adapter = _make_matrix_adapter()
    adapter.create_handoff_thread = AsyncMock(return_value=ROOT)
    loop = MagicMock()
    loop.is_running = lambda: True

    with patch("agent.async_utils.safe_schedule_threadsafe") as sched:
        sched.return_value.result.return_value = ROOT
        new_thread = _open_continuable_cron_thread(
            {"id": "j1", "name": "Nightly Digest"}, adapter, ROOM, loop,
        )

    assert new_thread == ROOT
    sched.assert_called_once()
    coro = sched.call_args.args[0]
    coro.close()  # never awaited — close to silence the warning
    adapter.create_handoff_thread.assert_called_once_with(
        ROOM, "[Cron] Nightly Digest"
    )


# ---------------------------------------------------------------------------
# 2. Matrix ROOM thread seed keys like the inbound in-thread reply
# ---------------------------------------------------------------------------

def test_matrix_room_thread_seed_key_matches_in_thread_reply_key():
    store = _seed(is_dm=False)

    seed_key = build_session_key(_seeded_source(store))
    reply_source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=ROOM,
        chat_type="group",
        user_id="@alice:example.org",  # real sender; must NOT affect the key
        thread_id=ROOT,
    )
    assert seed_key == build_session_key(reply_source), (
        "seeded key diverges from the Matrix in-thread reply's key — the "
        "brief lands in a row no reply ever resolves to (continuation amnesia)"
    )
    # Participant-shared: the reply's sender must not split the key.
    assert "system:cron" not in seed_key


def test_matrix_room_thread_seed_differs_from_flat_room_session():
    """The whole point: the thread seed must be a DIFFERENT row from the
    flat room session, and the flat session must remain untouched."""
    store = _seed(is_dm=False)
    seed_key = build_session_key(_seeded_source(store))

    flat_source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=ROOM,
        chat_type="group",
        user_id="@alice:example.org",
        thread_id=None,
    )
    assert seed_key != build_session_key(flat_source)


def test_matrix_room_thread_seed_key_stable_across_senders():
    """Two members replying in the same thread join ONE session (shared)."""
    keys = set()
    for sender in ("@alice:example.org", "@bob:example.org"):
        store = _seed(is_dm=False)
        keys.add(build_session_key(SessionSource(
            platform=Platform.MATRIX,
            chat_id=ROOM,
            chat_type="group",
            user_id=sender,
            thread_id=ROOT,
        )))
    assert len(keys) == 1


# ---------------------------------------------------------------------------
# 3. Matrix DM thread seed keys through the dm arm
# ---------------------------------------------------------------------------

def test_matrix_dm_thread_seed_key_matches_dm_reply_key():
    store = _seed(is_dm=True)

    seed_key = build_session_key(_seeded_source(store))
    reply_source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=ROOM,
        chat_type="dm",
        user_id="@alice:example.org",
        thread_id=ROOT,
    )
    assert seed_key == build_session_key(reply_source)


# ---------------------------------------------------------------------------
# 3b. Live adapter DM classification overrides job metadata
# ---------------------------------------------------------------------------

def test_matrix_seed_follows_adapter_dm_classification_not_metadata():
    """A small named room is a group to job metadata but a DM to the
    adapter's live classifier (member count + m.direct). The seed must
    follow the ADAPTER's verdict — the reply keys through whichever arm
    the inbound event actually takes."""
    store = _seed(is_dm=False, adapter_is_dm=True)  # metadata: group

    seed_key = build_session_key(_seeded_source(store))
    reply_source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=ROOM,
        chat_type="dm",
        user_id="@alice:example.org",
        thread_id=ROOT,
    )
    assert seed_key == build_session_key(reply_source)


def test_matrix_seed_group_classification_survives_when_adapter_says_room():
    store = _seed(is_dm=False, adapter_is_dm=False)
    seed_key = build_session_key(_seeded_source(store))
    assert ":group:" in seed_key


def test_non_matrix_seed_ignores_adapter_probe():
    """Only matrix gets the live re-resolution; other platforms keep the
    historical shape even if an adapter exposes a probe by coincidence."""
    store = _seed(platform="slack", chat_id="C12345678", is_dm=False)
    seed_key = build_session_key(_seeded_source(store))
    assert ":thread:" in seed_key and "user_id" not in seed_key


# ---------------------------------------------------------------------------
# 4. Non-Matrix platforms keep the historical seed shape
# ---------------------------------------------------------------------------

def test_slack_channel_thread_seed_shape_unchanged():
    store = _seed(platform="slack", chat_id="C0AAAA", thread_id="1787.448949")

    source = _seeded_source(store)
    assert source.chat_type == "thread"
    assert source.user_id == "system:cron"
    assert source.chat_id == "C0AAAA"


def test_telegram_thread_seed_shape_unchanged():
    store = _seed(platform="telegram", chat_id="555111", thread_id="42")

    source = _seeded_source(store)
    assert source.chat_type == "thread"
    assert source.user_id == "system:cron"
    assert source.chat_id == "555111"


def test_discord_thread_seed_keeps_thread_own_id():
    store = _seed(platform="discord", chat_id="987654", thread_id="111222")

    source = _seeded_source(store)
    assert source.chat_type == "thread"
    assert source.user_id == "system:cron"
    assert source.chat_id == "111222"  # thread's OWN id, not the parent


def test_matrix_dm_seed_keeps_system_cron_user():
    """DM arm keeps the historical synthetic user (dm keys embed user_id)."""
    store = _seed(is_dm=True)
    source = _seeded_source(store)
    assert source.chat_type == "dm"
    assert source.user_id == "system:cron"
