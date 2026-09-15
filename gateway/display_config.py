"""Per-platform display/verbosity resolver (``resolve_display_setting``).

Resolution order, first non-None wins:
``display.platforms.<platform>.chats.<chat_id>.<key>`` (only when the caller passes
``chat=``) → ``display.platforms.<platform>.<key>`` → ``display.<key>`` →
``_PLATFORM_DEFAULTS[platform][key]`` → ``_GLOBAL_DEFAULTS[key]``. Threads inherit their
parent chat's entry: lookup keys are ``chat_id`` → ``thread_id`` → ``parent_chat_id``
(same order as ``channel_overrides``), with the exact thread key beating the parent.
Exception: ``display.streaming`` is CLI-only; gateway streaming follows the top-level
``streaming`` config unless a per-platform override sets it. Legacy
``display.tool_progress_overrides`` is still read as a ``tool_progress`` fallback.
"""

from __future__ import annotations

from typing import Any

# Settings configurable per-platform; other display settings are CLI-only.
_GLOBAL_DEFAULTS: dict[str, Any] = {
    "tool_progress": "all",
    "tool_progress_grouping": "accumulate",  # "accumulate" = edit one bubble; "separate" = one msg per tool
    "show_reasoning": False,
    "reasoning_style": "code",  # "code" (💭 **Reasoning:** + fence), "blockquote" ("> "), "subtext" ("-# " Discord)
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
    # Gateway-only assistant/status chatter; mobile platforms opt down to final-answer-first.
    "interim_assistant_messages": True,
    "long_running_notifications": True,
    "busy_ack_detail": True,
    "busy_steer_ack_enabled": True,  # busy_input_mode=steer echo; the text still lands in the run
    # Delete tool-progress / "⏳ Working" bubbles after a SUCCESSFUL final response where deletion is
    # supported (Telegram); failed runs keep them as breadcrumbs.
    "cleanup_progress": False,
    # Working-state text on text-rendering indicators (Slack assistant status): "full"/true = verb +
    # argument preview, "verb" = verb only (keeps paths out of shared channels), "off"/false = static.
    "live_status": "full",
}

# Tiers: HIGH = editing, personal/team use; MEDIUM = editing but customer-facing;
# LOW = no edit support (progress messages are permanent); MINIMAL = batch delivery.
_TIER_HIGH = {
    "tool_progress": "all", "show_reasoning": False, "tool_preview_length": 40,
    "streaming": None,  # follow global
    "interim_assistant_messages": True, "long_running_notifications": True, "busy_ack_detail": True,
}
_TIER_MEDIUM = {**_TIER_HIGH, "tool_progress": "new"}
_TIER_LOW = {
    **_TIER_HIGH, "tool_progress": "off", "streaming": False,
    "interim_assistant_messages": False, "long_running_notifications": False, "busy_ack_detail": False,
}
_TIER_MINIMAL = {**_TIER_LOW, "tool_preview_length": 0}

_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    # Mobile inbox: quiet tool_progress / busy-ack, but keep interim commentary and heartbeats so it
    # doesn't look like "typing..." for 30 minutes.
    "telegram": {**_TIER_HIGH, "tool_progress": "off", "busy_ack_detail": False},
    "discord": {**_TIER_HIGH, "reasoning_style": "subtext"},  # "-# " subtext reads as metadata
    # Slack: Bolt posts cannot be edited like CLI; "new"/"all" spam permanent lines.
    "slack": {**_TIER_MEDIUM, "tool_progress": "off", "long_running_notifications": False, "busy_ack_detail": False},
    "mattermost": _TIER_MEDIUM,
    "matrix": _TIER_MEDIUM,
    "feishu": _TIER_MEDIUM,
    "buzz": _TIER_MEDIUM,  # Nostr: edits in place but channels are shared community spaces
    "signal": _TIER_LOW,
    "whatsapp": _TIER_MEDIUM,  # Baileys bridge supports /edit
    "whatsapp_cloud": _TIER_LOW,  # adapter lacks edit_message; promote once it lands
    "photon": _TIER_LOW,  # permanent-message iMessage inboxes (no edit)
    "bluebubbles": _TIER_LOW,
    "weixin": _TIER_LOW,
    # Non-editable, but its native "stream" msgtype gives a typing animation + cumulative updates.
    "wecom": {**_TIER_LOW, "streaming": True},
    "wecom_callback": _TIER_LOW,
    "dingtalk": _TIER_LOW,
    "email": _TIER_MINIMAL,
    "sms": _TIER_MINIMAL,
    "webhook": _TIER_MINIMAL,
    "homeassistant": _TIER_MINIMAL,
    "api_server": {**_TIER_HIGH, "tool_preview_length": 0},
}

# Canonical set of per-platform overrideable keys (for validation).
OVERRIDEABLE_KEYS = frozenset(_GLOBAL_DEFAULTS.keys())


def _chat_lookup_keys(chat: Any) -> list:
    """Ordered chat-id candidates for per-chat display overrides.

    ``chat`` is either a plain chat-id string or a ``SessionSource``-like object. The
    exact chat/thread id is tried first, then the parent channel, so threads inherit
    their parent chat's entry (same order as ``channel_overrides``).
    """
    if chat is None:
        return []
    if isinstance(chat, str):
        return [chat] if chat else []
    keys: list = []
    for attr in ("chat_id", "thread_id", "parent_chat_id"):
        val = getattr(chat, attr, None)
        if val is None or val == "":
            continue
        sk = str(val)
        if sk not in keys:
            keys.append(sk)
    return keys


def _chat_override_entries(user_config: dict, platform_key: str, chat: Any):
    """Yield chat-override mappings matching ``chat`` (most specific first)."""
    display_cfg = user_config.get("display") or {}
    platforms = display_cfg.get("platforms") or {}
    plat_overrides = platforms.get(platform_key)
    if not isinstance(plat_overrides, dict):
        return
    chats = plat_overrides.get("chats")
    if not isinstance(chats, dict) or not chats:
        return
    # Normalise keys to strings: numeric platform chat ids (e.g. Telegram's
    # -100… group ids) parse as YAML ints while lookup keys are strings.
    norm: dict = {}
    for k, v in chats.items():
        try:
            norm.setdefault(str(k), v)
        except Exception:
            continue
    for key in _chat_lookup_keys(chat):
        entry = norm.get(key)
        if entry is not None:
            yield entry


def _chat_display_value(user_config: dict, platform_key: str, setting: str, chat: Any) -> Any:
    """First non-None per-chat override value for ``setting`` (most specific chat wins)."""
    if chat is None:
        return None
    for entry in _chat_override_entries(user_config, platform_key, chat):
        if isinstance(entry, dict):
            val = entry.get(setting)
            if val is not None:
                return val
    return None


def has_chat_display_override(
    user_config: dict, platform_key: str, setting: str, chat: Any = None,
) -> bool:
    """True when ``display.platforms.<platform>.chats.<chat_id>`` explicitly sets ``setting`` for ``chat``."""
    if chat is None:
        return False
    for entry in _chat_override_entries(user_config, platform_key, chat):
        if isinstance(entry, dict) and setting in entry:
            return True
    return False


def resolve_display_setting(
    user_config: dict, platform_key: str, setting: str, fallback: Any = None, chat: Any = None,
) -> Any:
    """Resolve a display setting with per-platform/per-chat override support (see module docstring).

    ``platform_key`` is the platform config key (``"telegram"``; see ``_platform_config_key`` in
    gateway/run.py). Pass ``chat`` — a chat-id string or a ``SessionSource``-like object — to
    consult the per-chat layer (``display.platforms.<platform>.chats.<chat_id>``); threads
    inherit their parent chat's entry. Returns *fallback* when nothing is configured.
    """
    # 0. Per-chat override (display.platforms.<platform>.chats.<chat_id>.<key>)
    configured = _chat_display_value(user_config, platform_key, setting, chat)
    if configured is not None:
        return _normalise(setting, configured)
    # 1. Explicit per-platform / legacy / global value, then tier defaults.
    configured = _configured_display_value(user_config, platform_key, setting)
    if configured is not None:
        return _normalise(setting, configured)
    val = _PLATFORM_DEFAULTS.get(platform_key, {}).get(setting)
    if val is None:
        val = _GLOBAL_DEFAULTS.get(setting)
    return fallback if val is None else val


def _configured_display_value(user_config: dict, platform_key: str, setting: str) -> Any:
    """First non-None operator value, without introducing tier defaults."""
    display_cfg = user_config.get("display") or {}
    plat_overrides = (display_cfg.get("platforms") or {}).get(platform_key)
    if isinstance(plat_overrides, dict) and plat_overrides.get(setting) is not None:
        return plat_overrides[setting]
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict) and legacy.get(platform_key) is not None:
            return legacy[platform_key]
    if setting != "streaming":  # display.streaming is CLI-only
        return display_cfg.get(setting)
    return None


def resolve_tool_progress(
    user_config: dict, platform_key: str, env_mode: str | None = None, chat: Any = None,
) -> tuple[str, bool]:
    """Return (mode, explicit intent) from the same winning source.

    Non-None YAML wins over the legacy env bridge; a per-chat override counts as
    intent. Null inherits through to env, then tier defaults. A tier's off is not
    an operator request to disable cards.
    """
    configured = _chat_display_value(user_config, platform_key, "tool_progress", chat)
    if configured is not None:
        return _normalise("tool_progress", configured), True
    configured = _configured_display_value(user_config, platform_key, "tool_progress")
    if configured is not None:
        return _normalise("tool_progress", configured), True
    if env_mode:
        return _normalise("tool_progress", env_mode), True
    return resolve_display_setting(user_config, platform_key, "tool_progress"), False


# --- Normalisation of YAML quirks (bare ``off`` → False in YAML 1.1, etc.) ---

_TRUTHY = {"true", "1", "yes", "on"}
_FALSY = {"false", "0", "no"}


def _norm_tristate(on: str, off: str, choices: set, extra_truthy: set = frozenset()):
    """Normaliser for bool-or-keyword settings: bools/truthy tokens → *on*, falsy → *off*, else a known choice or *on*."""
    def norm(value: Any) -> str:
        if isinstance(value, bool):
            return on if value else off
        val = str(value).strip().lower()
        if val in _FALSY:
            return off
        if val in _TRUTHY | extra_truthy:
            return on
        return val if val in choices else on
    return norm


def _norm_bool(value: Any) -> bool:
    return value.strip().lower() in _TRUTHY | {"raw", "verbose"} if isinstance(value, str) else bool(value)


def _norm_long_running(value: Any) -> Any:
    return "generic" if isinstance(value, str) and value.strip().lower() == "generic" else _norm_bool(value)


def _norm_cleanup_progress(value: Any) -> bool:
    return value.lower() in _TRUTHY if isinstance(value, str) else bool(value)


def _norm_choice(choices: tuple[str, ...]) -> Any:
    def norm(value: Any) -> str:
        val = str(value).lower()
        return val if val in choices else choices[0]

    return norm


def _norm_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_NORMALISERS: dict[str, Any] = {
    "tool_progress": _norm_tristate("all", "off", {"off", "new", "all", "verbose", "log"}),
    "show_reasoning": _norm_bool,
    "streaming": _norm_bool,
    "interim_assistant_messages": _norm_bool,
    "long_running_notifications": _norm_long_running,
    "busy_ack_detail": _norm_bool,
    "busy_steer_ack_enabled": _norm_bool,
    "thinking_progress": _norm_bool,
    "cleanup_progress": _norm_cleanup_progress,
    "live_status": _norm_tristate("full", "off", {"full", "verb", "off"}, extra_truthy={"all"}),
    "tool_progress_grouping": _norm_choice(("accumulate", "separate")),
    "reasoning_style": _norm_choice(("code", "blockquote", "subtext")),
    "tool_preview_length": _norm_int,
}


def _normalise(setting: str, value: Any) -> Any:
    """Normalise a user-supplied value for *setting*; unknown settings pass through."""
    norm = _NORMALISERS.get(setting)
    return norm(value) if norm else value
