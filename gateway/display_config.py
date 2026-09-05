"""Per-platform display/verbosity resolver (``resolve_display_setting``).

Provides ``resolve_display_setting()`` — the single entry-point for reading
display settings with platform-specific overrides and sensible defaults.

Resolution order (first non-None wins):
    0. ``display.platforms.<platform>.chats.<chat_id>.<key>``  — per-chat user
       override (#31488), only consulted when the caller passes ``chat=``.
       Threads inherit their parent chat's entry: lookup keys are
       ``chat_id`` → ``thread_id`` → ``parent_chat_id`` (same order as
       ``channel_overrides``).
    1. ``display.platforms.<platform>.<key>``  — explicit per-platform user override
    2. ``display.<key>``                       — global user setting
    3. ``_PLATFORM_DEFAULTS[<platform>][<key>]``  — built-in sensible default
    4. ``_GLOBAL_DEFAULTS[<key>]``              — built-in global default

Exception: ``display.streaming`` is CLI-only.  Gateway streaming follows the
top-level ``streaming`` config unless ``display.platforms.<platform>.streaming``
sets an explicit per-platform override.

Backward compatibility: ``display.tool_progress_overrides`` is still read as a
fallback for ``tool_progress`` when no ``display.platforms`` entry exists.  A
config migration (version bump) automatically moves the old format into the new
``display.platforms`` structure.
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

    ``chat`` is either a plain chat-id string or a ``SessionSource``-like
    object.  Mirrors ``_channel_override_lookup_keys`` semantics: the exact
    chat/thread id first, then the parent channel (threads inherit the
    parent chat's override).
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


def has_chat_display_override(
    user_config: dict,
    platform_key: str,
    setting: str,
    chat: Any = None,
) -> bool:
    """True when ``display.platforms.<platform>.chats.<chat_id>`` explicitly
    sets ``setting`` for ``chat``.  Used by the per-chat display resolution
    (#31488) and by the "explicit override required" gates in gateway/run.py.
    """
    if chat is None:
        return False
    for entry in _chat_override_entries(user_config, platform_key, chat):
        if isinstance(entry, dict) and setting in entry:
            return True
    return False


def resolve_display_setting(
    user_config: dict,
    platform_key: str,
    setting: str,
    fallback: Any = None,
    chat: Any = None,
) -> Any:
    """Resolve a display setting with per-platform override support.

    Parameters
    ----------
    user_config : dict
        The full parsed config.yaml dict.
    platform_key : str
        Platform config key (e.g. ``"telegram"``, ``"slack"``).  Use
        ``_platform_config_key(source.platform)`` from gateway/run.py.
    setting : str
        Display setting name (e.g. ``"tool_progress"``, ``"show_reasoning"``).
    fallback : Any
        Fallback value when the setting isn't found anywhere.
    chat : Any, optional
        Chat context for per-chat overrides (#31488): a chat-id string or a
        ``SessionSource``-like object.  When given,
        ``display.platforms.<platform>.chats.<chat_id>.<setting>`` is
        consulted first; threads inherit their parent chat's entry.

    Returns
    -------
    The resolved value, or *fallback* if nothing is configured.
    """
    display_cfg = user_config.get("display") or {}

    # 0. Per-chat override (display.platforms.<platform>.chats.<chat_id>.<key>)
    if chat is not None:
        for entry in _chat_override_entries(user_config, platform_key, chat):
            if isinstance(entry, dict):
                val = entry.get(setting)
                if val is not None:
                    return _normalise(setting, val)

    # 1. Explicit per-platform override (display.platforms.<platform>.<key>)
    platforms = display_cfg.get("platforms") or {}
    plat_overrides = platforms.get(platform_key)
    if isinstance(plat_overrides, dict):
        val = plat_overrides.get(setting)
        if val is not None:
            return _normalise(setting, val)

    # 1b. Backward compat: display.tool_progress_overrides.<platform>
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict) and legacy.get(platform_key) is not None:
            return _normalise(setting, legacy[platform_key])
    if setting != "streaming" and display_cfg.get(setting) is not None:  # display.streaming is CLI-only
        return _normalise(setting, display_cfg[setting])
    val = _PLATFORM_DEFAULTS.get(platform_key, {}).get(setting)
    if val is None:
        val = _GLOBAL_DEFAULTS.get(setting)
    return fallback if val is None else val


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
