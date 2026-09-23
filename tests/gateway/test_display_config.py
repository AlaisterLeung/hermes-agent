"""Tests for gateway.display_config — per-platform display/verbosity resolver."""


# ---------------------------------------------------------------------------
# Resolver: resolution order
# ---------------------------------------------------------------------------

class TestToolProgressProvenance:
    def test_winning_source_controls_mode_and_intent(self):
        from gateway.display_config import resolve_tool_progress

        cases = [
            ({}, None, ("off", False)),
            ({}, "all", ("all", True)),
            ({"tool_progress": None}, "all", ("all", True)),
            ({"platforms": {"slack": {"tool_progress": None}}}, "off", ("off", True)),
            ({"tool_progress_overrides": {"slack": None}}, "new", ("new", True)),
            ({"tool_progress": False}, "all", ("off", True)),
            ({"tool_progress": "all", "platforms": {"slack": {"tool_progress": None}}}, "off", ("all", True)),
            ({"tool_progress": "off", "tool_progress_overrides": {"slack": "new"}}, "all", ("new", True)),
            ({"tool_progress_overrides": {"slack": "off"}, "platforms": {"slack": {"tool_progress": "all"}}}, None, ("all", True)),
        ]
        for display, env, expected in cases:
            assert resolve_tool_progress({"display": display}, "slack", env) == expected


class TestResolveDisplaySetting:
    """resolve_display_setting() resolves with correct priority."""

    def test_explicit_platform_override_wins(self):
        """display.platforms.<plat>.<key> takes top priority."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "platforms": {
                    "telegram": {"tool_progress": "verbose"},
                },
            }
        }
        assert resolve_display_setting(config, "telegram", "tool_progress") == "verbose"

    def test_global_setting_when_no_platform_override(self):
        """Falls back to display.<key> when no platform override exists."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "new",
                "platforms": {},
            }
        }
        assert resolve_display_setting(config, "telegram", "tool_progress") == "new"


    def test_platform_override_only_affects_that_platform(self):
        """Other platforms are unaffected by a specific platform override."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "platforms": {
                    "slack": {"tool_progress": "off"},
                },
            }
        }
        assert resolve_display_setting(config, "slack", "tool_progress") == "off"
        assert resolve_display_setting(config, "telegram", "tool_progress") == "all"


# ---------------------------------------------------------------------------
# Backward compatibility: tool_progress_overrides
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Legacy tool_progress_overrides is still respected as a fallback."""

    def test_legacy_overrides_read(self):
        """tool_progress_overrides is read when no platforms entry exists."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "tool_progress": "all",
                "tool_progress_overrides": {
                    "signal": "off",
                    "telegram": "verbose",
                },
            }
        }
        assert resolve_display_setting(config, "signal", "tool_progress") == "off"
        assert resolve_display_setting(config, "telegram", "tool_progress") == "verbose"


# ---------------------------------------------------------------------------
# YAML normalisation
# ---------------------------------------------------------------------------

class TestYAMLNormalisation:
    """YAML 1.1 quirks (bare off → False, on → True) are handled."""

    def test_tool_progress_false_normalised_to_off(self):
        """YAML's bare `off` parses as False — normalised to 'off' string."""
        from gateway.display_config import resolve_display_setting

        config = {"display": {"tool_progress": False}}
        assert resolve_display_setting(config, "telegram", "tool_progress") == "off"


    def test_only_long_running_visibility_accepts_generic_mode(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "whatsapp": {
                        "thinking_progress": "generic",
                        "interim_assistant_messages": "generic",
                        "long_running_notifications": "generic",
                    }
                }
            }
        }
        assert resolve_display_setting(config, "whatsapp", "thinking_progress") is False
        assert resolve_display_setting(config, "whatsapp", "interim_assistant_messages") is False
        assert resolve_display_setting(config, "whatsapp", "long_running_notifications") == "generic"

    def test_thinking_progress_string_false_normalised_to_false(self):
        from gateway.display_config import resolve_display_setting

        config = {"display": {"platforms": {"whatsapp": {"thinking_progress": "false"}}}}
        assert resolve_display_setting(config, "whatsapp", "thinking_progress") is False


# ---------------------------------------------------------------------------
# Built-in platform defaults (tier system)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Config migration: tool_progress_overrides → display.platforms
# ---------------------------------------------------------------------------

class TestConfigMigration:
    """Version 16 migration moves tool_progress_overrides into display.platforms."""

    def test_migration_creates_platforms_entries(self, tmp_path, monkeypatch):
        """Old overrides are migrated into display.platforms.<plat>.tool_progress."""
        import yaml

        config_path = tmp_path / "config.yaml"
        config = {
            "_config_version": 15,
            "display": {
                "tool_progress_overrides": {
                    "signal": "off",
                    "telegram": "all",
                },
            },
        }
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Re-import to pick up the new HERMES_HOME
        import importlib
        import hermes_cli.config as cfg_mod
        importlib.reload(cfg_mod)

        result = cfg_mod.migrate_config(interactive=False, quiet=True)
        # Re-read config
        updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        platforms = updated.get("display", {}).get("platforms", {})
        assert platforms.get("signal", {}).get("tool_progress") == "off"
        assert platforms.get("telegram", {}).get("tool_progress") == "all"


# ---------------------------------------------------------------------------
# Streaming per-platform (None = follow global)
# ---------------------------------------------------------------------------

class TestStreamingPerPlatform:
    """Streaming per-platform override semantics."""


    def test_explicit_false_disables(self):
        """Explicit False disables streaming for that platform."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {"telegram": {"streaming": False}},
            }
        }
        assert resolve_display_setting(config, "telegram", "streaming") is False





# ---------------------------------------------------------------------------
# cleanup_progress — opt-in deletion of temporary progress bubbles
# ---------------------------------------------------------------------------

class TestCleanupProgress:
    """``cleanup_progress`` is off by default and resolvable per-platform."""



    def test_yaml_true_string_normalises_to_true(self):
        """String 'true'/'yes'/'on' all resolve to True."""
        from gateway.display_config import resolve_display_setting

        for val in ("true", "yes", "on", "1"):
            config = {
                "display": {
                    "platforms": {"telegram": {"cleanup_progress": val}},
                }
            }
            assert resolve_display_setting(config, "telegram", "cleanup_progress") is True, val








# ---------------------------------------------------------------------------
# per-chat overrides — display.platforms.<plat>.chats.<chat_id>.<key>
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal SessionSource stand-in for resolver tests."""

    def __init__(self, chat_id, thread_id=None, parent_chat_id=None):
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.parent_chat_id = parent_chat_id


class TestPerChatDisplayOverrides:
    """display.platforms.<plat>.chats.<chat_id>.<key> per-chat layer."""

    def test_chat_override_beats_platform_override(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "matrix": {
                        "tool_progress": "new",
                        "chats": {
                            "!noisy:example.org": {"tool_progress": "off"},
                        },
                    },
                }
            }
        }
        assert (
            resolve_display_setting(
                config, "matrix", "tool_progress", chat="!noisy:example.org"
            )
            == "off"
        )

    def test_other_chats_fall_through_to_platform_setting(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "matrix": {
                        "tool_progress": "new",
                        "chats": {
                            "!noisy:example.org": {"tool_progress": "off"},
                        },
                    },
                }
            }
        }
        assert (
            resolve_display_setting(
                config, "matrix", "tool_progress", chat="!quiet:example.org"
            )
            == "new"
        )

    def test_no_chat_argument_ignores_chats_layer(self):
        """Legacy call shape (no chat=) must behave exactly as before."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "matrix": {
                        "chats": {"!noisy:example.org": {"tool_progress": "off"}},
                    },
                }
            }
        }
        # Built-in matrix default (Tier 2) wins — chats layer is invisible.
        assert resolve_display_setting(config, "matrix", "tool_progress") == "new"

    def test_interim_assistant_messages_per_chat(self):
        """interim_assistant_messages resolves per chat, not only per platform."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "telegram": {
                        "chats": {
                            "-1001234567890": {
                                "interim_assistant_messages": False,
                            },
                        },
                    },
                }
            }
        }
        assert (
            resolve_display_setting(
                config, "telegram", "interim_assistant_messages",
                chat="-1001234567890",
            )
            is False
        )
        # A different chat in the same platform keeps the default (True).
        assert (
            resolve_display_setting(
                config, "telegram", "interim_assistant_messages",
                chat="-1009999999999",
            )
            is True
        )

    def test_numeric_chat_ids_match_string_lookups(self):
        """Telegram numeric ids parse as YAML ints; lookups are strings."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "telegram": {
                        "chats": {
                            -1001234567890: {"tool_progress": "verbose"},
                        },
                    },
                }
            }
        }
        assert (
            resolve_display_setting(
                config, "telegram", "tool_progress", chat="-1001234567890"
            )
            == "verbose"
        )

    def test_chat_string_values_are_normalised(self):
        """YAML 1.1 'off' → False coercion applies inside chat entries too."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "slack": {
                        "chats": {
                            "C123": {"interim_assistant_messages": "false"},
                        },
                    },
                }
            }
        }
        assert (
            resolve_display_setting(
                config, "slack", "interim_assistant_messages", chat="C123"
            )
            is False
        )

    def test_source_object_thread_inherits_parent_chat(self):
        """A thread chat falls back to its parent chat's override."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "matrix": {
                        "chats": {
                            "!room:example.org": {"tool_progress": "off"},
                        },
                    },
                }
            }
        }
        src = _FakeSource(
            chat_id="$threadevt:example.org",
            thread_id="$threadevt:example.org",
            parent_chat_id="!room:example.org",
        )
        assert (
            resolve_display_setting(config, "matrix", "tool_progress", chat=src)
            == "off"
        )

    def test_thread_exact_key_beats_parent(self):
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "discord": {
                        "chats": {
                            "111": {"tool_progress": "verbose"},
                            "222": {"tool_progress": "off"},
                        },
                    },
                }
            }
        }
        src = _FakeSource(chat_id="222", thread_id="222", parent_chat_id="111")
        assert (
            resolve_display_setting(config, "discord", "tool_progress", chat=src)
            == "off"
        )

    def test_has_chat_display_override(self):
        from gateway.display_config import has_chat_display_override

        config = {
            "display": {
                "platforms": {
                    "matrix": {
                        "chats": {"!room:example.org": {"tool_progress": "off"}},
                    },
                }
            }
        }
        assert has_chat_display_override(
            config, "matrix", "tool_progress", chat="!room:example.org"
        )
        assert not has_chat_display_override(
            config, "matrix", "show_reasoning", chat="!room:example.org"
        )
        assert not has_chat_display_override(
            config, "matrix", "tool_progress", chat="!other:example.org"
        )
        assert not has_chat_display_override(config, "matrix", "tool_progress")

    def test_null_entry_is_ignored(self):
        """A chat entry that is not a mapping (e.g. empty YAML value) is skipped."""
        from gateway.display_config import resolve_display_setting

        config = {
            "display": {
                "platforms": {
                    "matrix": {
                        "chats": {"!room:example.org": None},
                    },
                }
            }
        }
        assert (
            resolve_display_setting(config, "matrix", "tool_progress", chat="!room:example.org")
            == "new"
        )


class TestPerChatDisplayTurnWiring:
    """_run_agent_display_settings() must consult the per-chat layer.

    The resolver tests above stay green even when a turn call site forgets ``chat=``;
    these pin the turn assembly so a per-chat override actually reaches the resolved
    turn fields (platform-level off/false must not win).
    """

    def _mixin(self):
        from gateway.run import GatewayRunner
        from gateway.run_turn import GatewayTurnMixin

        mixin = GatewayTurnMixin.__new__(GatewayTurnMixin)
        object.__setattr__(mixin, "_RunAgentDisplay", GatewayRunner._RunAgentDisplay)
        object.__setattr__(mixin, "_delivery_adapter_for", lambda source: None)
        object.__setattr__(
            mixin, "_resolve_turn_toolsets",
            lambda user_config, source, platform_key: (None, None),
        )
        return mixin

    def _source(self):
        from gateway.config import Platform
        from gateway.session import SessionSource

        return SessionSource(
            platform=Platform.MATRIX,
            chat_id="!dev:example.org",
            user_id="@boss:example.org",
        )

    def _config(self):
        return {
            "display": {
                "platforms": {
                    "matrix": {
                        "tool_progress": "off",
                        "interim_assistant_messages": False,
                        "chats": {
                            "!dev:example.org": {
                                "tool_progress": "all",
                                "interim_assistant_messages": True,
                            },
                        },
                    },
                }
            },
        }

    def test_per_chat_tool_progress_reaches_turn(self, monkeypatch):
        mixin = self._mixin()
        monkeypatch.setattr(
            "gateway.run._load_gateway_config", lambda: self._config(),
        )
        disp = mixin._run_agent_display_settings(self._source())
        assert disp.progress_mode == "all"
        assert disp.tool_progress_enabled is True

    def test_per_chat_interim_messages_reach_turn(self, monkeypatch):
        mixin = self._mixin()
        monkeypatch.setattr(
            "gateway.run._load_gateway_config", lambda: self._config(),
        )
        disp = mixin._run_agent_display_settings(self._source())
        assert disp.interim_assistant_messages_enabled is True

    def test_per_chat_override_is_configured_against_env_mode(self, monkeypatch):
        """A per-chat tool_progress entry counts as configured (env must not shadow it)."""
        mixin = self._mixin()
        monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {
                "display": {
                    "platforms": {
                        "matrix": {
                            "chats": {"!dev:example.org": {"tool_progress": "all"}},
                        },
                    },
                },
            },
        )
        disp = mixin._run_agent_display_settings(self._source())
        assert disp.progress_mode == "all"


