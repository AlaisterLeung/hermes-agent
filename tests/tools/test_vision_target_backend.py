"""Tests for execution-target support in the vision media resolver.

The resolver must key its confinement decision off the execution-target
resolution (named targets + ``terminal.default_target``), not a legacy env
sniff, and its in-backend reads must delegate to the file tools'
``ShellFileOperations.read_file_bytes`` — the same mechanism the ACP adapter
uses for pinned-target attachment reads.

Fictional target names only (``devbox``, ``staging``): this is a public-facing
fork — never real homelab hosts (contribution rule).
"""

import base64
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    importlib.reload(isrc)
    return isrc


def _targets_config(**overrides):
    """A minimal terminal config with named ssh targets (fictional hosts)."""
    terminal = {
        "backend": "local",
        "default_target": "local",
        "targets": {
            "local": {"backend": "local"},
            "devbox": {
                "backend": "ssh",
                "ssh_host": "203.0.113.10",
                "ssh_user": "deploy",
            },
        },
    }
    terminal.update(overrides)
    return {"terminal": terminal}


def _fake_ops(b64_payload=None, error=None, calls=None):
    """A stub file-ops object mirroring ShellFileOperations.read_file_bytes."""

    class _FakeOps:
        def read_file_bytes(self, path, max_bytes=None):
            if calls is not None:
                calls.append({"path": path, "max_bytes": max_bytes})
            if error:
                return SimpleNamespace(error=error, base64_content=None)
            return SimpleNamespace(
                error=None, base64_content=b64_payload, file_size=1, is_binary=True,
            )

    return _FakeOps()


class TestSchemaTargetParameter:
    def test_vision_analyze_schema_accepts_target(self):
        from tools.vision_tools import VISION_ANALYZE_SCHEMA

        props = VISION_ANALYZE_SCHEMA["parameters"]["properties"]
        assert props["target"]["type"] == "string"
        assert "terminal.default_target" in props["target"]["description"]
        assert "target" not in VISION_ANALYZE_SCHEMA["parameters"]["required"]

    def test_video_analyze_schema_accepts_target(self):
        from tools.vision_tools import VIDEO_ANALYZE_SCHEMA

        props = VIDEO_ANALYZE_SCHEMA["parameters"]["properties"]
        assert props["target"]["type"] == "string"
        assert "target" not in VIDEO_ANALYZE_SCHEMA["parameters"]["required"]

    def test_resolve_context_accepts_target(self):
        from tools.image_source import ResolveContext

        ctx = ResolveContext(task_id="t1", target="devbox")
        assert ctx.target == "devbox"


class TestTargetAwareConfinement:
    """The local-vs-sandbox decision follows the resolved execution target."""

    @pytest.mark.asyncio
    async def test_local_target_reads_host_path(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        importlib.reload(importlib.import_module("tools.terminal_tool"))
        import tools.execution_targets as et

        img = tmp_path / "host_only.png"
        img.write_bytes(PNG)

        with et.execution_target_config_scope(_targets_config()):
            monkeypatch.setenv("TERMINAL_ENV", "docker")  # legacy noise: ignored
            res = await isrc.resolve_image_source(
                str(img), isrc.ResolveContext(target="local"))

        assert res.origin == "file"
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_named_ssh_target_routes_host_path_to_backend_read(
        self, tmp_path, monkeypatch,
    ):
        """A path that exists on the host but not on the target backend must be
        read from the backend (origin 'container'), never from the host."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        import tools.execution_targets as et

        host_img = tmp_path / "host_only.png"
        host_img.write_bytes(PNG)
        calls = []
        fake = _fake_ops(b64_payload=base64.b64encode(PNG).decode(), calls=calls)

        with et.execution_target_config_scope(_targets_config()):
            with patch.object(isrc, "_backend_file_ops", return_value=fake) as bp:
                res = await isrc.resolve_image_source(
                    str(host_img), isrc.ResolveContext(task_id="t1", target="devbox"))

        assert res.origin == "container"
        assert res.data == PNG
        # Delegation happened, scoped to the named target's resolution and the
        # ingest cap.
        assert bp.call_count == 1
        resolution = bp.call_args.args[1]
        assert resolution.target == "devbox"
        assert resolution.backend == "ssh"
        assert calls[0]["max_bytes"] == isrc._MAX_INGEST_BYTES

    @pytest.mark.asyncio
    async def test_named_ssh_target_fails_closed_when_backend_cannot_read(
        self, tmp_path, monkeypatch,
    ):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        import tools.execution_targets as et

        host_secret = tmp_path / "id_rsa"
        host_secret.write_bytes(b"HOST-PRIVATE-KEY")
        fake = _fake_ops(error="File not found: /home/deploy/id_rsa")

        with et.execution_target_config_scope(_targets_config()):
            with patch.object(isrc, "_backend_file_ops", return_value=fake):
                with pytest.raises(isrc.SourceNotFound):
                    await isrc.resolve_image_source(
                        str(host_secret), isrc.ResolveContext(task_id="t1", target="devbox"))

    @pytest.mark.asyncio
    async def test_media_cache_path_still_host_reads_under_named_target(
        self, tmp_path, monkeypatch,
    ):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        import tools.execution_targets as et

        cached = home / "cache" / "images" / "inbound.png"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(PNG)

        with et.execution_target_config_scope(_targets_config()):
            res = await isrc.resolve_image_source(
                str(cached), isrc.ResolveContext(target="devbox"))

        assert res.origin == "file"
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_omitted_target_follows_default_target(self, tmp_path, monkeypatch):
        """No explicit target -> the resolver follows terminal.default_target.
        A remote default_target confines a non-cache path to the backend read."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        import tools.execution_targets as et

        host_img = tmp_path / "host_only.png"
        host_img.write_bytes(PNG)
        fake = _fake_ops(b64_payload=base64.b64encode(PNG).decode())

        config = _targets_config(default_target="devbox")
        with et.execution_target_config_scope(config):
            with patch.object(isrc, "_backend_file_ops", return_value=fake) as bp:
                res = await isrc.resolve_image_source(
                    str(host_img), isrc.ResolveContext(task_id="t1"))

        assert res.origin == "container"
        assert bp.call_args.args[1].target == "devbox"

    @pytest.mark.asyncio
    async def test_unknown_target_fails_closed(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        import tools.execution_targets as et

        img = tmp_path / "x.png"
        img.write_bytes(PNG)

        with et.execution_target_config_scope(_targets_config()):
            with pytest.raises(Exception) as excinfo:
                await isrc.resolve_image_source(
                    str(img), isrc.ResolveContext(target="no-such-target"))
        assert "no-such-target" in str(excinfo.value)


class TestLegacyModeUnchanged:
    """Deployments without terminal.targets keep the historical behavior."""

    @pytest.mark.asyncio
    async def test_legacy_local_reads_host_path(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "local.png"
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_legacy_non_local_exec_reads_via_file_ops(self, tmp_path, monkeypatch):
        """Legacy docker mode: same delegation, default resolution."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        host_img = tmp_path / "host.png"
        host_img.write_bytes(PNG)
        fake = _fake_ops(b64_payload=base64.b64encode(PNG).decode())

        with patch.object(isrc, "_backend_file_ops", return_value=fake) as bp:
            res = await isrc.resolve_image_source(
                str(host_img), isrc.ResolveContext(task_id="t1"))

        assert res.origin == "container"
        resolution = bp.call_args.args[1]
        assert resolution.named is False
        assert resolution.backend == "docker"


class TestReadResultHandling:
    @pytest.mark.asyncio
    async def test_error_result_raises_sourcenotfound(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        import tools.execution_targets as et
        fake = _fake_ops(error="File is too large (99 bytes, limit is 10)")
        with et.execution_target_config_scope(_targets_config()):
            with patch.object(isrc, "_backend_file_ops", return_value=fake):
                with pytest.raises(isrc.SourceNotFound) as excinfo:
                    await isrc.resolve_image_source(
                        "/big.png", isrc.ResolveContext(task_id="t1", target="devbox"))
        assert "too large" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_non_image_payload_raises_notanimage(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        import tools.execution_targets as et
        fake = _fake_ops(b64_payload=base64.b64encode(b"definitely not an image").decode())
        with et.execution_target_config_scope(_targets_config()):
            with patch.object(isrc, "_backend_file_ops", return_value=fake):
                with pytest.raises(isrc.NotAnImage):
                    await isrc.resolve_image_source(
                        "/etc/hosts", isrc.ResolveContext(task_id="t1", target="devbox"))

    @pytest.mark.asyncio
    async def test_empty_payload_raises_sourcenotfound(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        import tools.execution_targets as et
        fake = _fake_ops(b64_payload=None)
        with et.execution_target_config_scope(_targets_config()):
            with patch.object(isrc, "_backend_file_ops", return_value=fake):
                with pytest.raises(isrc.SourceNotFound):
                    await isrc.resolve_image_source(
                        "/gone.png", isrc.ResolveContext(task_id="t1", target="devbox"))


class TestTargetThreading:
    @pytest.mark.asyncio
    async def test_native_fast_path_forwards_target(self, tmp_path, monkeypatch):
        import tools.vision_tools as vt

        captured = {}

        async def fake_resolve(src, ctx, permitted=("image",)):
            captured["ctx"] = ctx
            return SimpleNamespace(data=PNG, mime="image/png", origin="file")

        async def fake_native(image_url, question, task_id=None, region=None, target=None):
            captured["target"] = target
            return {"_multimodal": True}

        monkeypatch.setattr(
            vt, "_should_use_native_vision_fast_path", lambda: True)
        monkeypatch.setattr(
            "tools.image_source.resolve_image_source", fake_resolve)
        monkeypatch.setattr(vt, "_vision_analyze_native", fake_native)

        await vt._handle_vision_analyze(
            {"image_url": "/x.png", "question": "what?", "target": "devbox"},
            task_id="t1",
        )
        assert captured["target"] == "devbox"
