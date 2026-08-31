"""Tests for tools/image_source.py — the unified vision image-source resolver.

Covers the delivery contract (data:/http/file/local/container source handling,
size cap, magic-byte sniff) AND the terminal-backend confinement security model
(GHSA-gpxw-6wxv-w3qq): under a non-local backend, host reads are confined to the
media caches and every other path is read inside the sandbox via exec-read.
"""

import base64
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# Minimal valid 1x1 PNG bytes. Resolver validation requires a decodable fixture.
PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
JPEG = b"\xff\xd8\xff" + b"\x00" * 64
CORRUPT_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAFElEQVR4nGP8z8Dwn4EIwESJ5gAAVQ4CH1evYJQAAAAASUVORK5CYII="
)


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    importlib.reload(isrc)
    return isrc


@pytest.fixture(autouse=True)
def _no_real_sandbox_bringup(monkeypatch):
    """Neutralize real sandbox bring-up so unit tests never spawn a real
    ssh/docker env. The resolver delegates to tools.file_tools._get_file_ops;
    patching that indirection (which _reload does not touch) covers the
    per-test image_source reload. The delegation tests override it."""
    import tools.image_source as isrc_src
    monkeypatch.setattr(
        isrc_src, "_backend_file_ops",
        lambda task_id, resolution, target=None: _FailingFileOps(),
        raising=False,
    )


class _FailingFileOps:
    """Default stub when a test doesn't expect a backend read at all."""

    def read_file_bytes(self, path, max_bytes=None):
        return SimpleNamespace(
            error=f"unexpected backend read of {path!r}", base64_content=None)


class TestDataUrl:
    @pytest.mark.asyncio
    async def test_valid_data_url_resolves_to_bytes(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(PNG).decode()
        res = await isrc.resolve_image_source(
            f"data:image/png;base64,{b64}", isrc.ResolveContext())
        assert res.data == PNG
        assert res.mime == "image/png"
        assert res.origin == "data"

    @pytest.mark.asyncio
    async def test_non_image_data_url_rejected(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(isrc.NotAnImage):
            await isrc.resolve_image_source(
                f"data:text/plain;base64,{b64}", isrc.ResolveContext())

    @pytest.mark.asyncio
    async def test_corrupt_png_rejected_at_resolver_boundary(self, tmp_path, monkeypatch):
        """A PNG-shaped but undecodable payload never becomes a resolved image."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "corrupt.png"
        img.write_bytes(CORRUPT_PNG)
        with pytest.raises(isrc.NotAnImage):
            await isrc.resolve_image_source(str(img), isrc.ResolveContext())


class TestLocalBackend:
    @pytest.mark.asyncio
    async def test_local_backend_reads_any_host_path(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "outside" / "pic.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(PNG)
        res = await isrc.resolve_image_source(str(img), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"


    @pytest.mark.asyncio
    async def test_bare_relative_path_resolves(self, tmp_path, monkeypatch):
        """A cwd-relative bare filename ('pic.png') is a valid local source —
        main accepted it; the resolver must not regress it (PR review)."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        img = tmp_path / "pic.png"
        img.write_bytes(PNG)
        monkeypatch.chdir(tmp_path)
        res = await isrc.resolve_image_source("pic.png", isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"


    @pytest.mark.asyncio
    async def test_svg_passes_through_for_rasterization(self, tmp_path, monkeypatch):
        """SVG has no raster magic bytes but is passed through with mime
        image/svg+xml so the vision call sites can rasterize it to PNG."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        svg = tmp_path / "art.svg"
        svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        svg.write_bytes(svg_bytes)
        res = await isrc.resolve_image_source(str(svg), isrc.ResolveContext())
        assert res.mime == "image/svg+xml"
        assert res.data == svg_bytes


class TestNonLocalBackendConfinement:
    """The security model: under a sandbox backend, host reads are confined to
    the media caches; every other path is read inside the sandbox."""

    @pytest.mark.asyncio
    async def test_media_cache_path_host_read(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        cached = home / "cache" / "images" / "inbound.png"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(PNG)
        # No sandbox env needed — a cache path is host-read directly.
        res = await isrc.resolve_image_source(str(cached), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_desktop_upload_images_dir_host_read(self, tmp_path, monkeypatch):
        """Desktop/clipboard uploads under ``HERMES_HOME/images`` are host-read.

        Regression for #69575: uploads land in the flat top-level ``images/``
        dir (not ``cache/images``). Under a sandbox backend the vision resolver
        must permit reading them host-side — otherwise it falls through to the
        task-id-less sandbox reader and fails with "not reachable inside the
        sandbox".
        """
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        upload = home / "images" / "upload_20260722_181019_1.png"
        upload.parent.mkdir(parents=True)
        upload.write_bytes(PNG)
        # No sandbox env: an uploads path must be host-read directly, not routed
        # to the in-sandbox exec-read.
        res = await isrc.resolve_image_source(str(upload), isrc.ResolveContext())
        assert res.data == PNG
        assert res.origin == "file"

    @pytest.mark.asyncio
    async def test_host_secret_outside_cache_routes_to_sandbox_not_host(self, tmp_path, monkeypatch):
        """A non-cache host path (e.g. /etc/passwd) must NOT be host-read — it
        routes to the in-backend read, which reads the CONTAINER's file."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        # A real host file outside the caches, holding a "secret".
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY-DO-NOT-LEAK")

        # Fake backend file-ops: its exec-read returns a *different* (container)
        # image, proving we read the container filesystem, not the host secret.
        container_png_b64 = base64.b64encode(PNG).decode()
        calls = {}

        class _FakeOps:
            def read_file_bytes(self, path, max_bytes=None):
                calls["path"] = path
                calls["max_bytes"] = max_bytes
                return SimpleNamespace(
                    error=None, base64_content=container_png_b64, is_binary=True)

        with patch("tools.image_source._backend_file_ops", return_value=_FakeOps()):
            res = await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))

        # Read came from the backend exec-read, returning the container image —
        # the host secret bytes never appear.
        assert res.origin == "container"
        assert res.data == PNG
        assert b"HOST-PRIVATE-KEY" not in res.data
        assert calls["max_bytes"] == isrc._MAX_INGEST_BYTES  # bounded read

    @pytest.mark.asyncio
    async def test_non_cache_path_fails_closed_without_sandbox(self, tmp_path, monkeypatch):
        """No reachable backend env -> refuse rather than fall back to a host read."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY")

        class _NoEnvOps:
            def read_file_bytes(self, path, max_bytes=None):
                return SimpleNamespace(
                    error="No terminal environment is available for task 't1'",
                    base64_content=None)

        with patch("tools.image_source._backend_file_ops", return_value=_NoEnvOps()):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))

    @pytest.mark.asyncio
    async def test_symlink_in_cache_pointing_outside_is_not_host_read(self, tmp_path, monkeypatch):
        """A symlink planted inside a cache dir that points at a host secret must
        not be host-read (resolve() escapes the cache) — it routes to sandbox."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        secret = tmp_path / "outside" / "id_rsa"
        secret.parent.mkdir(parents=True)
        secret.write_bytes(b"HOST-PRIVATE-KEY")
        cache_dir = home / "cache" / "images"
        cache_dir.mkdir(parents=True)
        link = cache_dir / "sneaky.png"
        try:
            link.symlink_to(secret)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unsupported")

        # Fails closed (no sandbox) rather than host-reading the symlink target.
        class _NoEnvOps:
            def read_file_bytes(self, path, max_bytes=None):
                return SimpleNamespace(error="no environment", base64_content=None)

        with patch("tools.image_source._backend_file_ops", return_value=_NoEnvOps()):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(str(link), isrc.ResolveContext(task_id="t1"))


class TestExecReadSafety:
    @pytest.mark.asyncio
    async def test_exec_read_is_bounded_and_redirect_safe(self, tmp_path, monkeypatch):
        """The read is delegated with the ingest byte cap; leading-dash paths
        are handed over as a plain string (ShellFileOperations handles shell
        escaping via its own arg-quoting, not here)."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        captured = {}

        class _FakeOps:
            def read_file_bytes(self, path, max_bytes=None):
                captured["path"] = path
                captured["max_bytes"] = max_bytes
                return SimpleNamespace(
                    error=None, base64_content=base64.b64encode(PNG).decode(),
                    is_binary=True)

        with patch("tools.image_source._backend_file_ops", return_value=_FakeOps()):
            await isrc.resolve_image_source(
                "/workspace/-i-etc-shadow.png", isrc.ResolveContext(task_id="t1"))
        assert captured["max_bytes"] == isrc._MAX_INGEST_BYTES
        # The path arrives verbatim — no quoting/option mangling at this layer.
        assert captured["path"] == "/workspace/-i-etc-shadow.png"


    @pytest.mark.asyncio
    async def test_exec_read_error_result_raises(self, tmp_path, monkeypatch):
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        class _FailOps:
            def read_file_bytes(self, path, max_bytes=None):
                return SimpleNamespace(
                    error="File not found: /workspace/nope.png", base64_content=None)

        with patch("tools.image_source._backend_file_ops", return_value=_FailOps()):
            with pytest.raises(isrc.SourceNotFound):
                await isrc.resolve_image_source(
                    "/workspace/nope.png", isrc.ResolveContext(task_id="t1"))

    @pytest.mark.asyncio
    async def test_exec_read_single_delegation_no_resolver_retry(self, tmp_path, monkeypatch):
        """Retry policy lives in the env bring-up (_get_file_ops), not the
        resolver: read_file_bytes is called exactly once per resolve."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        calls = {"n": 0}

        class _FakeOps:
            def read_file_bytes(self, path, max_bytes=None):
                calls["n"] += 1
                return SimpleNamespace(
                    error=None, base64_content=base64.b64encode(PNG).decode(),
                    is_binary=True)

        with patch("tools.image_source._backend_file_ops", return_value=_FakeOps()):
            res = await isrc.resolve_image_source(
                "/workspace/cold.png", isrc.ResolveContext(task_id="t1"))
        assert res.origin == "container"
        assert res.data == PNG
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_exec_read_failure_includes_diagnostic(self, tmp_path, monkeypatch):
        """When the backend read fails, its error text is surfaced so the user
        can tell 'no such file' from 'permission denied' from 'container never
        came up'."""
        home = tmp_path / "hermes"
        isrc = _reload(monkeypatch, home)
        monkeypatch.setenv("TERMINAL_ENV", "docker")

        class _FailOps:
            def read_file_bytes(self, path, max_bytes=None):
                return SimpleNamespace(
                    error="head: can't open '/workspace/missing.png': No such file or directory",
                    base64_content=None)

        with patch("tools.image_source._backend_file_ops", return_value=_FailOps()):
            with pytest.raises(isrc.SourceNotFound) as excinfo:
                await isrc.resolve_image_source(
                    "/workspace/missing.png", isrc.ResolveContext(task_id="t1"))
        # Diagnostic surfaced — the user can act on it.
        assert "No such file or directory" in str(excinfo.value)


class TestSvgNormalization:
    """SVG resolves end-to-end: the resolver passes it through as
    image/svg+xml and the vision call sites rasterize it to PNG via
    _normalize_to_supported_image (PR #52688, folded in)."""

    @pytest.mark.asyncio
    async def test_svg_rasterized_when_converter_available(self, tmp_path, monkeypatch):
        from tools import vision_tools as vt
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        svg = tmp_path / "art.svg"
        svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>')

        def fake_rasterize(svg_path, out_path):
            out_path.write_bytes(PNG)
            return True

        with patch.object(vt, "_rasterize_svg_to_png", side_effect=fake_rasterize):
            res = await isrc.resolve_image_source(str(svg), isrc.ResolveContext())
            assert res.mime == "image/svg+xml"
            path, mime, err = vt._normalize_to_supported_image(svg, "image/svg+xml")
        assert err is None
        assert mime == "image/png"
        assert path.read_bytes() == PNG
        path.unlink()

    def test_svg_actionable_error_when_no_converter(self, tmp_path, monkeypatch):
        from tools import vision_tools as vt
        _reload(monkeypatch, tmp_path / "hermes")
        svg = tmp_path / "art.svg"
        svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
        with patch.object(vt, "_rasterize_svg_to_png", return_value=False):
            path, mime, err = vt._normalize_to_supported_image(svg, "image/svg+xml")
        assert path is None
        assert "rasterizer" in err


class TestLazySandboxBringUp:
    """Issue #62825: under a non-local backend, the FIRST vision_analyze of a
    session (before any terminal command) must trigger the environment
    bring-up itself instead of failing with 'no active sandbox session'. The
    resolver satisfies this by delegating to _get_file_ops, which lazily
    creates the env — so the contract under test is that the delegation
    happens with the resolver's resolution and task context."""

    @pytest.mark.asyncio
    async def test_first_read_delegates_to_file_ops_bring_up(self, tmp_path, monkeypatch):
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "ssh")

        delegation = {}

        class _LazyOps:
            """Mimics _get_file_ops: bring-up happens on construction; the
            read then succeeds."""

            def __init__(self, task_id, resolution, target=None):
                delegation["task_id"] = task_id
                delegation["resolution"] = resolution
                delegation["target"] = target

            def read_file_bytes(self, path, max_bytes=None):
                delegation["path"] = path
                return SimpleNamespace(
                    error=None,
                    base64_content=base64.b64encode(PNG).decode(),
                    is_binary=True,
                )

        monkeypatch.setattr(
            "tools.image_source._backend_file_ops", _LazyOps, raising=False)

        res = await isrc.resolve_image_source(
            "/tmp/test.png", isrc.ResolveContext(task_id="t1"))

        assert delegation["task_id"] == "t1"
        assert delegation["path"] == "/tmp/test.png"
        assert delegation["resolution"].named is False  # legacy ssh resolution
        assert res.origin == "container"
        assert res.data == PNG

    @pytest.mark.asyncio
    async def test_bringup_that_yields_no_env_still_fails_closed(self, tmp_path, monkeypatch):
        """If the bring-up can't produce an env, the resolver still refuses
        rather than falling back to a host read."""
        isrc = _reload(monkeypatch, tmp_path / "hermes")
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        secret = tmp_path / "id_rsa"
        secret.write_bytes(b"HOST-PRIVATE-KEY")

        class _NoEnvOps:
            def read_file_bytes(self, path, max_bytes=None):
                return SimpleNamespace(
                    error="No terminal environment is available for task 't1'",
                    base64_content=None)

        monkeypatch.setattr(
            "tools.image_source._backend_file_ops",
            lambda task_id, resolution, target=None: _NoEnvOps(),
            raising=False,
        )

        with pytest.raises(isrc.SourceNotFound):
            await isrc.resolve_image_source(str(secret), isrc.ResolveContext(task_id="t1"))
