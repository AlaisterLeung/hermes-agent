"""Single resolver for every media source (data:/http(s)/file/local/container) -> bytes + mime.

<<<<<<< HEAD
Everything funnels through :func:`resolve_image_source` so size and magic-byte checks run
exactly once. Images are the default; video callers opt in via ``permitted=("video",)``.
Security (GHSA-gpxw-6wxv-w3qq): under a non-local backend vision is confined like the file
tools — host-read only inside a media cache (bind-mounted into the sandbox), anything else is
exec-read *inside the sandbox*, so ``vision_analyze('/etc/passwd')`` never reads the host.
=======
All source handling (data:/http(s)/file/local/container) funnels through
:func:`resolve_image_source` so size and magic-byte checks are enforced exactly
once.  Returns raw bytes (not a path): the downstream step is base64 -> data URL
(RFC 2397) and provider base64 content blocks.

Images are the default and the historical purpose. Callers whose argument
takes video opt in via ``permitted=("video",)`` — the same confinement and
credential-guard pipeline applies, and only the type check at the end differs
(extension-table typing plus an mp4 magic sniff, rather than image magic
bytes). Every existing call site keeps the image-only default unchanged.

Security (terminal-backend confinement, GHSA-gpxw-6wxv-w3qq): under a non-local
terminal backend the file tools are confined to the sandbox (SECURITY.md 2.2),
but vision read images host-side. This resolver enforces the same boundary:

  * local backend            -> read any host path (chosen posture, unchanged)
  * non-local backend:
      path in a media cache   -> host-read (the gateway/download caches live on
                                 the host and are bind-mounted into the sandbox)
      path anywhere else      -> read the bytes *inside the sandbox* via exec-read
                                 (the agent can already ``cat`` any container file;
                                 this stays within the sandbox boundary and never
                                 reaches the host's ``/etc/passwd`` / ``~/.ssh``).

The governing backend comes from the execution-target resolver (named targets
included): an explicit ``ResolveContext.target`` selects that named target, an
omitted target follows ``terminal.default_target`` (same semantics as the file
tools). Legacy deployments without ``terminal.targets`` keep the historical
``TERMINAL_ENV``-driven behavior — the resolver's legacy flat mode preserves
that precedence. In-sandbox reads delegate to the file tools'
``ShellFileOperations.read_file_bytes`` (stat probe, non-regular-file
rejection, byte cap, terminal-fence-leak stripping, validated base64), which
already knows how to look up or lazily create the environment for any target.

So a prompt-injected ``vision_analyze('/etc/passwd')`` under Docker reads the
*container's* file (what every other tool sees), not the host's — no escape —
while container-only images (tmpfs ``/workspace``, root-owned) are still
deliverable. This is the unified delivery + confinement model: the same
mechanism that fixes "vision can't see container files" also closes the escape.
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Raw-bytes INGEST budget: deliberately the 50MB download cap, NOT the 20MB provider
# payload cap — that one is enforced post-resize at the call sites.
_MAX_INGEST_BYTES = 50 * 1024 * 1024


class ImageResolutionError(Exception):
    def __init__(self, message: str, *, src: str = "", origin: str = ""):
        super().__init__(message)
        self.src, self.origin = src, origin


class UnsupportedScheme(ImageResolutionError): ...
class SourceUnsafe(ImageResolutionError): ...  # SSRF / path-allowlist
class SourceTooLarge(ImageResolutionError): ...
class SourceNotFound(ImageResolutionError): ...
class NotAnImage(ImageResolutionError): ...


@dataclass
class ResolveContext:
    task_id: Optional[str] = None
    # Optional named execution target (tools/execution_targets.py). None
    # follows ``terminal.default_target`` — the same semantics the file tools
    # apply to an omitted ``target`` argument.
    target: Optional[str] = None


@dataclass
class _BackendPlan:
    """Resolved execution target + the confinement decision it implies."""

    resolution: Any
    host_reads_any_path: bool


def _resolve_backend_plan(ctx: ResolveContext) -> _BackendPlan:
    """Resolve the governing execution target for this media read.

    ``resolve_execution_target`` reads configuration lazily (importing tool
    schemas never touches user config), so calling it per read is cheap and
    always current — live reloads and ACP/CLI target pinning are honored.
    Host path reads are permitted only when the resolved backend actually
    runs on this host (``backend == "local"``), whatever route selected it.
    """
    from tools.execution_targets import resolve_execution_target

    resolution = resolve_execution_target(ctx.target)
    host_reads_any_path = resolution.backend == "local"
    return _BackendPlan(resolution=resolution, host_reads_any_path=host_reads_any_path)


@dataclass
class ResolvedImage:
    data: bytes
    mime: str
    origin: str  # one of: data | http | file | local | container


# Explicit URL scheme ("ftp://", "s3://"). Bare Windows drive paths lack the "//".
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


async def resolve_image_source(
    src: str, ctx: ResolveContext, *, permitted: tuple = ("image",)) -> ResolvedImage:
    if not isinstance(src, str) or not src.strip():
        raise SourceNotFound("image_url is required", src=str(src))
    s = src.strip()
    if s.startswith("data:"):
        data, mime = _resolve_data_url(s)
        return _finalize(data, mime, "data", s, permitted)
    if s.startswith(("http://", "https://")):
        reason = _http_block_reason(s)
        if reason:
            raise SourceUnsafe(reason, src=s)
        return _finalize(await _download_to_bytes(s), "", "http", s, permitted)
    if _SCHEME_RE.match(s) and not s.lower().startswith("file://"):
        raise UnsupportedScheme(
            "Unrecognized image source scheme. Use an http(s) URL, a local "
            "file path, a file:// URI, or a data: URL.",
            src=s)
    # Everything else is a filesystem path — including bare relative names like "pic.png"
    # (a path-shape gate here regressed them once).
    candidate = s[len("file://"):] if s.lower().startswith("file://") else s
    p = Path(os.path.expanduser(candidate))
<<<<<<< HEAD
    host_target = _permitted_host_read_target(p, ctx)
=======
    # Confinement decision (see module docstring). Under a non-local backend
    # a path is host-readable ONLY if it lands in a media cache (after
    # translating a container-visible cache path back to its host mount);
    # every other path is read inside the sandbox via exec-read, so a host
    # path outside the caches never yields the host's bytes. The governing
    # backend comes from the execution-target resolution (named targets and
    # ``terminal.default_target`` included), not from a legacy env sniff.
    plan = _resolve_backend_plan(ctx)
    host_target = _permitted_host_read_target(p, ctx, plan)
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))
    if host_target is not None and host_target.is_file():
        _guard_credential_read(host_target, s)
        data = await asyncio.to_thread(host_target.read_bytes)
        return _finalize(data, "", "file", s, permitted)
<<<<<<< HEAD
    if _is_local_terminal_backend():
        # Any path was host-readable, so a miss means the file doesn't exist.
        raise SourceNotFound(f"media file not found: '{p}'", src=s, origin="file")
    return await _resolve_container_fallback(p, ctx, s, permitted)
=======
    if plan.host_reads_any_path:
        # Local backend: any path was host-readable, so a miss simply means
        # the file doesn't exist — no sandbox to fall back to.
        raise SourceNotFound(f"media file not found: '{p}'", src=s, origin="file")
    # Not a permitted host read (or the host file is absent) -> read the
    # bytes inside the backend. Under a sandbox this reads that backend's
    # filesystem, never the host's.
    return await _resolve_container_fallback(p, ctx, s, permitted, plan)
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))


def _guard_credential_read(host_target: Path, src: str) -> None:
    """Shared credential-read guard: refuse secret-bearing files (.env, auth.json) with a specific
    error. Guard import is best-effort; a real block always propagates."""
    try:
        # Shared credential-read guard (agent.file_safety, #57698): refuse secret-bearing files (.env,
        # auth.json, ...) with an intentional, specific error instead of relying on the magic-byte sniff to
        # reject them incidentally. Same chokepoint the image-gen/video-gen provider plugins enforce on
        # model-supplied local paths. Import is best-effort (guard unavailability must not break image
        # loading); a real block always propagates.
        from agent.file_safety import raise_if_read_blocked
    except Exception:  # noqa: BLE001 — guard unavailable: proceed
        return
    try:
        raise_if_read_blocked(str(host_target))
    except ValueError as exc:
        raise SourceUnsafe(str(exc), src=src, origin="file")


def _resolve_data_url(s: str) -> tuple[bytes, str]:
    header, _, payload = s.partition(",")
    if ";base64" not in header:
        raise NotAnImage("data: URL must be base64-encoded", src=s[:64])
    declared = header[len("data:"):].split(";", 1)[0].strip() or "application/octet-stream"
    # Cheap pre-decode size gate on the encoded length (~4/3 expansion).
    if (len(payload) * 3) // 4 > _MAX_INGEST_BYTES:
        raise SourceTooLarge("data: URL exceeds size limit", src=s[:64])
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise NotAnImage(f"invalid base64 in data: URL: {exc}", src=s[:64])
    return data, declared  # real mime verified in _finalize via magic bytes


def _http_block_reason(url: str) -> Optional[str]:
    """Block reason, or None when allowed. Refuses policy-blocked URLs BEFORE any network I/O;
    ``_download_image`` re-checks per attempt and against the final redirect target (intentional)."""
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access
    if not is_safe_url(url):
        return "blocked: unsafe or private URL"
    if blocked := check_website_access(url):
        return blocked.get("message") or "blocked by website policy"
    return None


async def _download_to_bytes(url: str) -> bytes:
    import tempfile
    from tools.vision_tools import _download_image
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        # Enforces the 50MB stream cap, redirect SSRF guard, and website policy.
        await _download_image(url, tmp)
        return await asyncio.to_thread(tmp.read_bytes)
    except PermissionError as exc:  # website policy block
        raise SourceUnsafe(str(exc), src=url, origin="http")
    finally:
        tmp.unlink(missing_ok=True)


def _is_local_terminal_backend() -> bool:
<<<<<<< HEAD
    """True when the terminal backend runs directly on the host (keys off ``TERMINAL_ENV``)."""
=======
    """True when the default terminal backend runs directly on the host.

    Legacy default-target sniff only — kept for tests and external callers.
    The resolver itself keys confinement off the execution-target resolution
    (see :func:`_resolve_backend_plan`), which honors named targets and
    ``terminal.default_target``; the legacy flat mode preserves the historical
    ``TERMINAL_ENV`` precedence.
    """
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))
    return os.getenv("TERMINAL_ENV", "local").strip().lower() in ("local", "")


# Host-side media caches: the only host paths vision may read under a non-local backend
# (gateway inbound media + the tools' own download temp dirs).
_MEDIA_CACHE_SUBDIRS = (
    "cache",  # cache/images, cache/vision, cache/video(s), cache/audio
    "images",  # desktop/clipboard/PDF uploads (tui_gateway)
    "image_cache", "audio_cache", "video_cache", "temp_vision_images", "temp_video_files")


def _media_cache_roots() -> list:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return [home / sub for sub in _MEDIA_CACHE_SUBDIRS]


<<<<<<< HEAD
def _permitted_host_read_target(p: Path, ctx: ResolveContext) -> Optional[Path]:
    """Host path to read, or ``None`` (caller exec-reads inside the sandbox instead).
=======
def _permitted_host_read_target(
    p: Path, ctx: ResolveContext, plan: Optional[_BackendPlan] = None,
) -> Optional[Path]:
    """Return the host path to read, or ``None`` if a host read is not permitted.
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))

    Local backend: any path. Non-local: only paths inside a media cache root (a
    container-visible cache path is first translated back to its host mount).
    """
    if plan is None:
        plan = _resolve_backend_plan(ctx)
    if plan.host_reads_any_path:
        try:
            return p.resolve()
        except Exception:  # noqa: BLE001 — unresolved path: let is_file() fail downstream
            return p
    from tools.credential_files import from_agent_visible_cache_path
    try:
        real = Path(from_agent_visible_cache_path(str(p))).resolve()
    except Exception:  # noqa: BLE001 — cannot resolve -> not a safe host read
        return None
    if any(real.is_relative_to(root.resolve()) for root in _media_cache_roots()):
        return real
    return None


<<<<<<< HEAD
def _get_active_env(task_id: Optional[str]):
    if not task_id:
        return None
    try:
        from tools.terminal_tool_lifecycle import get_active_env
        return get_active_env(task_id)
    except Exception:
        return None


def _ensure_container_env(task_id: Optional[str]) -> None:
    """Lazily bring up the sandbox before an in-sandbox read (vision may be a session's first
    action). Best-effort: failure leaves the env absent and the caller hits the fail-closed error.

    Unlike the terminal tool, vision never triggered environment creation, so a session whose first action
    is ``vision_analyze`` on a container-only path under a non-local backend found no active env and failed
    — until a terminal command happened to create one (issue #62825).
    """
    if not task_id:
        return
    try:
        from tools.terminal_tool import ensure_task_env
        ensure_task_env(task_id)
    except Exception:
        pass


async def _resolve_container_fallback(
    p: Path, ctx: ResolveContext, src: str, permitted: tuple = ("image",)) -> ResolvedImage:
    """Read the bytes inside the sandbox; fail-closed when no env exists (a non-cache host
    path under a sandbox must never leak via a host fallback).

    Cold-start retry: under Docker the first exec against a fresh container can fail (empty
    pipe) while a second succeeds. On final failure the container's output is folded into the
    error so "no such file" / "permission denied" / "never came up" are distinguishable.

    We retry once with a short delay before giving up, so callers don't see "could not read inside the
    sandbox" on a file that is verifiably readable on the immediate retry. See #76566.
    """
    import shlex
    # Bring the sandbox up on demand: without this, the first vision_analyze of a session (before any
    # terminal command) has no active env to read from under a non-local backend (issue #62825).
    _ensure_container_env(ctx.task_id)
    env = _get_active_env(ctx.task_id)
    if env is None:
=======
def _backend_file_ops(
    task_id: Optional[str], resolution: Any, target: Optional[str] = None,
) -> Any:
    """Fetch (or lazily create) the backend's ShellFileOperations.

    Module-level indirection so tests can stub the delegation without touching
    ``tools.file_tools`` internals. ``_get_file_ops`` shares the terminal
    tool's per-task creation locks, so a vision-triggered bring-up (issue
    #62825 behavior, now target-aware) can't race a concurrent terminal call.
    ``target`` must accompany ``resolution``: the create-time publish guard
    re-resolves the *named target* to detect config edits mid-creation, and
    omitting the name makes it re-resolve the default target instead (scope
    mismatch -> "Execution target changed while its environment was being
    created").
    """
    from tools.file_tools import _get_file_ops

    return _get_file_ops(
        task_id or "default", target, _resolution=resolution,
    )


async def _resolve_container_fallback(
    p: Path, ctx: ResolveContext, src: str, permitted: tuple = ("image",),
    plan: Optional[_BackendPlan] = None,
) -> ResolvedImage:
    """Read the image bytes inside the backend (fail-closed when none exists).

    Reached when a host read is not permitted or the host file is absent. The
    agent can already ``cat`` any file on the backend (file_tools reads
    root-owned mode-600 files this way), so this stays within the same sandbox
    boundary and never touches the host filesystem.

    The read delegates to the file tools' ``ShellFileOperations.read_file_bytes``
    via ``tools.file_tools._get_file_ops`` — the same mechanism the ACP adapter
    uses to inline attachments through a pinned execution target. That path
    already knows how to look up or lazily create the environment for any
    execution target (named or default), probes the file with ``stat`` (so
    directories and special files are rejected before a read is attempted),
    caps the read, strips terminal-fence leaks from the stream, and validates
    the base64. ``max_bytes`` bounds the transfer just like the historical
    in-line ``head -c`` pipeline it replaces.

    Fail-closed: if the backend env cannot be brought up we refuse rather than
    falling back to a host read, so a non-cache host path under a sandbox never
    leaks.
    """
    if plan is None:
        plan = _resolve_backend_plan(ctx)

    # _backend_file_ops looks up (or lazily creates, issue #62825) the
    # environment for this task under the resolved target. Blocking backend
    # I/O — keep it off the event loop so a multi-MB base64 read doesn't stall
    # every other coroutine.
    def _read() -> Any:
        ops = _backend_file_ops(
            ctx.task_id, plan.resolution, target=ctx.target,
        )
        return ops.read_file_bytes(str(p), max_bytes=_MAX_INGEST_BYTES)

    result = await asyncio.to_thread(_read)
    if getattr(result, "error", None):
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))
        raise SourceNotFound(
            f"could not read '{p}' inside the backend: {result.error}",
            src=src, origin="container")
<<<<<<< HEAD
    # Bound the read INSIDE the sandbox: head -c caps at ingest-limit+1 (+1 distinguishes "at the
    # cap" from "over") so /dev/zero can't stream unbounded base64 into host memory. The input
    # redirect avoids argv (leading-dash paths); tr -d instead of GNU-only base64 -w0 (BusyBox).
    cmd = f"head -c {_MAX_INGEST_BYTES + 1} < {shlex.quote(str(p))} | base64 | tr -d '\\n'"
    last_res: dict = {"returncode": 1, "output": ""}
    for attempt in range(2):
        last_res = await asyncio.to_thread(env.execute, cmd)
        if last_res.get("returncode", 1) == 0:
            break
        if attempt == 0:
            await asyncio.sleep(0.15)  # covers Docker exec warm-up in practice
    if last_res.get("returncode", 1) != 0:
        diag = (last_res.get("output") or "").strip().splitlines()
        first = next((ln.strip() for ln in diag if ln.strip()), "")
        suffix = f" ({first[:200]})" if first else ""
        raise SourceNotFound(f"could not read '{p}' inside the sandbox{suffix}", src=src, origin="container")
=======
    b64 = getattr(result, "base64_content", None)
    if not b64:
        raise SourceNotFound(
            f"backend returned no data for '{p}'",
            src=src, origin="container")
>>>>>>> c733415018 (feat(vision): add target argument for named execution target media resolution (#9))
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise NotAnImage(f"backend returned non-image data for '{p}': {exc}", src=src)
    return _finalize(data, "", "container", src, permitted)


def _finalize(
    data: bytes, declared_mime: str, origin: str, src: str, permitted: tuple = ("image",)
) -> ResolvedImage:
    """Chokepoint: 50MB ingest cap + type check. Images by magic bytes; video (opt-in) by
    extension + mp4 sniff — enough because every downstream consumer re-validates."""
    from tools.vision_tools_image_prep import _detect_image_mime_type_from_bytes
    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("media exceeds size limit", src=src, origin=origin)
    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is not None:
        if "image" not in permitted:
            raise NotAnImage("source is an image, but this argument takes a video", src=src, origin=origin)
        return ResolvedImage(data=data, mime=sniffed, origin=origin)
    if "image" in permitted and b"<svg" in data[:4096].lower():
        # Pass SVG through — call sites rasterize it to PNG before embedding.
        return ResolvedImage(data=data, mime="image/svg+xml", origin=origin)
    if "video" in permitted:
        video_mime = _detect_video_mime(data, src)
        if video_mime is not None:
            return ResolvedImage(data=data, mime=video_mime, origin=origin)
        raise NotAnImage("source is not a recognized video (mp4 expected)", src=src, origin=origin)
    raise NotAnImage("source is not a recognized image", src=src, origin=origin)


def _detect_video_mime(data: bytes, src: str) -> Optional[str]:
    """Video MIME from the extension table, else the ISO base-media ``ftyp`` magic at
    offset 4 (covers extensionless data: URLs / query-string URLs)."""
    from urllib.parse import urlsplit
    from tools.vision_tools import _detect_video_mime_type
    path_part = urlsplit(src).path if _SCHEME_RE.match(src) else src
    by_extension = _detect_video_mime_type(Path(path_part))
    if by_extension is not None:
        return by_extension
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    return None


async def resolve_local_source_to_data_url(
    src: str, task_id: Optional[str], *, permitted: tuple = ("image",)) -> str:
    """Convert a path-like media source into a ``data:`` URL via the resolver.

    Dispatch-layer chokepoint for generation tools so providers never read model-supplied paths
    off the HOST under a sandbox. URL-shaped sources pass through; callers apply this only under
    a non-local backend.
    """
    s = (src or "").strip()
    if not s or s.lower().startswith(("http://", "https://", "data:")):
        return src
    resolved = await resolve_image_source(s, ResolveContext(task_id=task_id), permitted=permitted)
    encoded = base64.b64encode(resolved.data).decode("ascii")
    mime = resolved.mime or "application/octet-stream"
    return f"data:{mime};base64,{encoded}"
