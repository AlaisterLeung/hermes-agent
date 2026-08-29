import base64

import pytest
from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

from acp_adapter.server import HermesACPAgent, _content_blocks_to_openai_user_content


def test_acp_image_blocks_convert_to_openai_multimodal_content():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="What is in this image?"),
        ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png"),
    ])

    assert content == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        },
    ]


def test_text_only_acp_blocks_stay_string_for_legacy_prompt_path():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="/help"),
    ])

    assert content == "/help"


def test_acp_resource_link_file_is_inlined_as_text(tmp_path):
    attached = tmp_path / "notes.md"
    attached.write_text("# Notes\n\nAttached file body", encoding="utf-8")

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="Please read this file"),
        ResourceContentBlock(
            type="resource_link",
            name="notes.md",
            title="Project notes",
            uri=attached.as_uri(),
            mimeType="text/markdown",
        ),
    ])

    assert content == (
        "Please read this file\n"
        "[Attached file: Project notes (notes.md)]\n"
        f"URI: {attached.as_uri()}\n\n"
        "# Notes\n\nAttached file body"
    )


def test_acp_resource_link_missing_locally_falls_back_to_named_target(
    tmp_path, monkeypatch,
):
    """The editor attaches workspace files by editor-host path; with a named SSH
    execution target pinned, the attachment must be read through the target
    backend instead of failing on the ACP server's local filesystem."""
    from types import SimpleNamespace

    import acp_adapter.server as server_mod

    remote_path = "/home/editor/Documents/repos/Centralcord/plans/architecture.md"
    read_calls: list[str] = []

    class _FakeReadResult:
        def __init__(self, content="", error=None, file_size=0):
            self.content = content
            self.error = error
            self.file_size = file_size

    class _FakeFileOps:
        def __init__(self, env):
            self.env = env

        def read_file(self, path, offset=1, limit=2000):
            read_calls.append(path)
            return _FakeReadResult(content="# Architecture\n\nremote body",
                                   file_size=28)

    fake_env = SimpleNamespace(cwd="/home/editor")
    monkeypatch.setattr(
        server_mod, "_pinned_execution_target_resolution",
        lambda: SimpleNamespace(target="cachy650", backend="ssh", named=True),
    )
    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id: _FakeFileOps(fake_env),
    )

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="See the attached plan"),
        ResourceContentBlock(
            type="resource_link",
            name="architecture.md",
            uri=f"file://{remote_path}",
            mimeType="text/markdown",
        ),
    ])

    assert read_calls == [remote_path]
    assert content == (
        "See the attached plan\n"
        "[Attached file: architecture.md]\n"
        f"URI: file://{remote_path}\n\n"
        "# Architecture\n\nremote body"
    )


def test_acp_resource_link_target_read_failure_reports_error(tmp_path, monkeypatch):
    """When the target read also fails, the prompt carries the reason
    instead of a silent success."""
    from types import SimpleNamespace

    import acp_adapter.server as server_mod

    class _FakeReadResult:
        content = ""
        error = "File not found: /gone.md"
        file_size = 0

    class _FakeFileOps:
        env = SimpleNamespace(cwd="/")

        def read_file(self, path, offset=1, limit=2000):
            return _FakeReadResult()

    monkeypatch.setattr(
        server_mod, "_pinned_execution_target_resolution",
        lambda: SimpleNamespace(target="cachy650", backend="ssh", named=True),
    )
    monkeypatch.setattr(
        "tools.file_tools._get_file_ops", lambda task_id: _FakeFileOps(),
    )

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="read it"),
        ResourceContentBlock(
            type="resource_link",
            name="gone.md",
            uri="file:///gone.md",
            mimeType="text/markdown",
        ),
    ])

    assert "File not found: /gone.md" in content


def test_acp_resource_link_local_read_still_wins_without_target(tmp_path):
    """No pinned named target → local filesystem read, unchanged behavior
    even when the local file exists."""
    attached = tmp_path / "local.md"
    attached.write_text("local body", encoding="utf-8")

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="hi"),
        ResourceContentBlock(
            type="resource_link",
            name="local.md",
            uri=attached.as_uri(),
            mimeType="text/markdown",
        ),
    ])

    assert "local body" in content




@pytest.mark.asyncio
async def test_initialize_advertises_image_prompt_capability():
    response = await HermesACPAgent().initialize()

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities.image is True


# 1x1 transparent PNG — smallest valid image payload for inlining tests.
_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)
