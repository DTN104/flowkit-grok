"""Tests for the grok media provider (Grok Imagine via the signed-in Grok CLI).

The `grok` binary is faked at asyncio.create_subprocess_exec: the fake reads
the argv the provider built, writes a media file where the real CLI does
(~/.grok/sessions/<cwd>/<session_id>/{images,videos}/), and prints the
streaming-messages-json lines the real CLI prints. That pins the contract the
provider depends on without spending Imagine quota.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.sdk.services import grok_provider as gp
from agent.sdk.services.provider_base import (
    KIND_AUDIO,
    KIND_EDIT_IMAGE,
    KIND_IMAGE,
    KIND_UPSCALE,
    KIND_VIDEO,
    KIND_VIDEO_REFS,
    ProviderJob,
)
from agent.sdk.services.registry import ProviderRegistry
from agent.sdk.services.result_handler import parse_result
from agent.services import grok_cli_media as gcm
from agent.worker._parsing import _is_uuid

JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8

QUOTA_ERROR = json.dumps({
    "error": "tool_execution_failed",
    "message": 'Image generation failed with HTTP 403 Forbidden: {"code":"personal-team-blocked:'
               'spending-limit","error":"You have run out of credits or need a Grok subscription."}',
})


def _proc(stdout: str, returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


ZDR_ERROR = json.dumps({
    "error": "tool_execution_failed",
    "message": "Video generation tools are unavailable under zero data retention (ZDR). "
               "To enable, either turn off /privacy mode to disable ZDR or supply a "
               "user-hosted storage bucket.",
})


SEND_ERROR = json.dumps({
    "error": "tool_execution_failed",
    "message": "Image generation API request failed: error sending request for url "
               "(https://api.x.ai/v1/images/generations)",
})


class FakeGrok:
    """Stands in for the grok CLI. ``outcome``: media | quota | zdr | no_call."""

    def __init__(self, grok_home: Path, outcome: str = "media"):
        self.grok_home = grok_home
        self.outcome = outcome
        self.argv: tuple = ()
        self.kwargs: dict = {}

    def arg(self, flag: str) -> str:
        return self.argv[self.argv.index(flag) + 1]

    def tool_input(self) -> dict:
        instruction = self.arg("-p")
        return json.loads(instruction.split("```json\n", 1)[1].split("\n```", 1)[0])

    async def __call__(self, *argv, **kwargs):
        self.argv, self.kwargs = argv, kwargs
        tool = self.arg("--tools")
        lines = [{"type": "system", "subtype": "init", "tools": [tool]}]
        if self.outcome == "no_call":
            lines.append({"type": "result", "subtype": "success", "result": "I won't do that."})
            return _proc("\n".join(json.dumps(x) for x in lines))

        lines.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "c1", "name": tool, "input": self.tool_input()}]}})
        if self.outcome == "quota":
            result = QUOTA_ERROR
        elif self.outcome == "zdr":
            result = ZDR_ERROR
        elif self.outcome == "send_error":
            result = SEND_ERROR
        else:
            sub, name, data = (("videos", "1.mp4", MP4) if tool == "reference_to_video"
                               else ("images", "1.jpg", JPG))
            out = self.grok_home / "sessions" / "%2Fwork" / self.arg("--session-id") / sub / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            result = "saved"
        lines.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": result}]}})
        lines.append({"type": "result", "subtype": "success", "result": "DONE"})
        return _proc("\n".join(json.dumps(x) for x in lines))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gcm.config, "GROK_HOME", tmp_path / "grok_home")
    monkeypatch.setattr(gcm.config, "GROK_WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(gcm.config, "GROK_AGENT_MODEL", "")
    monkeypatch.setattr(gcm.config, "GROK_UPLOAD_TO_FLOW", False)
    monkeypatch.setattr(gcm.config, "GROK_VIDEO_DURATIONS", (6, 10))
    monkeypatch.setattr(gcm.config, "GROK_VIDEO_SECONDS", 6)
    monkeypatch.setattr(gcm.config, "GROK_VIDEO_RESOLUTION", "720p")
    return tmp_path


@pytest.fixture
def provider(env):
    prov = gp.GrokProvider()
    prov.output_dir = env / "out"
    return prov


def fake(env, outcome="media"):
    grok = FakeGrok(env / "grok_home", outcome)
    return grok, patch("agent.services.grok_cli_media.asyncio.create_subprocess_exec",
                       new=AsyncMock(side_effect=grok.__call__))


@pytest.fixture
def image(tmp_path):
    def _make(name):
        p = tmp_path / name
        p.write_bytes(JPG)
        return "file://" + str(p)
    return _make


async def test_text_image_uses_image_gen_only(env, provider):
    grok, patched = fake(env)
    with patched:
        result = await provider.run(ProviderJob(
            job_id="img1", kind=KIND_IMAGE, prompt="a fox", orientation="VERTICAL",
            extra={"composition": "portrait full-body"}))

    gen = parse_result(result, "GENERATE_IMAGE")
    assert gen.success, result
    assert _is_uuid(gen.media_id)
    assert gen.url == "file://" + str((provider.output_dir / "img1.jpg").resolve())
    assert (provider.output_dir / "img1.jpg").read_bytes() == JPG

    assert grok.arg("--tools") == "image_gen"
    assert grok.arg("--allow") == "image_gen"
    assert grok.arg("--permission-mode") == "dontAsk"
    assert "Bash(*)" in grok.argv and "MCPTool(*)" in grok.argv
    assert "--always-approve" not in grok.argv
    assert grok.arg("--cwd") == str(env / "work")
    assert grok.kwargs["stdin"] is not None  # DEVNULL, never inherited
    args = grok.tool_input()
    assert args == {"prompt": "a fox\n\nComposition: portrait full-body", "aspect_ratio": "9:16"}


async def test_image_with_refs_uses_image_edit_with_local_paths(env, provider, image):
    grok, patched = fake(env)
    refs = (image("a.jpg"), image("b.jpg"))
    with patched:
        result = await provider.run(ProviderJob(
            job_id="img2", kind=KIND_IMAGE, prompt="they share a meal",
            orientation="HORIZONTAL", reference_urls=refs,
            extra={"reference_names": ["Mai", "Lan"]}))

    assert parse_result(result, "GENERATE_IMAGE").success, result
    assert grok.arg("--tools") == "image_edit"
    args = grok.tool_input()
    assert args["image"] == [str(Path(r[len("file://"):]).resolve()) for r in refs]
    assert args["aspect_ratio"] == "16:9"
    assert "Mai, Lan" in args["prompt"] and args["prompt"].endswith("they share a meal")
    assert "padding" not in args["prompt"]   # two refs: aspect_ratio is honoured, no canvas


async def test_single_ref_is_padded_to_the_scene_shape(env, provider, image):
    """image_edit keeps a lone source's aspect ratio, so a portrait ref for a
    landscape scene is letterboxed onto a 16:9 canvas first."""
    grok, patched = fake(env)
    pad = AsyncMock(return_value="/work/inputs/one_canvas.jpg")
    with patched, patch.object(gp, "pad_to_canvas", new=pad):
        await provider.run(ProviderJob(
            job_id="one", kind=KIND_IMAGE, prompt="p", orientation="HORIZONTAL",
            reference_urls=(image("a.jpg"),)))
    assert pad.call_args.args[2] == (1280, 720)
    args = grok.tool_input()
    assert args["image"] == ["/work/inputs/one_canvas.jpg"]
    assert "padding" in args["prompt"]


async def test_edit_image_puts_the_source_first(env, provider, image):
    grok, patched = fake(env)
    with patched:
        await provider.run(ProviderJob(
            job_id="ed1", kind=KIND_EDIT_IMAGE, prompt="make it night",
            source_url=image("src.jpg"), reference_urls=(image("ref.jpg"),)))
    args = grok.tool_input()
    assert [Path(p).name for p in args["image"]] == ["src.jpg", "ref.jpg"]
    assert args["prompt"] == "make it night"


async def test_chained_video_pins_first_and_last_frame(env, provider, image):
    grok, patched = fake(env)
    with patched:
        result = await provider.run(ProviderJob(
            job_id="vid1", kind=KIND_VIDEO, prompt="she walks left to right",
            orientation="VERTICAL", start_url=image("s.jpg"), end_url=image("e.jpg"),
            extra={"duration_s": 8}))

    gen = parse_result(result, "GENERATE_VIDEO")
    assert gen.success, result
    assert _is_uuid(gen.media_id)
    assert (provider.output_dir / "vid1.mp4").read_bytes() == MP4
    assert grok.arg("--tools") == "reference_to_video"
    args = grok.tool_input()
    assert Path(args["first_frame"]).name == "s.jpg"
    assert Path(args["last_frame"]).name == "e.jpg"
    assert args["duration"] == 10          # 8s rounded up to a length the CLI takes
    assert args["aspect_ratio"] == "9:16"
    assert args["resolution_name"] == "720p"
    assert "images" not in args


async def test_r2v_passes_refs_as_images(env, provider, image):
    grok, patched = fake(env)
    with patched:
        result = await provider.run(ProviderJob(
            job_id="r2v", kind=KIND_VIDEO_REFS, prompt="p", orientation="HORIZONTAL",
            reference_urls=(image("a.jpg"), image("b.jpg"))))
    assert parse_result(result, "GENERATE_VIDEO").success, result
    args = grok.tool_input()
    assert [Path(p).name for p in args["images"]] == ["a.jpg", "b.jpg"]
    assert args["duration"] == 6 and "first_frame" not in args


async def test_quota_error_is_reported_verbatim(env, provider):
    _, patched = fake(env, "quota")
    with patched:
        result = await provider.run(ProviderJob(job_id="q", kind=KIND_IMAGE, prompt="x"))
    assert result["error"].startswith("GROK_QUOTA: ")
    assert "spending-limit" in result["error"]


async def test_send_failure_is_retried_once_then_reported(env, provider):
    """The CLI's call to api.x.ai drops often and works when repeated."""
    calls = []

    async def flaky(*argv, **kwargs):
        calls.append(argv)
        grok = FakeGrok(env / "grok_home", "media" if len(calls) > 1 else "send_error")
        return await grok(*argv, **kwargs)

    with patch("agent.services.grok_cli_media.asyncio.create_subprocess_exec",
               new=AsyncMock(side_effect=flaky)):
        result = await provider.run(ProviderJob(job_id="f", kind=KIND_IMAGE, prompt="x"))
    assert parse_result(result, "GENERATE_IMAGE").success, result
    assert len(calls) == 2

    _, patched = fake(env, "send_error")
    with patched as spawn:
        result = await provider.run(ProviderJob(job_id="f2", kind=KIND_IMAGE, prompt="x"))
    assert "error sending request" in result["error"]
    assert spawn.await_count == 2          # one retry, not a loop


async def test_zdr_privacy_mode_is_named_as_such(env, provider, image):
    """Real answer from Grok CLI 1.0.41 with /privacy on (2026-10-11)."""
    _, patched = fake(env, "zdr")
    with patched:
        result = await provider.run(ProviderJob(
            job_id="z", kind=KIND_VIDEO, prompt="x", start_url=image("s.jpg")))
    assert result["error"].startswith("GROK_ZDR: ")
    assert "/privacy" in result["error"]


async def test_agent_that_never_calls_the_tool_is_an_error(env, provider):
    _, patched = fake(env, "no_call")
    with patched:
        result = await provider.run(ProviderJob(job_id="n", kind=KIND_IMAGE, prompt="x"))
    assert "did not call image_gen" in result["error"]


async def test_video_without_start_frame_errors_before_spawning(env, provider):
    grok, patched = fake(env)
    with patched as spawn:
        result = await provider.run(ProviderJob(job_id="v", kind=KIND_VIDEO, prompt="x"))
    assert result == {"error": "No vertical image for scene"}
    spawn.assert_not_called()


async def test_missing_input_file_is_an_error(env, provider):
    _, patched = fake(env)
    with patched:
        result = await provider.run(ProviderJob(
            job_id="v", kind=KIND_VIDEO, prompt="x", start_url="file:///nope/missing.jpg"))
    assert "image not found" in result["error"]


async def test_upload_to_flow_uses_flow_media_id(env, monkeypatch):
    monkeypatch.setattr(gcm.config, "GROK_UPLOAD_TO_FLOW", True)
    flow = MagicMock(connected=True)
    flow.upload_image = AsyncMock(return_value={"_mediaId": "11111111-2222-3333-4444-555555555555"})
    prov = gp.GrokProvider(flow)
    prov.output_dir = env / "out"
    _, patched = fake(env)
    with patched:
        result = await prov.run(ProviderJob(job_id="up", kind=KIND_IMAGE, prompt="x",
                                            extra={"project_id": "p1"}))
    gen = parse_result(result, "GENERATE_IMAGE")
    assert gen.media_id == "11111111-2222-3333-4444-555555555555"
    assert flow.upload_image.call_args.kwargs["project_id"] == "p1"


def test_upload_to_flow_needs_the_extension(env, monkeypatch):
    monkeypatch.setattr(gcm.config, "GROK_UPLOAD_TO_FLOW", True)
    with patch("agent.sdk.services.grok_provider.shutil.which", return_value="/bin/grok"):
        assert gp.GrokProvider(MagicMock(connected=False)).is_available() is False
        assert gp.GrokProvider(MagicMock(connected=True)).is_available() is True


def test_capabilities(provider):
    for kind in (KIND_IMAGE, KIND_EDIT_IMAGE, KIND_VIDEO, KIND_VIDEO_REFS):
        assert provider.capabilities.supports(kind)
    with patch("agent.sdk.services.grok_provider.shutil.which", return_value="/bin/grok"):
        assert "does not support" in provider.check(ProviderJob(job_id="u", kind=KIND_UPSCALE))
        assert "does not support" in provider.check(ProviderJob(job_id="a", kind=KIND_AUDIO))
    with patch("agent.sdk.services.grok_provider.shutil.which", return_value=None):
        assert "not on PATH" in provider.check(ProviderJob(job_id="i", kind=KIND_IMAGE))
        assert provider.is_available() is False


@pytest.mark.parametrize("requested,expected", [
    (None, 6), (4, 6), (6, 6), (8, 10), (10, 10), (15, 10), ("bad", 6),
])
def test_snap_duration(requested, expected):
    assert gp.snap_duration(requested, (6, 10), 6) == expected


async def test_register_existing_image_mints_uuid(provider, image):
    url = image("ref.jpg")
    result = await provider.register_existing_image(url, name="Mai")
    media = result["data"]["media"]
    assert _is_uuid(media[0]["name"]) and result["data"]["url"] == url


def test_parse_stream_flattens_tool_result_blocks():
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "image_gen", "input": {"prompt": "p"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content":
            json.dumps([{"type": "content", "content": {"type": "text", "text": "boom"}}])}]}},
        {"type": "result", "result": "done"},
    ]
    raw = "\n".join(json.dumps(e) for e in events) + "\nnot json, ignored"
    run = gcm.parse_stream(raw, "sid")
    assert run.tool_calls == [{"name": "image_gen", "input": {"prompt": "p"}}]
    assert run.tool_results == ["boom"]
    assert run.final_text == "done"


def test_registry_includes_grok():
    assert isinstance(ProviderRegistry().get("grok"), gp.GrokProvider)
