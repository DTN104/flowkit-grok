"""GrokProvider — MediaProvider adapter for Grok Imagine via the Grok CLI.

    Flow Kit worker ──> GrokProvider ──> grok -p ──> Grok Imagine (signed-in account)

Uses the account behind `grok login` (SuperGrok quota), never an API key. The
CLI plumbing lives in :mod:`agent.services.grok_cli_media`; this module maps a
:class:`ProviderJob` onto Imagine's tools and hands the result back in the
Flow-shaped dict the rest of the pipeline parses, so scenes rendered here look
exactly like Flow output downstream (review, concat, narration, upload).

Finished media is copied to ``output/_shared/grok/`` and recorded as a
``file://`` URL, which never expires. Media ids are minted UUIDs unless
``GROK_UPLOAD_TO_FLOW=1``, which uploads every image to Flow and stores Flow's
own media_id — so a Grok image can feed a Flow video, and Flow images feed
Grok videos either way (the provider reads them by URL).

Job kind → Imagine tool:

* image, no refs   → ``image_gen``          (prompt + aspect ratio)
* image with refs  → ``image_edit``         (entity refs as source images, ≤5)
* edit_image       → ``image_edit``         (source image first, then refs)
* video (i2v)      → ``reference_to_video`` (``first_frame``; a chained
                                             scene's end frame → ``last_frame``)
* video_refs (r2v) → ``reference_to_video`` (refs in ``images``, ≤14)
* upscale, audio   — not offered.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from agent import config
from agent.sdk.services.assistant_provider import _image_result, _video_result
from agent.sdk.services.provider_base import (
    KIND_EDIT_IMAGE,
    KIND_IMAGE,
    KIND_VIDEO,
    KIND_VIDEO_REFS,
    MediaProvider,
    ProviderCapabilities,
    ProviderJob,
)
from agent.services.grok_cli_media import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    GrokCliError,
    run_tool,
)
from agent.utils.paths import file_url_to_path

logger = logging.getLogger(__name__)

_ASPECT = {"VERTICAL": "9:16", "HORIZONTAL": "16:9"}
_MAX_EDIT_IMAGES = 5
_MAX_VIDEO_REFS = 14
# Canvas a lone reference is padded onto so image_edit renders the right frame.
_CANVAS = {"VERTICAL": (720, 1280), "HORIZONTAL": (1280, 720)}
_MIME = {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN") or "ffmpeg"


async def pad_to_canvas(src: str, dest: Path, size: tuple[int, int]) -> str:
    """Fit ``src`` inside a grey canvas of ``size``; return the new path.

    image_edit keeps a single source image's aspect ratio and ignores
    ``aspect_ratio`` (verified 2026-10-11: a portrait ref asked for 16:9 came
    back 9:16). Entity refs are portrait and scenes are often landscape, so a
    lone ref is letterboxed onto a canvas of the scene's shape first.
    """
    w, h = size
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x808080")
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg_bin(), "-y", "-v", "error", "-i", src, "-vf", vf, "-frames:v", "1", str(dest),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dest.is_file():
        raise OSError(f"ffmpeg could not pad {Path(src).name}: {err.decode()[-300:]}")
    return str(dest.resolve())


def snap_duration(requested, allowed: tuple, default: int) -> int:
    """Round a scene duration up to a clip length the CLI accepts."""
    try:
        want = float(requested) if requested else float(default)
    except (TypeError, ValueError):
        want = float(default)
    if not allowed:
        return max(1, int(round(want)))
    for d in allowed:
        if d >= want:
            return d
    return allowed[-1]


class GrokProvider(MediaProvider):
    """Renders media with Grok Imagine through the signed-in Grok CLI."""

    name = "grok"
    display_name = "Grok Imagine (Grok CLI account)"

    def __init__(self, flow_client=None) -> None:
        self._flow_client = flow_client
        self.output_dir = config.SHARED_OUTPUT_DIR / "grok"
        self.capabilities = ProviderCapabilities(
            generate_image=True,
            edit_image=True,
            generate_video_i2v=True,
            generate_video_r2v=True,
            upscale=False,
            generate_audio=False,
            max_concurrent=config.GROK_MAX_CONCURRENT,
            cooldown_s=config.GROK_COOLDOWN_S,
        )

    def is_available(self) -> bool:
        """Available while the grok binary is on PATH. Sign-in and quota are
        only knowable by trying — a failed job reports the CLI's own error."""
        if not shutil.which(config.GROK_BIN):
            return False
        if config.GROK_UPLOAD_TO_FLOW and not self._flow_connected():
            return False
        return True

    def check(self, job: ProviderJob) -> str | None:
        if not shutil.which(config.GROK_BIN):
            return f"grok provider: `{config.GROK_BIN}` not on PATH — install the Grok CLI and run `grok login`"
        return super().check(job)

    def _flow_connected(self) -> bool:
        try:
            return bool(self._flow_client and self._flow_client.connected)
        except Exception:
            return False

    # ── MediaProvider interface ──

    async def run(self, job: ProviderJob) -> dict:
        try:
            if job.kind in (KIND_IMAGE, KIND_EDIT_IMAGE):
                return await self._run_image(job)
            if job.kind in (KIND_VIDEO, KIND_VIDEO_REFS):
                return await self._run_video(job)
        except GrokCliError as e:
            logger.error("grok: job %s (%s) failed: %s", job.job_id[:12], job.kind, e)
            return {"error": str(e)}
        except (OSError, ValueError, httpx.HTTPError) as e:
            # Unreadable or unreachable input image, full disk …
            logger.error("grok: job %s (%s) failed: %s", job.job_id[:12], job.kind, e)
            return {"error": f"grok: {e}"}
        return {"error": f"grok provider does not support job kind '{job.kind}'"}

    async def _run_image(self, job: ProviderJob) -> dict:
        extra = job.extra or {}
        prompt = job.prompt
        composition = extra.get("composition")
        if composition:
            prompt = f"{prompt}\n\nComposition: {composition}"
        aspect = _ASPECT.get(job.orientation, "auto")

        sources: list[str] = []
        if job.kind == KIND_EDIT_IMAGE:
            if not job.source_url:
                return {"error": "grok: no source image to edit"}
            sources.append(job.source_url)
        sources += list(job.reference_urls)
        if len(sources) > _MAX_EDIT_IMAGES:
            logger.info("grok: job %s — %d images, image_edit takes %d; extra refs dropped",
                        job.job_id[:12], len(sources), _MAX_EDIT_IMAGES)
            sources = sources[:_MAX_EDIT_IMAGES]

        if not sources:
            tool, args = "image_gen", {"prompt": prompt, "aspect_ratio": aspect}
        else:
            paths = [await self._local_input(job.job_id, i, u) for i, u in enumerate(sources)]
            padded = job.kind == KIND_IMAGE and len(paths) == 1 and job.orientation in _CANVAS
            if padded:
                paths = [await pad_to_canvas(
                    paths[0], config.GROK_WORK_DIR / "inputs" / f"{job.job_id}_canvas.jpg",
                    _CANVAS[job.orientation])]
            if job.kind == KIND_IMAGE:
                # Flow Kit scene prompts describe action only; appearance comes
                # from the refs. Say so, or image_edit treats ref 1 as a photo
                # to retouch rather than a cast to stage a new scene with.
                names = [n for n in extra.get("reference_names") or [] if n]
                cast = f" ({', '.join(names)})" if names else ""
                prompt = (f"Create a new image of this scene, featuring the subject(s) from "
                          f"the reference image(s){cast} with their exact appearance, outfit "
                          f"and style. "
                          + ("Fill the entire frame with the scene — the grey canvas around "
                             "the reference is padding, not part of the picture. " if padded else "")
                          + f"Scene: {prompt}")
            tool, args = "image_edit", {"prompt": prompt, "image": paths, "aspect_ratio": aspect}

        run = await run_tool(tool, args, exts=IMAGE_EXTS)
        src = run.media[-1]
        url = await self._save(job.job_id, src)
        mid = await self._media_id(job, src)
        logger.info("grok: image %s ready via %s (media_id=%s)", job.job_id[:12], tool, mid[:8])
        return _image_result(mid, url)

    async def _run_video(self, job: ProviderJob) -> dict:
        extra = job.extra or {}
        prefix = "vertical" if job.orientation == "VERTICAL" else "horizontal"
        duration = snap_duration(job.duration_s or extra.get("duration_s"),
                                 config.GROK_VIDEO_DURATIONS, config.GROK_VIDEO_SECONDS)
        args: dict = {
            "prompt": job.prompt,
            "aspect_ratio": _ASPECT.get(job.orientation, "16:9"),
            "duration": duration,
            "resolution_name": config.GROK_VIDEO_RESOLUTION,
        }
        if job.kind == KIND_VIDEO_REFS:
            refs = list(job.reference_urls) or ([job.start_url] if job.start_url else [])
            if not refs:
                return {"error": "No valid reference images for r2v"}
            refs = refs[:_MAX_VIDEO_REFS]
            args["images"] = [await self._local_input(job.job_id, i, u)
                              for i, u in enumerate(refs)]
        else:
            if not job.start_url:
                return {"error": f"No {prefix} image for scene"}
            args["first_frame"] = await self._local_input(job.job_id, 0, job.start_url)
            if job.end_url:
                # Chaining works natively here: Imagine interpolates between
                # pinned first and last frames.
                args["last_frame"] = await self._local_input(job.job_id, 1, job.end_url)

        run = await run_tool("reference_to_video", args, exts=VIDEO_EXTS)
        url = await self._save(job.job_id, run.media[-1])
        mid = str(uuid.uuid4())
        logger.info("grok: video %s ready (%ss, media_id=%s)", job.job_id[:12], duration, mid[:8])
        return _video_result(mid, url)

    # ── helpers ──

    async def _local_input(self, job_id: str, index: int, ref: str) -> str:
        """Absolute local path for an input image: the tools read paths, and
        Flow's signed URLs are only fetchable from this machine anyway."""
        ref = (ref or "").strip()
        if not ref:
            raise ValueError("empty image reference")
        parsed = urlparse(ref)
        if parsed.scheme == "file" or not parsed.scheme:
            path = file_url_to_path(ref) if parsed.scheme == "file" else Path(ref)
            if not path or not path.is_file():
                raise FileNotFoundError(f"image not found: {ref}")
            return str(path.resolve())

        inputs = config.GROK_WORK_DIR / "inputs"
        if parsed.scheme == "data":
            header, _, payload = ref.partition(",")
            ext = ".png" if "png" in header else ".webp" if "webp" in header else ".jpg"
            data = base64.b64decode(payload)
        elif parsed.scheme in ("http", "https"):
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(ref)
                resp.raise_for_status()
            data = resp.content
            ctype = resp.headers.get("content-type", "")
            ext = ".png" if "png" in ctype else ".webp" if "webp" in ctype else ".jpg"
        else:
            raise ValueError(f"unsupported image reference: {ref[:80]}")

        dest = inputs / f"{job_id}_{index}{ext}"

        def _write() -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

        await asyncio.to_thread(_write)
        return str(dest.resolve())

    async def _save(self, job_id: str, src: Path) -> str:
        dest = self.output_dir / f"{job_id}{src.suffix.lower()}"

        def _copy() -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        await asyncio.to_thread(_copy)
        return "file://" + str(dest.resolve())

    async def _media_id(self, job: ProviderJob, image: Path) -> str:
        """A minted UUID, or Flow's media_id when GROK_UPLOAD_TO_FLOW is on."""
        if not config.GROK_UPLOAD_TO_FLOW:
            return str(uuid.uuid4())
        return await self._upload_to_flow(image, name=job.job_id,
                                          project_id=(job.extra or {}).get("project_id", ""))

    async def _upload_to_flow(self, image: Path, *, name: str, project_id: str) -> str:
        if not self._flow_connected():
            raise GrokCliError("grok: GROK_UPLOAD_TO_FLOW=1 but the Flow extension is not connected")
        data = await asyncio.to_thread(image.read_bytes)
        mime = _MIME.get(image.suffix.lower(), "image/jpeg")
        result = await self._flow_client.upload_image(
            base64.b64encode(data).decode(), mime_type=mime,
            project_id=project_id, file_name=f"{name}{image.suffix.lower()}")
        mid = result.get("_mediaId")
        if not mid:
            media = (result.get("data") or {}).get("media") if isinstance(result.get("data"), dict) else None
            mid = media.get("name") if isinstance(media, dict) else None
        if not mid:
            raise GrokCliError(f"grok: upload to Flow returned no media id: {str(result)[:300]}")
        logger.info("grok: uploaded %s to Flow (media_id=%s)", image.name, mid[:8])
        return mid

    async def register_existing_image(
        self, url: str, *, name: str = "", project_id: str = ""
    ) -> dict:
        """Give an image we already have a media id: a UUID, or Flow's own."""
        parsed = urlparse(url or "")
        if parsed.scheme in ("http", "https") and not config.GROK_UPLOAD_TO_FLOW:
            return {"data": {"media": [{"name": str(uuid.uuid4())}], "url": url}}
        try:
            local = Path(await self._local_input(uuid.uuid4().hex[:12], 0, url))
        except (OSError, ValueError, httpx.HTTPError) as e:
            return {"error": f"grok: image not readable for '{name}': {e}"}
        canonical = url if parsed.scheme in ("http", "https", "file") else "file://" + str(local)
        try:
            mid = (await self._upload_to_flow(local, name=name or local.stem, project_id=project_id)
                   if config.GROK_UPLOAD_TO_FLOW else str(uuid.uuid4()))
        except GrokCliError as e:
            return {"error": str(e)}
        logger.info("grok: registered existing image '%s' → media_id=%s", name, mid[:8])
        return {"data": {"media": [{"name": mid}], "url": canonical}}
