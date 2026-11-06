"""Run one Grok Imagine tool call through the signed-in Grok CLI.

    GrokProvider ──> grok -p (headless) ──> image_gen / image_edit / reference_to_video

Generation runs on the account behind `grok login` (SuperGrok quota), not an
API key. The CLI has no "just call this tool" mode, so every job is a
one-turn agent run that is allowed exactly one Imagine tool and told to call
it with arguments we wrote. What makes that safe and predictable — all
verified against Grok CLI 1.0.40/1.0.41 (see ~/AI/grok-cli-tools):

* ``--tools <tool>`` strips every other built-in tool, so the run cannot touch
  a shell, edit files or browse; ``--allow <tool>`` with
  ``--permission-mode dontAsk`` is what lets an Imagine tool run headlessly
  (without it the call is auto-cancelled: "User cancelled the execution").
* The tools take no output path. Results land in
  ``~/.grok/sessions/<cwd>/<session_id>/{images,videos}/``, so each run gets a
  session id we chose and the file is picked up from there.
* ``streaming-messages-json`` exposes the tool call and its result, so a
  rewritten prompt is logged and a tool error (quota, moderation) is reported
  verbatim instead of the agent's paraphrase of it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from agent import config

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".webm")

# Substrings of the errors Imagine returns when the account cannot generate.
_QUOTA_MARKERS = (
    "spending-limit", "spending limit", "out of credits", "supergrok",
    "rate limit", "rate_limit", "usage limit", "quota", "too many requests",
    "http 429", "upgrade notice",
)
_MODERATION_MARKERS = ("moderation", "content policy", "safety system", "unsafe")

# Codes the worker matches to fail a request once instead of retrying: neither
# an exhausted plan nor a moderation block is something a retry can reach, and
# each retry spends another agent turn on the account.
QUOTA_CODE = "GROK_QUOTA"
MODERATION_CODE = "GROK_MODERATION"
# Video tools refuse to run while the CLI is in zero-data-retention (/privacy)
# mode — a setting only the account owner should change.
ZDR_CODE = "GROK_ZDR"

# The CLI's own HTTP call to api.x.ai fails this way about one time in three
# (2026-10-11: 5 of ~12 media calls, across image_gen, image_edit and video),
# and the same call goes through when repeated. One immediate retry saves the
# worker's backoff for a blip that is not ours.
_TRANSIENT_MARKER = "error sending request for url"
TRANSIENT_RETRIES = 1


class GrokCliError(Exception):
    """A Grok CLI media run failed. ``quota`` is set for plan/limit errors."""

    def __init__(self, message: str, *, quota: bool = False) -> None:
        super().__init__(message)
        self.quota = quota


@dataclass
class GrokRun:
    """What one headless run did."""

    session_id: str
    tool_calls: list[dict] = field(default_factory=list)   # [{"name", "input"}]
    tool_results: list[str] = field(default_factory=list)  # text of each result
    final_text: str = ""
    media: list[Path] = field(default_factory=list)


def _content_text(value) -> str:
    """Flatten a tool_result's content (string, JSON string, block list) to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("[", "{"):
            try:
                return _content_text(json.loads(stripped)) or stripped
            except json.JSONDecodeError:
                pass
        return value
    if isinstance(value, list):
        return "\n".join(t for t in (_content_text(v) for v in value) if t)
    if isinstance(value, dict):
        if "error" in value or "message" in value:
            parts = [str(value.get(k)) for k in ("error", "message") if value.get(k)]
            return ": ".join(parts)
        for key in ("text", "content"):
            if key in value:
                return _content_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_stream(raw: str, session_id: str) -> GrokRun:
    """Parse ``--output-format streaming-messages-json`` (one JSON object per line)."""
    run = GrokRun(session_id=session_id)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind in ("assistant", "user"):
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    run.tool_calls.append({"name": block.get("name"),
                                           "input": block.get("input") or {}})
                elif block.get("type") == "tool_result":
                    run.tool_results.append(_content_text(block.get("content")))
        elif kind == "result":
            run.final_text = str(event.get("result") or "")
    return run


def session_media(session_id: str, exts: tuple, grok_home: Path | None = None) -> list[Path]:
    """Media files grok saved for ``session_id``, oldest first."""
    if not session_id or "/" in session_id or session_id.startswith("."):
        return []
    root = (grok_home or config.GROK_HOME) / "sessions"
    found = [p for p in root.glob(f"*/{session_id}/**/*")
             if p.is_file() and p.suffix.lower() in exts]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def _is_quota(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _QUOTA_MARKERS) or ("403" in low and "credit" in low)


def _is_moderation(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _MODERATION_MARKERS)


def build_instruction(tool: str, args: dict) -> str:
    return (
        f"Call the `{tool}` tool exactly once now, with exactly these arguments. "
        "Copy every value verbatim — do not rewrite, shorten, expand or translate the "
        "prompt, and do not add or drop arguments:\n\n"
        f"```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```\n\n"
        "Do not call any other tool. When it finishes reply with one line: DONE, "
        "or the exact error message if it failed."
    )


def build_command(tool: str, instruction: str, session_id: str, cwd: Path) -> list[str]:
    args = [
        config.GROK_BIN, "--no-auto-update",
        "-p", instruction,
        "--output-format", "streaming-messages-json",
        "--cwd", str(cwd),
        "--session-id", session_id,
        "--tools", tool,
        "--allow", tool,
        "--permission-mode", "dontAsk",
        "--deny", "Bash(*)",
        "--deny", "MCPTool(*)",
        "--no-subagents",
        "--no-memory",
        "--disable-web-search",
        "--max-turns", "3",
    ]
    if config.GROK_AGENT_MODEL:
        args += ["--model", config.GROK_AGENT_MODEL]
    if config.GROK_AGENT_EFFORT:
        args += ["--effort", config.GROK_AGENT_EFFORT]
    return args


async def run_tool(tool: str, args: dict, *, exts: tuple,
                   timeout_s: float | None = None) -> GrokRun:
    """Run ``tool`` with ``args``; return the run with its saved media.

    Repeats the run once when the CLI's request to api.x.ai never got through.
    Raises :class:`GrokCliError` when nothing was produced, carrying the tool's
    own error text when there is one.
    """
    for attempt in range(TRANSIENT_RETRIES + 1):
        try:
            return await _run_tool_once(tool, args, exts=exts, timeout_s=timeout_s)
        except GrokCliError as e:
            if _TRANSIENT_MARKER not in str(e) or attempt == TRANSIENT_RETRIES:
                raise
            logger.warning("grok: %s request did not reach api.x.ai, retrying once", tool)
    raise AssertionError("unreachable")


async def _run_tool_once(tool: str, args: dict, *, exts: tuple,
                         timeout_s: float | None = None) -> GrokRun:
    timeout_s = timeout_s or config.GROK_TIMEOUT_S
    session_id = str(uuid.uuid4())
    cwd = config.GROK_WORK_DIR
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = build_command(tool, build_instruction(tool, args), session_id, cwd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise GrokCliError(f"grok: could not start `{config.GROK_BIN}`: {e}") from e
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        # The render may still finish server-side; whatever reached the session
        # dir before the kill is still worth taking.
        media = session_media(session_id, exts)
        if media:
            logger.warning("grok: %s timed out after %.0fs but left media — using it",
                           tool, timeout_s)
            return GrokRun(session_id=session_id, media=media)
        raise GrokCliError(f"grok: {tool} timed out after {timeout_s:.0f}s")

    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace").strip()
    run = parse_stream(out, session_id)
    run.media = session_media(session_id, exts)

    called = [c for c in run.tool_calls if c["name"] == tool]
    if called:
        sent = called[0]["input"].get("prompt")
        if sent is not None and sent != args.get("prompt"):
            logger.warning("grok: %s prompt was rewritten by the agent: %r", tool, sent[:200])

    if run.media:
        return run

    detail = next((r for r in reversed(run.tool_results) if r.strip()), "")
    if proc.returncode != 0:
        detail = detail or err[-500:] or out.strip()[-500:] or f"exit code {proc.returncode}"
        hint = " — run `grok login`" if any(
            t in detail.lower() for t in ("auth", "login", "401")) else ""
        quota = _is_quota(detail)
        code = f"{QUOTA_CODE}: " if quota else "grok: "
        raise GrokCliError(f"{code}{tool} failed (rc={proc.returncode}): {detail}{hint}",
                           quota=quota)
    if not called:
        raise GrokCliError(
            f"grok: the agent did not call {tool}: {(run.final_text or err)[:300]}")
    detail = detail or run.final_text or "no media file was saved"
    if _is_quota(detail):
        raise GrokCliError(
            f"{QUOTA_CODE}: Imagine quota/plan limit on the signed-in Grok account: {detail}",
            quota=True)
    if "zero data retention" in detail.lower() or "(zdr)" in detail.lower():
        raise GrokCliError(
            f"{ZDR_CODE}: {tool} is unavailable while Grok CLI privacy (ZDR) mode is on — "
            f"turn off /privacy in `grok`, or configure a ZDR video bucket: {detail}")
    if _is_moderation(detail):
        raise GrokCliError(f"{MODERATION_CODE}: Imagine refused the request: {detail}")
    raise GrokCliError(f"grok: {tool} produced no media: {detail[:500]}")
