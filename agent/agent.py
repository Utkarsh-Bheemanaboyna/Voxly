"""Voxly writing agent (LiveKit Agents + a tool-calling LLM).

The agent joins the Voxly room, receives spoken commands as text, and lets the
LLM decide which tools to call. The tools are executed on the user's Mac via
LiveKit RPC (see client/agent_link.py), so the LLM can read the selection or
recent dictation and write the result back at the cursor.

Example: user selects a messy paragraph, holds Right Cmd, says "fix the grammar"
  LLM -> get_selected_text()          (RPC -> Mac reads selection via AX)
  LLM -> write_at_cursor(fixed_text)  (RPC -> Mac replaces selection via AX)
  LLM -> "Fixed grammar."             (status text back to the client)

Run (needs a LiveKit server, e.g. `livekit-server --dev`):
  python agent.py dev
Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY
     VOXLY_LLM (default gpt-4o-mini), VOXLY_LLM_BASE_URL (Groq/Ollama/any
     OpenAI-compatible endpoint, optional)
"""
from __future__ import annotations

import json
import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    room_io,
)
from livekit.plugins import openai

INSTRUCTIONS = """You are Voxly, a writing assistant inside a macOS dictation app.
The user's message is a spoken command, already transcribed (it may contain
small transcription errors). You edit text in the app the user is working in.

Tools:
- get_selected_text: the text the user has highlighted ("" if none)
- get_recent_dictation: what the user recently dictated
- write_at_cursor: insert text at the cursor; replaces the selection if any

How to work:
1. Fetch the source text first: use get_selected_text when the user says
   "this"/"selection"/"fix the grammar"; use get_recent_dictation when they say
   "what I just said"/"my last message". If the selection is empty, fall back to
   recent dictation.
2. Produce the result yourself:
   - grammar edits: minimal changes; fix grammar, spelling, punctuation; keep
     the user's meaning, tone and wording otherwise.
   - summaries: short and faithful; no new facts.
   - other rewrites (shorter, formal, bullet points): follow the instruction.
3. Call write_at_cursor exactly once with ONLY the final text (no quotes, no
   preamble, no explanation).
4. Then reply with a status of at most 6 words, e.g. "Fixed grammar."
If there is no text to work on, do not call write_at_cursor; say so briefly."""


class VoxlyAgent(Agent):
    def __init__(self, user_identity: str):
        super().__init__(instructions=INSTRUCTIONS)
        self.user_identity = user_identity

    async def _rpc(self, method: str, payload: dict | None = None) -> str:
        room = get_job_context().room
        return await room.local_participant.perform_rpc(
            destination_identity=self.user_identity,
            method=method,
            payload=json.dumps(payload or {}),
            response_timeout=5.0,
        )

    @function_tool
    async def get_selected_text(self, context: RunContext) -> str:
        """Return the text currently selected in the user's focused app, or an empty string."""
        return await self._rpc("get_selected_text")

    @function_tool
    async def get_recent_dictation(self, context: RunContext, count: int = 1) -> str:
        """Return the user's most recent dictated segments, oldest first.

        Args:
            count: how many recent segments to return (1-20).
        """
        return await self._rpc("get_recent_dictation", {"count": max(1, min(count, 20))})

    @function_tool
    async def write_at_cursor(self, context: RunContext, text: str) -> str:
        """Insert text at the user's cursor, replacing the current selection if there is one.

        Args:
            text: the final text to insert, with no quotes or commentary.
        """
        method = await self._rpc("write_at_cursor", {"text": text})
        return f"written via {method}"


# livekit-agents takes HTTPS_PROXY from the environment and ignores NO_PROXY.
# For a LiveKit server on localhost, set VOXLY_LOCAL_LIVEKIT=1 to connect directly.
server = AgentServer(http_proxy=None) if os.getenv("VOXLY_LOCAL_LIVEKIT") else AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    llm_kwargs = {"model": os.getenv("VOXLY_LLM", "gpt-4o-mini"), "temperature": 0.2}
    if os.getenv("VOXLY_LLM_BASE_URL"):
        llm_kwargs["base_url"] = os.environ["VOXLY_LLM_BASE_URL"]

    # Text-only session: no STT/TTS inside the agent. Speech recognition
    # happens in our own streaming Whisper server; the agent only orchestrates
    # the LLM and its tools.
    session = AgentSession(llm=openai.LLM(**llm_kwargs), max_tool_steps=5)
    await session.start(
        agent=VoxlyAgent(participant.identity),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            text_input=True,
            audio_input=False,
            audio_output=False,
            text_output=True,
            participant_identity=participant.identity,
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
