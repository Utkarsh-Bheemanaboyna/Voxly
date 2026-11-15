"""Client side of the LiveKit connection.

The Mac client joins a LiveKit room as a participant. The agent (agent/agent.py)
is dispatched into the same room. Communication:

  client -> agent : text stream on topic "lk.chat"   (the spoken command)
  agent  -> client: RPC calls the agent's *tools* make:
                      get_selected_text     -> read selection via Accessibility
                      get_recent_dictation  -> last dictated segments
                      write_at_cursor       -> insert via Accessibility
  agent  -> client: text stream on "lk.transcription" (short status reply)

Why RPC: the LLM runs in the agent process, but the user's text fields live on
the Mac. Tools are the LLM's only way to read or change them, and every tool
call becomes one request/response round-trip to the client.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from livekit import api, rtc

import insert


class AgentLink:
    def __init__(self, app):
        self.app = app
        self.room = rtc.Room()
        self.identity = "voxly-user"
        self.t_command = 0.0

    async def connect(self):
        url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
        token = (
            api.AccessToken(os.getenv("LIVEKIT_API_KEY", "devkey"),
                            os.getenv("LIVEKIT_API_SECRET", "secret"))
            .with_identity(self.identity)
            .with_grants(api.VideoGrants(room_join=True,
                                         room=os.getenv("VOXLY_ROOM", "voxly")))
            .to_jwt()
        )
        async def get_selected_text(data: rtc.RpcInvocationData) -> str:
            return await asyncio.to_thread(insert.get_selected_text)

        async def get_recent_dictation(data: rtc.RpcInvocationData) -> str:
            n = int(json.loads(data.payload or "{}").get("count", 1))
            return "\n".join(self.app.history[-n:])

        async def write_at_cursor(data: rtc.RpcInvocationData) -> str:
            text = json.loads(data.payload)["text"]
            ms = (time.perf_counter() - self.t_command) * 1000
            method = await asyncio.to_thread(insert.insert_text, text)
            print(f"\n✎ agent wrote {len(text)} chars via {method} "
                  f"\033[2m[{ms:.0f} ms after command]\033[0m")
            self.app.log.utterance(mode="agent_write", chars=len(text), method=method,
                                   command_to_write_ms=round(ms))
            return method

        def on_text(reader: rtc.TextStreamReader, participant_identity: str):
            async def read():
                msg = await reader.read_all()
                if participant_identity != self.identity and msg.strip():
                    print(f"\n🤖 {msg.strip()}")
            asyncio.create_task(read())

        self.room.register_text_stream_handler("lk.transcription", on_text)
        await self.room.connect(url, token)
        # RPC handlers can only be registered once connected
        lp = self.room.local_participant
        lp.register_rpc_method("get_selected_text", get_selected_text)
        lp.register_rpc_method("get_recent_dictation", get_recent_dictation)
        lp.register_rpc_method("write_at_cursor", write_at_cursor)
        print(f"LiveKit: joined room as {self.identity}")

    async def send_command(self, text: str):
        self.t_command = time.perf_counter()
        await self.room.local_participant.send_text(text, topic="lk.chat")
