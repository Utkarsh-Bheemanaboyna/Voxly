"""End-to-end test of the LiveKit path, runnable on Linux.

Uses the real client/agent_link.py with the macOS `insert` module replaced by
a fake text field. Needs: livekit-server --dev, agent.py running against
fake_openai.py (or a real LLM).
"""
import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client"))

field = {"selected": "i has seen teh results", "written": None}
fake_insert = types.ModuleType("insert")
fake_insert.get_selected_text = lambda: field["selected"]
fake_insert.insert_text = lambda t: field.__setitem__("written", t) or "ax"
sys.modules["insert"] = fake_insert

from agent_link import AgentLink  # noqa: E402


class FakeApp:
    history = ["hello world"]

    class log:
        @staticmethod
        def utterance(**kw):
            print("log:", kw)


async def main():
    link = AgentLink(FakeApp())
    await link.connect()
    await asyncio.sleep(3)  # let the agent get dispatched and join
    await link.send_command("fix the grammar")
    for _ in range(100):
        if field["written"]:
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(1)
    print("WRITTEN:", field["written"])
    await link.room.disconnect()
    assert field["written"] == "I have seen the results", field["written"]
    print("E2E OK")


asyncio.run(main())
