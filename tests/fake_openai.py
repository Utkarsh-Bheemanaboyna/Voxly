"""A scripted, OpenAI-compatible chat-completions server for testing the agent
without an API key. It plays the role of the LLM:
  user command          -> call get_selected_text
  selection came back   -> call write_at_cursor with a "fixed" version
  write confirmed       -> reply "Fixed grammar."
Run: python fake_openai.py  (listens on 127.0.0.1:8099, base_url .../v1)
"""
import json

from aiohttp import web


def chunk(delta, finish=None):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def tool_call(name, args):
    return {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{name}",
            "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


async def completions(request):
    body = await request.json()
    msgs = body["messages"]
    last = msgs[-1]
    print("LLM saw last message:", last.get("role"), str(last.get("content"))[:80], flush=True)
    if last["role"] == "tool":
        prev_call = next(m for m in reversed(msgs) if m.get("tool_calls"))
        name = prev_call["tool_calls"][0]["function"]["name"]
        if name == "get_selected_text":
            fixed = last["content"].replace("i has", "I have").replace("teh", "the")
            events = [chunk(tool_call("write_at_cursor", {"text": fixed})),
                      chunk({}, "tool_calls")]
        else:
            events = [chunk({"role": "assistant", "content": "Fixed grammar."}),
                      chunk({}, "stop")]
    else:
        events = [chunk(tool_call("get_selected_text", {})), chunk({}, "tool_calls")]

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for e in events:
        await resp.write(f"data: {json.dumps(e)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


app = web.Application()
app.router.add_post("/v1/chat/completions", completions)
web.run_app(app, host="127.0.0.1", port=8099)
