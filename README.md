# Voxly

Open-source, Wispr Flow–style dictation for macOS. Hold a key, speak, and text
appears at your cursor in any app. Hold a different key to give a spoken
command ("fix the grammar", "summarize what I just said"), and a tool-calling
LLM agent edits your text in place.

Architecture inspired by [BiteFlow](https://github.com/smallest-inc/BiteFlow) (MIT).
Voxly uses open-source Whisper with streaming partials, a LiveKit Agents
orchestration layer, and Accessibility-API text insertion.

## Architecture

```
 ┌──────────────────────── macOS client (client/) ────────────────────────┐
 │ Right Option / Right Cmd (pynput)                                      │
 │ mic ─ sounddevice: PCM16, 16 kHz mono, 50 ms blocks, 200 ms pre-roll   │
 │   │                                    ▲ partials (live overlay)       │
 │   ▼ WebSocket: binary PCM + start/stop  │ final transcript             │
 └───┼────────────────────────────────────┼───────────────────────────────┘
     ▼                                    │
 ┌── STT server (server/) ────────────────┴──┐
 │ rolling buffer → Whisper every 0.5 s       │  faster-whisper (CPU int8)
 │ LocalAgreement-2: commit stable words      │  or mlx-whisper (Apple GPU)
 │ trim buffer at last committed word         │
 │ on stop: transcribe remainder → final      │
 └────────────────────────────────────────────┘
     final transcript
       ├─ dictation → Accessibility API (AXUIElement selected-text)
       │              fallback: clipboard + simulated Cmd+V (clipboard restored)
       └─ command   → LiveKit room (text stream "lk.chat")
                        │
                 ┌──────▼─────── LiveKit agent (agent/) ───────────┐
                 │ AgentSession, text-only, LLM with tools:         │
                 │   get_selected_text()     ─┐                      │
                 │   get_recent_dictation(n)  ├─ LiveKit RPC → Mac   │
                 │   write_at_cursor(text)   ─┘   (runs AX calls)    │
                 └──────────────────────────────────────────────────┘
```

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Grant your terminal app (Terminal / iTerm) these in **System Settings → Privacy & Security**:
**Microphone**, **Accessibility** (text insertion), **Input Monitoring** (global hotkeys).
Restart the terminal after granting them.

### 1. Dictation only

```bash
# terminal 1: STT server (first run downloads the model)
python server/stt_server.py --backend mlx --model mlx-community/whisper-base.en-mlx
#   or: python server/stt_server.py --backend faster-whisper --model base.en

# terminal 2: client
python client/voxly.py
```
Click into TextEdit, hold **Right Option**, speak, release.

### 2. Agent commands (LiveKit)

```bash
brew install livekit
livekit-server --dev                       # terminal 3 (key=devkey, secret=secret)

cp .env.example .env  # fill in OPENAI_API_KEY (or Groq / Ollama, see file)
set -a; source .env; set +a
python agent/agent.py dev                  # terminal 4

python client/voxly.py --agent             # restart the client with --agent
```
Select some text, hold **Right Command**, say "fix the grammar".

## Measuring (do not quote numbers you have not measured)

```bash
cd bench
python record.py --out data/my_clips                   # 20 sentences, ~5 min
python stream_wav.py data/my_clips --runs 3 --tag base # needs the server running
python report.py results/latency_base.csv              # p50/p90/p95/p99 + WER
python report.py ~/.voxly/latency.csv                  # live usage latencies
python typing_vs_dictation.py                          # speed vs typing
```

Latency definitions:
- **final latency**: end of speech (stop sent) → final transcript received
- **partial latency**: audio chunk sent → partial covering that audio received
- **press-to-text** (in `~/.voxly/utterances.jsonl`): key press → text inserted

## Tests

```bash
python -m pytest tests -q     # streaming logic + server, with a fake Whisper
```
`tests/e2e_agent.py` + `tests/fake_openai.py` run the full LiveKit agent path
(LLM tool calls → RPC → insertion) without an API key.

## Known limitations

- Whisper is not natively streaming: partials come from re-transcription,
  which costs extra compute, and the tentative tail can flicker.
- Push-to-talk only (no VAD auto-endpointing).
- Accessibility insertion fails in some Electron apps and falls back to paste.
- English-only models by default.
- Latency numbers are single-user, single-machine measurements.

## License

MIT
