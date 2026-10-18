"""Local web app: run `python scripts/serve.py`, then open http://127.0.0.1:8000.

One WebSocket (/ws) carries everything. The page sends commands:
  {"cmd": "mic"}                                         listen to the microphone
  {"cmd": "replay", "split": "test", "dialogue": 240,    replay MELD lines in real time
   "first": 8, "last": 10}
  {"cmd": "stop"}
and receives the pipeline's events (see meldrt.pipeline) as JSON, plus
  {"type": "session_start", "source": ..., "lines": [...]}   (lines = MELD script for a replay)
  {"type": "session_end"}  and  {"type": "error", "message": ...}
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .conversation import Conversation
from .pipeline import Pipeline
from .sources import meld_dialogue, microphone, replay

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
shared: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    shared["pipeline"] = await asyncio.to_thread(Pipeline, CFG)  # load and warm up every model once
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    session: asyncio.Task | None = None
    stop = asyncio.Event()

    async def run(source, info: dict) -> None:
        await socket.send_json({"type": "session_start", **info})
        conv = Conversation(robot=CFG["pipeline"]["robot_name"])
        async for event in shared["pipeline"].run(source, conv):
            await socket.send_json(event)
        await socket.send_json({"type": "session_end"})

    try:
        while True:
            msg = await socket.receive_json()
            if msg.get("cmd") == "stop" and session and not session.done():
                stop.set()                      # the microphone hands over its last turn, then ends
                if msg.get("now") or session.get_name() == "replay":
                    session.cancel()
                continue
            if session and not session.done():  # a new command replaces the running session
                session.cancel()
            stop = asyncio.Event()
            if msg.get("cmd") == "mic":
                session = asyncio.create_task(run(microphone(stop), {"source": "microphone"}), name="mic")
            elif msg.get("cmd") == "replay":
                try:
                    clips, rows = meld_dialogue(CFG["paths"]["meld_root"], msg["split"], int(msg["dialogue"]),
                                                int(msg.get("first") or 0),
                                                None if msg.get("last") in (None, "") else int(msg["last"]))
                except (ValueError, KeyError) as e:
                    await socket.send_json({"type": "error", "message": str(e)})
                    continue
                except OSError:  # no MELD download: the microphone still works without it
                    await socket.send_json({"type": "error", "message": (
                        f"MELD data not found under {CFG['paths']['meld_root']}. Replaying MELD lines needs "
                        "the MELD download (README, Data); the microphone works without it.")})
                    continue
                lines = [{"clip_id": r.clip_id, "speaker": r.speaker, "emotion": r.emotion, "text": r.text}
                         for r in rows.itertuples()]
                info = {"source": f"MELD {msg['split']} dialogue {msg['dialogue']}", "lines": lines}
                session = asyncio.create_task(run(replay(clips), info), name="replay")
    except WebSocketDisconnect:
        if session:
            session.cancel()
