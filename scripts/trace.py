"""Follow one input through every stage of the live pipeline, with timings.

This is the brief's "one input travelling through the system to both outputs" demonstration.
The audio streams in real time through the same code as the live app (turn detection, voice
tag, speech-to-text, fused tag, reply), and every event is printed as it happens.

Usage:
    python scripts/trace.py --clip test/dia3_utt5              # one MELD clip, played in real time
    python scripts/trace.py --clip test/dia3_utt5 --context 2  # play the 2 lines before it first
    python scripts/trace.py --wav my_recording.wav             # any 16 kHz mono WAV
Writes reports/trace_<name>.json: every event plus the last turn's state (output 1) and reply (output 2).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.conversation import Conversation  # noqa: E402
from meldrt.pipeline import Pipeline  # noqa: E402
from meldrt.sources import SR, meld_dialogue, replay  # noqa: E402


def load_input(args, cfg):
    if args.wav:
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            sys.exit(f"{args.wav} is {sr} Hz; the pipeline expects 16 kHz mono")
        return [audio], [f"{Path(args.wav).name} ({len(audio) / SR:.1f} s)"], Path(args.wav).stem
    split, clip = args.clip.split("/")
    dialogue, utterance = (int(x) for x in clip.removeprefix("dia").split("_utt"))
    clips, rows = meld_dialogue(cfg["paths"]["meld_root"], split, dialogue,
                                first=max(0, utterance - args.context), last=utterance)
    lines = [f"{r.clip_id}  {r.speaker} [{r.emotion}]: {r.text}" for r in rows.itertuples()]
    return clips, lines, f"{split}_{clip}"


def show(event: dict, t0: float, first_token: set) -> None:
    clock = f"[{time.perf_counter() - t0:6.2f} s]"
    kind = event["type"]
    if kind == "speech_start":
        print(f"{clock} 1. turn detector  speech started")
    elif kind == "turn_end":
        print(f"{clock} 1. turn detector  turn over: {event['speech_seconds']} s of speech, then 400 ms "
              "of silence (the times below count from here)")
    elif kind == "state" and event["stage"] == "voice":
        e = event["emotion"]
        print(f"{clock} 2. voice tag      +{event['t_ms']:4.0f} ms  {e['label']} {e['confidence']:.2f}"
              "   (WavLM + audio head; no words yet)")
    elif kind == "state":
        e, by = event["emotion"], event["by_modality"]
        print(f"{clock} 3. speech-to-text +{event['latency_ms']['asr_done']:4.0f} ms  \"{event['transcript']}\"")
        print(f"{clock} 4. final tag      +{event['t_ms']:4.0f} ms  {e['label']} {e['confidence']:.2f}"
              f"   (words alone: {by['words']['label']} {by['words']['confidence']:.2f},"
              f" voice alone: {by['voice']['label']} {by['voice']['confidence']:.2f})")
    elif kind == "reply_token":
        if event["turn"] not in first_token:
            first_token.add(event["turn"])
            print(f"{clock} 5. reply          +{event['t_ms']:4.0f} ms  ", end="")
        print(event["text"], end="", flush=True)
    elif kind == "reply_done":
        print(f"\n{' ' * 13}   reply done     +{event['t_ms']:4.0f} ms\n")
    elif kind in ("ignored", "error"):
        print(f"{clock} {kind}: {event.get('reason') or event.get('message')}")


async def trace(pipe: Pipeline, clips, robot: str) -> list[dict]:
    events, first_token = [], set()
    t0 = time.perf_counter()
    async for event in pipe.run(replay(clips), Conversation(robot=robot)):
        show(event, t0, first_token)
        if event["type"] != "reply_token":
            events.append(event)
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="MELD clip as split/clip_id, e.g. test/dia3_utt5")
    src.add_argument("--wav", help="a 16 kHz mono WAV file")
    ap.add_argument("--context", type=int, default=0, help="also play this many earlier lines of the dialogue first")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    clips, lines, name = load_input(args, cfg)
    print("Input, played in real time:")
    for line in lines:
        print(f"  {line}")
    t = time.perf_counter()
    pipe = Pipeline(cfg)
    print(f"\nModels loaded and warmed up in {time.perf_counter() - t:.1f} s\n")

    events = asyncio.run(trace(pipe, clips, cfg["pipeline"]["robot_name"]))
    final = next((e for e in reversed(events) if e["type"] == "state" and e["stage"] == "final"), None)
    reply = next((e for e in reversed(events) if e["type"] == "reply_done"), None)
    print("Output 1, structured state for the last turn:")
    print(json.dumps(final, indent=2))
    print(f"\nOutput 2, reply: {reply['text'] if reply else '(none)'}")
    if reply:
        print(f"Reply prompt notes: {reply['notes']}")
        print("Latency from end of turn (ms):", reply["latency_ms"])

    out = ROOT / "reports" / f"trace_{name}.json"
    out.write_text(json.dumps({"input": lines, "events": events, "final_state": final,
                               "reply": reply}, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
