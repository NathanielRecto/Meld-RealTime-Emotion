"""Phase 4: latency and resource use of the live pipeline, measured on real MELD clips.

Plays whole test dialogues, turn by turn, through the same code as the live app
(Pipeline.process_turn: voice tag, speech-to-text, final tag, streamed reply), so the context and
conversation memory grow as they would live. Each stage is timed from the end of the turn. Also
records the turn detector's cost per audio chunk, GPU memory, process RAM and parameter counts.
Run it with nothing else using the GPU.

Usage:
    python scripts/benchmark_latency.py --turns 150
Writes reports/phase4_latency.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.conversation import Conversation  # noqa: E402
from meldrt.data import load_manifest  # noqa: E402
from meldrt.features import MAX_SECONDS, SR  # noqa: E402
from meldrt.pipeline import Pipeline  # noqa: E402
from meldrt.turns import TurnDetector  # noqa: E402

STAGES = ["voice_tag", "asr_done", "final_tag", "first_token", "reply_done"]
TARGETS = {"voice_tag": "voice_tag", "final_tag": "final_tag", "first_token": "first_token", "reply_done": "full_reply"}


def gpu_used_mb() -> float:
    import pynvml

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 2**20


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params(pipe: Pipeline, cfg: dict) -> dict:
    """Every learned parameter on the live inference path (the brief's 6B cap counts all of them)."""
    from transformers import AutoConfig, AutoModelForCausalLM, WhisperForConditionalGeneration

    with torch.device("meta"):  # shapes only: CTranslate2 and GGUF models aren't torch modules
        whisper = WhisperForConditionalGeneration(AutoConfig.from_pretrained(cfg["models"]["asr"]["hf_config"]))
        llm = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cfg["models"]["llm"]["hf_config"]))
    counts = {"turn_detector (Silero VAD)": n_params(TurnDetector().vad.model),
              "speech_to_text (Whisper small)": n_params(whisper),
              "voice_encoder (WavLM base+)": n_params(pipe.emotion.wavlm),
              f"emotion_models ({len(pipe.emotion.models)} x RoBERTa + heads)": sum(map(n_params, pipe.emotion.models)),
              "reply_model (Qwen2.5-1.5B)": n_params(llm)}
    counts["total"] = sum(counts.values())
    return counts


def pick_dialogues(df, n_turns: int, seed: int) -> list:
    """Whole dialogues, in random order, until there are at least n_turns usable lines."""
    ok = df[df.audio_ok & (df.duration_s <= MAX_SECONDS)]
    chosen, total = [], 0
    for d in np.random.default_rng(seed).permutation(ok.dialogue_id.unique()):
        rows = ok[ok.dialogue_id == d].sort_values("utterance_id")
        chosen.append(rows)
        total += len(rows)
        if total >= n_turns:
            break
    return chosen


def vad_cost_ms(audio: np.ndarray) -> float:
    detector = TurnDetector()
    chunks = [audio[i: i + 512] for i in range(0, len(audio) - 511, 512)]
    t0 = time.perf_counter()
    for c in chunks:
        detector.push(c)
    return (time.perf_counter() - t0) * 1000 / len(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))

    before_load = gpu_used_mb()
    t = time.perf_counter()
    pipe = Pipeline(cfg)
    load_s = time.perf_counter() - t
    after_load = peak = gpu_used_mb()

    dialogues = pick_dialogues(load_manifest(cfg["paths"]["meld_root"], "test"), args.turns, args.seed)
    timings = {s: [] for s in STAGES}
    ignored = done = 0
    for rows in dialogues:
        conv = Conversation(robot=cfg["pipeline"]["robot_name"])
        for r in rows.itertuples():
            audio, _ = sf.read(r.wav_path, dtype="float32", frames=int(MAX_SECONDS * SR))
            events: list[dict] = []
            pipe.process_turn(audio, time.perf_counter(), conv, events.append)  # timing starts at turn end
            last = events[-1]
            if last["type"] == "reply_done":
                for s in STAGES:
                    timings[s].append(last["latency_ms"][s])
            else:
                ignored += 1
            done += 1
            peak = max(peak, gpu_used_mb())
            print(f"\r{done} turns", end="", flush=True)
    print()

    def stats(values):
        v = np.array(values)
        return {"p50": round(float(np.median(v)), 1), "p95": round(float(np.percentile(v, 95)), 1),
                "max": round(float(v.max()), 1)}

    latency = {s: stats(v) for s, v in timings.items()}
    budget = cfg["realtime"]["budget_ms"]
    targets = {stage: {"target_ms": budget[key], "p95_ms": latency[stage]["p95"],
                       "met": latency[stage]["p95"] <= budget[key]} for stage, key in TARGETS.items()}
    props = torch.cuda.get_device_properties(0)
    report = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": {"gpu": props.name, "gpu_vram_mb": round(props.total_memory / 2**20),
                     "cpu_threads": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                     "os": platform.platform()},
        "emotion_models": cfg["models"]["emotion"]["runs"],
        "turns": done, "dialogues": len(dialogues), "ignored_no_words": ignored,
        "note": "times count from the end of the turn; the turn detector first waits "
                f"{cfg['realtime']['end_of_turn_silence_ms']} ms of silence to decide the turn is over",
        "latency_ms": latency,
        "targets": targets,
        "vad_ms_per_32ms_chunk": round(vad_cost_ms(sf.read(rows.wav_path.iloc[0], dtype="float32")[0]), 3),
        "model_load_s": round(load_s, 1),
        "gpu_memory_mb": {"before_load": round(before_load), "after_load": round(after_load), "peak": round(peak)},
        "process_ram_mb": round(psutil.Process().memory_info().rss / 2**20),
        "params": count_params(pipe, cfg),
    }
    out = ROOT / "reports" / "phase4_latency.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{done} turns from {len(dialogues)} dialogues ({ignored} had no recognisable words)")
    print(f"{'stage':<14}{'p50':>8}{'p95':>8}{'max':>8}  target (p95)")
    for s in STAGES:
        tgt = targets.get(s)
        verdict = f"{tgt['target_ms']} ms {'met' if tgt['met'] else 'MISSED'}" if tgt else ""
        print(f"{s:<14}{latency[s]['p50']:>8.0f}{latency[s]['p95']:>8.0f}{latency[s]['max']:>8.0f}  {verdict}")
    g = report["gpu_memory_mb"]
    print(f"GPU memory: {g['before_load']} MB before load, {g['after_load']} after, peak {g['peak']} "
          f"of {report['hardware']['gpu_vram_mb']} MB; parameters {report['params']['total'] / 1e9:.2f}B")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
