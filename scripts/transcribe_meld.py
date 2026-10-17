"""Whisper transcripts for MELD clips, made with the same settings the live system uses.

MELD's own transcripts were typed by people who heard each clip, so they carry tone in their
punctuation ("What?!"). Live, the robot only gets what speech-to-text writes. This script runs
the live ASR settings (configs/default.yaml, models.asr) over whole splits so the emotion
models can be scored on what the robot would actually read. It also measures speech-to-text
time on real clips and the word error rate against MELD's text.

Usage:
    python scripts/transcribe_meld.py dev test
Writes <meld_root>/asr/<split>.csv (clip_id, asr_text, audio_s, asr_ms) and
reports/asr_<split>.json (word error rate, punctuation, timing).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import SPLITS, load_manifest  # noqa: E402
from meldrt.runtime import expose_torch_cuda_dlls  # noqa: E402

SR = 16_000
MAX_SECONDS = 15.0  # same cap as the audio features: a few clips run for minutes (DESIGN section 7)


def words(text: str) -> list[str]:
    """Lowercase words without punctuation, for word error rate."""
    return re.sub(r"[^a-z0-9' ]+", " ", str(text).lower().replace("’", "'")).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def pct(mask) -> float:
    return round(float(np.mean(mask)) * 100, 1)


def transcribe_split(asr, cfg_asr: dict, root: Path, split: str) -> pd.DataFrame:
    df = load_manifest(root, split)
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc=split, unit=" clips"):
        if not r.audio_ok:
            rows.append({"clip_id": r.clip_id, "asr_text": "", "audio_s": np.nan, "asr_ms": np.nan})
            continue
        audio, _ = sf.read(r.wav_path, dtype="float32", frames=int(MAX_SECONDS * SR))
        t0 = time.perf_counter()
        segments, _ = asr.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                     without_timestamps=cfg_asr["without_timestamps"])
        text = " ".join(s.text.strip() for s in segments)  # transcribe() is lazy; this runs it
        rows.append({"clip_id": r.clip_id, "asr_text": text, "audio_s": round(len(audio) / SR, 3),
                     "asr_ms": round((time.perf_counter() - t0) * 1000, 1)})
    return df.merge(pd.DataFrame(rows), on="clip_id")


def summarise(df: pd.DataFrame, split: str, cfg_asr: dict) -> dict:
    ok = df[df.audio_ok]
    ref = [words(t) for t in ok.text]
    hyp = [words(t) for t in ok.asr_text]
    errors = sum(edit_distance(r, h) for r, h in zip(ref, hyp))
    return {
        "split": split,
        "asr": f"faster-whisper {cfg_asr['name']} ({cfg_asr['compute_type']})",
        "gpu": torch.cuda.get_device_name(0),
        "clips": int(len(ok)),
        "word_error_rate": round(errors / max(1, sum(len(r) for r in ref)), 4),
        "exact_match_pct": pct([r == h for r, h in zip(ref, hyp)]),
        "empty_transcript_pct": pct([len(h) == 0 for h in hyp]),
        # Punctuation that a listener adds from tone: how often each source writes it.
        "lines_with_exclamation_pct": {"meld": pct(ok.text.str.contains("!")),
                                       "whisper": pct(ok.asr_text.str.contains("!"))},
        "lines_with_question_pct": {"meld": pct(ok.text.str.contains(r"\?")),
                                    "whisper": pct(ok.asr_text.str.contains(r"\?"))},
        "asr_ms": {"p50": round(float(ok.asr_ms.median()), 1), "p95": round(float(ok.asr_ms.quantile(0.95)), 1),
                   "max": round(float(ok.asr_ms.max()), 1)},
        "audio_s_median": round(float(ok.audio_s.median()), 2),
        "real_time_factor_median": round(float((ok.asr_ms / 1000 / ok.audio_s).median()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("splits", nargs="+", choices=SPLITS)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])
    c = cfg["models"]["asr"]

    expose_torch_cuda_dlls()
    from faster_whisper import WhisperModel  # imported after the CUDA DLLs are findable

    asr = WhisperModel(c["name"], device=c["device"], compute_type=c["compute_type"])
    warm, _ = asr.transcribe(np.zeros(SR, dtype=np.float32), language="en")  # first call pays setup costs
    list(warm)
    out_dir = root / "asr"
    out_dir.mkdir(exist_ok=True)
    for split in args.splits:
        df = transcribe_split(asr, c, root, split)
        df[["clip_id", "asr_text", "audio_s", "asr_ms"]].to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8")
        report = summarise(df, split, c)
        (ROOT / "reports" / f"asr_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
