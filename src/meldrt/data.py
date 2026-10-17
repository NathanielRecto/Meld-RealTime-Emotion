"""Loading the Phase 1 tables and turning one utterance into model inputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_ID = {e: i for i, e in enumerate(EMOTIONS)}
SPLITS = ["train", "dev", "test"]
PREV_TURNS = 3  # earlier utterances of the same dialogue kept as context


def load_manifest(meld_root: str | Path, split: str) -> pd.DataFrame:
    # keep_default_na=False: a line like "NA" or "None" is dialogue, not a missing value.
    df = pd.read_csv(Path(meld_root) / "manifests" / f"{split}.csv", keep_default_na=False, na_values=[""])
    df["label"] = df.emotion.map(LABEL_ID)
    df["prev_turns"] = df.prev_turns.map(json.loads)
    return df


def previous_turns(df: pd.DataFrame, n: int = PREV_TURNS) -> list[list[dict]]:
    """For each row of a table sorted by dialogue and utterance, the n utterances before it in the
    same dialogue. Only speaker and words: earlier emotion labels would not be known in a live
    conversation."""
    out = []
    for _, g in df.groupby("dialogue_id", sort=False):
        turns = list(zip(g.speaker, g.text))
        for i in range(len(turns)):
            out.append([{"speaker": s, "text": t} for s, t in turns[max(0, i - n):i]])
    return out


def with_whisper_text(df: pd.DataFrame, meld_root: str | Path, split: str) -> pd.DataFrame:
    """Swap in Whisper's transcript (scripts/transcribe_meld.py) for every utterance and rebuild
    each context window from it: live, the earlier turns also come from speech-to-text."""
    asr = pd.read_csv(Path(meld_root) / "asr" / f"{split}.csv", keep_default_na=False, na_values=[""])
    df = df.copy()
    df["text"] = df.clip_id.map(asr.set_index("clip_id").asr_text).fillna("")
    df["prev_turns"] = previous_turns(df)
    return df


def format_text(speaker: str, text: str, prev_turns: list[dict], n_context: int) -> tuple[str, str]:
    """Return (target, context) strings for the text encoder.

    Speakers are renamed A, B, C... in order of appearance within this window, so the model
    learns who said what without memorising Friends characters. Live, the person and the
    robot simply become two letters."""
    window = prev_turns[-n_context:] if n_context > 0 else []
    names: dict[str, str] = {}

    def anon(name: str) -> str:
        return names.setdefault(name, chr(ord("A") + len(names)))

    context = " ".join(f"{anon(t['speaker'])}: {t['text']}" for t in window)
    target = f"{anon(speaker)}: {text}"
    return target, context


def class_weights(labels: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency class weights softened by `power` (0 = none, 1 = full inverse
    frequency), scaled so the average weight per training example is 1."""
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return w / w[labels].mean()
