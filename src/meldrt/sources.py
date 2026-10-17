"""Audio sources. Each yields 512-sample (32 ms) chunks of 16 kHz mono audio, in real time."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .data import load_manifest
from .features import MAX_SECONDS

SR = 16_000
CHUNK = 512


async def replay(clips: list[np.ndarray], gap_s: float = 1.0, lead_s: float = 0.5, speed: float = 1.0):
    """Play recorded clips at real speed with silence between them, as if spoken live."""
    silence = np.zeros(int(gap_s * SR), dtype=np.float32)
    parts = [np.zeros(int(lead_s * SR), dtype=np.float32)]
    for clip in clips:
        parts += [clip.astype(np.float32), silence]
    stream = np.concatenate(parts)
    stream = np.pad(stream, (0, (-len(stream)) % CHUNK))
    t0 = time.perf_counter()
    for i in range(0, len(stream), CHUNK):
        yield stream[i: i + CHUNK]
        # Wait until this chunk would have finished playing in real time.
        await asyncio.sleep(max(0.0, t0 + (i + CHUNK) / SR / speed - time.perf_counter()))


async def microphone(stop: asyncio.Event, device=None):
    """Live microphone audio until `stop` is set. Uses 16 kHz if the device allows it, otherwise
    records at 48 kHz and downsamples."""
    import sounddevice as sd
    from scipy.signal import resample_poly

    loop = asyncio.get_running_loop()
    blocks: asyncio.Queue = asyncio.Queue()
    try:
        sd.check_input_settings(device=device, samplerate=SR, channels=1)
        rate = SR
    except Exception:
        rate = 48_000

    def callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(blocks.put_nowait, indata[:, 0].copy())

    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=CHUNK * rate // SR,
                        device=device, callback=callback):
        while not stop.is_set():
            try:
                block = await asyncio.wait_for(blocks.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            yield block if rate == SR else resample_poly(block, 1, rate // SR).astype(np.float32)


def meld_dialogue(meld_root: str | Path, split: str, dialogue_id: int,
                  first: int = 0, last: int | None = None) -> tuple[list[np.ndarray], pd.DataFrame]:
    """The audio clips and table rows of one MELD dialogue (optionally a range of its utterances)."""
    df = load_manifest(meld_root, split)
    rows = df[(df.dialogue_id == dialogue_id) & df.audio_ok & (df.utterance_id >= first)]
    if last is not None:
        rows = rows[rows.utterance_id <= last]
    if rows.empty:
        raise ValueError(f"no usable utterances for {split} dialogue {dialogue_id}")
    clips = [sf.read(p, dtype="float32", frames=int(MAX_SECONDS * SR))[0] for p in rows.wav_path]
    return clips, rows.reset_index(drop=True)
