"""Simple, explainable voice cues (loudness, pitch, pace) for the state JSON and the reply prompt.

Raw numbers mean little across different voices, so the conversation compares each turn with
the same person's earlier turns ("louder than before") instead of with fixed thresholds."""
from __future__ import annotations

import numpy as np

SR = 16_000
FRAME = int(0.04 * SR)  # 40 ms analysis frames


def _frame_pitch(frame: np.ndarray, fmin: float = 70.0, fmax: float = 400.0) -> float | None:
    """Pitch of one frame by autocorrelation, or None if the frame isn't clearly periodic."""
    frame = frame - frame.mean()
    ac = np.correlate(frame, frame, "full")[len(frame) - 1:]
    if ac[0] <= 0:
        return None
    lo, hi = int(SR / fmax), int(SR / fmin)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SR / lag if ac[lag] / ac[0] >= 0.5 else None


def voice_cues(audio: np.ndarray, n_words: int | None = None) -> dict:
    cues = {"loudness_db": round(float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)), 1)}
    frames = audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME)
    if len(frames):
        rms = np.sqrt((frames ** 2).mean(axis=1))
        pitches = np.array([p for p in map(_frame_pitch, frames[rms > np.median(rms)]) if p])
        if len(pitches) >= 3:
            median = np.median(pitches)
            cues["pitch_hz"] = round(float(median), 1)
            cues["pitch_variation_semitones"] = round(float(np.std(12 * np.log2(pitches / median))), 2)
    if n_words:
        cues["words_per_second"] = round(n_words / (len(audio) / SR), 2)
    return cues


def compare(now: dict, earlier: list[dict]) -> dict:
    """This turn's cues relative to the average of the same person's earlier turns."""
    def base(key):
        values = [c[key] for c in earlier if key in c]
        return float(np.mean(values)) if values and key in now else None

    out = {}
    if (b := base("loudness_db")) is not None:
        d = now["loudness_db"] - b
        out["loudness"] = "louder" if d > 3 else "quieter" if d < -3 else "same"
    if (b := base("pitch_hz")) is not None:
        st = 12 * np.log2(now["pitch_hz"] / b)
        out["pitch"] = "higher" if st > 2 else "lower" if st < -2 else "same"
    if (b := base("pitch_variation_semitones")) is not None:
        d = now["pitch_variation_semitones"] - b
        out["expressiveness"] = "more animated" if d > 1 else "flatter" if d < -1 else "same"
    if (b := base("words_per_second")) is not None and b > 0:
        r = now["words_per_second"] / b
        out["pace"] = "faster" if r > 1.25 else "slower" if r < 0.8 else "same"
    return out
