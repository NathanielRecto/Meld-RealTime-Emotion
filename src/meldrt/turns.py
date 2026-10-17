"""Turn-taking: Silero VAD over a live 16 kHz stream, cut into one utterance per turn."""
from __future__ import annotations

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

SR = 16_000


class TurnDetector:
    """Feed 512-sample chunks with push(); it returns {"type": "speech_start"} when someone starts
    talking and {"type": "turn", "audio": ...} once they have been silent for `silence_ms`."""

    def __init__(self, threshold: float = 0.5, silence_ms: int = 400, pad_ms: int = 100,
                 max_turn_s: float = 15.0):
        self.vad = VADIterator(load_silero_vad(), threshold=threshold, sampling_rate=SR,
                               min_silence_duration_ms=silence_ms, speech_pad_ms=pad_ms)
        self.max_samples = int(max_turn_s * SR)
        self.audio = np.zeros(0, dtype=np.float32)  # recent audio, starting at absolute sample `base`
        self.base = 0
        self.vad_origin = 0  # absolute sample where the VAD's own counter last restarted
        self.start: int | None = None  # absolute start of the turn in progress

    def push(self, chunk: np.ndarray) -> list[dict]:
        self.audio = np.concatenate([self.audio, chunk.astype(np.float32)])
        now = self.base + len(self.audio)
        events = []
        hit = self.vad(torch.from_numpy(chunk))
        if hit and "start" in hit and self.start is None:
            self.start = self.vad_origin + hit["start"]
            events.append({"type": "speech_start"})
        elif hit and "end" in hit and self.start is not None:
            events.append(self._cut(self.vad_origin + hit["end"]))
        elif self.start is not None and now - self.start >= self.max_samples:
            events.append(self._cut(now))  # very long turn: cut it rather than wait forever
            self._restart_vad(now)
        if self.start is None and len(self.audio) > SR:
            # Between turns keep only the last second: enough for the speech padding to reach back.
            self.base += len(self.audio) - SR
            self.audio = self.audio[-SR:]
        return events

    def flush(self) -> list[dict]:
        """End of input: hand over a turn that was still in progress."""
        if self.start is None:
            return []
        now = self.base + len(self.audio)
        event = self._cut(now)
        self._restart_vad(now)
        return [event]

    def _cut(self, end: int) -> dict:
        begin = max(self.start, self.base) - self.base
        audio = self.audio[begin: end - self.base].copy()
        self.start = None
        return {"type": "turn", "audio": audio}

    def _restart_vad(self, now: int) -> None:
        self.vad.reset_states()  # also restarts its sample counter
        self.vad_origin = now
