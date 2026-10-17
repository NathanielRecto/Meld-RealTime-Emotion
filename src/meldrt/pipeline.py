"""The real-time loop: audio in -> turns -> two-stage emotion state -> streamed reply.

Events (plain dicts, JSON-ready), in order for each turn:
  speech_start                 the person started talking
  turn_end                     400 ms of silence: the turn is over (all timings start here)
  state, stage="voice"         emotion from the voice alone, about 0.1 s after turn_end
  state, stage="final"         fused text + voice + context emotion, after speech-to-text
  reply_token ... reply_done   the reply, streamed, then latency for every stage
"""
from __future__ import annotations

import asyncio
import threading
import time
import traceback
from typing import AsyncIterator

import numpy as np

from .conversation import Conversation
from .cues import voice_cues
from .reply import notes
from .runtime import expose_torch_cuda_dlls
from .turns import TurnDetector

SR = 16_000


class Pipeline:
    def __init__(self, cfg: dict):
        expose_torch_cuda_dlls()
        from faster_whisper import WhisperModel  # after the CUDA DLLs are findable

        from .emotion import EmotionRecognizer
        from .reply import ReplyGenerator

        self.cfg = cfg
        self._models_busy = threading.Lock()
        self.asr_cfg = cfg["models"]["asr"]
        self.asr = WhisperModel(self.asr_cfg["name"], device=self.asr_cfg["device"],
                                compute_type=self.asr_cfg["compute_type"])
        self.emotion = EmotionRecognizer(cfg)
        self.reply = ReplyGenerator(cfg)
        self._warm_up()

    def _warm_up(self) -> None:
        """Run every model once so the first real turn doesn't pay one-off GPU setup costs."""
        audio = (np.random.default_rng(0).standard_normal(SR) * 0.01).astype(np.float32)
        feats = self.emotion.voice_features(audio)
        self.emotion.voice_probs(feats)
        self.transcribe(audio)
        self.emotion.probs("hello", "User", [], feats)
        list(self.reply.stream([], "hello", "They clearly sound calm."))

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.asr.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                          without_timestamps=self.asr_cfg["without_timestamps"])
        return " ".join(s.text.strip() for s in segments).strip()

    def turn_detector(self) -> TurnDetector:
        rt, p = self.cfg["realtime"], self.cfg["pipeline"]
        return TurnDetector(threshold=p["vad_threshold"], silence_ms=rt["end_of_turn_silence_ms"],
                            pad_ms=p["speech_pad_ms"], max_turn_s=p["max_turn_seconds"])

    def process_turn(self, audio: np.ndarray, t_end: float, conv: Conversation, emit) -> None:
        """One finished turn, run in a worker thread. Emits both states and the streamed reply."""
        with self._models_busy:  # never two turns on the GPU models at once (llama.cpp isn't thread-safe)
            self._process_turn(audio, t_end, conv, emit)

    def _process_turn(self, audio: np.ndarray, t_end: float, conv: Conversation, emit) -> None:
        def ms() -> float:
            return round((time.perf_counter() - t_end) * 1000, 1)

        latency = {}
        turn = conv.start_turn()
        feats = self.emotion.voice_features(audio)
        voice = self.emotion.voice_probs(feats)
        latency["voice_tag"] = ms()
        emit({**conv.voice_state(turn, voice, voice_cues(audio), len(audio) / SR), "t_ms": latency["voice_tag"]})

        text = self.transcribe(audio)
        latency["asr_done"] = ms()
        if not text:
            emit({"type": "ignored", "turn": turn, "reason": "no words recognised", "t_ms": ms()})
            return
        history = conv.chat_history(self.cfg["pipeline"]["reply_history_turns"])
        probs = self.emotion.probs(text, conv.user, conv.context(self.emotion.context), feats)
        state = conv.add_user_turn(turn, text, probs, voice_cues(audio, n_words=len(text.split())))
        latency["final_tag"] = ms()
        emit({**state, "latency_ms": dict(latency), "t_ms": latency["final_tag"]})

        pieces = []
        for piece in self.reply.stream(history, text, notes(state)):
            if not pieces:
                latency["first_token"] = ms()
            pieces.append(piece)
            emit({"type": "reply_token", "turn": turn, "text": piece, "t_ms": ms()})
        reply = "".join(pieces).strip()
        latency["reply_done"] = ms()
        conv.add_robot_turn(reply)
        emit({"type": "reply_done", "turn": turn, "text": reply, "notes": notes(state),
              "latency_ms": latency, "t_ms": latency["reply_done"]})

    async def run(self, source: AsyncIterator[np.ndarray], conv: Conversation) -> AsyncIterator[dict]:
        """Listen to `source` and yield events until it ends. Listening never pauses: new turns
        queue up while an earlier one is still being processed."""
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        turns: asyncio.Queue = asyncio.Queue()
        detector = self.turn_detector()

        def emit(event) -> None:  # thread-safe: process_turn runs in a worker thread
            loop.call_soon_threadsafe(events.put_nowait, event)

        async def listen() -> None:
            try:
                async for chunk in source:
                    for ev in detector.push(chunk):
                        if ev["type"] == "speech_start":
                            emit({"type": "speech_start"})
                        else:
                            turns.put_nowait((ev["audio"], time.perf_counter()))
                for ev in detector.flush():
                    turns.put_nowait((ev["audio"], time.perf_counter()))
            finally:
                turns.put_nowait(None)

        async def think() -> None:
            while (item := await turns.get()) is not None:
                audio, t_end = item
                emit({"type": "turn_end", "speech_seconds": round(len(audio) / SR, 2)})
                try:
                    await loop.run_in_executor(None, self.process_turn, audio, t_end, conv, emit)
                except Exception as e:  # report and keep listening
                    traceback.print_exc()
                    emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            emit(None)

        tasks = [asyncio.create_task(listen()), asyncio.create_task(think())]
        try:
            while (event := await events.get()) is not None:
                yield event
        finally:
            for t in tasks:
                t.cancel()
