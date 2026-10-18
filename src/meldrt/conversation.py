"""Conversation memory, and the structured state (output 1) built for each turn."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cues import compare
from .data import EMOTIONS

# Rough pleasantness of each emotion, used only for the mood trend.
VALENCE = {"anger": -1, "disgust": -1, "fear": -1, "joy": 1, "neutral": 0, "sadness": -1, "surprise": 0}


def top(probs: np.ndarray) -> dict:
    i = int(np.argmax(probs))
    return {"label": EMOTIONS[i], "confidence": round(float(probs[i]), 3)}


def with_probs(probs: np.ndarray) -> dict:
    return {**top(probs), "probs": {e: round(float(p), 3) for e, p in zip(EMOTIONS, probs)}}


def sentiment(label: str) -> str:
    v = VALENCE[label]
    return "positive" if v > 0 else "negative" if v < 0 else "neutral"


@dataclass
class Conversation:
    user: str = "User"
    robot: str = "Pip"
    turns: list[dict] = field(default_factory=list)      # both sides, in order: {speaker, text}
    user_cues: list[dict] = field(default_factory=list)  # the person's earlier voice cues
    mood: float | None = None                            # smoothed pleasantness, -1 to 1
    n_turns: int = 0

    def start_turn(self) -> int:
        self.n_turns += 1
        return self.n_turns

    def context(self, n: int) -> list[dict]:
        """Earlier turns for the emotion model. Both sides count, like the multi-party MELD
        dialogues it was trained on; speaker names become A/B inside format_text."""
        return [dict(t) for t in self.turns[-n:]] if n else []

    def chat_history(self, n: int) -> list[dict]:
        """Earlier turns as chat messages for the reply model: the person's words as they said them."""
        return [{"role": "user" if t["speaker"] == self.user else "assistant", "content": t["text"]}
                for t in self.turns[-n:]]

    def voice_state(self, turn: int, voice_probs: np.ndarray, cues: dict, seconds: float) -> dict:
        """Stage 1 state: from the voice alone, sent before the words are known."""
        return {"type": "state", "stage": "voice", "turn": turn, "transcript": None,
                "speech_seconds": round(seconds, 2), "emotion": with_probs(voice_probs),
                "cues": {**cues, "vs_earlier": compare(cues, self.user_cues)}}

    def add_user_turn(self, turn: int, text: str, probs: dict, cues: dict) -> dict:
        """Stage 2 state from the fused model; also records the turn in memory."""
        fused, words, voice = with_probs(probs["fused"]), top(probs["text"]), top(probs["audio"])
        valence = float(sum(p * VALENCE[e] for e, p in zip(EMOTIONS, probs["fused"])))
        before = self.mood
        self.mood = valence if before is None else 0.5 * before + 0.5 * valence
        if before is None:
            trend = "starting"
        else:
            trend = "improving" if self.mood - before > 0.15 else "worsening" if self.mood - before < -0.15 else "steady"
        state = {
            "type": "state", "stage": "final", "turn": turn, "transcript": text,
            "emotion": fused, "sentiment": sentiment(fused["label"]),
            "by_modality": {"words": words, "voice": voice},
            "modalities_agree": words["label"] == voice["label"],
            "cues": {**cues, "vs_earlier": compare(cues, self.user_cues)},
            "mood": round(self.mood, 3), "mood_trend": trend,
        }
        self.user_cues.append(cues)
        self.turns.append({"speaker": self.user, "text": text})
        return state

    def add_robot_turn(self, text: str) -> None:
        self.turns.append({"speaker": self.robot, "text": text})
