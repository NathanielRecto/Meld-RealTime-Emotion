"""Live emotion recognition: WavLM voice features plus the Phase 2 fusion model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

from .data import format_text
from .features import wavlm_layer_means
from .model import EmotionModel

ROOT = Path(__file__).resolve().parents[2]


class EmotionRecognizer:
    def __init__(self, cfg: dict):
        audio_cfg, run = cfg["models"]["audio_encoder"], cfg["models"]["emotion"]["run"]
        trained = json.loads((ROOT / "reports" / f"phase2_{run}.json").read_text(encoding="utf-8"))
        if trained["mode"] != "fusion":
            raise ValueError(f"{run} is a {trained['mode']} model; the live pipeline needs a fusion model")
        self.run = run
        self.context = trained["context"]
        self.max_len = trained["hyperparams"]["max_len"]
        text_name = cfg["models"]["text_encoder"]["name"]
        self.extractor = AutoFeatureExtractor.from_pretrained(audio_cfg["name"])
        self.wavlm = AutoModel.from_pretrained(audio_cfg["name"], dtype=getattr(torch, audio_cfg["dtype"])).cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(text_name)
        self.model = EmotionModel("fusion", text_model=text_name).cuda().eval()
        self.model.load_state_dict(torch.load(ROOT / "checkpoints" / run / "model.pt", map_location="cuda"))

    def voice_features(self, audio: np.ndarray) -> np.ndarray:
        return wavlm_layer_means(self.wavlm, self.extractor, audio)

    @torch.inference_mode()
    def voice_probs(self, feats: np.ndarray) -> np.ndarray:
        """Stage 1: emotion from the voice alone, before any words are known."""
        return self.model.audio_only(torch.from_numpy(feats)[None].cuda()).softmax(-1)[0].cpu().numpy()

    @torch.inference_mode()
    def probs(self, text: str, speaker: str, context_turns: list[dict], feats: np.ndarray) -> dict:
        """Stage 2: fused, words-only and voice-only probabilities from one forward pass."""
        target, context = format_text(speaker, text, context_turns, self.context)
        enc = self.tok(target, context or None, truncation=True, max_length=self.max_len, return_tensors="pt").to("cuda")
        out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         audio=torch.from_numpy(feats)[None].cuda(), audio_present=torch.ones(1, device="cuda"))
        return {head: logits.softmax(-1)[0].cpu().numpy() for head, logits in out.items()}
