"""The emotion model: a text branch, an audio branch, and a fusion head on top.

One forward pass returns logits for three views over the 7 MELD emotions:
  fused  the answer the robot acts on
  text   what the words alone suggest
  audio  what the voice alone suggests
In fusion mode the two single-modality heads are trained alongside the fused one (auxiliary
losses). That keeps each branch informative on its own, and lets the state JSON say why, for
example "words say joy, voice says anger".
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

N_CLASSES = 7


class LayerPool(nn.Module):
    """Learned, softmax-weighted average of WavLM's hidden layers, as in the SUPERB benchmark.
    Different layers carry different information; the weights show which ones emotion uses."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers: torch.Tensor) -> torch.Tensor:  # (batch, n_layers, dim)
        w = torch.softmax(self.logits, dim=0)
        return (layers * w[None, :, None]).sum(dim=1)


class EmotionModel(nn.Module):
    def __init__(self, mode: str, text_model: str = "FacebookAI/roberta-base",
                 n_audio_layers: int = 13, audio_dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        assert mode in ("text", "audio", "fusion"), mode
        self.mode = mode
        self.use_text = mode in ("text", "fusion")
        self.use_audio = mode in ("audio", "fusion")
        if self.use_text:
            self.text_encoder = AutoModel.from_pretrained(text_model, add_pooling_layer=False)
            self.text_proj = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(self.text_encoder.config.hidden_size, hidden), nn.GELU())
            self.text_head = nn.Linear(hidden, N_CLASSES)
        if self.use_audio:
            self.layer_pool = LayerPool(n_audio_layers)
            self.audio_proj = nn.Sequential(
                nn.LayerNorm(audio_dim), nn.Dropout(dropout), nn.Linear(audio_dim, hidden), nn.GELU())
            self.audio_head = nn.Linear(hidden, N_CLASSES)
        if mode == "fusion":
            self.fusion_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))

    def audio_only(self, audio: torch.Tensor) -> torch.Tensor:
        """Logits from the audio branch alone: the fast "voice" tag, available before any words."""
        return self.audio_head(self.audio_proj(self.layer_pool(audio)))

    def forward(self, input_ids=None, attention_mask=None, audio=None, audio_present=None) -> dict:
        out = {}
        if self.use_text:
            # First token (<s>) summarises the sequence, like BERT's [CLS].
            cls = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            h_text = self.text_proj(cls)
            out["text"] = self.text_head(h_text)
        if self.use_audio:
            h_audio = self.audio_proj(self.layer_pool(audio))
            if audio_present is not None:  # broken clip, or audio hidden on purpose during training
                h_audio = h_audio * audio_present[:, None].to(h_audio.dtype)
            out["audio"] = self.audio_head(h_audio)
        out["fused"] = self.fusion_head(torch.cat([h_text, h_audio], dim=-1)) if self.mode == "fusion" else out[self.mode]
        return out
