"""WavLM voice features, computed the same way for training (scripts/extract_wavlm.py) and live use,
so the model never sees features made differently from the ones it learned on."""
from __future__ import annotations

import numpy as np
import torch

SR = 16_000
MAX_SECONDS = 15.0  # covers 99.7% of MELD clips untouched (p95 is 7.9 s)
MIN_SECONDS = 0.5   # WavLM's convolutional front end needs a minimum length


@torch.inference_mode()
def wavlm_layer_means(model, extractor, audio: np.ndarray) -> np.ndarray:
    """(13, 768) float32 array: each WavLM hidden layer averaged over time, for one 16 kHz clip."""
    audio = audio[: int(MAX_SECONDS * SR)]
    if len(audio) < MIN_SECONDS * SR:
        audio = np.pad(audio, (0, int(MIN_SECONDS * SR) - len(audio)))
    x = extractor(audio, sampling_rate=SR, return_tensors="pt").input_values.to(model.device, model.dtype)
    hidden = model(x, output_hidden_states=True).hidden_states  # 13 x (1, frames, 768)
    return torch.stack([h[0].mean(dim=0) for h in hidden]).float().cpu().numpy()
