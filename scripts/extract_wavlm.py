"""Phase 2, step 1: precompute WavLM features for every clip, once, so training is fast.

For each clip, frozen WavLM-base-plus runs over the audio, and each of its 13 hidden layers is
averaged over time, giving a (13, 768) array. Audio is capped at the first MAX_SECONDS: a few
MELD clips run for minutes because their subtitle timestamps are wrong (docs/DESIGN.md section 7).

Usage:
    python scripts/extract_wavlm.py
Writes <meld_root>/features/wavlm_<split>.npy (float16) and wavlm_<split>_ids.txt (row order).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import SPLITS, load_manifest  # noqa: E402

SR = 16_000
MAX_SECONDS = 15.0  # covers 99.7% of clips untouched (p95 is 7.9 s)
MIN_SECONDS = 0.5   # WavLM's convolutional front end needs a minimum length


@torch.inference_mode()
def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])
    c = cfg["models"]["audio_encoder"]
    extractor = AutoFeatureExtractor.from_pretrained(c["name"])
    model = AutoModel.from_pretrained(c["name"], dtype=getattr(torch, c["dtype"])).to("cuda").eval()
    n_layers, dim = model.config.num_hidden_layers + 1, model.config.hidden_size  # +1: input embeddings
    out_dir = root / "features"
    out_dir.mkdir(exist_ok=True)

    for split in SPLITS:
        df = load_manifest(root, split)
        feats = np.zeros((len(df), n_layers, dim), dtype=np.float16)  # missing audio stays all-zero
        t0 = time.perf_counter()
        for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=split)):
            if not row.audio_ok:
                continue
            audio, _ = sf.read(row.wav_path, dtype="float32", frames=int(MAX_SECONDS * SR))
            if len(audio) < MIN_SECONDS * SR:
                audio = np.pad(audio, (0, int(MIN_SECONDS * SR) - len(audio)))
            x = extractor(audio, sampling_rate=SR, return_tensors="pt").input_values.to("cuda", model.dtype)
            hidden = model(x, output_hidden_states=True).hidden_states  # n_layers x (1, frames, dim)
            feats[i] = torch.stack([h[0].mean(dim=0) for h in hidden]).float().cpu().numpy()
        np.save(out_dir / f"wavlm_{split}.npy", feats)
        (out_dir / f"wavlm_{split}_ids.txt").write_text("\n".join(df.clip_id), encoding="utf-8")
        print(f"{split}: {len(df)} clips in {time.perf_counter() - t0:.0f} s -> {out_dir / f'wavlm_{split}.npy'}")


if __name__ == "__main__":
    main()
