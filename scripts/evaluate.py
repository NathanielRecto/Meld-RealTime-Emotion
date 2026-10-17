"""Re-score trained Phase 2 models on Whisper's transcripts instead of MELD's typed ones.

For each run, loads checkpoints/<run>/model.pt with the settings in reports/phase2_<run>.json,
replaces every utterance's text, and the earlier turns used as context, with the Whisper
transcript from scripts/transcribe_meld.py, then scores dev and test exactly as in training.

Usage:
    python scripts/evaluate.py text_ctx0 text_ctx3 fusion_ctx3
Writes reports/eval_asr_<run>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import load_manifest, with_whisper_text  # noqa: E402
from meldrt.model import EmotionModel  # noqa: E402
from train_emotion import load_split, make_collate, predict, scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="Phase 2 run names that read text, e.g. text_ctx3 fusion_ctx3")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])
    text_name = cfg["models"]["text_encoder"]["name"]
    tok = AutoTokenizer.from_pretrained(text_name)
    collate = make_collate(tok.pad_token_id)

    tables = {split: with_whisper_text(load_manifest(root, split), root, split) for split in ("dev", "test")}
    short = (tables["test"].n_words <= 3).to_numpy()  # same slice as Phase 2: by MELD's word count

    rows = []
    for run in args.runs:
        trained = json.loads((ROOT / "reports" / f"phase2_{run}.json").read_text(encoding="utf-8"))
        mode = trained["mode"]
        if mode == "audio":
            sys.exit(f"{run} is audio-only and doesn't read text")
        settings = Namespace(mode=mode, context=trained["context"], limit=None,
                             max_len=trained["hyperparams"]["max_len"])
        model = EmotionModel(mode, text_model=text_name).cuda()
        model.load_state_dict(torch.load(ROOT / "checkpoints" / run / "model.pt", map_location="cuda"))
        heads = ["fused", "text", "audio"] if mode == "fusion" else ["fused"]

        report = {"run": run, "mode": mode, "context": settings.context, "text_source": "whisper"}
        for split in ("dev", "test"):
            df, items = load_split(root, split, settings, tok, df=tables[split])
            probs = predict(model, DataLoader(items, batch_size=64, collate_fn=collate), amp=False)
            y = df.label.to_numpy()
            report[split] = {h: scores(y, probs[h]) for h in heads}
        report["test_slices"] = {
            "3_words_or_less": scores(y[short], probs["fused"][short]),
            "4_words_or_more": scores(y[~short], probs["fused"][~short]),
        }
        out = ROOT / "reports" / f"eval_asr_{run}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for h in heads:
            rows.append((run if h == "fused" else f"  {h} head", trained["test"][h]["weighted_f1"],
                         report["test"][h]["weighted_f1"]))
        del model
        torch.cuda.empty_cache()

    print(f"\n{'test weighted F1':<18}{'MELD text':>11}{'Whisper':>9}{'change':>8}")
    for name, gold, whisper in rows:
        print(f"{name:<18}{gold:>11.3f}{whisper:>9.3f}{whisper - gold:>+8.3f}")


if __name__ == "__main__":
    main()
