"""Re-score trained Phase 2 models, optionally on Whisper's transcripts and as an ensemble.

For each run, loads checkpoints/<run>/model.pt with the settings in reports/phase2_<run>.json and
scores dev and test exactly as in training. With --text whisper (the default), every utterance's
text, and the earlier turns used as context, are swapped for the Whisper transcript from
scripts/transcribe_meld.py: what the robot would actually read. With --ensemble NAME the runs'
probabilities are also averaged and scored as one model, and the spread between the members shows
how much a score moves from one training run to the next.

Usage:
    python scripts/evaluate.py text_ctx0 text_ctx3 fusion_ctx3
    python scripts/evaluate.py fusion_ctx3 fusion_ctx3_seed43 fusion_ctx3_seed44 --ensemble fusion_x3
    python scripts/evaluate.py fusion_ctx3 fusion_ctx3_seed43 fusion_ctx3_seed44 --ensemble fusion_x3 --text meld
Writes reports/eval_{asr|meld}_<run>.json for each run and for the ensemble.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import load_manifest, with_whisper_text  # noqa: E402
from meldrt.model import EmotionModel  # noqa: E402
from train_emotion import load_split, make_collate, predict, scores  # noqa: E402

SPLITS = ("dev", "test")


def score_all(probs: dict, y: dict, short: np.ndarray) -> dict:
    """Scores for every head on dev and test, plus the test slices by utterance length."""
    out = {split: {head: scores(y[split], p) for head, p in probs[split].items()} for split in SPLITS}
    fused = probs["test"]["fused"]
    out["test_slices"] = {"3_words_or_less": scores(y["test"][short], fused[short]),
                          "4_words_or_more": scores(y["test"][~short], fused[~short])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="Phase 2 run names that read text, e.g. text_ctx3 fusion_ctx3")
    ap.add_argument("--text", choices=["whisper", "meld"], default="whisper",
                    help="score on Whisper's transcripts (live condition) or MELD's typed ones")
    ap.add_argument("--ensemble", metavar="NAME", help="also average the runs' probabilities and score that as NAME")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])
    text_name = cfg["models"]["text_encoder"]["name"]
    tok = AutoTokenizer.from_pretrained(text_name)
    collate = make_collate(tok.pad_token_id)
    tag = "asr" if args.text == "whisper" else "meld"

    tables = {split: load_manifest(root, split) for split in SPLITS}
    if args.text == "whisper":
        tables = {split: with_whisper_text(df, root, split) for split, df in tables.items()}
    y = {split: df.label.to_numpy() for split, df in tables.items()}
    short = (tables["test"].n_words <= 3).to_numpy()  # same slice as Phase 2: by MELD's word count

    all_probs, rows = {}, []
    for run in args.runs:
        trained = json.loads((ROOT / "reports" / f"phase2_{run}.json").read_text(encoding="utf-8"))
        mode = trained["mode"]
        if mode == "audio":
            sys.exit(f"{run} is audio-only and doesn't read text")
        settings = Namespace(mode=mode, context=trained["context"], limit=None,
                             max_len=trained["hyperparams"]["max_len"])
        model = EmotionModel(mode, text_model=text_name).cuda()
        model.load_state_dict(torch.load(ROOT / "checkpoints" / run / "model.pt", map_location="cuda"))
        probs = {}
        for split in SPLITS:
            _, items = load_split(root, split, settings, tok, df=tables[split])
            probs[split] = predict(model, DataLoader(items, batch_size=64, collate_fn=collate), amp=False)
            if mode != "fusion":
                probs[split] = {"fused": probs[split]["fused"]}
        del model
        torch.cuda.empty_cache()
        all_probs[run] = probs
        report = {"run": run, "mode": mode, "context": settings.context, "text_source": args.text,
                  **score_all(probs, y, short)}
        (ROOT / "reports" / f"eval_{tag}_{run}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append((run, report))

    if args.ensemble:
        heads = set.intersection(*(set(p["test"]) for p in all_probs.values()))
        mean = {split: {h: np.mean([all_probs[r][split][h] for r in args.runs], axis=0) for h in heads}
                for split in SPLITS}
        member_wf1 = [rep["test"]["fused"]["weighted_f1"] for _, rep in rows]
        report = {"run": args.ensemble, "members": args.runs, "text_source": args.text,
                  "member_test_wf1": {"values": member_wf1, "mean": round(float(np.mean(member_wf1)), 4),
                                      "std": round(float(np.std(member_wf1, ddof=1)), 4) if len(member_wf1) > 1 else None},
                  **score_all(mean, y, short)}
        (ROOT / "reports" / f"eval_{tag}_{args.ensemble}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append((f"{args.ensemble} (average)", report))

    print(f"\nScored on {args.text} text")
    print(f"{'run':<28}{'dev W-F1':>9}{'test W-F1':>10}{'macro':>7}{'acc':>7}")
    for name, rep in rows:
        t = rep["test"]["fused"]
        print(f"{name:<28}{rep['dev']['fused']['weighted_f1']:>9.3f}{t['weighted_f1']:>10.3f}"
              f"{t['macro_f1']:>7.3f}{t['accuracy']:>7.3f}")
    if args.ensemble and len(args.runs) > 1:
        m = rows[-1][1]["member_test_wf1"]
        print(f"members' test W-F1: mean {m['mean']:.3f}, std {m['std']:.3f} (run-to-run noise)")


if __name__ == "__main__":
    main()
