"""Phase 5: the tone test. Does the emotion tag follow the voice when the words say little?

Scores trained fusion runs on the CREMA-D tone sets from scripts/prepare_crema.py: actors and
sentences never seen in training, emotionally flat sentences, and only clips whose emotion
listeners could hear from the voice alone. MELD's own test can't show this, because there the
words usually carry the label.

Main number: balanced accuracy of the final (fused) tag, the average recall over the 6 emotions,
so the many neutral clips don't dominate. Also: recall on the emotional clips alone (does the
tag pick up a tone at all?), on the neutral ones (does it invent one?), and the same for the
voice-only head, which is the live voice tag.

Usage:
    python scripts/evaluate_tone.py fusion_ctx3 fusion_ctx3_crema_td25
    python scripts/evaluate_tone.py fusion_ctx3 fusion_ctx3_seed43 fusion_ctx3_seed44 --ensemble fusion_x3
Writes reports/tone_<run>.json for each run and for the ensemble.
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
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import EMOTIONS, LABEL_ID, load_manifest  # noqa: E402
from meldrt.model import EmotionModel  # noqa: E402
from train_emotion import LABELS, load_split, make_collate, predict  # noqa: E402

SPLITS = ("dev", "test")
HEADS = ("fused", "audio")  # the final tag and the voice tag; the words head has nothing to go on here
NEUTRAL = LABEL_ID["neutral"]


def tone_scores(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    present = [label for label in LABELS if (y == label).any()]
    recall = {EMOTIONS[label]: round(float((pred[y == label] == label).mean()), 4) for label in present}
    emotional = y != NEUTRAL
    return {
        "n": int(len(y)),
        "balanced_accuracy": round(float(np.mean(list(recall.values()))), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "emotional_recall": round(float((pred[emotional] == y[emotional]).mean()), 4),
        "neutral_recall": round(float((pred[~emotional] == NEUTRAL).mean()), 4),
        "says_neutral": round(float((pred == NEUTRAL).mean()), 4),
        "recall": recall,
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),  # rows = true, cols = predicted
    }


def score_all(probs: dict, y: dict) -> dict:
    return {split: {head: tone_scores(y[split], probs[split][head]) for head in HEADS} for split in SPLITS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="Phase 2 fusion run names")
    ap.add_argument("--ensemble", metavar="NAME", help="also average the runs' probabilities and score that as NAME")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["crema_root"])
    text_name = cfg["models"]["text_encoder"]["name"]
    tok = AutoTokenizer.from_pretrained(text_name)
    collate = make_collate(tok.pad_token_id)
    y = {split: load_manifest(root, split).label.to_numpy() for split in SPLITS}

    all_probs, rows = {}, []
    for run in args.runs:
        trained = json.loads((ROOT / "reports" / f"phase2_{run}.json").read_text(encoding="utf-8"))
        if trained["mode"] != "fusion":
            sys.exit(f"{run} is a {trained['mode']} model; the tone test scores fusion models")
        settings = Namespace(mode="fusion", context=trained["context"], limit=None,
                             max_len=trained["hyperparams"]["max_len"])
        model = EmotionModel("fusion", text_model=text_name).cuda()
        model.load_state_dict(torch.load(ROOT / "checkpoints" / run / "model.pt", map_location="cuda"))
        probs = {}
        for split in SPLITS:
            _, items = load_split(root, split, settings, tok, words_carry_label=False)
            probs[split] = predict(model, DataLoader(items, batch_size=64, collate_fn=collate), amp=False)
        del model
        torch.cuda.empty_cache()
        all_probs[run] = probs
        report = {"run": run, "meld_dev_weighted_f1": trained["dev"]["weighted_f1"],
                  "meld_test_weighted_f1": trained["test"]["fused"]["weighted_f1"], **score_all(probs, y)}
        (ROOT / "reports" / f"tone_{run}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append((run, report))

    if args.ensemble:
        mean = {split: {h: np.mean([all_probs[r][split][h] for r in args.runs], axis=0) for h in HEADS}
                for split in SPLITS}
        report = {"run": args.ensemble, "members": args.runs, **score_all(mean, y)}
        (ROOT / "reports" / f"tone_{args.ensemble}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append((f"{args.ensemble} (average)", report))

    print("\nTone test (CREMA-D, unseen actors and sentences). bal = balanced accuracy, "
          "emo = recall on emotional clips, neu = recall on neutral clips")
    print(f"{'run':<30}{'MELD dev':>9}{'dev bal':>9}{'test bal':>9}{'emo':>6}{'neu':>6}{'voice bal':>10}")
    for name, rep in rows:
        d, t, v = rep["dev"]["fused"], rep["test"]["fused"], rep["test"]["audio"]
        meld = f"{rep['meld_dev_weighted_f1']:.3f}" if "meld_dev_weighted_f1" in rep else "-"
        print(f"{name:<30}{meld:>9}{d['balanced_accuracy']:>9.3f}{t['balanced_accuracy']:>9.3f}"
              f"{t['emotional_recall']:>6.2f}{t['neutral_recall']:>6.2f}{v['balanced_accuracy']:>10.3f}")


if __name__ == "__main__":
    main()
