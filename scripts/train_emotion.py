"""Phase 2, step 2: train one emotion model on MELD, pick it on dev, then score it once on test.

Modes:
  text    RoBERTa-base fine-tuned on the utterance (plus earlier turns with --context N)
  audio   a small head on the frozen WavLM features from scripts/extract_wavlm.py
  fusion  both branches trained together: a fused head plus one head per modality

Usage:
    python scripts/train_emotion.py --mode fusion --context 3 --name fusion_ctx3
    python scripts/train_emotion.py --mode text --limit 300 --epochs 1 --name debug   # quick check

Writes reports/phase2_<name>.json and checkpoints/<name>/ (weights and test probabilities,
ignored by git). Runs with --limit keep their report in checkpoints/<name>/ instead.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import EMOTIONS, class_weights, format_text, load_manifest, with_whisper_text  # noqa: E402
from meldrt.model import EmotionModel  # noqa: E402

LABELS = list(range(len(EMOTIONS)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["text", "audio", "fusion"], required=True)
    ap.add_argument("--name", required=True, help="run name for the report and checkpoint folder")
    ap.add_argument("--context", type=int, default=3, help="earlier turns given to the text branch")
    ap.add_argument("--text-source", choices=["meld", "whisper"], default="meld",
                    help="train and score on MELD's typed transcripts or on Whisper's (scripts/transcribe_meld.py)")
    ap.add_argument("--epochs", type=int, default=None, help="default: 5 with text, 40 audio-only")
    ap.add_argument("--batch", type=int, default=None, help="default: 16 with text, 64 audio-only")
    ap.add_argument("--lr-encoder", type=float, default=1.5e-5, help="RoBERTa learning rate")
    ap.add_argument("--lr-head", type=float, default=5e-4, help="learning rate for everything else")
    ap.add_argument("--class-weight-power", type=float, default=0.5, help="0 = unweighted, 1 = inverse frequency")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="fusion: weight of the per-modality losses")
    ap.add_argument("--modality-dropout", type=float, default=0.15, help="fusion: chance audio is hidden per example")
    ap.add_argument("--max-len", type=int, default=192, help="max tokens for utterance + context")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows of each split (debugging)")
    # fp32 by default: on the GTX 1660 SUPER (no tensor cores) fp16 mixed precision measured 4x
    # slower per step (1.75 s vs 0.42 s, docs/DESIGN.md section 8). Newer GPUs should use --amp.
    ap.add_argument("--amp", action="store_true", help="train with fp16 mixed precision")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# --- data ----------------------------------------------------------------------------------


def load_split(root: Path, split: str, args, tok, df=None):
    """Model inputs for one split. Pass `df` to use a table with different text, e.g. Whisper's
    transcripts (scripts/evaluate.py); it must keep the manifest's row order."""
    df = load_manifest(root, split) if df is None else df
    feats = None
    if args.mode != "text":
        feats = np.load(root / "features" / f"wavlm_{split}.npy", mmap_mode="r")
        ids = (root / "features" / f"wavlm_{split}_ids.txt").read_text(encoding="utf-8").split("\n")
        assert ids == df.clip_id.tolist(), f"WavLM features for {split} are stale: rerun extract_wavlm.py"
    if args.limit:
        df = df.iloc[: args.limit]
    items = []
    for i, r in enumerate(df.itertuples()):
        item = {"label": int(r.label), "audio_ok": float(r.audio_ok)}
        if tok is not None:
            target, context = format_text(r.speaker, r.text, r.prev_turns, args.context)
            # truncation=True cuts the longer of the two first, which is almost always the context
            enc = tok(target, context or None, truncation=True, max_length=args.max_len)
            item["input_ids"] = enc["input_ids"]
        if feats is not None:
            item["audio"] = np.array(feats[i])  # float16 (13, 768); widened per batch
        items.append(item)
    return df.reset_index(drop=True), items


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        out = {"labels": torch.tensor([b["label"] for b in batch]),
               "audio_ok": torch.tensor([b["audio_ok"] for b in batch])}
        if "input_ids" in batch[0]:
            longest = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((len(batch), longest), pad_id)
            mask = torch.zeros((len(batch), longest), dtype=torch.long)
            for i, b in enumerate(batch):
                ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
                mask[i, : len(b["input_ids"])] = 1
            out["input_ids"], out["attention_mask"] = ids, mask
        if "audio" in batch[0]:
            out["audio"] = torch.from_numpy(np.stack([b["audio"] for b in batch])).float()
        return out
    return collate


# --- training ------------------------------------------------------------------------------


def forward(model, batch, amp: bool, audio_present=None) -> dict:
    kw = {}
    if "input_ids" in batch:
        kw["input_ids"] = batch["input_ids"].cuda()
        kw["attention_mask"] = batch["attention_mask"].cuda()
    if "audio" in batch:
        kw["audio"] = batch["audio"].cuda()
        kw["audio_present"] = (batch["audio_ok"] if audio_present is None else audio_present).cuda()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        return model(**kw)


def masked_ce(logits, labels, weights, mask=None):
    loss = F.cross_entropy(logits.float(), labels, weight=weights, reduction="none")
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def compute_loss(model, out, labels, present, weights, aux_weight):
    if model.mode == "text":
        return masked_ce(out["fused"], labels, weights)
    if model.mode == "audio":  # clips without audio carry no information here
        return masked_ce(out["fused"], labels, weights, present)
    return (masked_ce(out["fused"], labels, weights)
            + aux_weight * (masked_ce(out["text"], labels, weights)
                            + masked_ce(out["audio"], labels, weights, present)))


def train_epoch(model, loader, opt, sched, scaler, weights, args, amp: bool) -> float:
    model.train()
    total, seen = 0.0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        present = batch["audio_ok"].clone()
        if model.mode == "fusion" and args.modality_dropout > 0:
            # Sometimes hide the audio so the fused head can't lean on it blindly and still
            # works when a clip's audio is missing.
            present *= (torch.rand_like(present) >= args.modality_dropout).float()
        out = forward(model, batch, amp, present)
        labels = batch["labels"].cuda()
        loss = compute_loss(model, out, labels, present.cuda(), weights, args.aux_weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item() * len(labels)
        seen += len(labels)
    return total / seen


@torch.inference_mode()
def predict(model, loader, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    probs: dict[str, list] = {}
    for batch in loader:
        for head, logits in forward(model, batch, amp).items():
            probs.setdefault(head, []).append(logits.float().softmax(-1).cpu())
    return {head: torch.cat(p).numpy() for head, p in probs.items()}


def scores(y: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    per_class = f1_score(y, pred, average=None, labels=LABELS, zero_division=0)
    return {
        "n": int(len(y)),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", labels=LABELS, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "per_class_f1": {e: round(float(f), 4) for e, f in zip(EMOTIONS, per_class)},
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),  # rows = true, cols = predicted
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])
    text_name = cfg["models"]["text_encoder"]["name"]
    amp = args.amp
    uses_text = args.mode != "audio"
    epochs = args.epochs or (5 if uses_text else 40)
    batch_size = args.batch or (16 if uses_text else 64)

    tok = AutoTokenizer.from_pretrained(text_name) if uses_text else None
    tables = {s: load_manifest(root, s) for s in ("train", "dev", "test")}
    if uses_text and args.text_source == "whisper":
        tables = {s: with_whisper_text(df, root, s) for s, df in tables.items()}
    data = {s: load_split(root, s, args, tok, df=df) for s, df in tables.items()}
    y = {s: df.label.to_numpy() for s, (df, _) in data.items()}
    collate = make_collate(tok.pad_token_id if tok else 0)
    train_loader = DataLoader(data["train"][1], batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(data["dev"][1], batch_size=64, collate_fn=collate)
    test_loader = DataLoader(data["test"][1], batch_size=64, collate_fn=collate)

    weights_np = class_weights(y["train"], args.class_weight_power)
    weights = torch.tensor(weights_np, dtype=torch.float32, device="cuda")
    model = EmotionModel(args.mode, text_model=text_name).cuda()
    encoder = [p for n, p in model.named_parameters() if n.startswith("text_encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("text_encoder.")]
    groups = [{"params": rest, "lr": args.lr_head}] + ([{"params": encoder, "lr": args.lr_encoder}] if encoder else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = ROOT / "checkpoints" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.name}: mode={args.mode} context={args.context if uses_text else '-'} "
          f"text={args.text_source if uses_text else '-'} "
          f"train={len(data['train'][1])} epochs={epochs} batch={batch_size} amp={amp}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    history, best_f1, best_epoch = [], -1.0, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = train_epoch(model, train_loader, opt, sched, scaler, weights, args, amp)
        dev_f1 = scores(y["dev"], predict(model, dev_loader, amp)["fused"])["weighted_f1"]
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "dev_weighted_f1": dev_f1,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"  epoch {epoch}/{epochs}: loss {loss:.4f}  dev weighted F1 {dev_f1:.4f}  "
              f"({history[-1]['seconds']} s)", flush=True)
        if dev_f1 > best_f1:  # model selection uses dev only; test is touched once, below
            best_f1, best_epoch = dev_f1, epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
    train_minutes = round((time.perf_counter() - t_start) / 60, 1)

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location="cuda"))
    dev_probs, test_probs = predict(model, dev_loader, amp), predict(model, test_loader, amp)
    heads = ["fused", "text", "audio"] if args.mode == "fusion" else ["fused"]
    short = (data["test"][0].n_words <= 3).to_numpy()  # where the words alone say little
    report = {
        "name": args.name,
        "mode": args.mode,
        "context": args.context if uses_text else None,
        "text_source": args.text_source if uses_text else None,
        "hyperparams": {"epochs": epochs, "batch": batch_size, "lr_encoder": args.lr_encoder,
                        "lr_head": args.lr_head, "class_weight_power": args.class_weight_power,
                        "aux_weight": args.aux_weight, "modality_dropout": args.modality_dropout,
                        "max_len": args.max_len, "amp": amp, "seed": args.seed, "limit": args.limit},
        "class_weights": {e: round(float(w), 3) for e, w in zip(EMOTIONS, weights_np)},
        "history": history,
        "best_epoch": best_epoch,
        "dev": scores(y["dev"], dev_probs["fused"]),
        "test": {h: scores(y["test"], test_probs[h]) for h in heads},
        "test_slices": {
            "3_words_or_less": scores(y["test"][short], test_probs["fused"][short]),
            "4_words_or_more": scores(y["test"][~short], test_probs["fused"][~short]),
        },
        "params": int(sum(p.numel() for p in model.parameters())),
        "train_minutes": train_minutes,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu": torch.cuda.get_device_name(0),
    }
    if model.use_audio:
        layer_w = torch.softmax(model.layer_pool.logits.detach().float(), 0).cpu().numpy()
        report["wavlm_layer_weights"] = [round(float(w), 3) for w in layer_w]
    np.savez(out_dir / "test_probs.npz", clip_id=data["test"][0].clip_id.to_numpy(),
             **{h: test_probs[h] for h in heads})

    out = (out_dir if args.limit else ROOT / "reports") / f"phase2_{args.name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["test"]["fused"]
    print(f"\n{args.name}: best epoch {best_epoch}, dev W-F1 {report['dev']['weighted_f1']}")
    print(f"  test: weighted F1 {t['weighted_f1']}  macro F1 {t['macro_f1']}  accuracy {t['accuracy']}")
    print(f"  test, <=3 words: weighted F1 {report['test_slices']['3_words_or_less']['weighted_f1']}")
    for h in heads[1:]:
        print(f"  test, {h} head alone: weighted F1 {report['test'][h]['weighted_f1']}")
    print(f"  {train_minutes} min, peak VRAM {report['peak_vram_mb']} MB -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
