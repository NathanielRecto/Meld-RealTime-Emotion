"""Phase 4: does the emotion state actually change Pip's replies?

Takes a stratified sample of test lines (default 5 per emotion) and builds the final state exactly
as the live path would: Whisper's transcript and context turns, the clip's WavLM features, the
fused model, voice cues. It then asks the reply model twice with the same random seed: once with
the plain-language notes, once with notes that carry no emotion information. Both replies go side
by side, in random order, into a blind A/B rating sheet. The key to unblind it is in the JSON.

Usage:
    python scripts/reply_check.py --per-class 5
Writes reports/phase4_replies.json and reports/phase4_reply_sheet.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.conversation import Conversation  # noqa: E402
from meldrt.cues import voice_cues  # noqa: E402
from meldrt.data import load_manifest, with_whisper_text  # noqa: E402
from meldrt.features import MAX_SECONDS, SR  # noqa: E402
from meldrt.reply import notes  # noqa: E402
from meldrt.runtime import expose_torch_cuda_dlls  # noqa: E402

NO_INFO = "Nothing is known about how they feel."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])

    expose_torch_cuda_dlls()
    from meldrt.emotion import EmotionRecognizer  # after the CUDA DLLs are findable
    from meldrt.reply import ReplyGenerator

    emotion, reply = EmotionRecognizer(cfg), ReplyGenerator(cfg)
    df = with_whisper_text(load_manifest(root, "test"), root, "test")
    df = df[df.audio_ok & (df.text.str.split().str.len() >= 3)]  # a reply needs some words to work with
    sample = pd.concat([g.sample(min(len(g), args.per_class), random_state=args.seed)
                        for _, g in df.groupby("emotion")])
    feats_all = np.load(root / "features" / "wavlm_test.npy", mmap_mode="r")
    ids = (root / "features" / "wavlm_test_ids.txt").read_text(encoding="utf-8").split("\n")
    row_of = {clip: i for i, clip in enumerate(ids)}
    order = random.Random(args.seed)

    items = []
    for i, r in enumerate(sample.itertuples(), 1):
        feats = np.asarray(feats_all[row_of[r.clip_id]], dtype=np.float32)
        audio, _ = sf.read(r.wav_path, dtype="float32", frames=int(MAX_SECONDS * SR))
        probs = emotion.probs(r.text, r.speaker, r.prev_turns, feats)
        state = Conversation(robot=cfg["pipeline"]["robot_name"]).add_user_turn(
            1, r.text, probs, voice_cues(audio, n_words=len(r.text.split())))
        with_notes = "".join(reply.stream([], r.text, notes(state), seed=args.seed)).strip()
        without_notes = "".join(reply.stream([], r.text, NO_INFO, seed=args.seed)).strip()
        items.append({"clip_id": r.clip_id, "gold": r.emotion, "transcript": r.text,
                      "emotion": state["emotion"]["label"], "confidence": state["emotion"]["confidence"],
                      "notes": notes(state), "reply_with_notes": with_notes, "reply_without_notes": without_notes,
                      "a_is": order.choice(["with_notes", "without_notes"])})
        print(f"\r{i}/{len(sample)} lines", end="", flush=True)
    print()

    report = {"seed": args.seed, "per_class": args.per_class, "no_info_notes": NO_INFO,
              "predicted_correctly": sum(it["emotion"] == it["gold"] for it in items),
              "replies_identical": sum(it["reply_with_notes"] == it["reply_without_notes"] for it in items),
              "items": items}
    (ROOT / "reports" / "phase4_replies.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Reply check: blind A/B sheet", "",
             "For each line: read what the person said and how they actually felt (MELD's label), then",
             "mark which reply fits better, **A**, **B**, or **=** if neither is better. Which reply had",
             "the emotion notes is hidden; the key is `a_is` in `reports/phase4_replies.json`.", ""]
    for i, it in enumerate(items, 1):
        a, b = ((it["reply_with_notes"], it["reply_without_notes"]) if it["a_is"] == "with_notes"
                else (it["reply_without_notes"], it["reply_with_notes"]))
        lines += [f"## {i}. They felt **{it['gold']}** ({it['clip_id']})", "",
                  f"> \"{it['transcript']}\"", "", f"- **A:** {a}", f"- **B:** {b}", "- **Better:** ___", ""]
    (ROOT / "reports" / "phase4_reply_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(items)} lines; emotion right on {report['predicted_correctly']}; "
          f"identical reply pairs: {report['replies_identical']}")
    print("Wrote reports/phase4_replies.json and reports/phase4_reply_sheet.md")


if __name__ == "__main__":
    main()
