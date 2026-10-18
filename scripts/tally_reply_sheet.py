"""Phase 4: unblind and tally the A/B reply sheet once a person has rated it.

reports/phase4_reply_sheet.md lists 35 test lines, each with two replies in random order: one
written with the emotion notes and one with notes that say nothing about feelings. The rater
replaces each "___" after **Better:** with A, B or = (neither is better), without opening the key.
Only then does this script read the key (`a_is` in reports/phase4_replies.json) and count how
often the reply with the notes won.

Usage:
    python scripts/tally_reply_sheet.py
Writes reports/phase4_reply_ratings.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATING = re.compile(r"^- \*\*Better:\*\*\s*(.*?)\s*$", re.M)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value for wins vs losses (ties left out)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def count(results: list[dict]) -> dict:
    c = {"with_notes_better": 0, "without_notes_better": 0, "same": 0}
    for r in results:
        c[{"with_notes": "with_notes_better", "without_notes": "without_notes_better"}.get(r["winner"], "same")] += 1
    c["n"] = len(results)
    c["sign_test_p"] = round(sign_test(c["with_notes_better"], c["without_notes_better"]), 3)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet", type=Path, default=ROOT / "reports" / "phase4_reply_sheet.md")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "phase4_reply_ratings.json")
    args = ap.parse_args()
    ratings = [r.upper() for r in RATING.findall(args.sheet.read_text(encoding="utf-8"))]
    unrated = [i for i, r in enumerate(ratings, 1) if r not in ("A", "B", "=")]
    if unrated:
        sys.exit(f"Lines {', '.join(map(str, unrated))} aren't rated yet: write A, B or = after **Better:**")

    key = json.loads((ROOT / "reports" / "phase4_replies.json").read_text(encoding="utf-8"))
    items = key["items"]
    if len(ratings) != len(items):
        sys.exit(f"The sheet has {len(ratings)} ratings but the key has {len(items)} lines")
    results = []
    for i, (rating, it) in enumerate(zip(ratings, items), 1):
        b_is = "without_notes" if it["a_is"] == "with_notes" else "with_notes"
        winner = {"A": it["a_is"], "B": b_is, "=": "same"}[rating]
        results.append({"line": i, "clip_id": it["clip_id"], "gold": it["gold"], "predicted": it["emotion"],
                        "emotion_right": it["emotion"] == it["gold"], "rating": rating, "winner": winner})

    right = [r for r in results if r["emotion_right"]]
    wrong = [r for r in results if not r["emotion_right"]]
    report = {"rated_sheet": args.sheet.name, "all": count(results),
              "emotion_predicted_right": count(right), "emotion_predicted_wrong": count(wrong),
              "by_gold_emotion": {e: count([r for r in results if r["gold"] == e])
                                  for e in sorted({r["gold"] for r in results})},
              "lines": results}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'':<28}{'notes better':>13}{'no-notes better':>16}{'same':>6}{'p':>7}")
    for name, c in [("all lines", report["all"]), ("emotion predicted right", report["emotion_predicted_right"]),
                    ("emotion predicted wrong", report["emotion_predicted_wrong"])]:
        label = f"{name} ({c['n']})"
        print(f"{label:<28}{c['with_notes_better']:>13}{c['without_notes_better']:>16}{c['same']:>6}"
              f"{c['sign_test_p']:>7}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
