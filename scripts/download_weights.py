"""Download the trained emotion models, so the demo runs without MELD, CREMA-D or any training.

The live model's three fusion runs (configs/default.yaml, models.emotion.runs) are published as
assets of this repo's GitHub release `weights-v1`. Each file is checked against its SHA-256 before
it replaces anything, and files already in place are skipped.

Usage:
    python scripts/download_weights.py          # about 1.5 GB, into checkpoints/<run>/model.pt
Then run `python scripts/serve.py` and use the microphone.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/NathanielRecto/Meld-RealTime-Emotion/releases/download/weights-v1"
# run -> (bytes, SHA-256) of checkpoints/<run>/model.pt, published as <run>.pt
WEIGHTS = {
    "fusion_ctx3_crema_td25":
        (498423523, "e1b0693c751c599f1a158219d3335501069a0e2b19435a560b52a4a6dd5a6a81"),
    "fusion_ctx3_crema_td25_seed43":
        (498423523, "48c84e548689e05e184908dc60198e3765450ac57c27fcb315d178953a161c5c"),
    "fusion_ctx3_crema_td25_seed44":
        (498423523, "587915c32066998cafc1f6b51d802c35b7471b10497a84012cd8bb0218fc6f34"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, tmp: Path, size: int) -> None:
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        done = 0
        while block := response.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r  {done / 2**20:,.0f} / {size / 2**20:,.0f} MB", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=RELEASE, help="where the <run>.pt files are (default: the GitHub release)")
    ap.add_argument("--dest", type=Path, default=ROOT / "checkpoints", help="checkpoint folder")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    runs = cfg["models"]["emotion"]["runs"]
    missing = [run for run in runs if run not in WEIGHTS]
    if missing:
        sys.exit(f"No published weights for {', '.join(missing)}: train them with scripts/train_emotion.py (README)")

    for run in runs:
        size, digest = WEIGHTS[run]
        dest = args.dest / run / "model.pt"
        if dest.exists() and dest.stat().st_size == size and sha256(dest) == digest:
            print(f"{run}: already in place")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name("model.pt.part")
        print(f"{run}: downloading")
        fetch(f"{args.url}/{run}.pt", tmp, size)
        if sha256(tmp) != digest:
            tmp.unlink()
            sys.exit(f"{run}: checksum mismatch, so the download was discarded. Try again.")
        tmp.replace(dest)  # only a verified file ever becomes model.pt
        print(f"{run}: verified -> {dest}")
    print("Done. Start the demo with: python scripts/serve.py")


if __name__ == "__main__":
    main()
