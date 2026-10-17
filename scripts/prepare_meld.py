"""Phase 1: turn the raw MELD download into clean, model-ready tables.

Steps (each skips work that is already done, so it is safe to re-run or resume):
  extract   MELD.Raw.tar.gz, then the train/dev/test video archives inside it
  audio     every video clip -> 16 kHz mono WAV, the format the models and the live mic use
  manifest  one CSV per split: labels, cleaned transcript, previous turns, audio path, duration
  stats     class balance, durations, broken clips -> reports/phase1_data.json

Usage:
    python scripts/prepare_meld.py [all|extract|audio|manifest|stats] [--workers 8]

Everything derived lives next to the download (paths.meld_root in configs/default.yaml):
    MELD.Raw/              extracted CSVs and video folders
    audio16k/<split>/      dia<D>_utt<U>.wav
    manifests/<split>.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import previous_turns  # noqa: E402  (shared with training and evaluation)

SR = 16_000
SPLITS = ["train", "dev", "test"]
VIDEO_DIR = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MANIFEST_COLUMNS = [
    "split", "clip_id", "dialogue_id", "utterance_id", "speaker", "text", "text_raw", "emotion",
    "sentiment", "n_words", "prev_turns", "wav_path", "audio_ok", "duration_s",
    "subtitle_duration_s", "season", "episode",
]


def clip_id(dialogue_id: int, utterance_id: int) -> str:
    return f"dia{dialogue_id}_utt{utterance_id}"  # MELD's own video file naming


# --- extract -------------------------------------------------------------------------------


def _keep(member: tarfile.TarInfo, dest: Path) -> bool:
    """Archives are untrusted input: only plain files and folders that land inside `dest`
    (no absolute paths, '..' tricks, links or device files). macOS '._' metadata files are
    skipped because they aren't data."""
    name = Path(member.name)
    if name.name.startswith("._") or "__MACOSX" in name.parts:
        return False
    if not (member.isfile() or member.isdir()):
        raise RuntimeError(f"refusing non-regular archive member {member.name!r}")
    target = (dest / name).resolve()
    if target != dest and dest not in target.parents:
        raise RuntimeError(f"refusing archive member outside {dest}: {member.name!r}")
    return True


def extract_archive(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    marker = dest / f".extracted_{archive.name}"
    if marker.exists():
        print(f"  {archive.name}: already extracted")
        return
    n_files = 0
    with tarfile.open(archive, "r|gz") as tar:  # stream mode: one sequential pass over the file
        for member in tqdm(tar, desc=f"  {archive.name}", unit=" entries"):
            if _keep(member, dest):
                tar.extract(member, dest, set_attrs=False)
                n_files += member.isfile()
    marker.write_text(f"{n_files} files\n")


def step_extract(root: Path) -> None:
    print("[extract]")
    extract_archive(root / "MELD.Raw.tar.gz", root)
    raw = root / "MELD.Raw"
    for split in SPLITS:
        extract_archive(raw / f"{split}.tar.gz", raw)


# --- audio ---------------------------------------------------------------------------------


def read_raw_csv(root: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(root / "MELD.Raw" / f"{split}_sent_emo.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"sr no.": "sr_no", "utterance": "text_raw",
                            "starttime": "start_time", "endtime": "end_time"})
    df["clip_id"] = [clip_id(d, u) for d, u in zip(df.dialogue_id, df.utterance_id)]
    return df


def convert(job: tuple[Path, Path, str]) -> tuple[str, str | None]:
    """One clip -> 16 kHz mono 16-bit WAV. Returns (clip_id, error or None)."""
    src, dst, ffmpeg = job
    if dst.exists():
        return dst.stem, None
    if not src.exists():
        return dst.stem, "video file missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:  # 44 B = empty WAV
        tmp.unlink(missing_ok=True)
        lines = result.stderr.strip().splitlines() or ["no audio stream"]
        return dst.stem, lines[-1][:200]
    tmp.replace(dst)  # rename only once complete, so an interrupted run never leaves half a file
    return dst.stem, None


def step_audio(root: Path, workers: int) -> None:
    print("[audio]")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    failures = {}
    for split in SPLITS:
        videos = root / "MELD.Raw" / VIDEO_DIR[split]
        jobs = [(videos / f"{c}.mp4", root / "audio16k" / split / f"{c}.wav", ffmpeg)
                for c in read_raw_csv(root, split).clip_id]
        with ThreadPoolExecutor(workers) as pool:  # threads are enough: ffmpeg does the work
            results = list(tqdm(pool.map(convert, jobs), total=len(jobs), desc=f"  {split}", unit=" clips"))
        failures[split] = {c: err for c, err in results if err}
        print(f"  {split}: {len(jobs) - len(failures[split])}/{len(jobs)} clips converted")
    out = ROOT / "reports" / "phase1_audio_failures.json"
    out.write_text(json.dumps(failures, indent=2), encoding="utf-8")


# --- manifest ------------------------------------------------------------------------------


# MELD's CSVs store Windows-1252 punctuation bytes as raw control characters ("company\x92s").
# Map them, and their proper Unicode forms, to plain ASCII so training text looks like what
# Whisper produces live ("company's").
PUNCT_FIXES = str.maketrans({
    "\x91": "'", "\x92": "'", "‘": "'", "’": "'",
    "\x93": '"', "\x94": '"', "“": '"', "”": '"',
    "\x85": "...", "…": "...",
    "\x96": "-", "–": "-", "\x97": " - ", "—": " - ",
    "\xa0": " ",
})


def clean_text(text: str) -> str:
    return " ".join(str(text).translate(PUNCT_FIXES).split())


def subtitle_seconds(t: str) -> float:
    h, m, s = str(t).replace(",", ".").split(":")  # e.g. "00:16:16,059"
    return int(h) * 3600 + int(m) * 60 + float(s)


def wav_seconds(path: Path) -> float | None:
    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError):
        return None
    return info.frames / info.samplerate if info.frames > 0 else None


def step_manifest(root: Path) -> None:
    print("[manifest]")
    out_dir = root / "manifests"
    out_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        df = read_raw_csv(root, split).sort_values(["dialogue_id", "utterance_id"]).reset_index(drop=True)
        df["split"] = split
        df["text"] = df.text_raw.map(clean_text)
        df["emotion"] = df.emotion.str.strip().str.lower()
        df["sentiment"] = df.sentiment.str.strip().str.lower()
        df["n_words"] = df.text.str.split().str.len()
        df["prev_turns"] = [json.dumps(p, ensure_ascii=False) for p in previous_turns(df)]
        df["wav_path"] = [str(root / "audio16k" / split / f"{c}.wav") for c in df.clip_id]
        durations = [wav_seconds(Path(p)) for p in tqdm(df.wav_path, desc=f"  {split}", unit=" wavs")]
        df["audio_ok"] = [d is not None for d in durations]
        df["duration_s"] = [round(d, 3) if d is not None else np.nan for d in durations]
        df["subtitle_duration_s"] = (df.end_time.map(subtitle_seconds) - df.start_time.map(subtitle_seconds)).round(3)
        df[MANIFEST_COLUMNS].to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8")
        print(f"  {split}: {len(df)} rows, {int(df.audio_ok.sum())} with audio -> {out_dir / f'{split}.csv'}")


# --- stats ---------------------------------------------------------------------------------


def _ints(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.items()}


def split_stats(df: pd.DataFrame) -> dict:
    dur = df.loc[df.audio_ok, "duration_s"]
    counts = df.emotion.value_counts().reindex(EMOTIONS, fill_value=0)
    return {
        "utterances": len(df),
        "dialogues": int(df.dialogue_id.nunique()),
        "speakers": int(df.speaker.nunique()),
        "emotion_counts": _ints(counts),
        "emotion_pct": {k: round(100 * v / len(df), 1) for k, v in _ints(counts).items()},
        "sentiment_counts": _ints(df.sentiment.value_counts()),
        "audio_ok": int(df.audio_ok.sum()),
        "audio_missing_or_broken": df.loc[~df.audio_ok, "clip_id"].tolist(),
        "duration_s": {"mean": round(float(dur.mean()), 2), "median": round(float(dur.median()), 2),
                       "p95": round(float(dur.quantile(0.95)), 2), "max": round(float(dur.max()), 2)},
        "clips_over_30s": int((dur > 30).sum()),  # longer than one Whisper window
        "clips_off_subtitle_by_2s": int(((df.duration_s - df.subtitle_duration_s).abs() > 2).sum()),
        "words_median": float(df.n_words.median()),
        "pct_3_words_or_less": round(float((df.n_words <= 3).mean() * 100), 1),
        "unknown_emotion_labels": sorted(set(df.emotion) - set(EMOTIONS)),
    }


def print_stats(report: dict) -> None:
    s = report["splits"]
    print(f"\n{'':<10}" + "".join(f"{split:>10}" for split in SPLITS))
    rows = [("utterances", lambda x: x["utterances"]), ("dialogues", lambda x: x["dialogues"]),
            ("audio ok", lambda x: x["audio_ok"])]
    rows += [(e, lambda x, e=e: f"{x['emotion_pct'][e]}%") for e in EMOTIONS]
    rows += [("median s", lambda x: x["duration_s"]["median"]), ("p95 s", lambda x: x["duration_s"]["p95"]),
             ("max s", lambda x: x["duration_s"]["max"]), (">30 s", lambda x: x["clips_over_30s"]),
             ("<=3 words", lambda x: f"{x['pct_3_words_or_less']}%")]
    for label, get in rows:
        print(f"{label:<10}" + "".join(f"{get(s[split])!s:>10}" for split in SPLITS))
    b = report["majority_baseline_test"]
    print(f"\nAlways-'neutral' baseline on test: weighted F1 {b['weighted_f1']}, macro F1 {b['macro_f1']}")


def step_stats(root: Path) -> None:
    print("[stats]")
    frames = {split: pd.read_csv(root / "manifests" / f"{split}.csv") for split in SPLITS}
    test = frames["test"]
    majority = ["neutral"] * len(test)
    report = {
        "splits": {split: split_stats(df) for split, df in frames.items()},
        "majority_baseline_test": {
            "predict": "neutral",
            "weighted_f1": round(f1_score(test.emotion, majority, average="weighted", zero_division=0), 3),
            "macro_f1": round(f1_score(test.emotion, majority, average="macro", zero_division=0), 3),
        },
    }
    out = ROOT / "reports" / "phase1_data.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_stats(report)
    print(f"\nWrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", nargs="?", default="all", choices=["all", "extract", "audio", "manifest", "stats"])
    ap.add_argument("--workers", type=int, default=8, help="parallel ffmpeg processes for the audio step")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["meld_root"])

    if args.step in ("all", "extract"):
        step_extract(root)
    if args.step in ("all", "audio"):
        step_audio(root, args.workers)
    if args.step in ("all", "manifest"):
        step_manifest(root)
    if args.step in ("all", "stats"):
        step_stats(root)


if __name__ == "__main__":
    main()
