"""Phase 5: turn CREMA-D into extra voice-training data and a tone test set.

CREMA-D (github.com/CheyneyComputerScience/CREMA-D, ODbL 1.0 / DbCL 1.0): 7,442 clips of 91 actors
reading 12 emotionally flat sentences in 6 emotions. The words carry no emotion, so the label can
only come from the voice. That is what MELD can't teach or test: there, the words usually win.

Clips kept: those where listeners rating the voice alone (processedResults/summaryTable.csv,
VoiceVote) agreed with the emotion the actor intended, about 3,100 of 7,442. Most of the others
sound neutral to people, and the robot should react to tone a person can actually hear. Four
clips the README lists as broken or mislabelled are dropped too.

Split, so that no actor and no sentence is shared between training and the tone sets:
  actors     shuffled with a fixed seed within each sex: 60% train, 13% dev, 27% test
  sentences  4 of the 11 one-take sentences held out; IEO (several intensity levels) stays in training
  train = train actors x the 8 training sentences
  dev   = dev actors x the 4 held-out sentences    (to choose between training recipes)
  test  = test actors x the 4 held-out sentences   (scored once, at the end)

Usage:
    python scripts/prepare_crema.py
Reads <crema_root>/repo (the git clone, AudioWAV pulled with Git LFS). Writes
<crema_root>/manifests/<split>.csv, in the columns the MELD loaders read, and reports/crema_data.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from meldrt.data import SPLITS  # noqa: E402

SEED = 0
SR = 16_000
SENTENCES = {  # CREMA-D README
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}
EMOTION = {"A": "anger", "D": "disgust", "F": "fear", "H": "joy", "N": "neutral", "S": "sadness"}
# README "Audio Files" notes: no audio, no duration, or a minute of every emotion in one file. The
# raters heard the broken MP3s, so the votes for these can't be trusted either.
BROKEN = {"1076_MTI_NEU_XX", "1076_MTI_SAD_XX", "1064_TIE_SAD_XX", "1064_IEO_DIS_MD"}
N_HELD_OUT_SENTENCES = 4
ACTOR_SHARE = {"dev": 0.13, "test": 0.27}  # the rest train


def wav_seconds(path: Path) -> float | None:
    """Length of a 16 kHz mono WAV, or None if it's unreadable, empty or in another format."""
    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError):
        return None
    if info.frames == 0 or info.samplerate != SR or info.channels != 1:
        return None
    return info.frames / info.samplerate


def split_actors(demo: pd.DataFrame, rng: np.random.Generator) -> dict[int, str]:
    """Actor -> split, shuffled within each sex so every split has a similar mix of voices."""
    out = {}
    for _, g in demo.groupby("Sex"):
        actors = rng.permutation(g.ActorID.to_numpy())
        n_test = round(ACTOR_SHARE["test"] * len(actors))
        n_dev = round(ACTOR_SHARE["dev"] * len(actors))
        for i, a in enumerate(actors):
            out[int(a)] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    return out


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    root = Path(cfg["paths"]["crema_root"])
    repo = root / "repo"
    rng = np.random.default_rng(SEED)

    votes = pd.read_csv(repo / "processedResults" / "summaryTable.csv")
    demo = pd.read_csv(repo / "VideoDemographics.csv")
    df = votes[["FileName", "VoiceVote", "VoiceLevel"]].rename(
        columns={"FileName": "clip_id", "VoiceVote": "voice_vote", "VoiceLevel": "voice_vote_pct"})
    parts = df.clip_id.str.split("_", expand=True)
    df["actor"], df["sentence"], df["level"] = parts[0].astype(int), parts[1], parts[3]
    df["intended"] = parts[2].str[0]  # ANG -> A, matches VoiceVote's letters
    n_all = len(df)
    df = df[(df.voice_vote == df.intended) & ~df.clip_id.isin(BROKEN)].copy()  # ties ("N:S") never match

    held_out = sorted(rng.choice(sorted(set(SENTENCES) - {"IEO"}), N_HELD_OUT_SENTENCES, replace=False).tolist())
    actor_split = split_actors(demo, rng)
    df["actor_split"] = df.actor.map(actor_split)
    df["held_out_sentence"] = df.sentence.isin(held_out)
    df["split"] = np.where(df.actor_split == "train",
                           np.where(df.held_out_sentence, None, "train"),
                           np.where(df.held_out_sentence, df.actor_split, None))
    df = df[df.split.notna()].copy()

    df["emotion"] = df.intended.map(EMOTION)
    df["speaker"] = "person"  # anonymised to "A" by format_text, like the live user
    df["text"] = df.sentence.map(SENTENCES)
    df["n_words"] = df.text.str.split().str.len()
    df["prev_turns"] = "[]"  # single sentences, no conversation
    df["wav_path"] = [str(repo / "AudioWAV" / f"{c}.wav") for c in df.clip_id]
    durations = [wav_seconds(Path(p)) for p in df.wav_path]
    df["audio_ok"] = [d is not None for d in durations]
    df["duration_s"] = [round(d, 3) if d is not None else np.nan for d in durations]
    df["sex"] = df.actor.map(demo.set_index("ActorID").Sex)

    out_dir = root / "manifests"
    out_dir.mkdir(exist_ok=True)
    columns = ["split", "clip_id", "actor", "sex", "sentence", "level", "speaker", "text", "emotion",
               "n_words", "prev_turns", "wav_path", "audio_ok", "duration_s", "voice_vote_pct"]
    stats = {"source": "CREMA-D (github.com/CheyneyComputerScience/CREMA-D), ODbL 1.0 / DbCL 1.0",
             "clips_total": n_all, "kept_rule": "listeners' voice-only majority vote == intended emotion",
             "held_out_sentences": {s: SENTENCES[s] for s in held_out}, "seed": SEED, "splits": {}}
    for split in SPLITS:
        part = df[df.split == split].sort_values("clip_id")
        part[columns].to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8")
        stats["splits"][split] = {
            "clips": len(part), "audio_ok": int(part.audio_ok.sum()),
            "actors": int(part.actor.nunique()),
            "actors_by_sex": {k: int(v) for k, v in part.groupby("sex").actor.nunique().items()},
            "sentences": sorted(part.sentence.unique().tolist()),
            "emotions": {k: int(v) for k, v in part.emotion.value_counts().items()},
            "seconds_median": round(float(part.duration_s.median()), 2),
        }
        print(f"{split}: {len(part)} clips, {part.actor.nunique()} actors, {part.audio_ok.sum()} with audio "
              f"-> {out_dir / f'{split}.csv'}")
    print(f"held-out sentences: {', '.join(held_out)}")
    out = ROOT / "reports" / "crema_data.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
