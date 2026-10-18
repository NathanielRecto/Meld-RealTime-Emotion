# MELD Realtime Emotion

A small real-time prototype for an emotion-aware character robot. It listens to a person, works
out how they feel from **what they say and how they say it**, and returns:

1. a structured state (JSON) with one of MELD's 7 emotions, and
2. a short reply that is streamed as it is generated.

Built on the [MELD](https://affective-meld.github.io/) dataset, with
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) added so the tag also follows the tone of
voice when the words say little. Everything runs locally on one consumer GPU, within a 6B
total-parameter budget.

## Results

| | Measured (details in [docs/DESIGN.md](docs/DESIGN.md)) |
|---|---|
| Emotion, MELD test, weighted F1 | **0.651** on MELD's typed transcripts, **0.557** on Whisper's live-style transcripts. The always-"neutral" floor is 0.313. |
| Tone, CREMA-D (flat sentences, emotional voices, unseen speakers) | Final tag follows the voice: **0.663** balanced accuracy, up from 0.127 before CREMA-D training (chance 0.167) |
| Latency, p95 from the end of the turn | voice tag **67 ms** · final tag **935 ms** · first reply word **1.33 s** · full reply **1.62 s** |
| Size | **2.25B** parameters (the limit is 6B); 5.4 GB peak GPU memory on a 6 GB GTX 1660 SUPER |

How one turn flows: turn detection (Silero VAD) → **voice-only emotion tag** (WavLM + a trained
head) → speech-to-text (Whisper small) → **final emotion tag** from words + voice + the last 3
turns (3 fine-tuned RoBERTa fusion models, averaged) → a short reply streamed from Qwen2.5-1.5B
(llama.cpp). The design doc explains the decisions, the definition of "real-time", what was tried
and didn't work, and the limitations.

## Setup (Windows, NVIDIA GPU)

1. Create and activate a virtual environment. Use Python 3.10–3.12, the versions with prebuilt
   llama.cpp GPU wheels:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install the GPU build of PyTorch. It bundles the CUDA libraries, so no separate CUDA install is
   needed:
   ```powershell
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
3. Install everything else. This includes llama.cpp's prebuilt CUDA wheel, fetched from its own
   package index listed in the file:
   ```powershell
   pip install -r requirements.txt
   ```
4. Check that every model loads and fits on your GPU. The first run downloads about 2.5 GB of model
   weights:
   ```powershell
   python scripts/smoke_test.py
   ```
   Results are written to `reports/phase0_smoke.json`.

## Quick start: run the demo without any training

The trained emotion models are published with the GitHub release
[`weights-v1`](https://github.com/NathanielRecto/Meld-RealTime-Emotion/releases/tag/weights-v1),
so the live demo needs neither MELD nor CREMA-D. After the setup above:

```powershell
python scripts/download_weights.py   # 3 files, about 1.5 GB, each checked against its SHA-256
python scripts/serve.py              # then open http://127.0.0.1:8000 and click Start microphone
```

To follow one recording of your own (16 kHz mono WAV) through every stage in the terminal:
`python scripts/trace.py --wav your_clip.wav`. Replaying MELD lines in the web page, and everything
under Data and Training below, does need the datasets.

## Data

MELD (about 10.9 GB) is kept **outside** the repo. The default location is `G:/datasets/MELD`;
change `paths.meld_root` in `configs/default.yaml` for your machine.

1. Download `MELD.Raw.tar.gz` into that folder:
   <https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz>
2. Prepare it. This extracts the archives, converts every clip to 16 kHz mono WAV, and builds one
   table per split:
   ```powershell
   python scripts/prepare_meld.py
   ```
   Each step can also run on its own (`extract`, `audio`, `manifest`, `stats`), and every step skips
   work that is already done, so an interrupted run can simply be restarted.

What it produces, next to the download:

```
MELD.Raw/              extracted label CSVs and video clips
audio16k/<split>/      dia<D>_utt<U>.wav, 16 kHz mono
manifests/<split>.csv  one row per utterance: speaker, text, emotion, sentiment,
                       previous 3 turns, WAV path, duration
```

Data statistics land in `reports/phase1_data.json`.

### CREMA-D (extra voice data and the tone test)

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (ODbL 1.0 / DbCL 1.0) has 91 actors
reading 12 emotionally flat sentences in 6 emotions, so its labels are in the voice only. Only the
WAV folder is needed (606 MB of the 7.5 GB repo). It is kept outside the repo too; set
`paths.crema_root` in `configs/default.yaml`:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git G:/datasets/CREMA-D/repo
git -C G:/datasets/CREMA-D/repo lfs pull -I "AudioWAV/*"
python scripts/prepare_crema.py                  # clips listeners agree on; actor + sentence split
python scripts/extract_wavlm.py --dataset crema  # ~1 min
```

## Training the emotion models

The live system uses three seeds of the fusion model, trained on MELD plus the CREMA-D clips
above, with the words hidden from the fused head on 25% of examples. Times are on the GTX 1660
SUPER:

```powershell
python scripts/extract_wavlm.py                                                      # ~11 min, once
python scripts/train_emotion.py --mode fusion --context 3 --crema --text-dropout 0.25 --name fusion_ctx3_crema_td25   # ~20 min
python scripts/train_emotion.py --mode fusion --context 3 --crema --text-dropout 0.25 --name fusion_ctx3_crema_td25_seed43 --seed 43
python scripts/train_emotion.py --mode fusion --context 3 --crema --text-dropout 0.25 --name fusion_ctx3_crema_td25_seed44 --seed 44
```

The comparison runs in the design doc use the same script: `--mode audio`, `--mode text`
(`--context 0` or `3`), `--text-source whisper` or `mixed` (these need the Whisper transcripts
below), and the Phase 4 live recipe without `--crema` and `--text-dropout`. Each run writes
`reports/phase2_<name>.json`; weights go to `checkpoints/<name>/`, which git ignores.

To score models on what the robot would actually read, Whisper's transcripts instead of MELD's
typed ones:

```powershell
python scripts/transcribe_meld.py dev test        # ~30 min; add train for the Whisper-text runs
python scripts/evaluate.py fusion_ctx3_crema_td25 fusion_ctx3_crema_td25_seed43 fusion_ctx3_crema_td25_seed44 --ensemble fusion_crema_x3
```

The tone test: does the final tag follow the voice when the words are flat? It uses CREMA-D clips
from actors and sentences never seen in training:

```powershell
python scripts/evaluate_tone.py fusion_ctx3_crema_td25 fusion_ctx3_crema_td25_seed43 fusion_ctx3_crema_td25_seed44 --ensemble fusion_crema_x3
```

## Run the demo

Needs the three trained fusion models: download them (Quick start) or train them (above).

```powershell
python scripts/serve.py      # then open http://127.0.0.1:8000
```

Click **Start microphone** and talk, or replay MELD lines (e.g. test dialogue 240, lines 8–10).
Each turn gets a voice-only emotion tag in well under 0.1 s after you stop talking, then the final
tag from words + voice + context, then Pip's reply, streamed word by word.

To follow one input through every stage in the terminal, with timings:

```powershell
python scripts/trace.py --clip test/dia240_utt10 --context 2
```

## Evidence (Phase 4)

```powershell
python scripts/benchmark_latency.py --turns 150   # latency per stage, GPU memory, parameters
python scripts/reply_check.py --per-class 5       # replies with vs without the emotion notes
```

`reports/phase4_reply_sheet.md` is a blind A/B sheet: rate it without looking at the key to judge
whether the emotion notes make Pip's replies fit better.

## Layout

```
configs/default.yaml        model names, paths, real-time targets
docs/DESIGN.md              decisions, assumptions, evidence
reports/                    measured results (JSON) and the reply rating sheet
scripts/smoke_test.py       Phase 0: do all models fit and run on the GPU together?
scripts/download_weights.py  the trained emotion models from the GitHub release, checksum-verified
scripts/prepare_meld.py     Phase 1: extract MELD, convert audio, build the tables
scripts/prepare_crema.py    Phase 5: CREMA-D clips for voice training and the tone test
scripts/extract_wavlm.py    Phase 2: precompute frozen WavLM features (MELD or CREMA-D)
scripts/train_emotion.py    Phase 2: train and score text / audio / fusion models
scripts/transcribe_meld.py  Whisper transcripts with the live settings, plus ASR time and errors
scripts/evaluate.py         re-score models on Whisper or MELD text, single or as an ensemble
scripts/evaluate_tone.py    Phase 5: does the tag follow the voice when the words are flat?
scripts/trace.py            Phase 3: one input through every stage, with timings
scripts/serve.py            Phase 3: the local web demo (web/index.html)
scripts/benchmark_latency.py  Phase 4: latency and resources on real clips
scripts/reply_check.py      Phase 4: do the emotion notes change the replies?
src/meldrt/                 shared code: data, emotion model, and the live pipeline
                            (turns, emotion, cues, conversation, reply, pipeline, server)
tests/assets/               a tiny synthetic speech clip used by the tests
```
