# MELD Realtime Emotion

A small real-time prototype for an emotion-aware character robot. It listens to a person, works
out how they feel from **what they say and how they say it**, and returns:

1. a structured state (JSON) with one of MELD's 7 emotions, and
2. a short reply that is streamed as it is generated.

Built on the [MELD](https://affective-meld.github.io/) dataset. Everything runs locally on one
consumer GPU, within a 6B total-parameter budget.

> **Status:** Phase 2 done. Emotion models trained on MELD: text + audio fusion reaches test
> weighted F1 0.631 on MELD's typed transcripts (text alone 0.638, audio alone 0.477,
> always-"neutral" 0.313). On live-style Whisper transcripts it's 0.536 against 0.530 for text
> alone. Everything runs on a 6 GB GTX 1660 SUPER within 2.00B parameters. See [docs/DESIGN.md](docs/DESIGN.md) for the
> decisions, the definition of "real-time", and the measurements.

## Training the emotion models

After the data steps below:

```powershell
python scripts/extract_wavlm.py                                         # ~11 min, once
python scripts/train_emotion.py --mode audio  --name audio_only         # ~1 min
python scripts/train_emotion.py --mode text   --context 0 --name text_ctx0
python scripts/train_emotion.py --mode text   --context 3 --name text_ctx3
python scripts/train_emotion.py --mode fusion --context 3 --name fusion_ctx3
```

Each run writes `reports/phase2_<name>.json`. Weights go to `checkpoints/<name>/`, which git
ignores. Times on the GTX 1660 SUPER: 11–23 min per text or fusion run.

To score the trained models on what the robot would actually read, Whisper's transcripts
instead of MELD's typed ones:

```powershell
python scripts/transcribe_meld.py dev test                   # ~30 min; also measures ASR time
python scripts/evaluate.py text_ctx0 text_ctx3 fusion_ctx3   # writes reports/eval_asr_<run>.json
```

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

## Layout

```
configs/default.yaml   model names, paths, real-time targets
docs/DESIGN.md         decisions, assumptions, evidence
reports/               measured results (JSON)
scripts/smoke_test.py  Phase 0: do all models fit and run on the GPU together?
scripts/prepare_meld.py   Phase 1: extract MELD, convert audio, build the tables
scripts/extract_wavlm.py  Phase 2: precompute frozen WavLM features
scripts/train_emotion.py  Phase 2: train and score text / audio / fusion models
scripts/transcribe_meld.py  Whisper transcripts with the live settings, plus ASR time and errors
scripts/evaluate.py       re-score trained models on Whisper transcripts
src/meldrt/            shared code: data loading and input formatting, the emotion model
tests/assets/          a tiny synthetic speech clip used by the tests
```
