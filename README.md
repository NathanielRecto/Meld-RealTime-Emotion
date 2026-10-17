# MELD Realtime Emotion

A small real-time prototype for an emotion-aware character robot. It listens to a person, works
out how they feel from **what they say and how they say it**, and returns:

1. a structured state (JSON) with one of MELD's 7 emotions, and
2. a short reply that is streamed as it is generated.

Built on the [MELD](https://affective-meld.github.io/) dataset. Everything runs locally on one
consumer GPU, within a 6B total-parameter budget.

> **Status:** Phase 0 done. All models fit and run together on a 6 GB GTX 1660 SUPER (2.00B
> parameters, 3.6 GB of GPU memory). See [docs/DESIGN.md](docs/DESIGN.md) for the decisions, the
> definition of "real-time", and the measurements.

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

Download: <https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz>
(a download-and-prepare script comes in Phase 1).

## Layout

```
configs/default.yaml   model names, paths, real-time targets
docs/DESIGN.md         decisions, assumptions, evidence
reports/               measured results (JSON)
scripts/smoke_test.py  Phase 0: do all models fit and run on the GPU together?
tests/assets/          a tiny synthetic speech clip used by the tests
```
