# Design notes

A living record of decisions, assumptions and evidence, updated each phase. The brief asks for
decisions, trade-offs, limitations, hardware and resource use, external components, and what was
left out. Those sections fill in as the work lands.

## 1. Framing

A character robot hears one person speaking. On every turn it must produce:

1. **Structured state:** a JSON object with one of MELD's 7 emotions (anger, disgust, fear, joy,
   neutral, sadness, surprise), a confidence, and the cues behind it.
2. **A short reply** (at most 30 words), streamed as it is generated, that responds to both what
   was said and how it was said.

**Primary track: Text + Audio.** Vision is the optional extension.

- In MELD most of the emotion signal is in the words: text-only systems reach about 0.63–0.67
  weighted F1. Audio adds tone, which is the brief's own example of words meaning something else
  (a sarcastic "oh great").
- MELD video is a poor fit for a vision-first model. Faces are often several per frame, small,
  off-angle, and there are no labels for who is speaking.
- A live webcam showing one frontal face is a much easier setting, so vision joins later as a
  third signal in the live demo.

*Note on the brief:* the track line reads "Text and Audio and Vision" as one bold phrase. We read
it as two tracks, *Text + Audio* or *Vision*, with the optional extension combining them.

## 2. What "real-time" means here

**Assumed interaction:** one person talking face to face with the robot, turn by turn, in English,
in a reasonably quiet room. The robot replies after the person finishes and is not interrupted.

| Output (measured from end-of-turn detection) | Target, p95 |
|---|---|
| Emotion tag / state JSON | ≤ 500 ms |
| First word of the reply | ≤ 1.5 s |
| Full reply (≤ 30 words) | ≤ 3.0 s |

- **End of turn** means 400 ms of silence according to the voice detector (tunable). From the
  person's side, the wait is this silence plus the budget above. A shorter threshold feels snappier
  but cuts people off when they pause mid-sentence.
- **Streaming input:** audio is processed in 32 ms chunks as it arrives, never as a finished file.
- **Keeps up:** each turn is processed faster than it was spoken (real-time factor below 1), so no
  backlog builds over a conversation.
- **Two speeds on purpose:** the tag arrives first, so the robot can react with its face or lights
  while the words are still being generated.
- **Stretch goal:** a provisional emotion estimate about every second while the person is still
  talking.

Why these numbers: in conversation people usually leave around 200 ms between turns (Stivers et
al., 2009, *PNAS*), and gaps much past a second start to feel awkward. Speech-to-text plus an LLM
on a 6 GB GPU can't hit 200 ms, so the design is fast where it's cheap (the tag) and streamed where
it isn't (the reply).

**Intentionally out of scope:** word-by-word live captions, barge-in (the person interrupting the
robot), several people talking at once, languages other than English.

## 3. Pipeline

```
  Live mic (later: webcam)      OR      a MELD clip replayed at real speed
                  └──────────────┬──────────────┘
                                 ▼
 [1] Audio stream (16 kHz, 32 ms chunks)
 [2] Voice activity detector (Silero VAD): start/end of speech → one utterance
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
 [3] Speech-to-text (Whisper)        [4] Audio encoder (WavLM)
     "what was said"                     + cues: loudness, pitch, speaking rate
                 ▼                           "how it was said"
 [5] Text encoder (RoBERTa, fine-tuned,       │
     + previous 2–3 turns as context)         │
                 └───────────────┬────────────┘
                                 ▼
 [6] Fusion head → probabilities for the 7 emotions + confidence
                                 ▼
 [7] Conversation state (history, mood trend, timings) ──► OUTPUT 1: JSON state
                                 ▼
 [8] Small local LLM (transcript + emotion + cues + history)
                                 ▼
                       OUTPUT 2: short reply, streamed
```

## 4. Parameter budget (cap: 6B total learned parameters)

Counted with `parameters()` on the real models, with tied weights counted once. Quantizing a model
does not change its count. Measured values come from `scripts/smoke_test.py` (section 6).

| Component | Model | Measured |
|---|---|---|
| Voice detector | Silero VAD | 0.5M |
| Speech-to-text | Whisper small (faster-whisper, int8 weights) | 241.7M |
| Text encoder | RoBERTa-base | 124.6M |
| Audio encoder | WavLM-base-plus | 94.4M |
| Fusion head | small MLP | < 1M (Phase 2) |
| Reply LLM | Qwen2.5-1.5B-Instruct, 4-bit GGUF via llama.cpp | 1.54B |
| **Total** | | **2.00B of 6B** |

Headroom: a 3B reply model would bring the total to about 3.5B, if reply quality needs it.

## 5. Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6 GB (Turing, compute 7.5: no tensor cores, no bf16) |
| CPU | AMD Ryzen 3 3100, 4 cores / 8 threads |
| RAM | 16 GB |
| OS | Windows 10 Home |
| Python | 3.10.11 |

## 6. Phase 0 smoke test

`scripts/smoke_test.py` loads every model at once, as the live app will, and runs a 5.4 s
synthetic speech clip (`tests/assets/sample_en.wav`) through each. Warm timings, median of
repeated runs. Full numbers: `reports/phase0_smoke.json`.

**First run (2026-10-16) found three problems:**

| Problem | Evidence | Cause | Fix | After |
|---|---|---|---|---|
| Speech-to-text too slow | 1,284 ms per 5.4 s clip | fp16 is slow without tensor cores | int8 weights + no timestamps | 485 ms, same transcript |
| WavLM output was all NaN | 100% NaN at fp16 | fp16 overflow on the GTX 16-series | run WavLM in fp32 | clean, and faster (174 → 44 ms) |
| Reply LLM too slow | 10–13 tok/s, even with the GPU to itself | per-token Python overhead in transformers on this CPU/GPU | llama.cpp, 4-bit GGUF | ~54–92 tok/s, first token 100 ms |

Speech-to-text options compared (same clip, all transcribed it perfectly):

| Whisper | Weights | ms |
|---|---|---|
| small | float16 | 1,027 |
| small | float16, no timestamps | 980 |
| **small** | **int8_float16, no timestamps** | **454** |
| base | float16, no timestamps | 293 |
| distil-small.en | float16, no timestamps | 805 |

**Final Phase 0 result:** all five models fit together in **3.6 of 6.1 GB** of GPU memory.
1.2 GB of that was already used by Windows and open apps before loading anything. The
parameter total is **2.00B**.

**Latency outlook vs. the section 2 budget:**
- The emotion tag currently costs about ASR 485 + RoBERTa 25 + WavLM 44 ms ≈ 0.55 s for a 5.4 s
  clip, slightly over the 500 ms target. MELD utterances are shorter (about 3 s on average), and
  WavLM can run in parallel with ASR, so real MELD clips may land under it; Phase 4 measures this.
- Fallbacks if it doesn't: Whisper base (293 ms), or transcribing while the person is still
  talking so only the last second is left at end of turn.
- The reply is well inside budget: first token at 100 ms, a 13-token reply in 0.24 s.

**Reply quality is not solved yet.** The sample reply ("You're not angry, you're just frustrated
with the timing.") argues with the person's feelings. Prompt design is Phase 3 work, and the
reply check in Phase 4 will measure it.

## 7. Phase 1: data preparation

`scripts/prepare_meld.py` (extract → audio → manifest → stats), run 2026-10-16. Converting all
clips took about 10 minutes with 8 parallel ffmpeg processes and produced 1.4 GB of WAVs. Full
numbers: `reports/phase1_data.json`.

| | train | dev | test |
|---|---|---|---|
| Utterances | 9,989 | 1,109 | 2,610 |
| Dialogues | 1,038 | 114 | 280 |
| With usable audio | 9,988 | 1,108 | 2,610 |
| Median / p95 clip length | 2.5 / 7.9 s | 2.4 / 7.6 s | 2.6 / 7.8 s |
| Utterances of 3 words or fewer | 28.0% | 28.3% | 27.4% |

Emotion balance (% of utterances):

| | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---|---|---|---|---|---|---|
| train | 11.1 | 2.7 | 2.7 | 17.4 | 47.2 | 6.8 | 12.1 |
| dev | 13.8 | 2.0 | 3.6 | 14.7 | 42.4 | 10.0 | 13.5 |
| test | 13.2 | 2.6 | 1.9 | 15.4 | 48.1 | 8.0 | 10.8 |

**Floor to beat:** always predicting "neutral" scores weighted F1 **0.313** and macro F1 **0.093**
on test.

**Data problems found, and how they are handled:**
- **Two utterances have no usable audio:** `train/dia125_utt3` (corrupt video) and
  `dev/dia110_utt7` (video missing from the archive). They stay in the tables with
  `audio_ok = false`. Audio models skip them; text models can still use them.
- **Garbled punctuation:** the transcripts store Windows-1252 punctuation as raw control
  characters (`company\x92s`), about 3,900 times in train alone. They are mapped to plain ASCII,
  matching Whisper's output style. The original is kept in `text_raw`.
- **Some clips don't match their line because MELD's subtitle timestamps are wrong.** For
  example, `test/dia38_utt4` runs 305 s for a 7-word line. 3 clips are longer than Whisper's 30 s
  window, and 44 clips (42 of them in test) differ from their subtitle times by more than 2 s.
  The official splits are kept unchanged so scores stay comparable with published work; Phase 2
  caps how much audio each model sees.
- **141 extra video clips have no label row** (4 in dev, 137 in test). They are ignored.

**What this means for the plan:**
- Clips are short (median 2.5 s), so the latency outlook in section 6, measured on a 5.4 s clip,
  should improve on real data.
- 28% of utterances are 3 words or fewer, where the words alone say little. Audio should help
  most on that slice, so Phase 2 reports it separately.
- Fear and disgust have under 300 training examples each, so class weighting matters and macro
  F1 will stay low.

## 8. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-16 | Text + Audio primary track, vision as the extension | Section 1 |
| 2026-10-16 | Real-time = turn-based, with the budgets in section 2 | Conversational turn-taking norms |
| 2026-10-16 | MELD lives outside the repo (`G:/datasets/MELD`) and is never committed | 10.9 GB, and the clips are copyrighted TV footage |
| 2026-10-16 | ffmpeg comes from the `imageio-ffmpeg` pip package | No system install; same setup on any machine |
| 2026-10-16 | ~~Reply LLM through Hugging Face transformers~~ | Replaced the same day: 10–13 tok/s (section 6) |
| 2026-10-16 | Reply LLM through llama.cpp (`llama-cpp-python` 0.3.4, CUDA 12.4 wheel), Qwen2.5-1.5B Q4_K_M | ~7x faster, about 2 GB less VRAM, prebuilt for Windows |
| 2026-10-16 | Whisper small with int8 weights | 2.3x faster than fp16 on this card with the same transcript; base is the fallback |
| 2026-10-16 | WavLM in fp32 | fp16 gives NaN on the GTX 16-series, and fp32 is faster here anyway |
| 2026-10-16 | Keep MELD's official splits unchanged, misaligned clips included | Scores stay comparable with published results |
| 2026-10-16 | Normalise transcript punctuation to ASCII (`text`), keep the original (`text_raw`) | Training text should look like Whisper's live output |
| 2026-10-16 | Context = the previous 3 utterances, speaker and words only, no earlier labels | Earlier emotions aren't known during a live conversation |
| 2026-10-16 | Tar extraction in Python with a member filter (plain files only, inside the target folder) | Archives are untrusted input, and Python 3.10.11 lacks `tarfile`'s built-in filter |

## 9. External components

Licences checked 2026-10-16 from each model's Hugging Face card or source repository.
"Changed" and "checked" fill in as each component is trained or evaluated.

| Component | Source | Licence | Changed by us | How we checked it |
|---|---|---|---|---|
| MELD dataset | [declare-lab/MELD](https://github.com/declare-lab/MELD) | GPL-3.0 (repo); clips are *Friends* footage | nothing (kept outside the repo) | archive size matches server; `gzip -t` |
| Silero VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | no | smoke test |
| Whisper small | [openai/whisper-small](https://huggingface.co/openai/whisper-small) via [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | Apache-2.0 / MIT | no | smoke test |
| RoBERTa-base | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | MIT | fine-tuned (Phase 2) | smoke test |
| WavLM-base-plus | [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) | CC BY-SA 3.0 (per its card) | frozen; small head trained on top (Phase 2) | smoke test |
| Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | Apache-2.0 | prompt only | smoke test |
| llama.cpp runtime | [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 0.3.4 | MIT | no | smoke test |

## 10. Limitations and what was left out

*Filled in at the end.*
