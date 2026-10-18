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
| Voice tag: state JSON from the voice alone (stage 1) | ≤ 500 ms |
| Final tag: state JSON from words + voice + context (stage 2) | ≤ 1.0 s |
| First word of the reply | ≤ 1.5 s |
| Full reply (≤ 30 words) | ≤ 3.0 s |

*Revised 2026-10-17.* The first version had a single 500 ms target for the emotion tag. Measuring
Whisper on real clips (section 9: p95 694 ms for speech-to-text alone) showed that a tag which
needs the words can't meet it on this GPU. So the tag now comes in two stages: a voice-only tag
inside the original 500 ms, then the full tag once the words are in (section 10).

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
| Text encoder | RoBERTa-base, fine-tuned, without the unused pooler | 124.1M |
| Audio encoder | WavLM-base-plus (frozen) | 94.4M |
| Emotion heads | fusion + per-modality heads + WavLM layer weighting (Phase 2) | 0.5M |
| *Phase 4 ensemble* | *the live system runs 3 seeds of RoBERTa + heads (section 11)* | *373.8M in total* |
| Reply LLM | Qwen2.5-1.5B-Instruct, 4-bit GGUF via llama.cpp | 1.54B |
| **Total** | | **2.00B of 6B** |

Headroom: a 3B reply model would bring the total to about 3.5B, if reply quality needs it.
With the Phase 4 ensemble the measured total is **2.25B** (`reports/phase4_latency.json`).

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

## 8. Phase 2: emotion models

**One model, three answers.** `src/meldrt/model.py` has a text branch (RoBERTa-base, fine-tuned)
and an audio branch (frozen WavLM-base-plus features plus a small trained head), joined by a
fusion head. In fusion mode each branch also keeps its own classifier head, trained alongside the
fused one with an auxiliary loss (weight 0.5). So one forward pass returns "words say X, voice
says Y, combined says Z", which the state JSON needs. The auxiliary losses also stop the fused
head from ignoring the weaker audio branch.

How the inputs are built:
- **Text:** the utterance first, then the previous 3 turns as a second segment, e.g.
  `B: So let's talk a little bit about your duties.` + `A: … B: You must've had your hands full.
  A: That I did.` Speakers are renamed A, B, C… within each window, so the model learns turn-taking
  instead of memorising *Friends* characters. Live, the person and the robot are just two letters.
  Median length is 49 tokens and the max is 153, so nothing hits the 192-token cap.
- **Audio:** `scripts/extract_wavlm.py` runs frozen WavLM once per clip (10.7 min for all 13.7k
  clips) and stores each of its 13 layers averaged over time. The audio branch learns a softmax
  weighting over those layers, as in the SUPERB benchmark. Audio is capped at the first 15 s,
  which leaves 99.7% of clips untouched and stops the misaligned multi-minute clips from
  dominating.

Training (`scripts/train_emotion.py`):
- **Loss:** cross-entropy with class weights proportional to (1 / frequency)^0.5. That's a
  middle ground: full inverse frequency would weight fear about 17x neutral and hurt weighted F1.
- **Modality dropout:** in fusion mode, a training example's audio is hidden 15% of the time, so
  the fused head still works when audio is missing. Two clips have no audio at all.
- **Optimiser:** AdamW. RoBERTa at 1.5e-5 and the heads at 5e-4, with linear warmup (10%) then
  decay, 5 epochs at batch 16. Audio-only uses 40 epochs at batch 64, because it trains just a
  small head.
- **Selection:** the epoch with the best dev weighted F1 is kept, and test is scored exactly once
  with that checkpoint. Nothing is tuned on test.
- **Precision:** fp32. fp16 mixed precision was **4x slower** per step on this card (1.75 s vs
  0.42 s over 100 steps), for the same no-tensor-core reason as WavLM in section 6.

Runs, each one compared on the same test set:

| Run | What it tests |
|---|---|
| `audio_only` | how much emotion the voice carries on its own |
| `text_ctx0` | words of the utterance alone |
| `text_ctx3` | what conversation history adds |
| `fusion_ctx3` | what tone adds on top of words + history (the model the robot uses) |

**Results** (2026-10-17, one seed each; full numbers in `reports/phase2_<run>.json`):

| Run | Dev W-F1 | Test W-F1 | Test macro F1 | ≤ 3 words | 4+ words | Train time |
|---|---|---|---|---|---|---|
| Always "neutral" | – | 0.313 | 0.093 | – | – | – |
| `audio_only` | 0.473 | 0.477 | 0.312 | 0.483 | 0.473 | 0.8 min |
| `text_ctx0` | 0.597 | **0.638** | 0.477 | 0.724 | 0.606 | 10.9 min |
| `text_ctx3` | 0.616 | 0.635 | **0.486** | 0.705 | 0.608 | 23.2 min |
| `fusion_ctx3` | **0.623** | 0.631 | 0.479 | 0.704 | 0.604 | 19.0 min |

Per-class test F1:

| Run | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---|---|---|---|---|---|---|
| `audio_only` | 0.380 | 0.126 | 0.094 | 0.339 | 0.648 | 0.237 | 0.361 |
| `text_ctx0` | 0.510 | 0.272 | 0.243 | 0.606 | 0.782 | 0.365 | 0.559 |
| `text_ctx3` | 0.498 | 0.327 | 0.254 | 0.607 | 0.772 | 0.405 | 0.540 |
| `fusion_ctx3` | 0.481 | 0.262 | 0.277 | 0.605 | 0.770 | 0.407 | 0.550 |

What the numbers say:
1. **Text carries most of the signal.** About 0.64 weighted F1 is in line with published
   RoBERTa-base results on MELD (about 0.62–0.65); the best published systems reach about 0.65–0.70.
2. **Audio alone is far above the floor** (0.48 vs 0.31) **and is right in different places.** On
   10.3% of test lines the audio model is right where the text model (`text_ctx0`) is wrong; on
   26.2% both are wrong.
3. **On MELD's own transcripts, neither context nor audio gave a clear test gain.** The three
   text-based runs sit within 0.7 points of each other, which is inside run-to-run noise for a
   single seed. Fusion scored best on dev (+0.7 over `text_ctx3`) and is best on fear, the rarest
   class, but it doesn't beat text-only on test overall. Inside the fusion model, the text head
   alone scores 0.624 and the audio head alone 0.464.
4. **Likely reason: MELD's transcripts already carry tone.** The transcriber heard the clip and
   wrote "What?!" or "Hey!". The 3-word slice, expected to be where audio helps, is where text
   does *best* (0.72). It's mostly greetings and reactions ("hey!", "hi!", "what?"), with the
   same class mix as the full test set. Whisper punctuates from tone less reliably, so audio's
   real value should show on **Whisper transcripts**. That check moves up to run next.
5. **WavLM layers:** the learned weights lean slightly towards layers 8–10. That matches the
   SUPERB finding that emotion information peaks mid-to-upper network, but the weights are
   nearly flat (0.067–0.095).

**Model for the robot: `fusion_ctx3`.** It is selected by the dev rule (best dev weighted F1),
ties text-only on test, and is the only run that gives the per-modality views the state JSON
needs. Size: 124.6M parameters, i.e. RoBERTa-base without its unused pooler (124.1M) plus 0.5M
of heads, so the section 4 total stays at 2.00B.

**Not done (yet):** several seeds per run to measure the noise band, and a probability-level
late-fusion baseline.

## 9. Whisper transcripts vs MELD's typed text

Live, the robot only reads what speech-to-text writes, so the Phase 2 models were re-scored on
Whisper transcripts (2026-10-17). `scripts/transcribe_meld.py` ran the live ASR settings (Whisper
small, int8, greedy decoding, audio capped at 15 s) over dev and test. `scripts/evaluate.py` then
swapped Whisper's text in for each utterance *and* its context turns and scored the unchanged
checkpoints. Full numbers: `reports/asr_{dev,test}.json` and `reports/eval_asr_<run>.json`.

**What Whisper writes:**

| | dev | test |
|---|---|---|
| Word error rate vs MELD's text | 27.5% | 30.7% |
| Lines transcribed exactly (ignoring punctuation and case) | 30.0% | 30.2% |
| Lines with "!": MELD / Whisper | 31.5% / 13.3% | 30.7% / 10.7% |
| Lines with "?": MELD / Whisper | 26.4% / 26.2% | 26.3% / 26.5% |
| ASR time per clip, p50 / p95 | 451 / 680 ms | 447 / 694 ms |

- **"!" nearly vanishes but "?" doesn't.** A question shows in its word order. An exclamation is
  heard, so MELD's typists were writing tone into the text.
- **The error rate is high because *Friends* audio is hard:** laugh tracks, overlapping speech,
  and clips that catch the end of the previous line.
- **ASR time hardly depends on clip length.** Whisper always encodes a 30 s window, so a 2.5 s
  clip costs about the same as the 5.4 s smoke-test clip.

**Emotion scores on Whisper text (test weighted F1):**

| Model | MELD text | Whisper text | Change |
|---|---|---|---|
| `text_ctx0` | 0.638 | 0.530 | −0.108 |
| `text_ctx3` | 0.635 | 0.529 | −0.106 |
| `fusion_ctx3` | 0.631 | **0.536** | **−0.095** |
| `fusion_ctx3`, audio head alone | 0.464 | 0.464 | 0 (doesn't read text) |

Per-class change for `text_ctx0` → its Whisper score: anger −0.16, disgust −0.04, fear −0.11,
**joy −0.21**, neutral −0.09, sadness 0.00, surprise −0.09. Fusion on Whisper text beats
text-only on joy (0.428 vs 0.395) and fear (0.177 vs 0.130).

What this means:
1. **Real speech-to-text costs about 11 points.** About 0.53 is the honest estimate for the
   live system; the 0.64 from section 8 was helped by human punctuation.
2. **The excited emotions lose most.** Joy and anger, the ones typists mark with "!", drop the
   most, and sadness doesn't move. The 3-word lines lose their advantage too (0.72 → 0.52),
   so that advantage was mostly punctuation.
3. **Fusion degrades least, so it's the best of the three on realistic input, but only just.**
   It leads by 0.6 points on test, while `text_ctx3` is ahead on dev (0.529 vs 0.519). With one
   seed that's a tie, not a win. Audio's help shows where expected (joy, fear), just not yet
   overall.
4. **The models learned on "!"-rich text and are tested on plain text.** That train/test
   mismatch is the likely limiter. The fix is to train on Whisper transcripts of the training set
   too, so the text branch can't lean on punctuation and the fusion head learns when to trust
   the voice. Cost: about 80 min to transcribe train and about 45 min to retrain two models.
5. **Latency: speech-to-text alone breaks the tag budget.** Its p95 of 694 ms exceeds the 500 ms
   emotion-tag target (section 2) before RoBERTa or WavLM even run. Phase 3 has to fix this,
   using some combination of Whisper base (293 ms in section 6), transcribing while the person
   is still talking, or a speech-to-text model whose cost scales with clip length.

**Retraining on Whisper text** (2026-10-17). To test point 4, the training set was transcribed
too (`scripts/transcribe_meld.py train`: 9,988 clips, word error rate 31.7%, "!" on 10.1% of
lines vs MELD's 30.3%). The text and fusion models were then retrained with
`--text-source whisper`, so they were trained, picked on dev and scored on Whisper text throughout.

| Model, scored on Whisper text | Trained on | Dev W-F1 | Test W-F1 | Test macro F1 |
|---|---|---|---|---|
| `text_ctx3` | MELD text | 0.528 | 0.529 | 0.373 |
| `text_ctx3_whisper` | Whisper text | 0.516 | 0.523 | 0.357 |
| `fusion_ctx3` | MELD text | 0.519 | **0.536** | **0.389** |
| `fusion_ctx3_whisper` | Whisper text | 0.525 | 0.513 | 0.369 |

**It didn't help.** Matching the training text to the live text made the text model slightly
worse on both dev and test. The fusion model gained 0.6 points on dev but lost 2.3 on test, and
both lost macro F1. So point 4's guess was wrong, or at least this isn't the way to fix it. A
likely reason: the labels were made from the clean lines and the video, so pairing them with
transcripts that get about a third of the words wrong mostly adds noise. The model trained on clean
text learned what the words mean emotionally and still uses that whenever Whisper gets them right.
The retrained fusion model did read anger better (0.43 vs 0.35) but lost on neutral and fear.

**Decision: the live system keeps `fusion_ctx3`.** The only dev gain (+0.6 for fusion) is inside
single-seed noise, so there's no evidence for switching, and test points the same way.
`fusion_ctx3` stays the best of the four on realistic input, though by a margin that is itself
within noise.

## 10. Phase 3: the real-time pipeline

**How one turn flows** (`src/meldrt/pipeline.py`):

1. **Audio in.** Either the microphone (`sources.microphone`: 16 kHz, or 48 kHz downsampled) or a
   MELD dialogue replayed at real speed with 1 s gaps (`sources.replay`). Both deliver 32 ms chunks.
2. **Turn detection** (`turns.TurnDetector`). Silero VAD runs on every chunk. A turn ends after
   400 ms of silence, keeps 100 ms of padding on each side, and is cut at 15 s. Listening never
   pauses: new turns queue up while an earlier one is still being processed.
3. **Stage 1, voice tag.** WavLM features (the same function as training, `features.py`) go
   through the fusion model's audio head. A `state` event with `stage: "voice"` goes out.
4. **Speech-to-text:** Whisper small, int8.
5. **Stage 2, final tag.** RoBERTa reads the transcript plus the previous 3 turns (both sides of
   the conversation), together with the same WavLM features. One pass gives fused, words-only and
   voice-only probabilities, and a `state` event with `stage: "final"` goes out. This is output 1.
6. **Reply (output 2).** The state is turned into plain-language notes, and Qwen2.5-1.5B
   (llama.cpp) streams a reply of at most 30 words, token by token.

**Output 1, trimmed from `reports/trace_test_dia240_utt10.json`:**

```json
{"stage": "final", "turn": 4, "transcript": "Is this a real person?",
 "emotion": {"label": "surprise", "confidence": 0.885, "probs": {"surprise": 0.885, "neutral": 0.061, "...": "..."}},
 "sentiment": "neutral",
 "by_modality": {"words": {"label": "surprise", "confidence": 0.903},
                 "voice": {"label": "anger", "confidence": 0.4}},
 "modalities_agree": false,
 "cues": {"loudness_db": -29.0, "pitch_hz": 275.9, "words_per_second": 2.78,
          "vs_earlier": {"pitch": "higher", "expressiveness": "flatter", "pace": "slower"}},
 "mood_trend": "steady",
 "latency_ms": {"voice_tag": 45.0, "asr_done": 736.0, "final_tag": 787.3}}
```

**Design choices:**
- **Two-stage tag** (section 2). The voice tag lets the robot react within about 50 ms; the final
  tag follows once the words are known.
- **The reply is grounded through plain-language notes, not numbers** ("They clearly sound
  surprised. Compared with earlier, their voice is higher, slower."). A words/voice mismatch is
  only mentioned when the fused model **sided with the voice**, which is the sarcasm case. In the
  first trace a weak voice-only guess (anger 0.40) made the reply hedge ("I'm not sure if you're
  joking"), even though the fused model was confident in surprise. The rule changed after that.
- **Voice cues are compared with the same person's earlier turns,** not with fixed thresholds, so
  they work for any voice without calibration.
- **The emotion model's context includes the robot's own replies,** so it sees a two-party
  conversation, like the multi-party MELD dialogues it learned from.
- **One lock around the models:** there is one GPU, and llama.cpp isn't thread-safe.

**Demo tools:**
- `python scripts/trace.py --clip test/dia240_utt10 --context 2` follows one input through every
  stage with timings: the brief's "one input to both outputs" demonstration. It saves
  `reports/trace_<clip>.json`.
- `python scripts/serve.py`, then open http://127.0.0.1:8000. The page takes a microphone or a
  MELD replay and shows a live card per turn: provisional voice tag, final tag with probability
  bars, words vs voice, cues, mood, the streamed reply, latency, and the raw state JSON.

**First measurements** (the trace above, while a background Whisper job shared the GPU, so these
are pessimistic): voice tag 40–63 ms, final tag 0.79–1.38 s, first reply word 1.09–1.64 s, reply
done 1.5–2.3 s. Clean numbers on an idle GPU come in Phase 4.

**Known issues so far:**
- The 400 ms silence threshold split "No. No, I'm Joey…" at its first pause. The 0.39 s "No."
  got no transcript and was skipped.
- There's no speaker identification, so a replayed multi-person MELD scene is treated as one
  person talking to the robot.
- The microphone path is implemented but hasn't been tested with live speech yet.

## 11. Phase 4: evidence

### Two last accuracy experiments (2026-10-17)

Section 9 left the live system at about 0.54 weighted F1 on Whisper text, with every comparison
so far inside an unknown noise band. Two cheap experiments, both scored with
`scripts/evaluate.py` on dev and test with Whisper text unless noted:

- **A. Mixed-text training** (`--text-source mixed`). Every training line is used twice, once as
  MELD typed it and once as Whisper heard it. Unlike Whisper-only training (section 9), the model
  keeps the clean text to learn word meanings from.
- **B. Three seeds of the live recipe** (`fusion_ctx3` with seeds 42, 43 and 44). This measures
  run-to-run noise, and averaging the three gives an ensemble.

| Model | Dev W-F1 | Dev macro F1 | Test W-F1 | Test macro F1 | Test accuracy |
|---|---|---|---|---|---|
| `fusion_ctx3` (seed 42, the previous live model) | 0.519 | 0.415 | 0.536 | 0.389 | 54.8% |
| seed 43 / seed 44 | 0.530 / 0.530 | – | 0.529 / 0.528 | 0.373 / 0.374 | 54.2% / 53.5% |
| **3-seed average (`fusion_x3`)** | 0.531 | **0.422** | **0.538** | **0.390** | **55.0%** |
| A. mixed text (`fusion_ctx3_mixed`) | **0.534** | 0.391 | 0.531 | 0.354 | 55.0% |

On MELD's typed text, the same three seeds score 0.631 / 0.637 / 0.642 on test, and the ensemble
**0.648** (macro F1 0.500, accuracy 64.6%), the best result of the project.

What this shows:
1. **Run-to-run noise is now measured.** Single seeds of the same recipe differ by a standard
   deviation of **0.004** in test weighted F1 on Whisper text and **0.006** on MELD text. Earlier
   differences under about 1 point were indeed noise.
2. **The ensemble beats every single seed on both text sources** (+0.7 over the seeds' mean on
   Whisper text, +1.1 on MELD text), and it has the best dev macro F1.
3. **Mixed-text training ties the ensemble on dev weighted F1** (0.534 vs 0.531, inside the noise)
   **but is clearly worse on dev macro F1** (0.391 vs 0.422), so it handles the rare emotions worse.

**Decision: the live system uses the 3-seed ensemble** (`configs/default.yaml`,
`models.emotion.runs`). The decision is based on dev only: weighted F1 is tied, and macro F1 is
clearly better. Each turn runs the three models and averages their probabilities. That costs about
+250M parameters (still far under 6B) and a few tens of milliseconds on the final tag (measured
below).

### Latency and resources on an idle GPU

`scripts/benchmark_latency.py` (2026-10-17) played 158 real test turns from 16 whole dialogues,
turn by turn, through `Pipeline.process_turn` with the 3-seed live model, so the conversation
memory and context grew as they would live. Full numbers: `reports/phase4_latency.json`.

| Stage, timed from the end of the turn | p50 | p95 | Max | Target (p95) |
|---|---|---|---|---|
| Voice tag | 28 ms | 67 ms | 106 ms | ≤ 500 ms ✅ |
| Words (speech-to-text done) | 524 ms | 875 ms | 13.0 s | – |
| Final tag | 563 ms | 935 ms | 13.1 s | ≤ 1,000 ms ✅ |
| First reply word | 814 ms | 1,331 ms | 13.4 s | ≤ 1,500 ms ✅ |
| Full reply | 1,102 ms | 1,618 ms | 13.6 s | ≤ 3,000 ms ✅ |

- **All four targets are met at p95.** The final tag has the least room (935 vs 1,000 ms).
- **The 13 s maximum is a single turn where Whisper looped on a long clip.** The 15 s turn cap
  bounds it, but a shorter time limit on speech-to-text would be safer live.
- **Before any of these timings start, the turn detector waits 400 ms of silence** to decide the
  turn is over. Its own compute is 0.9 ms per 32 ms of audio, under 3% of real time.
- **3 of the 158 turns had no recognisable words** and got no reply.

| Resource | Measured |
|---|---|
| Parameters | **2.25B of 6B:** Silero VAD 0.5M, Whisper small 241.7M, WavLM 94.4M, 3 × (RoBERTa + heads) 373.8M, Qwen2.5-1.5B 1,543.7M |
| GPU memory | 5.4 GB peak of 6.1 GB, including 0.8 GB already used by Windows before loading |
| RAM | 2.7 GB |
| Start-up | 23 s to load and warm up all models |

### Does the emotion state change the replies?

`scripts/reply_check.py` takes 35 test lines (5 per emotion, Whisper transcripts) and answers
each one twice with the same random seed: once with the plain-language emotion notes, and once
with notes that say nothing about feelings. Full replies are in `reports/phase4_replies.json`,
and a blind A/B sheet is in `reports/phase4_reply_sheet.md`.

The first run exposed a prompt problem in *both* conditions, so the prompt was changed and the
check re-run. The first run is kept as `reports/phase4_replies_prompt_v1.json`.

| Replies with notes (35) | Prompt v1 | Prompt v2 |
|---|---|---|
| Talks about the person in the third person ("It's clear they're surprised") | 12 | **1** |
| Reads its instructions back ("I'll respond gently") | 6 | **0** |

Prompt v1 handed the reply model `They said: "…"` plus notes written in the third person, and
the 1.5B model copied that framing. Prompt v2 passes the person's own words as the message and
keeps the notes in the system prompt.

**Do the notes make replies fit better?** I read the 35 prompt-v2 pairs against MELD's label.
This wasn't blind: I could see which reply had the notes. The notes were better in 5, worse in 6,
and made no clear difference in 24.
- **The replies do change** (33 of 35 differ). The notes help when the emotion is clear and
  right: a frantic "It won't stop crying!" gets "I'm here now, sweetie. Let's try to calm it down"
  instead of "I can't tell what you're feeling".
- **A wrong emotion misleads the reply,** e.g. "You look excited!" to an angry line. The emotion
  model is wrong on 20 of these 35 lines; the sample has equal numbers of each emotion, so rare,
  hard emotions are over-represented.
- **With prompt v1 the notes looked clearly helpful** (12 better, 4 worse). That was mostly
  because the no-notes replies suffered from the same prompt flaws.

**Conclusion:** the emotion state does steer Pip's replies, but with a 1.5B reply model and the
current emotion accuracy, a clear benefit isn't shown. Two ways to settle it: the blind sheet,
rated by someone who can't see which reply had the notes (still to do), and a stronger reply
model. Qwen2.5-3B would still fit the budget at about 3.8B total.

## 12. Phase 5: making the voice count

### The problem

In the live microphone test the emotion tag followed the **words, not the tone**: a cheerful voice
saying something harsh was tagged anger. On MELD test (typed text), the Phase 4 3-seed model's
words head and voice head disagree on half of the lines (1,315 of 2,610). On those lines the final
tag sides with the words 92% of the time and with the voice 4%. That is what MELD rewards: there,
the words match the label 52% of the time and the voice 23%, because MELD's labels were given by
people who heard the words. So MELD can't measure this problem, and training on MELD alone can't
fix it.

### What changed

1. **A tone test set.** [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) has 91 actors
   reading 12 emotionally flat sentences ("The surface is slick.") in 6 emotions, so the emotion is
   in the voice only. `scripts/prepare_crema.py` keeps the 3,099 of 7,442 clips where listeners
   rating the voice alone picked the emotion the actor intended. Most of the rest sound neutral to
   people (only 26% of "happy" and 16% of "sad" clips are heard that way), and the robot should
   react to tone a person can hear. Actors are split 54 / 12 / 25 (train / dev / test, balanced by
   sex), and 4 of the 12 sentences are held out of training entirely. The tone dev set (121 clips)
   and test set (226 clips) therefore have **new voices and new sentences**. Without the held-out
   sentences, a model could memorise "these 12 sentences → trust the voice", which wouldn't carry
   over to live speech. `scripts/evaluate_tone.py` scores the final tag by **balanced accuracy**
   (the mean recall over the 6 emotions, so the 42% neutral clips don't dominate), and reports
   recall on the emotional clips and on the neutral ones separately.
2. **CREMA-D in training** (`train_emotion.py --crema`). The 1,260 training clips are added to
   MELD's 9,989 lines. They train the fused head and the voice head; the words-only head skips
   them, because their words carry no emotion. Their audio is never hidden in training, since the
   label would then be unknowable. Model selection and every MELD score stay on MELD's own dev and
   test sets.
3. **Hiding the words in training** (`--text-dropout 0.25`). On a quarter of the examples where
   the audio is kept, the fused head doesn't see the words, so it can't always lean on them. This
   mirrors the existing audio dropout (15%).
4. **The reply rule** (`src/meldrt/reply.py`, `notes()`). Pip's notes now also mention a
   words-vs-voice mismatch when the voice alone is at least 60% sure of a non-calm emotion, even
   if the final tag went with the words. The wording depends on the words. Plain words get "Their
   words are matter-of-fact, but their voice sounds angry." Words with an emotion of their own get
   the sarcasm note, "…so they may not mean the words literally". The old rule used the sarcasm
   note for both, which is wrong for "It's eleven o'clock." said angrily. The blind A/B sheet in
   section 11 was made before this change and is left as it was.

**How the recipe is chosen, fixed before any results.** Three single-seed recipes (seed 42):
CREMA-D + hidden words, hidden words only, and CREMA-D only. The winner has the best tone-dev
balanced accuracy among those whose MELD dev weighted F1 is at least 0.613, i.e. no more than 0.01
below the current recipe's 0.623 at the same seed. If none qualifies, the live model stays as it
is. The winner then gets seeds 43 and 44, and the 3-seed ensemble is scored once on both test sets.

### Choosing the recipe (seed 42, single runs)

| Recipe | MELD dev W-F1 | MELD test W-F1 | Tone dev bal. acc. | Tone test bal. acc. | Emotional clips caught (test) | Neutral kept neutral (test) |
|---|---|---|---|---|---|---|
| Current (`fusion_ctx3`) | 0.623 | 0.631 | 0.131 | 0.127 | 0% | 76% |
| Hidden words only (`fusion_ctx3_td25`) | 0.627 | 0.630 | 0.131 | 0.127 | 0% | 76% |
| CREMA-D only (`fusion_ctx3_crema`) | 0.618 | 0.652 | 0.565 | 0.540 | 52% | 88% |
| **CREMA-D + hidden words** (`fusion_ctx3_crema_td25`) | **0.634** | 0.634 | **0.653** | **0.648** | **64%** | **99%** |

All three pass the MELD guard (dev ≥ 0.613), and **CREMA-D + hidden words** has the best tone-dev
score, so it was chosen. Test scores are shown for every row, but only dev was used to choose.
Chance on the tone set is 0.167 (6 emotions).

What this shows:
1. **The current model never follows the voice when the words are flat.** On the tone test the
   Phase 4 live ensemble's final tag only says *neutral* or *surprise*: 32 of 42 angry-sounding
   clips come out neutral, even though its own voice head hears anger in 17 of them. That is the
   demo complaint, now measured, and it scores *below* chance (0.127).
2. **Hiding the words alone changes nothing on tone** (0.131, identical to the current model).
   MELD's labels still follow the words, so the fused head learns to wait for them.
3. **CREMA-D alone gets most of the way** (0.565), and **both together do best** (0.653). CREMA-D
   teaches what an emotional voice sounds like, and hiding the words makes the fused head rely on
   that signal. Together they also keep neutral clips neutral (99% vs 88%).
4. **MELD doesn't suffer:** dev weighted F1 0.634 vs 0.623, test 0.634 vs 0.631, both within about
   2 seed standard deviations.
5. **Joy is the weak spot** (2 of 17 happy-sounding test clips caught; most read as neutral or
   fear). Only 141 of the CREMA-D training clips are happy ones that listeners also heard as happy.

**On MELD, the fused tag still mostly follows the words.** Where its words head and voice head
disagree on MELD test, the chosen recipe sides with the voice on 10% of those lines (seed 42 of the
current recipe: 5%) and with the words on 85% (91%). On those lines MELD's label matches the words
56% of the time and the voice 20%, so that is still the right call for MELD. The model now follows
the tone when the words say little, and the words when they carry the emotion.

### The new live model (3 seeds, scored once on test)

Seeds 43 and 44 of the chosen recipe were trained, and the three are averaged as before
(`configs/default.yaml`, `models.emotion.runs`). Scored with `scripts/evaluate.py` (MELD and
Whisper text) and `scripts/evaluate_tone.py`:

| | Phase 4 live model (`fusion_x3`) | **Phase 5 live model (`fusion_crema_x3`)** |
|---|---|---|
| MELD typed text: dev W-F1 / **test W-F1** | 0.630 / 0.648 | 0.642 / **0.651** |
| MELD typed text: test macro F1 / accuracy | 0.500 / 64.6% | 0.489 / 65.3% |
| Whisper text: dev W-F1 / **test W-F1** | 0.531 / 0.538 | 0.533 / **0.557** |
| Whisper text: test macro F1 / accuracy | 0.390 / 55.0% | 0.405 / 57.1% |
| Voice tag alone, MELD test W-F1 | 0.473 | 0.465 |
| **Tone test: final tag, balanced accuracy** | **0.127** | **0.663** |
| Tone test: emotional clips caught / neutral kept neutral | 0% / 76% | 64% / 95% |
| Tone test: voice tag alone, balanced accuracy | 0.260 | 0.706 |

Tone test recall by emotion for the new model: anger 88% (37 of 42), disgust 88% (14 of 16),
sadness 69% (9 of 13), fear 52% (22 of 42), **joy 6% (1 of 17)**, neutral 95%. The old model's
recall is 0% for every emotion except neutral.

What this shows, and how far to trust it:
1. **The tone problem is fixed on the tone set** (0.127 → 0.663) **with no loss on MELD**:
   weighted F1 is level on typed text (0.648 → 0.651) and on Whisper's dev (0.531 → 0.533).
   Whisper's test rose 0.538 → 0.557, almost 3 seed standard deviations. With dev flat, I read
   that as "no loss, maybe a small gain" rather than a proven improvement.
2. **Rare MELD emotions dipped slightly on typed text:** macro F1 0.500 → 0.489, fear 0.29 → 0.24,
   disgust 0.31 → 0.26. CREMA-D adds many angry, disgusted and fearful clips, which lowers those
   classes' training weights, and MELD's test only has 50 fear and 68 disgust lines. On Whisper
   text, macro F1 went *up* (0.390 → 0.405).
3. **Tone scores vary more between seeds than MELD scores do.** The three seeds score 0.648, 0.592
   and 0.582 on tone test. Seed 42, the one the recipe was chosen on, was the best of the three,
   so the 3-seed figure (0.663) is the fair one.
4. **The tone set is acted studio speech.** Even with unseen actors and sentences it is closer to
   CREMA-D's training clips than to a person talking to a robot through a laptop microphone. The
   live microphone check below bears that out.
5. **Joy in the voice is still mostly missed.** Few CREMA-D clips are heard as happy, and a happy
   voice is often read as neutral or fear.
6. **Same size and speed.** The architecture is unchanged (3 × RoBERTa + heads, 2.25B parameters
   in total), so section 11's latency figures still apply. One end-to-end trace of an unseen
   angry clip (`reports/trace_1037_IWW_ANG_XX.json`, "I wonder what this is about.") gave voice tag
   anger 0.90 after 36 ms, and a final tag of anger 0.71 after 497 ms, with the words alone saying
   neutral.

**Reply rule, measured on MELD test with the new model:** a mismatch note fires on 85 of 2,610
lines (3.3%). The new "voice is sure" condition accounts for 15 of them (0.6%), and on those the
voice matches MELD's label 47% of the time and the words 40%. A single seed's voice head is more
overconfident: it fires on 34 extra lines, with the voice matching on only 26%, mostly "anger" on
loud TV audio. So averaging the seeds matters here too. `VOICE_SURE` in `reply.py` is the knob if
the live notes over-flag.

**The reply model is now the weak link for tone.** Same-seed replies to flat sentences in an
angry voice were compared across three note wordings (the new one, the old sarcasm one, and the
tag alone). The 1.5B model rarely acknowledges the anger under any of them, and sometimes just
echoes the line ("The surface is slick. I need to be extra careful."). The emotion state is now
right on these lines; making Pip act on it needs a stronger reply model (section 15).

### Live microphone check (2026-10-17)

Nine lines were spoken into the web demo (`scripts/serve.py`, laptop microphone, one speaker) with
the new live model. Words alone and voice alone are the model's two single-modality heads.

| Turn | Said | How | Final tag | Words alone | Voice alone | Right? |
|---|---|---|---|---|---|---|
| 14 | "It's 11 o'clock!" | angry | **anger 70%** | joy 37% | anger 52% | ✅ the voice beat the words |
| 21 | "It's 11 o'clock." | sad | neutral 51% (sadness 25%) | neutral 72% | sadness 27% | ❌ |
| 10 | "I'm on my way to the store." | flat | **neutral 90%** | neutral 77% | neutral 66% | ✅ |
| 22 | "Thanks a lot." | sarcastic | joy 60% | joy 53% | sadness 27% | ❌ |
| 26 | "Ha ha ha ha ha ha, you're the worst." | laughing | **joy 56%** | joy 50% | joy 26% | ✅ via the transcript: Whisper wrote the laughter as words |
| 24 | "Haha, you're the worst." | playful | disgust 56% | disgust 70% | sadness 27% | ❌ |
| 35 | "You're the worst." | playful | disgust 67% | disgust 85% | surprise 36% | ❌ |
| 27 | "You got the job!" | excited | **joy 90%** | joy 90% | sadness 31% | ✅ from the words |
| 29 | "You got the job!" | excited | **joy 94%** | joy 95% | surprise 27% | ✅ from the words |

What this shows:
1. **Angry tone over neutral or positive words now works live**, which was the original
   complaint (turn 14). Flat speech stays neutral (turn 10).
2. **Every other tone was missed.** Sadness, sarcasm and a playful voice (turns 21, 22, 24 and 35)
   all lost to the words. Where the tone should change the meaning (6 lines), the voice won once
   (turn 14), the transcript rescued one (turn 26), and 4 were missed.
3. **The voice head is unsure on this microphone.** Its top guess is 25–36% on every line except
   the angry one (52%), against about 52% on average on the CREMA-D tone test. "Sadness ~27%" comes
   up on lines that aren't sad.
4. **It isn't loudness.** The live turns were about 8 dB louder than the training audio (median
   −22 dB against −30.6 dB for both MELD and CREMA-D), and WavLM's input isn't level-normalised.
   But replaying the CREMA-D tone test from 10 dB quieter to 14 dB louder keeps the voice head's
   balanced accuracy between 0.68 and 0.71. The gap is the recording conditions (laptop
   microphone, room) and natural rather than acted delivery.
5. **Pip's replies are now the most visible weakness.** They are generic, often close the
   conversation ("Have a good night!"), apologise when the person is joking ("I'm sorry if my
   responses weren't clear"), and never mention the tone.
6. **Speed held:** final tag 441–553 ms, first reply word 0.67–0.88 s and full reply 0.90–1.23 s
   after the end of each turn, all inside the targets.

Nine lines from one speaker is a spot check, not a measurement. A recorded live tone test (section
15) would turn it into one.

**Fixing the help-desk replies.** The nine lines were replayed through the reply model with the
notes their live states produced, 2 seeds each (18 replies per condition):

| Reply history | Prompt | Say goodbye | Help-desk offer | Apologise |
|---|---|---|---|---|
| Pip's actual live replies | v2 (unchanged) | 6 | 3 | 4 |
| Pip's actual live replies | v3 ("like a friend, not an assistant", "don't say goodbye unless they do", "play along with teasing") | 6 | 5 | 4 |
| **Pip's actual live replies, cleaned** | **v2** | **2** | **0** | **1** |
| Pip's own new replies, cleaned | v2 | 1 | 0 | 0 |

- **The habits snowball.** Once Pip says "Let me know if you need anything else", that reply sits
  in the history the reply model sees next turn, and the 1.5B model copies itself. Replayed
  without that history, the same prompt rarely does it.
- **A stricter prompt didn't help** (v3), so it was dropped.
- **Cleaning the history did.** `Conversation.chat_history` now passes Pip's earlier replies
  through `remembered()`, which drops sign-off and help-desk sentences ("Have a good night!",
  "Let me know if…"). What the person heard is unchanged.
- **It doesn't make Pip act on the tone.** On the angry "It's 11 o'clock!" it still talks about
  the time ("Well, it's already late. Time for some snacks and a movie!"). That still needs a
  stronger reply model.

### A ~3B reply model (2026-10-17)

Can a bigger reply model act on the tone, and does it fit? Full numbers and every reply are in
`reports/phase5_reply_model_3b.json`. The setup:
- Prompt v2 and the cleaned history (commit `53ff156`), 3 seeds.
- Two test sets: the 9 live lines as one conversation, and 4 flat sentences in an angry voice (the
  live turn 14 plus 3 CREMA-D lines).
- Replies scored with model names hidden and the order shuffled. The Q3 row was scored afterwards,
  not blind.

| Reply model (Q4_K_M unless noted) | Licence | Total params | Fits the 6 GB card? | Angry voice acknowledged (of 15) | Teasing played along (of 9) | Spoke as if it were the person (of 39) |
|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B (live)** | Apache-2.0 | 2.25B | yes, 5.6 GB peak | 2 | 1 | 11 |
| Llama-3.2-3B | Llama 3.2 Community | 3.92B | **no:** spills into system RAM; turns take minutes | 6 | 3 | 2 |
| **Qwen2.5-3B** | Qwen Research (non-commercial) | 3.80B | **barely:** only with llama.cpp's micro-batch cut to 256 (5.8–5.95 GB peak); without it, spills | **11** | 1 | 0 |
| Qwen2.5-3B, Q3_K_M | Qwen Research (non-commercial) | 3.80B | yes, 5.8 GB peak | 5 | 1 | 0 |

Phi-3.5-mini (3.8B, MIT) wasn't downloaded. It would need about 1.9 GB more GPU memory than the
1.5B (its KV cache alone is about 0.75 GB), and only about 0.56 GB is free.

What this shows:
1. **A bigger model does act on the tone.** Qwen2.5-3B acknowledged the angry voice in 11 of 15
   replies ("I understand, but your tone suggests you're frustrated."), against 2 for the 1.5B.
   It also stopped answering as if it were the person ("Oh no! I'm late.").
2. **It doesn't fit this card reliably.** The only Q4 setup that stays on the GPU leaves about
   0.2–0.3 GB spare, depending on what Windows is using. In 7 latency runs on the 158 MELD test
   turns, its first-word p95 was 1,375–1,542 ms and missed the 1,500 ms target in 3 runs. The
   1.5B met it in all 7 (1,066–1,311 ms).
3. **The smaller Q3 quantisation fits, but loses most of the gain** (5 of 15).
4. **None of them reliably plays along with teasing** (at best 3 of 9, Llama). The Qwen models
   mostly apologise to a playful "you're the worst".
5. **Moving the notes after the history** (so llama.cpp can reuse its cached prompt;
   `reports/phase5_reply_prompt_layout.json`) didn't help. End-to-end p95 latency didn't improve
   reliably, and the 3B acknowledged the tone less often (8 vs 11 of 15). The prompt layout stays
   as it is.

**Decision: keep Qwen2.5-1.5B.** The 3B is the clear quality upgrade for Pip's replies, but on a
6 GB card it trades a reliable latency target for one it sometimes misses, and its licence is
non-commercial. It's the first change to make on an 8 GB or larger GPU.

## 13. Decision log

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
| 2026-10-16 | Fusion = one jointly trained model with per-modality auxiliary heads | Per-modality explanations come from the same forward pass, and no second-stage combiner has to be trained on text features that are already overfit to train |
| 2026-10-16 | Speakers renamed A/B/C within each context window | Avoids memorising *Friends* characters; matches a two-party live conversation |
| 2026-10-16 | Train in fp32 | fp16 mixed precision was 4x slower per step on the GTX 1660 SUPER |
| 2026-10-16 | Audio capped at the first 15 s | Misaligned multi-minute clips; 99.7% of clips are unaffected |
| 2026-10-17 | Emotion tag in two stages: voice-only, then words + voice + context | Whisper's p95 (694 ms) alone breaks a 500 ms tag; the voice tag lands in about 50 ms |
| 2026-10-17 | ~~Reply notes mention a words/voice mismatch only when the fused model sided with the voice~~ | A weak voice-only guess made replies hedge when the fused model was sure. Replaced in Phase 5: it almost never fired, because the fused tag follows the words |
| 2026-10-17 | Voice cues relative to the same person's earlier turns | Works for any voice without calibrating thresholds |
| 2026-10-17 | One shared WavLM feature function for training and live use | Live features can't drift from the training ones (checked: within float16 rounding) |
| 2026-10-17 | Keep `fusion_ctx3` (trained on MELD text) live; don't adopt the Whisper-text retrains | No dev gain beyond noise; test 0.513 vs 0.536 for fusion (section 9) |
| 2026-10-17 | Live model = 3-seed ensemble of `fusion_ctx3`; mixed-text training not adopted | Dev weighted F1 tied; dev macro F1 clearly better (0.422 vs 0.391); +250M parameters, final tag p95 still 935 ms (section 11) |
| 2026-10-17 | Reply prompt v2: the person's words are the message, the notes go in the system prompt | Third-person replies 12 → 1 of 35, instructions read back 6 → 0 (section 11) |
| 2026-10-17 | Add CREMA-D to training, for the voice only; MELD stays the foundation and the scoreboard | In the live demo the tag followed the words over the tone; MELD can't teach or test tone, because its labels follow the words (section 12) |
| 2026-10-17 | Use only CREMA-D clips whose voice-only listener vote matches the intended emotion (3,099 of 7,442) | The robot should react to tone a person can hear; most other clips sound neutral to listeners |
| 2026-10-17 | Tone test = CREMA-D clips from unseen actors *and* unseen sentences | With only 12 sentences, reusing them would let the model memorise "these words → trust the voice" |
| 2026-10-17 | Recipe choice fixed before any results: best tone-dev balanced accuracy with MELD dev weighted F1 ≥ 0.613 (current model − 0.01) | Stops the tone gain being bought with MELD accuracy, and keeps test for the final score |
| 2026-10-17 | Reply notes also mention a mismatch when the voice alone is ≥ 60% sure of a non-calm emotion; plain words get "matter-of-fact" wording instead of the sarcasm note | The person hears the tone even when the fused tag goes with the words; flat words said angrily aren't sarcasm |
| 2026-10-17 | Reply history drops Pip's own sign-off and help-desk sentences; the prompt stays v2 | They snowballed in the live check. With the live history, 18 replies: goodbye 6 → 2, help-desk 3 → 0, apology 4 → 1. A stricter prompt (v3) didn't help (section 12) |
| 2026-10-17 | Keep Qwen2.5-1.5B as the reply model; Qwen2.5-3B tested and not adopted | 3B acknowledged an angry voice in 11 of 15 replies vs 2, but on 6 GB it fits only with a reduced micro-batch, about 0.2–0.3 GB spare, and missed the 1.5 s first-word p95 in 3 of 7 runs; non-commercial licence (section 12) |
| 2026-10-17 | Live model = 3-seed ensemble of `fusion_ctx3_crema_td25` (CREMA-D + words hidden 25%) | Chosen by the fixed dev rule (tone dev 0.653, MELD dev 0.634); test: tone 0.127 → 0.663, MELD 0.648 → 0.651, Whisper 0.538 → 0.557 (section 12) |

## 14. External components

Licences checked 2026-10-16 from each model's Hugging Face card or source repository.
"Changed" and "checked" fill in as each component is trained or evaluated.

| Component | Source | Licence | Changed by us | How we checked it |
|---|---|---|---|---|
| MELD dataset | [declare-lab/MELD](https://github.com/declare-lab/MELD) | GPL-3.0 (repo); clips are *Friends* footage | nothing (kept outside the repo) | archive size matches server; `gzip -t` |
| CREMA-D dataset (Phase 5) | [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | ODbL 1.0 / DbCL 1.0 | nothing; only the WAV folder, kept outside the repo | all 7,442 WAVs read as 16 kHz mono; the 4 files the README lists as broken are dropped |
| Silero VAD | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT | no | smoke test |
| Whisper small | [openai/whisper-small](https://huggingface.co/openai/whisper-small) via [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | Apache-2.0 / MIT | no | smoke test |
| RoBERTa-base | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | MIT | fine-tuned (Phase 2) | smoke test |
| WavLM-base-plus | [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) | CC BY-SA 3.0 (per its card) | frozen; small head trained on top (Phase 2) | smoke test |
| Qwen2.5-1.5B-Instruct (Q4_K_M GGUF) | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | Apache-2.0 | prompt only | smoke test |
| llama.cpp runtime | [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 0.3.4 | MIT | no | smoke test |

**AI-assisted development.** The code and experiments were built with an AI coding assistant
(Claude Code). Every number in this document comes from a script in this repo, and the JSON files
in `reports/` are the source of truth for them. I reviewed the design decisions and can explain
and change each part of the pipeline.

## 15. Limitations and what was left out

**Completed:** the Text + Audio track end to end. That covers data preparation, emotion models
(text, voice, fusion), a live pipeline with a two-stage emotion tag and a streamed reply, a trace
demo and a web demo, and an evaluation on live-style Whisper transcripts as well as MELD's typed
ones. Phase 5 added CREMA-D voice training and a tone test, so the tag also follows the voice
when the words say little.

**Limitations:**
- **Accuracy.** About 0.56 weighted F1 on Whisper transcripts of MELD, 0.65 on MELD's typed
  ones. Rare emotions are recognised poorly: fear and disgust are about 2–3% of the data, with
  F1 of 0.1–0.3.
- **Tone is measured on acted speech only.** The tone test (0.663 balanced accuracy) uses
  CREMA-D's studio recordings of flat sentences. It doesn't cover the other half of the demo
  complaint, harsh words in a cheerful voice, because there is no labelled set for that, and a
  happy voice is still mostly missed (1 of 17). In a 9-line live microphone check (section 12),
  only an angry tone reliably beat the words; sadness, sarcasm and playfulness did not.
- **MELD isn't a robot conversation.** It's acted TV dialogue with a laugh track and overlapping
  speakers. One person talking to a robot in a quiet room should be easier for speech-to-text,
  though less exaggerated emotionally. Accuracy on live speech is not measured.
- **The voice-only tag is a fast first guess, not a reliable signal** (about 0.46 weighted F1
  on its own).
- **Turn-taking is silence-based.** A pause over 400 ms ends a turn, even mid-sentence. There is
  no barge-in and no speaker identification, so several voices are treated as one person.
- **Replies come from a 1.5B model.** They are short and on topic, sometimes too literal, and in
  the Phase 4 check (section 11) the emotion notes changed them without a clear gain in fit. A
  wrong emotion can mislead the reply, and a right one is often ignored: with an angry voice on
  flat words, the notes now say so, but Pip rarely acknowledges it (section 12).
- **Hardware-specific choices.** fp32 everywhere on the GTX 1660 SUPER (fp16 was slower or
  broken), and a prebuilt llama.cpp wheel that needs Python 3.10–3.12.
- **Not in the repo:** the MELD and CREMA-D data, features and model weights. Reproducing the
  results needs both downloads plus about 2 h of preparation and training on a similar GPU.

**Intentionally left out:**
- **Vision,** the optional extension. The timebox went into making text + audio solid and
  measuring it honestly instead.
- **Reinforcement learning,** also optional. Not attempted.
- **Word-by-word live captions, barge-in, multi-speaker handling, and languages other than
  English.**

**What I would do next,** in order of expected value:
1. **A stronger reply model on a bigger GPU.** Qwen2.5-3B (3.8B parameters in total) acknowledged
   an angry voice in 11 of 15 replies, against 2 for the 1.5B, but doesn't fit a 6 GB card
   reliably (section 12). With 8 GB or more it's the first change to make; note its
   non-commercial licence.
2. **A small live tone test:** 20–30 lines recorded through the demo microphone, including harsh
   words said cheerfully and kind words said angrily. That's the case CREMA-D can't test.
3. **Voice features built for emotion** (emotion2vec+), and more happy-voice data. The voice side
   still has the most headroom, and joy is its blind spot.
4. **Better speech-to-text** (Whisper large-v3-turbo, or transcribing while the person talks).
   It costs latency on this GPU, so it needs measuring as a trade-off.
5. **RoBERTa-large for the words,** trained on a rented or free cloud GPU, because it doesn't fit
   training in 6 GB.
6. **A webcam branch** (a facial-expression model feeding the same fusion head) for the vision
   extension.
