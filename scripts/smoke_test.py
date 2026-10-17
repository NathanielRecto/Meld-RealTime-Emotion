"""Phase 0 smoke test: can every model on the planned inference path run on this GPU at once?

It loads all models together, pushes one input through each, and reports:
  - parameter count per model (the brief caps the total at 6B)
  - load time and warm inference time
  - GPU memory in use (the target card has 6 GB)
  - a sample LLM reply with time-to-first-token and tokens/sec

Usage:
    python scripts/smoke_test.py [--config configs/default.yaml]

Writes reports/phase0_smoke.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000
SAMPLE_WAV = ROOT / "tests" / "assets" / "sample_en.wav"
SAMPLE_TEXT = "Oh great, that's just perfect. Now we're going to be late again."
PARAM_BUDGET = 6_000_000_000


def expose_torch_cuda_dlls() -> None:
    """faster-whisper (CTranslate2) needs the cuBLAS and cuDNN DLLs. PyTorch's Windows wheel
    already ships them in torch/lib, so point the DLL search there instead of installing the
    CUDA toolkit separately."""
    if sys.platform == "win32":
        torch_lib = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(torch_lib)


def n_params(model) -> int:
    # parameters() yields tied weights once, so this is the true learned total.
    return sum(p.numel() for p in model.parameters())


def gpu_used_mb() -> float:
    """GPU memory in use across the whole device. Unlike torch.cuda counters this also sees
    CTranslate2 (Whisper) and whatever the Windows desktop is using."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def warm_ms(fn, repeats: int = 5) -> float:
    """Median time of `fn` in ms after one warm-up call (the first CUDA call pays one-off
    setup costs that a running system never sees again)."""
    fn()
    sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 1)


def load_sample_audio() -> tuple[np.ndarray, str]:
    if SAMPLE_WAV.exists():
        audio, sr = sf.read(SAMPLE_WAV, dtype="float32")
        assert sr == SR and audio.ndim == 1, f"{SAMPLE_WAV} must be 16 kHz mono"
        return audio, SAMPLE_WAV.name
    # No speech sample: low-level noise still exercises every model.
    return (np.random.default_rng(0).standard_normal(3 * SR) * 0.01).astype(np.float32), "noise"


# --- one check per pipeline component; each returns (model, info) -------------------------


def check_vad(cfg, audio):
    from silero_vad import get_speech_timestamps, load_silero_vad

    t0 = time.perf_counter()
    vad = load_silero_vad()
    load_s = time.perf_counter() - t0
    chunk = torch.from_numpy(audio[:512])  # the streaming unit: 512 samples = 32 ms
    ms = warm_ms(lambda: vad(chunk, SR), repeats=50)
    vad.reset_states()
    speech = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR)
    return vad, {
        "model": "silero-vad", "device": "cpu", "params": n_params(vad),
        "load_s": round(load_s, 2), "warm_ms": ms, "per": "32 ms chunk",
        "check": f"{len(speech)} speech segment(s) found in sample",
    }


def check_asr(cfg, audio):
    from faster_whisper import WhisperModel
    from transformers import AutoConfig, WhisperForConditionalGeneration

    c = cfg["models"]["asr"]
    t0 = time.perf_counter()
    asr = WhisperModel(c["name"], device=c["device"], compute_type=c["compute_type"])
    load_s = time.perf_counter() - t0
    out = {}

    def run():
        segments, _ = asr.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                     without_timestamps=c["without_timestamps"])
        out["text"] = " ".join(s.text.strip() for s in segments)  # transcribe() is lazy

    ms = warm_ms(run, repeats=3)
    # CTranslate2 models aren't torch modules, so build the same architecture from its
    # Hugging Face config on the "meta" device (shapes only, no weights) to count it.
    with torch.device("meta"):
        ref = WhisperForConditionalGeneration(AutoConfig.from_pretrained(c["hf_config"]))
    return asr, {
        "model": f"faster-whisper {c['name']} ({c['compute_type']})", "device": c["device"],
        "params": n_params(ref), "load_s": round(load_s, 2), "warm_ms": ms,
        "per": f"{len(audio) / SR:.1f} s utterance", "check": f"transcript: {out['text']!r}",
    }


def check_text_encoder(cfg):
    from transformers import AutoModel, AutoTokenizer

    c = cfg["models"]["text_encoder"]
    name = c["name"]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, dtype=getattr(torch, c["dtype"])).to("cuda").eval()
    load_s = time.perf_counter() - t0
    # Previous turn as context, then the target utterance (the Phase 2 input format).
    batch = tok("Ross: We were on a break!", SAMPLE_TEXT, return_tensors="pt").to("cuda")
    out = {}

    @torch.inference_mode()
    def run():
        out["h"] = model(**batch).last_hidden_state

    ms = warm_ms(run)
    finite = bool(torch.isfinite(out["h"]).all())
    return model, {
        "model": name, "device": f"cuda {c['dtype']}", "params": n_params(model),
        "load_s": round(load_s, 2), "warm_ms": ms, "per": f"{batch['input_ids'].shape[1]} tokens",
        "check": "finite output" if finite else "NaN/inf in output (fp16 problem)",
    }


def check_audio_encoder(cfg, audio):
    from transformers import AutoModel

    c = cfg["models"]["audio_encoder"]
    name, dtype = c["name"], getattr(torch, c["dtype"])
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(name, dtype=dtype).to("cuda").eval()
    load_s = time.perf_counter() - t0
    wav = torch.from_numpy(audio).to("cuda", dtype)[None]
    out = {}

    @torch.inference_mode()
    def run():
        out["h"] = model(wav).last_hidden_state

    ms = warm_ms(run)
    finite = bool(torch.isfinite(out["h"]).all())
    return model, {
        "model": name, "device": f"cuda {c['dtype']}", "params": n_params(model),
        "load_s": round(load_s, 2), "warm_ms": ms, "per": f"{len(audio) / SR:.1f} s audio",
        "check": "finite output" if finite else "NaN/inf in output (fp16 problem)",
    }


def check_llm(cfg):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    from transformers import AutoConfig, AutoModelForCausalLM

    c = cfg["models"]["llm"]
    t0 = time.perf_counter()
    path = hf_hub_download(c["gguf_repo"], c["gguf_file"])
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=c["n_ctx"], verbose=False)  # all layers on GPU
    load_s = time.perf_counter() - t0

    system = {"role": "system", "content": (
        "You are Pip, a warm, playful character robot. Reply in ONE short sentence "
        "(under 25 words) that responds to what the person said and how they feel.")}

    def generate(user_text: str):
        messages = [system, {"role": "user", "content": user_text}]
        t0 = time.perf_counter()
        first, text, n = None, "", 0
        for chunk in llm.create_chat_completion(messages, max_tokens=48, temperature=0.0, stream=True):
            piece = chunk["choices"][0]["delta"].get("content", "")
            if piece:  # llama.cpp streams one token per chunk
                n += 1
                text += piece
                if first is None and piece.strip():
                    first = time.perf_counter() - t0
        return text.strip(), n, first, time.perf_counter() - t0

    # Warm up on a different turn: llama.cpp reuses the cached system prompt (as it would
    # in a real conversation) but the measured turn's text is new, so timing stays honest.
    generate('Transcript: "Hi Pip, how are you?"\nEmotion: joy (confidence 0.71).')
    text, n_new, ttft, total = generate(
        f'Transcript: "{SAMPLE_TEXT}"\n'
        "Emotion: anger (confidence 0.58). The words alone read as joy, but the voice "
        "is loud and tense, so it is likely sarcasm.")
    # The GGUF file is the same network quantized to 4 bits; count its unquantized twin.
    with torch.device("meta"):
        ref = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(c["hf_config"]))
    garbled = not text or text.count("!") > len(text) / 2
    return llm, {
        "model": f"{c['gguf_file']} (llama.cpp)", "device": "cuda", "params": n_params(ref),
        "load_s": round(load_s, 2), "warm_ms": round(total * 1000, 1), "per": f"{n_new}-token reply",
        "first_token_ms": round(ttft * 1000, 1) if ttft else None,
        "tokens_per_s": round(n_new / total, 1),
        "check": "garbled output" if garbled else "reply looks normal",
        "reply": text,
    }


# --- environment checks ------------------------------------------------------------------


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def versions() -> dict:
    v = {"python": platform.python_version(), "cuda": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version()}
    for pkg in ["torch", "transformers", "faster-whisper", "ctranslate2", "silero-vad", "llama-cpp-python"]:
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    return v


def check_ffmpeg() -> dict:
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        first = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        return {"found": True, "version": first}
    except Exception as e:
        return {"found": False, "error": str(e)}


def check_microphone() -> dict:
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        return {"found": True, "name": dev["name"], "default_samplerate": dev["default_samplerate"]}
    except Exception as e:
        return {"found": False, "error": str(e)}


def fmt_params(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def print_summary(r: dict) -> None:
    print("\n" + "=" * 86)
    print(f"{'Component':<14} {'Model':<38} {'Params':>8} {'Load s':>7} {'Warm ms':>8}")
    for c in r["components"]:
        if "error" in c:
            print(f"{c['role']:<14} FAILED: {c['error']}")
            continue
        print(f"{c['role']:<14} {c['model'][:38]:<38} {fmt_params(c['params']):>8} "
              f"{c['load_s']:>7.1f} {c['warm_ms']:>8.1f}  ({c['per']})")
        print(f"{'':<14} -> {c['check']}")
    verdict = "OK" if r["within_budget"] else "OVER BUDGET"
    print(f"\nTotal parameters: {fmt_params(r['total_params'])} of 6B budget -> {verdict}")
    g = r["gpu_memory_mb"]
    print(f"GPU memory: {g['baseline_before_models']} MB before, {g['with_all_models']} MB with all "
          f"models (+{g['added_by_models']} MB) of {r['hardware']['gpu_vram_mb']} MB")
    llm = next((c for c in r["components"] if c["role"] == "Reply LLM" and "error" not in c), None)
    if llm:
        print(f"LLM: first token {llm['first_token_ms']} ms, {llm['tokens_per_s']} tok/s")
        print(f"     reply: {llm['reply']!r}")
    print(f"ffmpeg: {r['ffmpeg'].get('version', r['ffmpeg'])}")
    print(f"microphone: {r['microphone'].get('name', r['microphone'])}")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles choke on some characters
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if not torch.cuda.is_available():
        sys.exit("PyTorch can't see a CUDA GPU - see README setup step 2.")
    expose_torch_cuda_dlls()

    audio, audio_src = load_sample_audio()
    baseline_mb = gpu_used_mb()
    loaded, components = [], []  # keep every model alive so they share the GPU like the real app
    steps = [
        ("VAD", lambda: check_vad(cfg, audio)),
        ("ASR", lambda: check_asr(cfg, audio)),
        ("Text encoder", lambda: check_text_encoder(cfg)),
        ("Audio encoder", lambda: check_audio_encoder(cfg, audio)),
        ("Reply LLM", lambda: check_llm(cfg)),
    ]
    for role, step in steps:
        print(f"[{role}] loading and running...", flush=True)
        try:
            model, info = step()
            loaded.append(model)
        except Exception as e:  # record the failure and keep testing the rest
            traceback.print_exc()
            info = {"error": f"{type(e).__name__}: {e}"}
        components.append({"role": role, **info, "gpu_used_mb_after": round(gpu_used_mb())})

    final_mb = gpu_used_mb()
    props = torch.cuda.get_device_properties(0)
    total = sum(c.get("params", 0) for c in components)
    report = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": {
            "gpu": props.name,
            "gpu_vram_mb": round(props.total_memory / 2**20),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "cpu": cpu_name(),
            "cpu_threads": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "os": platform.platform(),
        },
        "versions": versions(),
        "sample_audio": audio_src,
        "components": components,
        "total_params": total,
        "param_budget": PARAM_BUDGET,
        "within_budget": total <= PARAM_BUDGET,
        "gpu_memory_mb": {
            "baseline_before_models": round(baseline_mb),
            "with_all_models": round(final_mb),
            "added_by_models": round(final_mb - baseline_mb),
            "torch_peak_allocated": round(torch.cuda.max_memory_allocated() / 2**20),
        },
        "process_ram_mb": round(psutil.Process().memory_info().rss / 2**20),
        "ffmpeg": check_ffmpeg(),
        "microphone": check_microphone(),
    }
    out = ROOT / "reports" / "phase0_smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
