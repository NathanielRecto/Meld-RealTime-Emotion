"""The reply (output 2): a short, in-character response streamed from a small local LLM.

The model never sees probabilities. The state is turned into plain-language notes ("they
clearly sound angry; their words sound happy but their voice sounds angry"), which is what
grounds the reply in the multimodal input."""
from __future__ import annotations

from typing import Iterator

ADJECTIVE = {"anger": "angry", "disgust": "disgusted", "fear": "worried or scared", "joy": "happy",
             "neutral": "calm", "sadness": "sad", "surprise": "surprised"}

SYSTEM = (
    "You are {name}, a small, warm character robot chatting with one person. "
    "Reply in one or two short sentences, under {max_words} words in total. "
    "Respond to what they said and to how they seem to feel, as described in the notes. "
    "Never argue with or correct their feelings, never mention the notes or how you noticed anything, "
    "and don't use emojis. If the notes say you're unsure how they feel, respond to their words and stay gentle."
)


def notes(state: dict) -> str:
    """Plain-language description of the final state, for the reply prompt."""
    e = state["emotion"]
    sure = "clearly" if e["confidence"] >= 0.6 else "probably" if e["confidence"] >= 0.4 else "maybe (unsure)"
    parts = [f"They {sure} sound {ADJECTIVE[e['label']]}."]
    words, voice = state["by_modality"]["words"], state["by_modality"]["voice"]
    # Flag a mismatch only when the fused model itself sided with the voice over the words (the
    # sarcasm case). If it went with the words, the voice head (the weaker one) was probably wrong.
    if words["label"] != voice["label"] == e["label"] and voice["confidence"] >= 0.4:
        parts.append(f"Their words sound {ADJECTIVE[words['label']]} but their voice sounds "
                     f"{ADJECTIVE[voice['label']]}, so they may not mean the words literally.")
    changes = [v for v in state["cues"].get("vs_earlier", {}).values() if v != "same"]
    if changes:
        parts.append("Compared with earlier, their voice is " + ", ".join(changes) + ".")
    if state["mood_trend"] in ("improving", "worsening"):
        parts.append(f"Their mood is {state['mood_trend']} over the conversation.")
    return " ".join(parts)


class ReplyGenerator:
    def __init__(self, cfg: dict):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama  # needs the CUDA DLLs on PATH (meldrt.runtime)

        c, rt = cfg["models"]["llm"], cfg["realtime"]
        self.llm = Llama(model_path=hf_hub_download(c["gguf_repo"], c["gguf_file"]),
                         n_gpu_layers=-1, n_ctx=c["n_ctx"], verbose=False)
        self.system = SYSTEM.format(name=cfg["pipeline"]["robot_name"], max_words=rt["max_reply_words"])
        self.max_tokens = 2 * rt["max_reply_words"]

    def stream(self, history: list[dict], text: str, notes_text: str) -> Iterator[str]:
        messages = [{"role": "system", "content": self.system}, *history,
                    {"role": "user", "content": f'They said: "{text}"\nNotes: {notes_text}'}]
        for chunk in self.llm.create_chat_completion(messages, max_tokens=self.max_tokens,
                                                     temperature=0.7, top_p=0.9, stream=True):
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece
