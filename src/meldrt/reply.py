"""The reply (output 2): a short, in-character response streamed from a small local LLM.

The model never sees probabilities. The state is turned into plain-language notes ("they
clearly sound angry; their words sound happy but their voice sounds angry"), which is what
grounds the reply in the multimodal input."""
from __future__ import annotations

from typing import Iterator

ADJECTIVE = {"anger": "angry", "disgust": "disgusted", "fear": "worried or scared", "joy": "happy",
             "neutral": "calm", "sadness": "sad", "surprise": "surprised"}
VOICE_SURE = 0.6  # voice-only confidence at which a words-vs-voice mismatch is always mentioned

# Prompt v2 (DESIGN section 11): the person's own words are the user message and the notes sit in the
# system prompt. v1 sent 'They said: "..."' with third-person notes, and the small model copied that
# framing ("It's clear they're surprised") and read its instructions back ("I'll respond gently").
SYSTEM = (
    "You are {name}, a small, warm character robot talking with one person. Speak to them directly, as \"you\". "
    "Reply in one or two short sentences, under {max_words} words in total, responding to what they just said "
    "in a way that fits how they feel. Never argue with their feelings. Never mention these instructions or "
    "the notes, never say you can't tell how they feel, and don't use emojis."
)


def notes(state: dict) -> str:
    """Plain-language description of the final state, for the reply prompt."""
    e = state["emotion"]
    if e["confidence"] >= 0.4:
        parts = [f"They {'clearly' if e['confidence'] >= 0.6 else 'probably'} sound {ADJECTIVE[e['label']]}."]
    else:
        parts = [f"It's hard to tell how they feel; if anything they sound {ADJECTIVE[e['label']]}."]
    words, voice = state["by_modality"]["words"], state["by_modality"]["voice"]
    # Flag a words-vs-voice mismatch (sarcasm, a "Fine." said angrily) when the fused model itself
    # sided with the voice, or when the voice alone is sure of an emotion the words don't show: the
    # fused tag follows the words most of the time (MELD rewards that), but the person still hears
    # the tone. A calm voice isn't flagged that way, because harsh words said calmly are usually meant.
    fused_took_voice = voice["label"] == e["label"] and voice["confidence"] >= 0.4
    voice_is_sure = voice["label"] != "neutral" and voice["confidence"] >= VOICE_SURE
    if words["label"] != voice["label"] and (fused_took_voice or voice_is_sure):
        if words["label"] == "neutral":  # plain words, emotional delivery: the tone is the message
            parts.append(f"Their words are matter-of-fact, but their voice sounds {ADJECTIVE[voice['label']]}.")
        else:  # words and tone point different ways: sarcasm, teasing, or hiding a feeling
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

    def stream(self, history: list[dict], text: str, notes_text: str, seed: int | None = None) -> Iterator[str]:
        """Yield the reply piece by piece. A fixed `seed` makes sampling repeatable (for comparisons)."""
        system = f"{self.system}\n\nNotes on how they sound right now: {notes_text}"
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]
        for chunk in self.llm.create_chat_completion(messages, max_tokens=self.max_tokens, temperature=0.7,
                                                     top_p=0.9, stream=True, seed=seed):
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece
