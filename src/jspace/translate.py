"""Live token translation to English.

The J-lens frequently surfaces non-English tokens (e.g. Chinese 巴黎 for "Paris",
Japanese 京 for "capital"). This module glosses such tokens into English so the
grid is readable, using the already-loaded multilingual model itself as the
translator (no extra dependency or download).

Design:
  * Tokens that are already plain ASCII / whitespace / punctuation are returned
    unchanged (fast path -- most tokens need no translation).
  * The rest are translated in one batched ``generate`` call with a compact
    few-shot prompt, then cached in memory so repeat lookups are free.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import torch

from .model import LensModel

logger = logging.getLogger(__name__)

# A token needs translating only if it contains a non-ASCII letter (CJK,
# Cyrillic, Arabic, accented Latin, etc.). Pure ASCII, digits, punctuation and
# whitespace are left as-is.
_NON_ASCII = re.compile(r"[^\x00-\x7f]")

_FEWSHOT = (
    "Translate each token to a short English word or phrase. "
    "Reply with only the English meaning, nothing else.\n"
    "Token: 巴黎\nEnglish: Paris\n"
    "Token: 京\nEnglish: capital\n"
    "Token: مرحبا\nEnglish: hello\n"
    "Token: {tok}\nEnglish:"
)


class Translator:
    def __init__(self, model: LensModel):
        self.model = model
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        # Hard cap on how many NEW (uncached) tokens a single request may
        # translate, so a huge token list cannot pin the GPU with sequential
        # generate calls.
        self.max_new_per_request = 128

    @staticmethod
    def needs_translation(token: str) -> bool:
        return bool(_NON_ASCII.search(token or ""))

    def translate_batch(self, tokens: list[str], max_new_tokens: int = 8) -> dict[str, str]:
        """Return a {token: english} map for the given tokens.

        ASCII tokens map to themselves. Non-ASCII tokens are translated once and
        cached. Only genuinely new non-ASCII tokens hit the model.
        """
        result: dict[str, str] = {}
        todo: list[str] = []
        for t in tokens:
            if not self.needs_translation(t):
                result[t] = t
            elif t in self._cache:
                result[t] = self._cache[t]
            else:
                todo.append(t)
        # De-duplicate while preserving order.
        todo = list(dict.fromkeys(todo))
        if not todo:
            return result
        if len(todo) > self.max_new_per_request:
            # Translate the first N new tokens; leave the rest as raw tokens.
            for t in todo[self.max_new_per_request:]:
                result[t] = t
            todo = todo[: self.max_new_per_request]

        with self._lock:
            # Re-check cache under lock (another thread may have filled it).
            pending = [t for t in todo if t not in self._cache]
            for t in pending:
                try:
                    self._cache[t] = self._translate_one(t, max_new_tokens)
                except Exception as exc:  # never let translation break the grid
                    logger.debug("translate failed for %r: %s", t, exc)
                    self._cache[t] = t
        for t in todo:
            result[t] = self._cache.get(t, t)
        return result

    def _translate_one(self, token: str, max_new_tokens: int) -> str:
        # Sanitize: collapse newlines and cap length so a token cannot inject
        # extra few-shot lines (e.g. "English: x\nToken: y") to steer output.
        safe = (token or "").replace("\n", " ").replace("\r", " ").strip()
        safe = safe[:64] or token
        prompt = _FEWSHOT.format(tok=safe)
        tok = self.model.tokenizer
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.model.device)
        attention_mask = enc.attention_mask.to(self.model.device)
        with torch.no_grad():
            out = self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None
                              else tok.eos_token_id),
            )
        gen = out[0][input_ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        # Keep the first line / first few words only.
        text = text.strip().splitlines()[0].strip() if text.strip() else token
        # Trim trailing artefacts like "Token:" the model may continue with.
        for stop in ("Token:", "English:", "\n"):
            idx = text.find(stop)
            if idx > 0:
                text = text[:idx].strip()
        # Cap length.
        if len(text) > 40:
            text = text[:40] + "…"
        return text or token
