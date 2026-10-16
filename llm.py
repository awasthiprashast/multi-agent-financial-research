"""
llm.py — language-model backends.

An LLM is used in two places, and in both its output is untrusted:

  * `rephrase` — optional fluency pass over an already-verified sentence;
  * `generate` — the LLM analyst agents (agents/llm_analysts.py) ask it to write
    research claims from retrieved evidence. Those claims go through the same
    Verification Agent as everything else, which is the point: this is the
    "LLM hallucination" the verifier exists to catch.

Order of preference (override with FINRESEARCH_LLM = groq | ollama | none):
  1. Groq free tier if GROQ_API_KEY is set (model: GROQ_MODEL, default
     openai/gpt-oss-120b; list what your key can use with the groq models API)
  2. a local Ollama server (model from OLLAMA_MODEL, else the first one installed)
  3. none — template analysts only, deterministic and free
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import config  # noqa: F401  (loads .env into the environment)

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLM:
    name = "template"

    def rephrase(self, text: str) -> str:
        return text

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        raise LLMUnavailable("no language model configured")

    @property
    def available(self) -> bool:
        return False


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqLLM(LLM):
    def __init__(self, model: Optional[str] = None):
        from groq import Groq  # noqa: WPS433

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        model = model or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        self._client = Groq(api_key=key)
        self._model = model
        self.name = f"groq/{model}"

    @property
    def available(self) -> bool:
        return True

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        if "gpt-oss" in self._model:
            # reasoning model: reasoning tokens count against the budget, so give it
            # room and keep the reasoning short
            kwargs.update(reasoning_effort="low", max_completion_tokens=3000)
        else:
            kwargs.update(max_tokens=900)
        r = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2, **kwargs)
        return (r.choices[0].message.content or "").strip()

    def rephrase(self, text: str) -> str:
        return self.generate(
            "Rewrite the analyst sentence more fluently. Do not add, remove or "
            "alter any number. Do not add any fact. Return one sentence.", text)


class OllamaLLM(LLM):
    def __init__(self, model: Optional[str] = None,
                 host: str = "http://localhost:11434"):
        import requests  # noqa: WPS433

        self._requests = requests
        self._host = host
        tags = requests.get(f"{host}/api/tags", timeout=3)
        tags.raise_for_status()
        installed = [m["name"] for m in tags.json().get("models", [])]
        model = model or os.environ.get("OLLAMA_MODEL") or (installed[0] if installed else None)
        if not model:
            raise RuntimeError("no Ollama model installed")
        self._model = model
        self.name = f"ollama/{model}"

    @property
    def available(self) -> bool:
        return True

    def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self._model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0.2, "num_predict": 700,
                        "seed": int(os.environ.get("FINRESEARCH_LLM_SEED", "7"))},
        }
        if json_mode:
            body["format"] = "json"
        r = self._requests.post(f"{self._host}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def rephrase(self, text: str) -> str:
        return self.generate(
            "Rewrite this analyst sentence more fluently. Do not change any number "
            "and do not add facts. Return one sentence.", text)


def get_llm(prefer: Optional[str] = None) -> LLM:
    prefer = (prefer or os.environ.get("FINRESEARCH_LLM", "")).lower() or None
    if prefer == "none":
        return LLM()
    order = [prefer] if prefer else ["groq", "ollama"]
    for choice in order:
        try:
            if choice == "groq":
                return GroqLLM()
            if choice == "ollama":
                return OllamaLLM()
        except Exception as exc:
            log.info("LLM backend %s unavailable: %s", choice, exc)
    return LLM()
