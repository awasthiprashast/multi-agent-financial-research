"""
Collect chatbot answers about each company, with no data supplied to the model (it answers
from memory, like a consumer chatbot):  python -m benchmark.collect
Writes benchmark/raw_answers.json. Models are queried once; the answers are then fixed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import config  # noqa: F401  (loads .env)
from data.universe import BY_TICKER, CURATED_TICKERS

OUT = Path(__file__).resolve().parent / "raw_answers.json"
MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
SYSTEM = "You are a helpful financial assistant."
PROMPT = ("Write a short analyst-style note on {name} ({ticker}) for a retail investor. Cover "
          "recent price action and technical indicators, valuation and profitability, and "
          "risks. Use specific figures. 6 to 9 sentences, plain prose, no tables or headings.")


def ask(model: str, prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kw = {"reasoning_effort": "low", "max_completion_tokens": 3000} if "gpt-oss" in model \
        else {"max_completion_tokens": 900}              # qwen's free tier caps output tokens/minute
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.2,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}], **kw)
            break
        except Exception as exc:                          # rate limit: wait and retry
            if "rate_limit" not in str(exc) or attempt == 5:
                raise
            time.sleep(20 * (attempt + 1))
    text = r.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main() -> None:
    answers = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(a["ticker"], a["model"]) for a in answers}
    for ticker in CURATED_TICKERS:
        for model in MODELS:
            if (ticker, model) in done:
                continue
            name = BY_TICKER[ticker].name
            text = ask(model, PROMPT.format(name=name, ticker=ticker))
            answers.append({"ticker": ticker, "model": model, "collected_on": time.strftime("%Y-%m-%d"),
                            "prompt": PROMPT.format(name=name, ticker=ticker), "answer": text})
            OUT.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
            print(ticker, model, len(text), "chars", file=sys.stderr)


if __name__ == "__main__":
    main()
