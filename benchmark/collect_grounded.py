"""
Collect chatbot answers when the model IS given the company's data sheet (the way a
research tool would use it):  python -m benchmark.collect_grounded
Appends to benchmark/raw_answers.json with setting="grounded".
"""

from __future__ import annotations

import json
import os
import sys
import time

from benchmark.collect import MODELS, OUT, ask
from benchmark.frozen import FrozenPipeline, data_sheet
from data.universe import BY_TICKER, CURATED_TICKERS

PROMPT = ("Facts about {name} ({ticker}):\n{facts}\n\nUsing these facts, write a short "
          "analyst-style note for a retail investor covering price action and technical "
          "indicators, valuation and profitability, and risks. 6 to 9 sentences, plain prose, "
          "no tables or headings.")


def main() -> None:
    os.environ.setdefault("FINRESEARCH_EMBEDDER", "tfidf")
    os.environ.setdefault("FINRESEARCH_SCORER", "lexicon")
    answers = json.loads(OUT.read_text(encoding="utf-8"))
    for a in answers:
        a.setdefault("setting", "ungrounded")
    done = {(a["ticker"], a["model"], a["setting"]) for a in answers}
    pipe = FrozenPipeline()
    for ticker in CURATED_TICKERS:
        facts = "\n".join("- " + line for line in data_sheet(pipe._prepare(ticker)))
        for model in MODELS:
            if (ticker, model, "grounded") in done:
                continue
            prompt = PROMPT.format(name=BY_TICKER[ticker].name, ticker=ticker, facts=facts)
            text = ask(model, prompt)
            answers.append({"ticker": ticker, "model": model, "setting": "grounded",
                            "collected_on": time.strftime("%Y-%m-%d"), "prompt": prompt,
                            "answer": text})
            OUT.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
            print(ticker, model, len(text), "chars", file=sys.stderr)


if __name__ == "__main__":
    main()
