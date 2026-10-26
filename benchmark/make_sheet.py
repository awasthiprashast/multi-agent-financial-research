"""
Write benchmark/label_sheet.md: for every company the frozen data sheet (what a sentence may be
checked against) followed by the numbered sentences of each chatbot answer.
    python -m benchmark.make_sheet
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agents.auditor import normalise, split_sentences
from benchmark.frozen import FrozenPipeline, data_sheet
from data.universe import NAME

HERE = Path(__file__).resolve().parent


def sentences_of(answer: str) -> list:
    return split_sentences(normalise(answer))


def main() -> None:
    os.environ.setdefault("FINRESEARCH_EMBEDDER", "tfidf")
    os.environ.setdefault("FINRESEARCH_SCORER", "lexicon")
    answers = json.loads((HERE / "raw_answers.json").read_text(encoding="utf-8"))
    for a in answers:
        a.setdefault("setting", "ungrounded")
    pipe = FrozenPipeline()
    lines, total = [], 0
    tickers = sorted({a["ticker"] for a in answers}, key=[a["ticker"] for a in answers].index)
    for ticker in tickers:
        lines += ["", f"## {NAME[ticker]} ({ticker})", "", "### Data sheet (frozen)", ""]
        lines += ["- " + line for line in data_sheet(pipe._prepare(ticker))]
        for a in answers:
            if a["ticker"] != ticker:
                continue
            lines += ["", f"### {a['setting']} answer by {a['model']}", ""]
            for i, s in enumerate(sentences_of(a["answer"]), 1):
                lines.append(f"{ticker}|{a['model']}|{a['setting']}|{i}  {s}")
                total += 1
    (HERE / "label_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    print("sentences:", total)


if __name__ == "__main__":
    main()
