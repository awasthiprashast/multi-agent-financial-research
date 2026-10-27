"""
Build benchmark/review_sheet.md: the sentences to re-check, most important first.

    python -m benchmark.review

Priority 1  the verifier and the label disagree (a wrong label here changes the headline numbers)
Priority 2  labels marked ambiguous (A) or partly checkable (P)
Priority 3  a random sample of the remaining labels, to estimate how reliable the rest are

Each entry shows the label, what the verifier said and why, the sentence, and the company's
frozen figures. To change a label:  python -m benchmark.relabel "<id>" <C|W|P|N|A>
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ["FINRESEARCH_EMBEDDER"] = "tfidf"
os.environ["FINRESEARCH_SCORER"] = "lexicon"
os.environ["FINRESEARCH_LLM"] = "none"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.frozen import FrozenPipeline, data_sheet  # noqa: E402
from benchmark.labels import DEV, NAMES, TEST  # noqa: E402
from eval import benchmark_real as b  # noqa: E402

OUT = Path(__file__).resolve().parent / "review_sheet.md"
SAMPLE = 30


def disagreement(r: dict) -> bool:
    g, p = r["gold"], r["pred"]
    if g in "CW":
        return p != g                      # wrong verdict or an abstention on a checkable sentence
    if g in "NP":
        return p in "CW"                   # committed where the label says no clean verdict
    return False


def main() -> None:
    rows = b.run(DEV + TEST)
    pipe = FrozenPipeline()
    facts = {}
    for t in DEV + TEST:
        lines = data_sheet(pipe._prepare(t))
        facts[t] = lines[0] + " | " + " ".join(x for x in lines[1:] if x.startswith(("[fundamentals_", "[financials_")))

    first = [r for r in rows if disagreement(r)]
    second = [r for r in rows if r["gold"] in "AP" and r not in first]
    rest = [r for r in rows if r not in first and r not in second]
    random.Random(7).shuffle(rest)
    third = rest[:SAMPLE]

    def block(title: str, note: str, items: list) -> list:
        out = [f"\n## {title} ({len(items)})", "", note, ""]
        for r in items:
            split = "held-out" if r["ticker"] in TEST else "dev"
            out += [f"### `{r['id']}`  ({split})",
                    f"- **My label:** {r['gold']} ({NAMES[r['gold']]})   **Verifier:** {r['status']}"
                    + (f" — {r['reason'][:240]}" if r["reason"] else ""),
                    f"- **Sentence:** {r['text']}",
                    f"- **Figures:** {facts[r['ticker']][:900]}",
                    "- **Your label:** ____", ""]
        return out

    lines = ["# Label review sheet", "",
             "Labels: **C** correct, **W** wrong, **P** partly checkable, **N** nothing the data sheet can settle, "
             "**A** ambiguous (excluded). A sentence is **W** if any figure or direction in it that the "
             "figures below can settle is wrong; **C** if everything it states that they can settle is right "
             "(\"about/around/near\" allows 2 %, otherwise round to the precision written; units count: "
             "₹1.63 trillion is not ₹16.3 trillion)."]
    lines += block("1. Verifier and label disagree",
                   "Decide who is right. If the label is wrong, fix it; if the verifier is wrong, leave it: "
                   "that is a real weakness and belongs in the report.", first)
    lines += block("2. Ambiguous or partly checkable labels",
                   "Pick the label you would defend in front of an examiner, or keep A.", second)
    lines += block(f"3. Random sample of {SAMPLE} other labels",
                   "If more than 2 of these are wrong, re-read the rest of that kind of sentence.", third)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"priority 1: {len(first)}  priority 2: {len(second)}  priority 3: {len(third)}  -> {OUT}")


if __name__ == "__main__":
    main()
