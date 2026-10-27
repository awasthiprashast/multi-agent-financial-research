"""
Change one label safely:

    python -m benchmark.relabel "LT.NS|openai/gpt-oss-120b|grounded|3" W

The id is the one printed in the review sheet. The code is C, W, P, N or A. The file
benchmark/labels.py is edited in place; run `python -m eval.benchmark_real` afterwards.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from benchmark.labels import LABELS

PATH = Path(__file__).resolve().parent / "labels.py"
MODELS = {"openai/gpt-oss-120b": "GPT", "qwen/qwen3.8-27b": "QWEN"}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[2] not in list("CWPNA"):
        raise SystemExit(__doc__)
    ticker, model, setting, idx = sys.argv[1].split("|")
    key, i = (ticker, model, setting), int(idx) - 1
    codes = LABELS[key]
    if not 0 <= i < len(codes):
        raise SystemExit(f"{sys.argv[1]}: this answer has {len(codes)} sentences")
    new = codes[:i] + sys.argv[2] + codes[i + 1:]
    text = PATH.read_text(encoding="utf-8")
    pat = re.compile(r'(\("' + re.escape(ticker) + r'", ' + MODELS[model] + r', "' + setting + r'"\): ")'
                     + re.escape(codes) + '"')
    if not pat.search(text):
        raise SystemExit("could not find that entry in benchmark/labels.py")
    PATH.write_text(pat.sub(lambda m: m.group(1) + new + '"', text, count=1), encoding="utf-8")
    print(f"{sys.argv[1]}: {codes[i]} -> {sys.argv[2]}   ({codes} -> {new})")


if __name__ == "__main__":
    main()
