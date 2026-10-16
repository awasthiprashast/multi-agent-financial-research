"""Test configuration.

The unit tests use the deterministic offline backends (TF-IDF embedder, lexicon
sentiment scorer). They are fast and identical on every machine; the neural
backends are exercised by `python -m eval.run_all --embedder neural`.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("FINRESEARCH_EMBEDDER", "tfidf")
os.environ.setdefault("FINRESEARCH_SCORER", "lexicon")
os.environ.setdefault("FINRESEARCH_OFFLINE", "1")          # never touch the network in tests
os.environ.setdefault("FINRESEARCH_ANALYSTS", "template")
os.environ["FINRESEARCH_LLM"] = "none"                     # nor a model, even if keys exist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
