"""
logging_setup.py — one place that configures logging for the CLI, API and UI.

  FINRESEARCH_LOG_LEVEL   DEBUG | INFO (default) | WARNING | ERROR
  FINRESEARCH_LOG_JSON    1 for one-JSON-object-per-line output (log aggregators)

Libraries log through `logging.getLogger(__name__)`; nothing else in the project
configures handlers.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            out.update(extra)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(level: str | None = None, json_output: bool | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("FINRESEARCH_LOG_LEVEL") or "INFO").upper()
    if json_output is None:
        json_output = os.environ.get("FINRESEARCH_LOG_JSON", "") in ("1", "true", "yes")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # chatty third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                  "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
