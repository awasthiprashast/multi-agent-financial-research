"""Freeze live data for the 12 curated companies:  python -m benchmark.freeze_all"""

from benchmark.frozen import freeze
from data.universe import CURATED_TICKERS

if __name__ == "__main__":
    for t in CURATED_TICKERS:
        print(t, freeze(t))
