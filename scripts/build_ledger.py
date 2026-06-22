"""Build the unified experiment ledger (standalone wrapper).

Thin CLI over :func:`backdoord.results.ledger.write_ledger`. The ledger is also
regenerated automatically as the final step of ``bdd results consolidate``; use
this script to rebuild it from the current ``results/`` CSVs without re-syncing.

    uv run python scripts/build_ledger.py [--results-dir results]
"""

import argparse
import logging
import sys
from pathlib import Path

from backdoord.results.ledger import write_ledger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def main() -> None:
    """Rebuild ``<results-dir>/ledger.csv`` from the current result CSVs."""
    parser = argparse.ArgumentParser(
        description="Build the unified attack×defense ledger"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    out = write_ledger(args.results_dir)

    sys.stdout = sys.__stdout__
    print(out)  # noqa: T201


if __name__ == "__main__":
    main()
