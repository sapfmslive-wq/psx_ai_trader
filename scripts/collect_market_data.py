"""Thin wrapper so the live collector can be started as
`python scripts/collect_market_data.py` in addition to
`python -m src.main run-collector` — matches the project's documented
script layout (Section 54/91) without duplicating any logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run-collector", *sys.argv[1:]]))
