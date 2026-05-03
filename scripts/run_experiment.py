from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full ML experiment.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    run_dir = run_experiment(args.config)
    print(f"Experiment completed: {run_dir}")


if __name__ == "__main__":
    main()
