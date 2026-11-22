"""One-command runner for every project comparison case.

Default: rebuild features and run all 16 experiment cases (8 unsupervised and
8 supervised). Every model receives the complete chronological 70% training
partition. Use --resume-engineered when the raw-data processing stages already
succeeded and only models must be rerun.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def call(arguments: list[str]) -> None:
    print("\n>>>", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-engineered", action="store_true",
                        help="skip cleaning/feature engineering and reuse engineered Parquet files")
    parser.add_argument("--supervised-train-size", type=int, default=0,
                        help="0 uses every row in the 70%% training partition")
    args = parser.parse_args()
    baseline = [sys.executable, "-m", "src.rebuild_project", "--prepare-only"]
    if args.resume_engineered:
        baseline.append("--resume-engineered")
    call(baseline)
    call([sys.executable, "-m", "src.large_training"])
    call([sys.executable, "-m", "src.supervised_compare", "--train-size", str(args.supervised_train_size)])
    print("\nCompleted all all-row cases. Open larger_training_output/comparison.csv and supervised_output/comparison.csv.")


if __name__ == "__main__":
    main()
