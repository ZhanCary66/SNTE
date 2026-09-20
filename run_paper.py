#!/usr/bin/env python3
"""Emit the exact commands behind Tables 1--3 of the paper.

Three groups of runs are defined:

- ``table1``: main results. Six models on three datasets, each under the five
  random cross-subject splits (split seeds 101-105), training seed 2.
- ``table2``: match-head comparison. Three heads under the same five random
  splits.
- ``table3``: component ablation. Six encoder variants on the fixed split,
  training seeds 0-2.

Examples:
    # print every command for Table 1
    python run_paper.py --table 1 --emit sh

    # write one slurm script per run
    python run_paper.py --table all --emit slurm --slurm-dir slurm

    # run a single job locally (for debugging)
    python run_paper.py --table 3 --emit sh | head -1 | bash
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from config import MODEL_NAMES
from dataset import DATASETS

ROOT = Path(__file__).resolve().parent
# Random cross-subject splits used by Tables 1 and 2.
SPLIT_SEEDS = (101, 102, 103, 104, 105)
# Training seed used by Tables 1 and 2 (the split comparison fixes it).
SPLIT_TRAIN_SEED = 2
# Training seeds used by Table 3 on the fixed split.
FIXED_SEEDS = (0, 1, 2)

# Table 2: head name -> extra flags.
HEADS = {
    "perstat": [],
    "concat": ["--head", "concat"],
    "timecos": ["--head", "timecos"],
}

# Table 3: variant name -> extra flags.
VARIANTS = {
    "full": [],
    "no-window-norm": ["--no-standardize"],
    "neural-linear": ["--neural-encoder", "linear"],
    "speech-linear": ["--speech-encoder", "linear"],
    "tied-params": ["--tied-encoder"],
    "no-dilation": ["--neural-dilations", "1,1,1", "--speech-dilations", "1,1,1"],
}


@dataclass(frozen=True)
class Run:
    """One training job."""

    table: str
    label: str
    model: str
    dataset: str
    seed: int
    split_seed: int | None
    flags: tuple[str, ...]

    @property
    def tag(self) -> str:
        return f"{self.table}/{self.label}"

    def command(self, python: str = "python3") -> list[str]:
        """The full command line for this run."""
        output = ROOT / "results" / self.table / self.label / f"{self.dataset}_seed{self.seed}.json"
        command = [
            python, "-u", "main.py",
            "--model", self.model,
            "--dataset", self.dataset,
            "--seed", str(self.seed),
            "--output", str(output),
        ]
        if self.split_seed is not None:
            command += ["--split-seed", str(self.split_seed)]
        return command + list(self.flags)

    def shell(self, python: str = "python3") -> str:
        return " ".join(self.command(python))

    def slurm(self, python: str = "python3") -> str:
        """A self-contained slurm script for this run."""
        name = f"{self.table}_{self.label}_{self.dataset}_s{self.seed}"
        body = " ".join(self.command(python))
        return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -euo pipefail
cd {ROOT}
{body}
"""


def build_runs(table: str, datasets: list[str]) -> list[Run]:
    """All runs of one table (or of every table when ``table == "all"``)."""
    runs: list[Run] = []
    if table in ("1", "all"):
        for model in MODEL_NAMES:
            for dataset in datasets:
                for split_seed in SPLIT_SEEDS:
                    runs.append(Run("table1", f"{model}_split{split_seed}", model, dataset,
                                    SPLIT_TRAIN_SEED, split_seed, ()))
    if table in ("2", "all"):
        for head, flags in HEADS.items():
            for dataset in datasets:
                for split_seed in SPLIT_SEEDS:
                    runs.append(Run("table2", f"{head}_split{split_seed}", "snte", dataset,
                                    SPLIT_TRAIN_SEED, split_seed, tuple(flags)))
    if table in ("3", "all"):
        for variant, flags in VARIANTS.items():
            for dataset in datasets:
                for seed in FIXED_SEEDS:
                    runs.append(Run("table3", variant, "snte", dataset, seed, None, tuple(flags)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--emit", choices=("sh", "slurm"), default="sh")
    parser.add_argument("--slurm-dir", type=Path, default=ROOT / "slurm")
    parser.add_argument("--python", default=sys.executable or "python3",
                        help="interpreter used inside the emitted commands")
    args = parser.parse_args()

    datasets = [d for d in DATASETS if d in [x.strip() for x in args.datasets.split(",")]]
    runs = build_runs(args.table, datasets)
    if not runs:
        raise SystemExit("no runs selected")

    if args.emit == "slurm":
        args.slurm_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            sub = args.slurm_dir / run.table
            sub.mkdir(parents=True, exist_ok=True)
            path = sub / f"{run.label}_{run.dataset}_s{run.seed}.slurm"
            path.write_text(run.slurm(args.python), encoding="utf-8")
        print(f"wrote {len(runs)} slurm scripts under {args.slurm_dir}", file=sys.stderr)
    else:
        for run in runs:
            print(run.shell(args.python))

    counts: dict[str, int] = {}
    for run in runs:
        counts[run.table] = counts.get(run.table, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"\n{len(runs)} runs ({summary})", file=sys.stderr)


if __name__ == "__main__":
    main()
