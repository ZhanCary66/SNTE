#!/usr/bin/env python3
"""Emit (never submit) the unique training jobs behind paper Tables 1--3."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from config import MODEL_NAMES
from dataset import DATASETS, SHARED_ROOT

ROOT = Path(__file__).resolve().parent
SPLIT_SEEDS = (101, 102, 103, 104, 105)
SPLIT_TRAIN_SEED = 2
FIXED_SEEDS = (0, 1, 2)
HEADS = {
    "perstat": [],
    "concat": ["--head", "concat"],
    "timecos": ["--head", "timecos"],
}
VARIANTS = {
    "full": [],
    "no-window-norm": ["--no-standardize"],
    "neural-linear": ["--neural-encoder", "linear"],
    "speech-linear": ["--speech-encoder", "linear"],
    "tied-params": ["--tied-encoder"],
    "no-dilation": ["--neural-dilations", "1,1,1", "--speech-dilations", "1,1,1"],
}


def validate_datasets(datasets: list[str]) -> list[str]:
    """Preserve requested order, rejecting typos, empty selections and repeats."""
    if not datasets or any(name not in DATASETS for name in datasets):
        raise ValueError(f"datasets must be a nonempty selection from {DATASETS}")
    if len(set(datasets)) != len(datasets):
        raise ValueError("duplicate datasets are not allowed")
    return list(datasets)


@dataclass(frozen=True)
class Run:
    """One logical table reference; use canonical_run for its physical job."""

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

    def command(self, python: str = "python3", output_root: Path | None = None,
                data_root: Path | None = None) -> list[str]:
        run = canonical_run(self)
        output = (Path(output_root) if output_root is not None else ROOT / "results").resolve()
        data = (Path(data_root) if data_root is not None else SHARED_ROOT).resolve()
        output /= f"{run.table}/{run.label}/{run.dataset}_seed{run.seed}.json"
        command = [python, "-u", str(ROOT / "main.py"), "--model", run.model,
                   "--dataset", run.dataset, "--seed", str(run.seed),
                   "--output", str(output), "--data-root", str(data)]
        if run.split_seed is not None:
            command += ["--split-seed", str(run.split_seed)]
        return command + list(run.flags)

    def shell(self, python: str = "python3", output_root: Path | None = None,
              data_root: Path | None = None) -> str:
        return shlex.join(self.command(python, output_root, data_root))

    def slurm(self, python: str = "python3", output_root: Path | None = None,
              data_root: Path | None = None, partition: str = "GPUA800") -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
            raise ValueError("invalid Slurm partition")
        run = canonical_run(self)
        name = f"{run.table}_{run.label}_{run.dataset}_s{run.seed}"
        return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -euo pipefail
cd {shlex.quote(str(ROOT))}
{run.shell(python, output_root, data_root)}
"""


def canonical_run(run: Run) -> Run:
    """Table 2/perstat is exactly Table 1/SNTE, not an extra experiment."""
    if (run.table == "table2" and run.model == "snte"
            and run.label == f"perstat_split{run.split_seed}" and not run.flags):
        return replace(run, table="table1", label=f"snte_split{run.split_seed}")
    return run


def run_aliases(run: Run) -> tuple[Run, ...]:
    """Historical alternate locations for a canonical run (not including it)."""
    run = canonical_run(run)
    if (run.table == "table1" and run.model == "snte"
            and run.label == f"snte_split{run.split_seed}" and not run.flags):
        return (replace(run, table="table2", label=f"perstat_split{run.split_seed}"),)
    return ()


def logical_table_runs(table: str, datasets: list[str]) -> list[Run]:
    """All table references, including the 15 reused Table 2/perstat results."""
    datasets = validate_datasets(datasets)
    if table not in ("1", "2", "3", "all"):
        raise ValueError(f"unknown table {table!r}")
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


def build_runs(table: str, datasets: list[str]) -> list[Run]:
    """Unique jobs; with all three datasets, 174 for all tables or 45 for Table 2."""
    return list(dict.fromkeys(canonical_run(run) for run in logical_table_runs(table, datasets)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--emit", choices=("sh", "slurm"), default="sh")
    parser.add_argument("--slurm-dir", type=Path, default=ROOT / "slurm")
    parser.add_argument("--partition", default="GPUA800")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument("--data-root", type=Path, default=SHARED_ROOT)
    parser.add_argument("--python", default=sys.executable or "python3")
    args = parser.parse_args()
    try:
        runs = build_runs(args.table, [name.strip() for name in args.datasets.split(",")])
        if args.emit == "slurm":
            scripts = {
                args.slurm_dir / run.table / f"{run.label}_{run.dataset}_s{run.seed}.slurm":
                run.slurm(args.python, args.output_root, args.data_root, args.partition)
                for run in runs
            }
            occupied = [str(path) for path in scripts if path.exists() or path.is_symlink()]
            if occupied:
                raise ValueError("refusing to overwrite scripts: " + ", ".join(occupied))
            for path, text in scripts.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(text)
            print(f"wrote {len(runs)} slurm scripts under {args.slurm_dir}", file=sys.stderr)
        else:
            for run in runs:
                print(run.shell(args.python, args.output_root, args.data_root))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.table] = counts.get(run.table, 0) + 1
    summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
    print(f"\n{len(runs)} unique runs ({summary})", file=sys.stderr)


if __name__ == "__main__":
    main()
