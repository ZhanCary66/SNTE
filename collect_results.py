#!/usr/bin/env python3
"""Validate selected paper results before printing official subject-macro tables."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from config import MODEL_CONFIGS
from dataset import DATASETS
from results_io import ResultError, load_subject_map, preflight_runs, run_paths
from run_paper import HEADS, VARIANTS, Run, canonical_run, logical_table_runs

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def macro(result: dict) -> float:
    """Test subject-macro accuracy in percent, after shared validation."""
    return 100.0 * result["test"]["subject_macro_accuracy"]


def describe(values: list[float]) -> str:
    if len(values) < 2:
        raise ResultError("official summaries require the complete repeated-run sample")
    return f"{statistics.mean(values):.2f}±{statistics.stdev(values):.2f}"


def print_table(table: str, references: list[Run], loaded: dict[Run, dict],
                datasets: list[str], coverage: bool) -> None:
    rows = ({name: config.display_name for name, config in MODEL_CONFIGS.items()}
            if table == "table1" else
            {label: label for label in (HEADS if table == "table2" else VARIANTS)})
    titles = {"table1": "Table 1 - main results", "table2": "Table 2 - match head",
              "table3": "Table 3 - component ablation"}
    suffix = ("coverage only (n/expected); not an official summary" if coverage else
              "fixed split, mean ± std over seeds 0-2" if table == "table3" else
              "mean ± std over 5 random cross-subject splits")
    print(f"\n{titles[table]} ({suffix})")
    print(f"{'':26s} " + "  ".join(f"{dataset:>16s}" for dataset in datasets))
    for label, display in rows.items():
        cells = []
        for dataset in datasets:
            runs = [canonical_run(run) for run in references if run.table == table
                    and run.dataset == dataset
                    and (run.label == label if table == "table3" else
                         run.label.rsplit("_split", 1)[0] == label)]
            values = [macro(loaded[run]) for run in runs if run in loaded]
            cell = f"{len(values)}/{len(runs)}" if coverage else describe(values)
            cells.append(f"{cell:>16s}")
        print(f"{display:26s} " + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--table", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--partial", action="store_true", help="report coverage only; never summarize missing runs")
    parser.add_argument("--subject-map", type=Path, help="explicit root-relative legacy metadata JSON")
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",")]
    try:
        references = logical_table_runs(args.table, datasets)
        mapping = load_subject_map(args.subject_map) if args.subject_map else None
        loaded, missing = preflight_runs(args.results, references, mapping)
    except (ValueError, OSError) as error:
        parser.exit(1, f"error: {error}\n")
    if missing:
        print(f"missing {len(missing)} expected result(s):", file=sys.stderr)
        for run in missing:
            print("  " + " OR ".join(str(path) for path in run_paths(args.results, run)), file=sys.stderr)
        if not args.partial:
            parser.exit(1, "no official tables printed; use --partial for coverage only\n")
    print("Result coverage (n/expected)." if args.partial else
          "Test subject-macro accuracy (%), chance level 20%.")
    tables = ("table1", "table2", "table3") if args.table == "all" else (f"table{args.table}",)
    for table in tables:
        print_table(table, references, loaded, datasets, args.partial)


if __name__ == "__main__":
    main()
