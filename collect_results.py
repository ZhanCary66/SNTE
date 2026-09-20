#!/usr/bin/env python3
"""Print the paper's Tables 1--3 from the JSON files written by ``main.py``.

Run ``run_paper.py`` first; this script only aggregates. The metric is test
subject-macro accuracy in percent (the JSON stores it as a fraction, and the
chance level is 20%).

    python collect_results.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from config import MODEL_CONFIGS
from dataset import DATASETS
from run_paper import FIXED_SEEDS, HEADS, SPLIT_SEEDS, VARIANTS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(table: str, label: str, dataset: str, seed: int) -> dict | None:
    """Read one result JSON, or None when the run has not been done."""
    path = RESULTS / table / label / f"{dataset}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def macro(result: dict) -> float:
    """Test subject-macro accuracy in percent."""
    return 100.0 * result["test"]["subject_macro_accuracy"]


def describe(values: list[float]) -> str:
    """``mean±std``, or ``--`` when nothing has been run yet."""
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f}±{statistics.stdev(values):.2f}"


def over_splits(table: str, label: str, dataset: str) -> list[float]:
    """Accuracies of one row across the random cross-subject splits."""
    return [
        macro(r) for split_seed in SPLIT_SEEDS
        if (r := load(table, f"{label}_split{split_seed}", dataset, 2)) is not None
    ]


def over_seeds(table: str, label: str, dataset: str) -> list[float]:
    """Accuracies of one row across the training seeds (fixed split)."""
    return [
        macro(r) for seed in FIXED_SEEDS
        if (r := load(table, label, dataset, seed)) is not None
    ]


def print_table(title: str, rows: dict[str, str], collect) -> None:
    """Print one table: ``rows`` maps a result label to its display name."""
    print(f"\n{title}")
    print(f"{'':26s} " + "  ".join(f"{d:>16s}" for d in DATASETS))
    for label, display in rows.items():
        cells = [f"{describe(collect(label, dataset)):>16s}" for dataset in DATASETS]
        print(f"{display:26s} " + "  ".join(cells))


def main() -> None:
    print("Test subject-macro accuracy (%), chance level 20%.")

    print_table(
        "Table 1 - main results (mean ± std over 5 random cross-subject splits)",
        {name: config.display_name for name, config in MODEL_CONFIGS.items()},
        lambda label, dataset: over_splits("table1", label, dataset),
    )
    print_table(
        "Table 2 - match head (mean ± std over 5 random cross-subject splits)",
        {head: head for head in HEADS},
        lambda label, dataset: over_splits("table2", label, dataset),
    )
    print_table(
        "Table 3 - component ablation (fixed split, mean ± std over seeds 0-2)",
        {variant: variant for variant in VARIANTS},
        lambda label, dataset: over_seeds("table3", label, dataset),
    )


if __name__ == "__main__":
    main()
