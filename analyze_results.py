#!/usr/bin/env python3
"""Paper statistics from complete, validated result JSONs; no raw data is read.

Table 1 uses nominal (subject, repartition) Wilcoxon comparisons. Repeated
participants are related observations, not independent folds. Optional
subject-collapsed comparisons average repeats, but do not establish independence.
Table 2 uses distinct-subject averages across repartitions (the archive
convention), and Table 3 averages each subject across the three training seeds.
All differences and margins are in percentage points, not relative percent.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from config import MODEL_NAMES
from dataset import DATASETS
from results_io import ResultError, load_run, load_subject_map, subject_values, validate_subject_spellings
from run_paper import FIXED_SEEDS, HEADS, SPLIT_SEEDS, SPLIT_TRAIN_SEED, VARIANTS, Run, logical_table_runs

# Two-tailed Nemenyi critical values divided by sqrt(2), alpha=0.05.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}
BlockKey = tuple[str, int]


def block_key(run: Run) -> BlockKey:
    """Dataset plus repartition seed, or training seed for a fixed split."""
    return run.dataset, run.split_seed if run.split_seed is not None else run.seed


def row_label(run: Run) -> str:
    if run.table == "table1":
        return run.model
    if run.table == "table2":
        return run.label.removesuffix(f"_split{run.split_seed}")
    return run.label


def _aligned_keys(columns: dict) -> list[BlockKey]:
    """Reject different block sets even when their cardinalities agree."""
    if not columns:
        raise ResultError("no columns to align")
    reference = next(iter(columns))
    expected = set(columns[reference])
    if not expected:
        raise ResultError(f"no blocks for {reference}")
    for label, column in columns.items():
        actual = set(column)
        if actual != expected:
            raise ResultError(
                f"block mismatch {reference} vs {label}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    return sorted(expected)


def align_blocks(columns: dict[str, dict[BlockKey, float]]) -> tuple[list[BlockKey], np.ndarray]:
    keys = _aligned_keys(columns)
    matrix = np.asarray([[column[key] for column in columns.values()] for key in keys], dtype=float)
    if not np.isfinite(matrix).all():
        raise ResultError("non-finite block accuracy")
    return keys, matrix


def _same_subjects(a: dict, b: dict, context: str) -> None:
    if set(a) != set(b):
        raise ResultError(
            f"subject mismatch {context}: missing={sorted(set(a) - set(b))}, "
            f"extra={sorted(set(b) - set(a))}"
        )


def preflight(root: Path, table: str, datasets: list[str], subject_map: dict | None = None) -> dict:
    """Load every selected logical run before reporting any statistics.

    Logical Table 2 references are retained for every selected dataset. The
    shared loader resolves perstat to the canonical Table 1/SNTE result and
    checks both copies when an alias exists. Missing seeds or splits cause an error.
    """
    tables = ("1", "2", "3") if table == "all" else (table,)
    runs = [run for selected in tables for run in logical_table_runs(selected, datasets)]
    if not runs:
        raise ResultError("no runs selected")
    results, errors = {}, []
    subjects_by_block: dict[tuple, dict] = defaultdict(dict)
    fixed_subjects: dict[tuple, dict] = defaultdict(dict)
    for run in runs:
        context = f"{run.table}/{run.label}/{run.dataset}_seed{run.seed}.json"
        try:
            result = load_run(root, run, subject_map=subject_map, require_subjects=True)
            subjects = subject_values(result)
            results[run] = result
            subjects_by_block[(run.table, *block_key(run))][row_label(run)] = subjects
            if run.table == "table3":
                fixed_subjects[(run.dataset, row_label(run))][run.seed] = subjects
        except (ResultError, OSError) as exc:
            errors.append(f"{context}: {exc}")
    for key, rows in subjects_by_block.items():
        reference = next(iter(rows))
        for label, subjects in rows.items():
            try:
                _same_subjects(rows[reference], subjects, f"{key}, {reference} vs {label}")
            except ResultError as exc:
                errors.append(str(exc))
    for key, seeds in fixed_subjects.items():
        reference = next(iter(seeds))
        for seed, subjects in seeds.items():
            try:
                _same_subjects(seeds[reference], subjects, f"fixed {key}, seeds {reference} vs {seed}")
            except ResultError as exc:
                errors.append(str(exc))
    try:
        validate_subject_spellings(results.items())
    except ResultError as exc:
        errors.append(str(exc))
    if errors:
        raise ResultError("preflight failed; selected paper runs must be complete:\n  " + "\n  ".join(errors))
    return results


def macro(result: dict) -> float:
    return 100.0 * result["test"]["subject_macro_accuracy"]


def block_means(results: dict, table: str, label: str, datasets: list[str]) -> dict[BlockKey, float]:
    return {block_key(run): macro(result) for run, result in results.items()
            if run.table == table and row_label(run) == label and run.dataset in datasets}


def paired_subject_diffs(a: dict, b: dict) -> dict[tuple[str, str], list[float]]:
    """Pair by explicit block and exact subject set, never by intersection."""
    by_subject = defaultdict(list)
    for key in _aligned_keys({"a": a, "b": b}):
        _same_subjects(a[key], b[key], str(key))
        for subject in sorted(a[key]):
            by_subject[(key[0], subject)].append(a[key][subject] - b[key][subject])
    return dict(by_subject)


def subject_diffs(results: dict, table: str, label_a: str, label_b: str,
                  datasets: list[str]) -> dict:
    columns = []
    for label in (label_a, label_b):
        columns.append({block_key(run): subject_values(result)
                        for run, result in results.items()
                        if run.table == table and row_label(run) == label and run.dataset in datasets})
    return paired_subject_diffs(*columns)


def _values(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite one-dimensional array")
    return values


# Percentage-point differences can inherit cancellation error from [0, 100] accuracies.
ROUNDOFF_PP = 8 * np.finfo(float).eps * 100


def _spread(values: np.ndarray) -> float:
    # Treat roundoff-only variation as a point mass, not evidence of real variance.
    return 0.0 if np.ptp(values) <= ROUNDOFF_PP else float(values.std(ddof=1))


def _t_test(values: np.ndarray, null: float = 0.0, greater: bool = False) -> tuple:
    if len(values) < 2:
        return None, None
    delta = float(values.mean()) - null
    spread = _spread(values)
    if spread == 0:
        # Explicit point-mass convention: no evidence against an equal null.
        if abs(delta) <= ROUNDOFF_PP:
            return 0.0, 1.0
        return float(np.copysign(np.inf, delta)), 0.0 if not greater or delta > 0 else 1.0
    statistic = delta / (spread / np.sqrt(len(values)))
    p = stats.t.sf(statistic, len(values) - 1) if greater else 2 * stats.t.sf(abs(statistic), len(values) - 1)
    return float(statistic), float(p)


def paired_stats(values) -> dict:
    """Two-sided paired t (on differences) and Wilcoxon signed-rank tests."""
    values = _values(values)
    n = len(values)
    spread = _spread(values) if n >= 2 else None
    statistic, p_t = _t_test(values)
    if n == 0:
        w_stat, p_w = None, None
    elif np.all(values == 0):
        w_stat, p_w = 0.0, 1.0
    else:
        wilcoxon = stats.wilcoxon(values, alternative="two-sided", method="auto")
        w_stat, p_w = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    notes = []
    if n < 2:
        notes.append("paired t and standardized effect undefined with n<2")
    elif n == 2:
        notes.append("n=2: tests computable, very low power")
    if spread == 0:
        notes.append("zero variance: paired d_z undefined; t p uses a point-mass convention")
    return {"n": n, "delta": float(values.mean()) if n else None, "sd": spread,
            "d": float(values.mean() / spread) if spread is not None and spread > 0 else None,
            "t": statistic, "p_t": p_t, "w": w_stat, "p_w": p_w, "notes": notes}


def nemenyi_cd(models: int, blocks: int, alpha: float = 0.05) -> float:
    if alpha != 0.05:
        raise ValueError("Nemenyi critical values are available only for alpha=0.05")
    if models not in Q05 or blocks < 1:
        raise ValueError("Nemenyi requires 2-8 models and at least one block")
    return Q05[models] * float(np.sqrt(models * (models + 1) / (6.0 * blocks)))


def friedman_stats(columns: dict[str, dict[BlockKey, float]]) -> dict:
    keys, matrix = align_blocks(columns)
    if matrix.shape[1] < 3 or len(keys) < 2:
        raise ValueError("Friedman requires at least three models and two blocks")
    ranks = np.asarray([stats.rankdata(-row, method="average") for row in matrix])
    if np.all(matrix == matrix[:, :1]):
        statistic, p = 0.0, 1.0
    else:
        statistic, p = stats.friedmanchisquare(*matrix.T)
    return {"n": len(keys), "statistic": float(statistic), "p": float(p),
            "mean_ranks": dict(zip(columns, ranks.mean(axis=0))),
            "cd": nemenyi_cd(matrix.shape[1], len(keys))}


def margin_stats(columns: dict[str, dict[BlockKey, float]], reference: str) -> dict:
    keys, matrix = align_blocks(columns)
    labels = list(columns)
    index = labels.index(reference)
    peers = [i for i in range(len(labels)) if i != index]
    if not peers:
        raise ValueError("margin requires at least one competitor")
    best = max(peers, key=lambda i: float(matrix[:, i].mean()))
    margins = matrix[:, index] - matrix[:, peers].max(axis=1)
    return {"keys": keys, "margins": margins, "mean_margin": float(margins.mean()),
            "sd_margin": _spread(margins) if len(keys) >= 2 else None,
            "best_baseline": labels[best],
            "mean_best_baseline_delta": float((matrix[:, index] - matrix[:, best]).mean()),
            "p_gt0": _t_test(margins, 0.0, greater=True)[1],
            "p_gt2": _t_test(margins, 2.0, greater=True)[1]}


def block_outcomes(columns: dict[str, dict[BlockKey, float]], reference: str) -> dict:
    keys, matrix = align_blocks(columns)
    labels = list(columns)
    index = labels.index(reference)
    ranks = {key: dict(zip(labels, stats.rankdata(-row, method="average")))
             for key, row in zip(keys, matrix)}
    pairs = {}
    for i, label in enumerate(labels):
        if label == reference:
            continue
        deltas = matrix[:, index] - matrix[:, i]
        pairs[label] = {"wins": [key for key, d in zip(keys, deltas) if d > 0],
                        "ties": [key for key, d in zip(keys, deltas) if d == 0],
                        "losses": [key for key, d in zip(keys, deltas) if d < 0]}
    return {"ranks": ranks, "pairs": pairs}


def _number(value, fmt: str = ".4g") -> str:
    return "undefined" if value is None else format(value, fmt)


def report_paired(diffs: dict, label: str, convention: str) -> dict:
    if convention == "nominal":
        values = [value for repeats in diffs.values() for value in repeats]
        tag = "nominal subject-repartition; repeats related"
    elif convention == "collapsed":
        values = [float(np.mean(repeats)) for repeats in diffs.values()]
        tag = "subject-collapsed; independence not established"
    else:
        raise ValueError(f"unknown paired convention {convention!r}")
    result = paired_stats(values)
    print(f"  {label} [{tag}] n={result['n']} delta={_number(result['delta'], '+.3f')} pp "
          f"paired-d_z={_number(result['d'], '+.3f')} "
          f"paired-t(two-sided)-p={_number(result['p_t'])} "
          f"Wilcoxon(two-sided,auto)-p={_number(result['p_w'])}")
    for note in result["notes"]:
        print(f"      {note}")
    return result


def report_friedman(columns: dict, title: str) -> None:
    result = friedman_stats(columns)
    print(f"  {title}: blocks={result['n']} chi2={result['statistic']:.3f} "
          f"p={result['p']:.4g}; Nemenyi CD(alpha=0.05)={result['cd']:.3f}")
    print("      mean ranks (1=best): " + ", ".join(
        f"{label}={rank:.3f}" for label, rank in result["mean_ranks"].items()))


def report_margins(columns: dict, reference: str) -> None:
    result = margin_stats(columns, reference)
    print(f"  {reference} minus mean-best baseline ({result['best_baseline']}): "
          f"{result['mean_best_baseline_delta']:+.3f} pp")
    print(f"  {reference} minus per-block best competitor: "
          f"{result['mean_margin']:+.3f} pp; SD={_number(result['sd_margin'])}")
    print(f"      one-sided one-sample t on paired block margins: H0 mean<=0 pp, H1 mean>0 pp, "
          f"p={_number(result['p_gt0'])}; H0 mean<=2 pp, H1 mean>2 pp, p={_number(result['p_gt2'])}")
    if len(result["keys"]) == 2:
        print("      n=2: tests computable, very low power")
    if result["sd_margin"] == 0:
        print("      zero-variance margins: t p uses a point-mass convention")
    print("      margins by block: " + ", ".join(
        f"{key[0]}/split{key[1]}={margin:+.3f}" for key, margin in zip(result["keys"], result["margins"])))


def report_blocks(columns: dict, reference: str) -> None:
    result = block_outcomes(columns, reference)
    for key, ranks in result["ranks"].items():
        print(f"  {key[0]}/split{key[1]} ranks: " + ", ".join(
            f"{label}={rank:g}" for label, rank in ranks.items()))
    for label, outcome in result["pairs"].items():
        wins = ",".join(str(key[1]) for key in outcome["wins"]) or "none"
        print(f"  {reference} vs {label}: wins={len(outcome['wins'])}/{len(result['ranks'])} "
              f"(split seeds {wins}), ties={len(outcome['ties'])}, losses={len(outcome['losses'])}")


def _columns(results: dict, table: str, labels, datasets: list[str]) -> dict:
    return {label: block_means(results, table, label, datasets) for label in labels}


def analyze_table1(results: dict, datasets: list[str], subject_collapsed: bool = False) -> None:
    print(f"\nTable 1: {len(SPLIT_SEEDS)} random subject repartitions, training seed {SPLIT_TRAIN_SEED}")
    print(f"Nominal two-sided Wilcoxon: {len(datasets) * (len(MODEL_NAMES) - 1)} comparisons; "
          "subject-repartition observations can repeat participants and are related.")
    for dataset in datasets:
        print(f"\n{dataset}")
        columns = _columns(results, "table1", MODEL_NAMES, [dataset])
        report_friedman(columns, "Friedman (models)")
        report_blocks(columns, "snte")
        report_margins(columns, "snte")
        for label in MODEL_NAMES:
            if label == "snte":
                continue
            diffs = subject_diffs(results, "table1", "snte", label, [dataset])
            report_paired(diffs, f"snte - {label}", "nominal")
            if subject_collapsed:
                report_paired(diffs, f"snte - {label}", "collapsed")
    print(f"\npooled selected datasets ({len(datasets) * len(SPLIT_SEEDS)} dataset-repartition blocks)")
    pooled = _columns(results, "table1", MODEL_NAMES, datasets)
    report_friedman(pooled, "Friedman (pooled)")
    report_margins(pooled, "snte")


def analyze_table2(results: dict, datasets: list[str]) -> None:
    print(f"\nTable 2: match heads, {len(SPLIT_SEEDS)} random subject repartitions, "
          f"training seed {SPLIT_TRAIN_SEED}; perstat reuses canonical Table 1/SNTE")
    for dataset in datasets:
        print(f"\n{dataset}")
        columns = _columns(results, "table2", HEADS, [dataset])
        report_friedman(columns, "Friedman (heads)")
        report_blocks(columns, "perstat")
        for label in HEADS:
            if label != "perstat":
                diffs = subject_diffs(results, "table2", "perstat", label, [dataset])
                report_paired(diffs, f"perstat - {label}", "collapsed")
    print("\npooled distinct (dataset, subject) averages across repartitions: archive convention; "
          "n is measured from actual IDs, not hardcoded")
    for label in HEADS:
        if label != "perstat":
            diffs = subject_diffs(results, "table2", "perstat", label, datasets)
            report_paired(diffs, f"perstat - {label}", "collapsed")


def analyze_table3(results: dict, datasets: list[str]) -> None:
    print(f"\nTable 3: fixed split, training seeds {','.join(map(str, FIXED_SEEDS))}; variant - full")
    print("Paired subject means average all three seeds; training repeats do not create new subjects.")
    for dataset in datasets:
        print(f"\n{dataset}")
        full = block_means(results, "table3", "full", [dataset])
        for variant in VARIANTS:
            if variant == "full":
                continue
            columns = {variant: block_means(results, "table3", variant, [dataset]), "full": full}
            keys, matrix = align_blocks(columns)
            delta = matrix[:, 0] - matrix[:, 1]
            print(f"  {variant} - full macro across {len(keys)} seeds: mean delta={delta.mean():+.3f} pp; "
                  + ", ".join(f"seed{key[1]}={value:+.3f}" for key, value in zip(keys, delta)))
            diffs = subject_diffs(results, "table3", variant, "full", [dataset])
            report_paired(diffs, f"{variant} - full (subject means across seeds)", "collapsed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="explicit result root; no raw data needed")
    parser.add_argument("--table", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated dataset names")
    parser.add_argument("--subject-map", type=Path, help="explicit legacy identity/config mapping JSON")
    parser.add_argument("--subject-collapsed", action="store_true", help="also report Table 1 subject-collapsed tests")
    args = parser.parse_args(argv)
    datasets = [item.strip() for item in args.datasets.split(",")]
    if not all(datasets) or len(set(datasets)) != len(datasets) or any(d not in DATASETS for d in datasets):
        parser.error(f"--datasets must select distinct names from {','.join(DATASETS)}")
    try:
        if not args.results.is_dir():
            raise ResultError(f"no results directory at {args.results}")
        subject_map = load_subject_map(args.subject_map) if args.subject_map is not None else None
        results = preflight(args.results, args.table, datasets, subject_map)
    except (ResultError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"results: {args.results}; complete selected-run preflight passed")
    print("accuracy in percent; differences in percentage points; paired tests two-sided unless labeled")
    if args.table in ("1", "all"):
        analyze_table1(results, datasets, args.subject_collapsed)
    if args.table in ("2", "all"):
        analyze_table2(results, datasets)
    if args.table in ("3", "all"):
        analyze_table3(results, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
