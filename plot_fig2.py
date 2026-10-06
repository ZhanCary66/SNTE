#!/usr/bin/env python3
"""Plot Figure 2 from nine complete Table 3/full runs, without raw data.

Points are actual subjects' three-seed means, not seed-level macro accuracies.
Boxes summarize subject means; vertical point ranges show training-seed variation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dataset import DATASETS
from results_io import ResultError, load_run, load_subject_map, subject_values
from run_paper import FIXED_SEEDS, ROOT, build_runs

TEST_COUNTS = dict(zip(DATASETS, (17, 5, 2)))
DISPLAY_NAMES = dict(zip(DATASETS, ("SparrKULee", "PKUEEG", "SEM4Lang")))
BLUE, INK, MUTED, SURFACE = "#2a78d6", "#262626", "#626262", "#fcfcfb"


def prepare_plot_data(results: Path, subject_map: dict | None = None) -> dict:
    """Return dataset -> actual subject ID -> percentages in seed 0, 1, 2 order.

    All nine expected runs must pass the shared strict loader. Subject positions
    may differ across files: alignment uses identities, never dictionary position.
    """
    prepared = {}
    runs = [run for run in build_runs("3", list(DATASETS)) if run.label == "full"]
    for dataset in DATASETS:
        seeds = {}
        for run in (run for run in runs if run.dataset == dataset):
            result = load_run(Path(results), run, subject_map=subject_map)
            seeds[run.seed] = subject_values(result)
        if set(seeds) != set(FIXED_SEEDS):
            raise ResultError(f"{dataset}: expected training seeds {FIXED_SEEDS}")
        subjects = seeds[FIXED_SEEDS[0]]
        if len(subjects) != TEST_COUNTS[dataset]:
            raise ResultError(f"{dataset}: expected {TEST_COUNTS[dataset]} test subjects, "
                              f"found {len(subjects)}")
        for seed, values in seeds.items():
            if set(values) != set(subjects):
                raise ResultError(f"{dataset} seed {seed}: test subject identities differ")
        prepared[dataset] = {subject: [seeds[seed][subject] for seed in FIXED_SEEDS]
                             for subject in subjects}
    return prepared


def box_statistics(subjects: dict[str, list[float]]) -> dict:
    """Matplotlib bxp statistics: IQR and median of three-seed subject means.

    Whiskers are the actual min/max means, not Tukey fences or seed extremes.
    """
    means = np.asarray(list(subjects.values()), dtype=float).mean(axis=1)
    q1, median, q3 = np.quantile(means, (0.25, 0.5, 0.75))
    return dict(q1=float(q1), med=float(median), q3=float(q3),
                whislo=float(means.min()), whishi=float(means.max()), fliers=[])


def make_figure(prepared: dict):
    """Create a static paper figure; local jitter does not touch training RNGs."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with plt.rc_context({"font.size": 10, "font.family": "DejaVu Sans",
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED}):
        fig, ax = plt.subplots(figsize=(6.3, 4.7), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.6)
        stats = [dict(box_statistics(prepared[d]),
                      label=f"{DISPLAY_NAMES[d]}\n(n = {len(prepared[d])})")
                 for d in DATASETS]
        ax.bxp(stats, widths=0.46, showfliers=False, patch_artist=True,
               boxprops={"facecolor": matplotlib.colors.to_rgba(BLUE, 0.10),
                         "edgecolor": BLUE, "linewidth": 1.0},
               medianprops={"color": INK, "linewidth": 1.5},
               whiskerprops={"color": BLUE, "linewidth": 1.0},
               capprops={"color": BLUE, "linewidth": 1.0})
        rng = np.random.default_rng(7)
        for position, dataset in enumerate(DATASETS, 1):
            values = np.asarray(list(prepared[dataset].values()), dtype=float)
            x = position + rng.uniform(-0.16, 0.16, len(values))
            ax.vlines(x, values.min(axis=1), values.max(axis=1),
                      color=BLUE, alpha=0.55, linewidth=0.8)
            ax.scatter(x, values.mean(axis=1), s=30, color=BLUE,
                       edgecolors=SURFACE, linewidths=0.7, zorder=3)
        ax.axhline(20, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
        ax.text(0.99, 20.8, "20% chance", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color=MUTED, fontsize=9)
        ax.set(ylim=(0, 100), yticks=np.arange(0, 101, 20),
               ylabel="Test accuracy (%)", title="SNTE · Fixed-split test subjects")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set(color="#d0d0cd", linewidth=0.6)
        ax.tick_params(length=0, pad=7)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.28)
        fig.text(0.12, 0.06,
                 "Points: subject means over seeds 0–2; vertical lines: seed min–max.\n"
                 "Boxes: median and IQR of subject means; whiskers: their min–max.\n"
                 "Dashed line: five-way chance (20%).",
                 color=MUTED, fontsize=9, linespacing=1.5)
        return fig


def validate_targets(output: Path, csv_path: Path | None, overwrite: bool) -> None:
    """Check every target before reading runs or writing either export."""
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ResultError("--output must end in .png, .pdf, or .svg")
    if csv_path is not None and csv_path.suffix.lower() != ".csv":
        raise ResultError("--csv must end in .csv")
    targets = [output] + ([csv_path] if csv_path is not None else [])
    if len({path.resolve() for path in targets}) != len(targets):
        raise ResultError("figure and CSV targets must be different")
    for path in targets:
        if path.is_dir():
            raise ResultError(f"output target is a directory: {path}")
        if not overwrite and (path.exists() or path.is_symlink()):
            raise ResultError(f"output already exists: {path}; use --overwrite explicitly")
        for parent in path.parents:
            if parent.exists() and not parent.is_dir():
                raise ResultError(f"output parent is not a directory: {parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-map", type=Path,
                        help="explicit identity/architecture mapping for legacy results")
    parser.add_argument("--csv", type=Path, help="export all point and range values")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        validate_targets(args.output, args.csv, args.overwrite)
        subject_map = load_subject_map(args.subject_map) if args.subject_map else None
        prepared = prepare_plot_data(args.results, subject_map)
        fig = make_figure(prepared)
        from matplotlib import pyplot as plt
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("wb" if args.overwrite else "xb") as stream:
                fig.savefig(stream, format=args.output.suffix.lower()[1:], dpi=300)
            if args.csv is not None:
                args.csv.parent.mkdir(parents=True, exist_ok=True)
                with args.csv.open("w" if args.overwrite else "x", newline="",
                                   encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["dataset", "subject_id", *[f"seed{s}_percent" for s in FIXED_SEEDS],
                                     "mean_percent", "min_percent", "max_percent"])
                    for dataset, subjects in prepared.items():
                        for subject, values in subjects.items():
                            writer.writerow([dataset, subject, *values,
                                             float(np.mean(values)), min(values), max(values)])
        finally:
            plt.close(fig)
    except (ResultError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
