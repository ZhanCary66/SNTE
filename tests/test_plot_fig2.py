"""Synthetic Figure 2 fixtures: no paper performance claims or raw data."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_fig2 as plot
from config import get_model_config
from dataset import DATASETS, FIXED_CANDIDATE_SEEDS, PROTOCOL_VERSION
from model import SNTEConfig
from results_io import ResultError, SCHEMA_VERSION, effective_architecture, subject_values
from run_paper import build_runs

COUNTS = {"SparKULee": (54, 14, 17), "PKUEEG": (15, 5, 5), "SEM4Lang": (8, 2, 2)}


def fixture(run):
    counts = dict(zip(("train", "val", "test"), COUNTS[run.dataset]))
    names, start = {}, 1
    for split, count in counts.items():
        names[split] = sorted(str(index) for index in range(start, start + count))
        start += count
    def metrics(split):
        values = [0.4] * counts[split]
        if split == "test":
            values[0] = (0.1, 0.2, 0.3)[run.seed]
            values[1] = (0.6, 0.9, 0.9)[run.seed]
        return {"loss": 1.0, "segment_accuracy": float(np.mean(values)),
                "subject_macro_accuracy": float(np.mean(values)),
                "per_subject_accuracy": {str(i): value for i, value in enumerate(values)},
                "segments": counts[split]}
    config = get_model_config(run.model)
    return {**config.to_dict(), "schema_version": SCHEMA_VERSION, "status": "complete",
            "protocol": PROTOCOL_VERSION, "integration": False, "dataset": run.dataset,
            "seed": run.seed, "split_seed": run.split_seed,
            "architecture": effective_architecture(run.model, SNTEConfig(
                embed_dim=config.embed_dim, dropout=config.dropout)),
            "split_unit_names": names, "candidate_seeds": FIXED_CANDIDATE_SEEDS,
            "split_files": counts, "split_segments": counts, "parameters": 1,
            "epochs_ran": 1, "best_epoch": 1, "test": metrics("test"),
            "validation": metrics("val")}


class Figure2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="snte synthetic figure ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = [run for run in build_runs("3", list(DATASETS)) if run.label == "full"]
        self.paths = []
        for run in self.runs:
            path = self.root / run.table / run.label / f"{run.dataset}_seed{run.seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fixture(run)), encoding="utf-8")
            self.paths.append(path)

    def replace(self, index, mutate):
        result = json.loads(self.paths[index].read_text())
        mutate(result)
        self.paths[index].write_text(json.dumps(result))

    def cli(self, *flags):
        return subprocess.run([sys.executable, str(ROOT / "plot_fig2.py"),
                               "--results", str(self.root), *map(str, flags)],
                              cwd=self.root, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                              capture_output=True, text=True, timeout=60)

    def test_actual_identity_alignment_and_manual_means(self):
        def reversed_values(result):
            values = subject_values(result)
            return dict(reversed(list(values.items()))) if result["seed"] else values
        with mock.patch.object(plot, "subject_values", side_effect=reversed_values):
            data = plot.prepare_plot_data(self.root)
        self.assertEqual([len(data[d]) for d in DATASETS], [17, 5, 2])
        for dataset in DATASETS:
            ids = fixture(next(r for r in self.runs if r.dataset == dataset))["split_unit_names"]["test"]
            self.assertEqual(data[dataset][ids[0]], [10.0, 20.0, 30.0])
            self.assertEqual(data[dataset][ids[1]], [60.0, 90.0, 90.0])
            np.testing.assert_allclose([np.mean(data[dataset][s]) for s in ids[:2]], [20, 80])

    def test_box_statistics_use_subject_means_not_seed_extremes(self):
        stats = plot.box_statistics({"2": [0, 30, 60], "10": [30, 90, 90]})
        self.assertEqual(stats, dict(q1=40, med=50, q3=60, whislo=30, whishi=70, fliers=[]))

    def test_points_seed_ranges_and_local_jitter(self):
        data, state = plot.prepare_plot_data(self.root), np.random.get_state()
        first, second = plot.make_figure(data), plot.make_figure(data)
        from matplotlib import pyplot as plt
        self.addCleanup(plt.close, first)
        self.addCleanup(plt.close, second)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state())))
        for index, dataset in enumerate(DATASETS):
            values = np.array(list(data[dataset].values()))
            a, b = first.axes[0].collections, second.axes[0].collections
            np.testing.assert_array_equal(a[2 * index + 1].get_offsets(), b[2 * index + 1].get_offsets())
            np.testing.assert_allclose(a[2 * index + 1].get_offsets()[:, 1], values.mean(axis=1))
            np.testing.assert_allclose(np.array(a[2 * index].get_segments())[:, :, 1],
                                       np.column_stack((values.min(axis=1), values.max(axis=1))))

    def test_missing_seed_never_filled(self):
        self.paths[0].unlink()
        with self.assertRaises(FileNotFoundError):
            plot.prepare_plot_data(self.root)
        output = self.root / "missing.png"
        self.assertNotEqual(self.cli("--output", output).returncode, 0)
        self.assertFalse(output.exists())

    def test_same_count_different_actual_id_rejected(self):
        self.replace(1, lambda r: r["split_unit_names"]["test"].__setitem__(0, "069"))
        with self.assertRaisesRegex(ResultError, "identities differ"):
            plot.prepare_plot_data(self.root)

    def test_wrong_seed_and_subject_key_and_metadata_rejected(self):
        changes = [lambda r: r.update(seed=2),
                   lambda r: r["test"]["per_subject_accuracy"].pop("0"),
                   lambda r: r.update(integration=True),
                   lambda r: r["architecture"].update(head="concat"),
                   lambda r: r["split_unit_names"]["test"].pop()]
        for mutate in changes:
            with self.subTest(mutate=mutate):
                self.paths[0].write_text(json.dumps(fixture(self.runs[0])))
                self.replace(0, mutate)
                with self.assertRaises(ResultError):
                    plot.prepare_plot_data(self.root)

    def test_legacy_requires_explicit_map_and_cli_reads_it(self):
        original = fixture(self.runs[0])
        overlay = {key: original[key] for key in ("split_unit_names", "architecture")}
        self.replace(0, lambda r: [r.pop(k) for k in ("schema_version", *overlay)])
        with self.assertRaises(ResultError):
            plot.prepare_plot_data(self.root)
        mapping = {self.paths[0].relative_to(self.root).as_posix(): overlay}
        self.assertEqual(len(plot.prepare_plot_data(self.root, mapping)[DATASETS[0]]), 17)
        map_path, output = self.root / "mapping.json", self.root / "legacy.png"
        map_path.write_text(json.dumps(mapping))
        result = self.cli("--subject-map", map_path, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.exists())

    def test_deterministic_marks_and_vector_exports(self):
        from matplotlib import pyplot as plt
        data = plot.prepare_plot_data(self.root)
        fig1, fig2 = plot.make_figure(data), plot.make_figure(data)
        self.addCleanup(plt.close, fig1)
        self.addCleanup(plt.close, fig2)
        ax = fig1.axes[0]
        self.assertEqual(ax.get_ylim(), (0, 100))
        self.assertTrue(any(np.array_equal(line.get_ydata(), [20, 20]) for line in ax.lines))
        np.testing.assert_allclose(ax.collections[5].get_offsets()[:, 1], [20, 80])
        np.testing.assert_equal(ax.collections[5].get_offsets(), fig2.axes[0].collections[5].get_offsets())
        np.testing.assert_allclose(ax.collections[4].get_segments()[0][:, 1], [10, 30])
        for suffix, signature in ((".pdf", b"%PDF"), (".svg", b"<?xml")):
            with self.subTest(format=suffix):
                output = self.root / f"synthetic figure{suffix}"
                result = self.cli("--output", output)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(output.read_bytes().startswith(signature))

    def test_cli_png_csv_and_all_target_preflight(self):
        output, table = self.root / "figure.png", self.root / "points.csv"
        result = self.cli("--output", output, "--csv", table)
        self.assertEqual(result.returncode, 0, result.stderr)
        from matplotlib.image import imread
        self.assertEqual(imread(output).ndim, 3)
        with table.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertAlmostEqual(float(rows[0]["mean_percent"]), 20)
        fresh = self.root / "fresh.png"
        rejected = self.cli("--output", fresh, "--csv", table)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(fresh.exists())
        accepted = self.cli("--output", fresh, "--csv", table, "--overwrite")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue(fresh.exists())
        for target in (output, self.root / "bad.txt"):
            self.assertNotEqual(self.cli("--output", target).returncode, 0)
        directory = self.root / "directory.csv"
        directory.mkdir()
        self.assertNotEqual(self.cli("--output", self.root / "absent.png", "--csv", directory).returncode, 0)
        self.assertFalse((self.root / "absent.png").exists())


if __name__ == "__main__":
    unittest.main()
