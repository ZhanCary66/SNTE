"""Synthetic analysis checks only; no paper performance claims or real data."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import analyze_results as analysis
from results_io import ResultError
from run_paper import FIXED_SEEDS, HEADS, SPLIT_SEEDS, VARIANTS, build_runs
from test_results_io import result_for, write_run


class AlignmentTests(unittest.TestCase):
    def test_same_count_different_blocks_rejected(self):
        a = {("synthetic", 101): 50, ("synthetic", 102): 60}
        b = {("synthetic", 101): 45, ("synthetic", 103): 55}
        with self.assertRaisesRegex(ResultError, "block mismatch"):
            analysis.align_blocks({"a": a, "b": b})
        with self.assertRaisesRegex(ResultError, "block mismatch"):
            analysis.paired_subject_diffs(
                {key: {"01": value} for key, value in a.items()},
                {key: {"01": value} for key, value in b.items()},
            )

    def test_aligns_keys_not_insertion_order(self):
        keys, matrix = analysis.align_blocks({
            "a": {("synthetic", 102): 60, ("synthetic", 101): 50},
            "b": {("synthetic", 101): 45, ("synthetic", 102): 55},
        })
        self.assertEqual(keys, [("synthetic", 101), ("synthetic", 102)])
        np.testing.assert_array_equal(matrix, [[50, 45], [60, 55]])

    def test_same_count_different_subjects_rejected(self):
        key = ("synthetic", 101)
        with self.assertRaisesRegex(ResultError, "subject mismatch"):
            analysis.paired_subject_diffs(
                {key: {"01": 50, "02": 60}}, {key: {"01": 40, "03": 50}})

    def test_nominal_and_collapsed_preserve_dataset_namespace(self):
        a = {("a", 101): {"01": 60, "02": 50},
             ("a", 102): {"01": 70, "03": 55}, ("b", 101): {"01": 65}}
        b = {("a", 101): {"01": 50, "02": 45},
             ("a", 102): {"01": 50, "03": 50}, ("b", 101): {"01": 45}}
        diffs = analysis.paired_subject_diffs(a, b)
        self.assertEqual(diffs[("a", "01")], [10, 20])
        self.assertEqual(diffs[("b", "01")], [20])
        with contextlib.redirect_stdout(io.StringIO()):
            nominal = analysis.report_paired(diffs, "invented", "nominal")
            collapsed = analysis.report_paired(diffs, "invented", "collapsed")
        self.assertEqual(nominal["n"], 5)
        self.assertEqual(collapsed["n"], 4)
        self.assertEqual(nominal["delta"], 12)
        self.assertEqual(collapsed["delta"], 11.25)


class StatisticsTests(unittest.TestCase):
    def test_t_wilcoxon_and_effect_match_direct_calculation(self):
        values = np.asarray([1., 2., 4., -1., 3.])
        result = analysis.paired_stats(values)
        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["delta"], 1.8)
        self.assertAlmostEqual(result["d"], values.mean() / values.std(ddof=1))
        self.assertAlmostEqual(result["p_t"], stats.ttest_1samp(values, 0).pvalue)
        self.assertAlmostEqual(result["p_w"], stats.wilcoxon(values, method="auto").pvalue)

    def test_all_zero_and_constant_nonzero(self):
        zeros = analysis.paired_stats([0., 0., 0.])
        self.assertEqual(zeros["delta"], 0)
        self.assertEqual(zeros["p_t"], 1)
        self.assertEqual(zeros["p_w"], 1)
        self.assertIsNone(zeros["d"])
        constant = analysis.paired_stats([2.] * 5)
        self.assertEqual(constant["delta"], 2)
        self.assertEqual(constant["p_t"], 0)
        self.assertEqual(constant["p_w"], 0.0625)
        self.assertIsNone(constant["d"])
        decimal = analysis.paired_stats([0.1] * 5)
        self.assertEqual(decimal["sd"], 0)
        self.assertIsNone(decimal["d"])

    def test_small_n_computable_and_labeled(self):
        two = analysis.paired_stats([1., 3.])
        self.assertAlmostEqual(two["p_t"], stats.ttest_1samp([1, 3], 0).pvalue)
        self.assertEqual(two["p_w"], 0.5)
        self.assertIn("very low power", " ".join(two["notes"]))
        one = analysis.paired_stats([2.])
        self.assertIsNone(one["p_t"])
        self.assertIsNone(one["d"])
        empty = analysis.paired_stats([])
        self.assertIsNone(empty["delta"])
        with self.assertRaises(ValueError):
            analysis.paired_stats([1, np.nan])

    def test_friedman_ranks_and_all_ties(self):
        columns = {"a": {("x", 101): 60, ("x", 102): 65},
                   "b": {("x", 101): 50, ("x", 102): 65},
                   "c": {("x", 101): 40, ("x", 102): 30}}
        result = analysis.friedman_stats(columns)
        self.assertEqual(result["mean_ranks"], {"a": 1.25, "b": 1.75, "c": 3.0})
        expected = stats.friedmanchisquare([60, 65], [50, 65], [40, 30])
        self.assertAlmostEqual(result["statistic"], expected.statistic)
        self.assertAlmostEqual(result["p"], expected.pvalue)
        ties = analysis.friedman_stats({label: {("x", 101): 50, ("x", 102): 60}
                                       for label in ("a", "b", "c")})
        self.assertEqual(ties["statistic"], 0)
        self.assertEqual(ties["p"], 1)
        self.assertEqual(ties["mean_ranks"], {"a": 2, "b": 2, "c": 2})

    def test_nemenyi_only_supported_alpha(self):
        self.assertAlmostEqual(analysis.nemenyi_cd(6, 15), 2.850 * np.sqrt(42 / 90))
        with self.assertRaisesRegex(ValueError, "alpha=0.05"):
            analysis.nemenyi_cd(6, 15, alpha=0.1)

    def test_mean_best_baseline_differs_from_per_block_margin(self):
        keys = [("x", seed) for seed in (101, 102, 103)]
        columns = {"snte": dict(zip(keys, [80, 80, 80])),
                   "a": dict(zip(keys, [79, 60, 79])),
                   "b": dict(zip(keys, [60, 79, 60]))}
        result = analysis.margin_stats(columns, "snte")
        self.assertEqual(result["best_baseline"], "a")
        self.assertAlmostEqual(result["mean_best_baseline_delta"], 22 / 3)
        self.assertEqual(result["mean_margin"], 1)
        np.testing.assert_array_equal(result["margins"], [1, 1, 1])
        self.assertEqual(result["p_gt0"], 0)
        self.assertEqual(result["p_gt2"], 1)
        zeros = analysis.margin_stats({"snte": dict(zip(keys, [50] * 3)),
                                       "a": dict(zip(keys, [50] * 3))}, "snte")
        self.assertEqual(zeros["p_gt0"], 1)
        self.assertEqual(zeros["p_gt2"], 1)

    def test_exact_two_pp_float_cancellation_is_not_significant(self):
        keys = [("x", seed) for seed in SPLIT_SEEDS]
        margin = 100 * (60 / 100) - 100 * (58 / 100)
        self.assertGreater(margin, 2)
        result = analysis.margin_stats({"snte": dict.fromkeys(keys, 60.),
                                        "a": dict.fromkeys(keys, 100 * (58 / 100))}, "snte")
        self.assertEqual(result["p_gt2"], 1)
        mixed = analysis.margin_stats({"snte": dict(zip(keys, [60.] * 3 + [61.] * 2)),
                                       "a": dict(zip(keys, [100 * (58 / 100)] * 3 + [100 * (59 / 100)] * 2))}, "snte")
        self.assertGreater(mixed["mean_margin"], 2)
        self.assertEqual(mixed["sd_margin"], 0)
        self.assertEqual(mixed["p_gt2"], 1)
        near = np.asarray([2 + 1e-10, 2 + 2e-10, 2 + 3e-10])
        self.assertAlmostEqual(analysis._t_test(near, 2, greater=True)[1],
                               stats.ttest_1samp(near, 2, alternative="greater").pvalue)
        self.assertEqual(analysis._t_test(np.repeat(2 + 1e-10, 5), 2, greater=True)[1], 0)

    def test_variable_margin_one_sided_tests(self):
        keys = [("x", seed) for seed in (101, 102, 103, 104, 105)]
        margins = np.asarray([1., 2., 3., 4., 7.])
        columns = {"snte": dict(zip(keys, 50 + margins)), "a": dict(zip(keys, [50] * 5))}
        result = analysis.margin_stats(columns, "snte")
        self.assertAlmostEqual(result["p_gt0"], stats.ttest_1samp(margins, 0, alternative="greater").pvalue)
        self.assertAlmostEqual(result["p_gt2"], stats.ttest_1samp(margins, 2, alternative="greater").pvalue)

    def test_wins_ties_losses_and_specific_split_seeds(self):
        keys = [("x", seed) for seed in SPLIT_SEEDS]
        columns = {"perstat": dict(zip(keys, [60] * 5)),
                   "concat": dict(zip(keys, [55, 60, 65, 50, 70])),
                   "timecos": dict(zip(keys, [59] * 5))}
        result = analysis.block_outcomes(columns, "perstat")
        self.assertEqual(result["pairs"]["concat"],
                         {"wins": [keys[0], keys[3]], "ties": [keys[1]], "losses": [keys[2], keys[4]]})
        self.assertEqual(result["pairs"]["timecos"]["wins"], keys)
        self.assertEqual(result["ranks"][keys[1]]["perstat"], 1.5)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analysis.report_blocks(columns, "perstat")
        self.assertIn("wins=5/5 (split seeds 101,102,103,104,105)", output.getvalue())


class PreflightTests(unittest.TestCase):
    @staticmethod
    def fake_result(run):
        return {"test": {"subject_macro_accuracy": .5}, "identities": {"01": 50., "02": 50.}}

    def test_missing_runs_all_reported_before_statistics(self):
        def missing(root, run, **kwargs):
            raise ResultError(f"missing split {run.split_seed}")
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(analysis, "load_run", side_effect=missing):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = analysis.main(["--results", temp, "--table", "2", "--datasets", "SEM4Lang"])
        self.assertEqual(status, 1)
        self.assertEqual(out.getvalue(), "")
        for head in HEADS:
            for seed in SPLIT_SEEDS:
                self.assertIn(f"{head}_split{seed}", err.getvalue())

    def test_preflight_checks_exact_subject_sets(self):
        def differing(root, run, **kwargs):
            result = self.fake_result(run)
            if run.label.startswith("concat"):
                result["identities"] = {"01": 50., "03": 50.}
            return result
        with mock.patch.object(analysis, "load_run", side_effect=differing), \
                mock.patch.object(analysis, "subject_values", side_effect=lambda result, **kw: result["identities"]):
            with self.assertRaisesRegex(ResultError, "subject mismatch"):
                analysis.preflight(Path("unused"), "2", ["SEM4Lang"])

    def test_fixed_split_identity_must_agree_across_seeds(self):
        def differing(root, run, **kwargs):
            result = self.fake_result(run)
            if run.seed == FIXED_SEEDS[-1]:
                result["identities"] = {"01": 50., "03": 50.}
            return result
        with mock.patch.object(analysis, "load_run", side_effect=differing), \
                mock.patch.object(analysis, "subject_values", side_effect=lambda result, **kw: result["identities"]):
            with self.assertRaisesRegex(ResultError, "fixed .* seeds"):
                analysis.preflight(Path("unused"), "3", ["SEM4Lang"])

    def test_preflight_retains_all_logical_head_rows(self):
        with mock.patch.object(analysis, "load_run", side_effect=lambda root, run, **kw: self.fake_result(run)) as loader, \
                mock.patch.object(analysis, "subject_values", side_effect=lambda result, **kw: result["identities"]):
            results = analysis.preflight(Path("unused"), "2", ["SEM4Lang"])
        self.assertEqual(len(results), 15)
        self.assertEqual(loader.call_count, 15)
        self.assertEqual(sum(run.label.startswith("perstat") for run in results), 5)
        columns = analysis._columns(results, "table2", HEADS, ["SEM4Lang"])
        self.assertTrue(all(len(column) == 5 for column in columns.values()))

    def test_results_root_is_required_and_dataset_typo_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing:
                analysis.main([])
            with self.assertRaises(SystemExit) as typo:
                analysis.main(["--results", "unused", "--datasets", "SEM4Lang,not-a-dataset"])
        self.assertEqual(missing.exception.code, 2)
        self.assertEqual(typo.exception.code, 2)


class FormalCliTests(unittest.TestCase):
    @staticmethod
    def write_results(root, table="all", datasets=None):
        datasets = datasets or ["SparKULee", "PKUEEG", "SEM4Lang"]
        for run in build_runs(table, datasets):
            result = result_for(run)
            # Invented offsets keep SNTE above the two alternate scoring heads.
            offset = -.04 if run.label.startswith("concat") else -.02 if run.label.startswith("timecos") else 0
            for split in ("validation", "test"):
                metrics = result[split]
                metrics["per_subject_accuracy"] = {
                    key: value + offset for key, value in metrics["per_subject_accuracy"].items()}
                metrics["subject_macro_accuracy"] = float(np.mean(list(metrics["per_subject_accuracy"].values())))
            write_run(root, run, result)

    def test_actual_schema1_all_tables_cli_canonical_only_any_cwd(self):
        with tempfile.TemporaryDirectory(prefix="analysis result space ") as temp:
            root = Path(temp) / "result root"
            self.write_results(root)
            self.assertEqual(len(list(root.rglob("*.json"))), 174)
            self.assertFalse(list((root / "table2").glob("perstat*")))
            command = [sys.executable, str(PROJECT_ROOT / "analyze_results.py"),
                       "--results", str(root), "--table", "all", "--subject-collapsed"]
            completed = subprocess.run(command, cwd=temp, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                       text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = completed.stdout
            self.assertIn("complete selected-run preflight passed", output)
            self.assertIn("Nominal two-sided Wilcoxon: 15 comparisons", output)
            self.assertIn("15 dataset-repartition blocks", output)
            self.assertIn("perstat reuses canonical Table 1/SNTE", output)
            self.assertIn("wins=5/5 (split seeds 101,102,103,104,105)", output)
            self.assertIn("n=2: tests computable, very low power", output)
            self.assertIn("Table 3: fixed split, training seeds 0,1,2", output)
            self.assertNotIn("nan", output.lower())
            self.assertNotIn("inf", output.lower())
            loaded = analysis.preflight(root, "2", ["SparKULee", "PKUEEG", "SEM4Lang"])
            collapsed = analysis.subject_diffs(loaded, "table2", "perstat", "concat",
                                               ["SparKULee", "PKUEEG", "SEM4Lang"])
            expected_n = sum(len(set().union(*(set(result["split_unit_names"]["test"])
                              for run, result in loaded.items()
                              if run.dataset == dataset and run.label.startswith("perstat"))))
                             for dataset in ("SparKULee", "PKUEEG", "SEM4Lang"))
            self.assertEqual(len(collapsed), expected_n)
            self.assertIn(f"n={expected_n} delta=+4.000 pp", output)
            values = [float(np.mean(repeats)) for repeats in collapsed.values()]
            self.assertAlmostEqual(float(np.mean(values)), 4)

    def test_cross_repartition_id_spelling_fails_before_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for run in build_runs("2", ["PKUEEG"]):
                result = result_for(run)
                if run.split_seed == 102:
                    for split, names in result["split_unit_names"].items():
                        metrics = result["validation" if split == "val" else split]
                        old = {name: metrics["per_subject_accuracy"][str(index)] for index, name in enumerate(names)}
                        padded = sorted(name.zfill(2) for name in names)
                        result["split_unit_names"][split] = padded
                        metrics["per_subject_accuracy"] = {str(index): old[str(int(name))]
                                                            for index, name in enumerate(padded)}
                write_run(root, run, result)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = analysis.main(["--results", temp, "--table", "2", "--datasets", "PKUEEG"])
            self.assertEqual(status, 1)
            self.assertEqual(out.getvalue(), "")
            self.assertIn("inconsistent subject ID spelling", err.getvalue())

    def test_exact_two_pp_mixed_cancellation_cli_uses_equal_null_convention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for run in build_runs("1", ["PKUEEG"]):
                result = result_for(run)
                baseline = 58 if run.split_seed <= 103 else 59
                accuracy = (baseline + (2 if run.model == "snte" else 0)) / 100
                metrics = result["test"]
                metrics["per_subject_accuracy"] = dict.fromkeys(metrics["per_subject_accuracy"], accuracy)
                metrics["subject_macro_accuracy"] = float(np.mean(list(metrics["per_subject_accuracy"].values())))
                metrics["segment_accuracy"] = accuracy
                write_run(root, run, result)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = analysis.main(["--results", temp, "--table", "1", "--datasets", "PKUEEG"])
            self.assertEqual(status, 0, err.getvalue())
            self.assertIn("H0 mean<=2 pp, H1 mean>2 pp, p=1", out.getvalue())
            self.assertIn("zero-variance margins", out.getvalue())

    def test_actual_missing_fixed_seed_zero_fails_without_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_results(root, "3", ["SEM4Lang"])
            missing = root / "table3" / "full" / "SEM4Lang_seed0.json"
            missing.unlink()
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = analysis.main(["--results", temp, "--table", "3", "--datasets", "SEM4Lang"])
        self.assertEqual(status, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("SEM4Lang_seed0.json", err.getvalue())

    def test_actual_schema0_explicit_map_is_applied_once_at_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            self.write_results(root, "2", ["SEM4Lang"])
            mapping = {}
            for path in root.rglob("*.json"):
                result = json.loads(path.read_text(encoding="utf-8"))
                mapping[path.relative_to(root).as_posix()] = {
                    "split_unit_names": result.pop("split_unit_names"),
                    "architecture": result.pop("architecture")}
                result.pop("schema_version")
                path.write_text(json.dumps(result), encoding="utf-8")
            map_path = Path(temp) / "identity map.json"
            map_path.write_text(json.dumps(mapping), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = analysis.main(["--results", str(root), "--table", "2", "--datasets", "SEM4Lang",
                                        "--subject-map", str(map_path)])
            self.assertEqual(status, 0, err.getvalue())
            self.assertIn("delta=+4.000 pp", out.getvalue())
            self.assertIn("wins=5/5", out.getvalue())

    def test_fixed_variant_macro_and_subject_mean_delta_across_three_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for run in build_runs("3", ["SEM4Lang"]):
                result = result_for(run)
                if run.label != "full":
                    per = result["test"]["per_subject_accuracy"]
                    per["0"] += (run.seed + 1) / 100
                    per["1"] -= (run.seed + 1) / 200
                    result["test"]["subject_macro_accuracy"] = float(np.mean(list(per.values())))
                write_run(root, run, result)
            loaded = analysis.preflight(root, "3", ["SEM4Lang"])
            variant = next(label for label in VARIANTS if label != "full")
            diffs = analysis.subject_diffs(loaded, "table3", variant, "full", ["SEM4Lang"])
            values = sorted(float(np.mean(repeats)) for repeats in diffs.values())
            np.testing.assert_allclose(values, [-1, 2], atol=1e-12)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                analysis.analyze_table3(loaded, ["SEM4Lang"])
            self.assertIn("mean delta=+0.500 pp", out.getvalue())
            self.assertIn("n=2 delta=+0.500 pp", out.getvalue())
            self.assertIn("seed0=+0.250, seed1=+0.500, seed2=+0.750", out.getvalue())


if __name__ == "__main__":
    unittest.main()
