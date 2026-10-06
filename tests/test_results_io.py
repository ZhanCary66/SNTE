"""Synthetic result-contract tests; no raw data, checkpoint or training access."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from config import get_model_config
from dataset import FIXED_CANDIDATE_SEEDS, PROTOCOL_VERSION
from model import SNTEConfig
from results_io import (
    ResultError, SCHEMA_VERSION, SPLIT_COUNTS, effective_architecture,
    expected_architecture, load_run, load_subject_map, preflight_runs,
    run_path, run_paths, subject_values, validate_result,
)
from run_paper import Run, build_runs, canonical_run, run_aliases


def result_for(run):
    """Full declared populations: 85/25/12 IDs, exact split counts and metadata."""
    counts = SPLIT_COUNTS[run.dataset]
    units = [str(index) for index in range(1, sum(counts) + 1)]
    order = (np.arange(len(units)) if run.split_seed is None else
             np.random.default_rng(run.split_seed).permutation(len(units)))
    train, val, _ = counts
    names = {split: [] for split in ("train", "val", "test")}
    for rank, index in enumerate(order):
        split = "train" if rank < train else "val" if rank < train + val else "test"
        names[split].append(units[index])
    names = {split: sorted(values) for split, values in names.items()}
    segments = {split: len(values) * 20 for split, values in names.items()}
    result = {
        **get_model_config(run.model).to_dict(), "schema_version": SCHEMA_VERSION,
        "dataset": run.dataset, "seed": run.seed, "split_seed": run.split_seed,
        "protocol": PROTOCOL_VERSION, "integration": False, "status": "complete",
        "architecture": expected_architecture(run), "split_unit_names": names,
        "candidate_seeds": dict(FIXED_CANDIDATE_SEEDS),
        "split_files": {split: len(values) * 2 for split, values in names.items()},
        "split_segments": segments, "parameters": 1234, "best_epoch": 3, "epochs_ran": 4,
        "elapsed_seconds": 1.5, "checkpoint": "/irrelevant/weights.pt",
        "sample_rate_hz": 64, "window_seconds": 5, "candidates": 5,
        "speech_features": ["wav2vec_l14_pca64_64Hz", "mel10_64Hz"],
    }
    for split, values in names.items():
        accuracies = {str(index): 0.2 + (index % 5) / 20 for index in range(len(values))}
        result["validation" if split == "val" else split] = {
            "loss": 1.2, "segment_accuracy": 0.3,
            "subject_macro_accuracy": sum(accuracies.values()) / len(accuracies),
            "per_subject_accuracy": accuracies, "segments": segments[split],
        }
    return result


def write_run(root, run, result=None):
    path = run_path(root, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result if result is not None else result_for(run)), encoding="utf-8")
    return path


class ResultsContractTests(unittest.TestCase):
    def setUp(self):
        self.run = build_runs("1", ["PKUEEG"])[0]
        self.result = result_for(self.run)

    def rejects(self, field, value):
        result = copy.deepcopy(self.result)
        result[field] = value
        with self.assertRaises(ResultError, msg=field):
            validate_result(result, self.run)

    def test_complete_populations_and_all_architecture_variants(self):
        for run in build_runs("all", ["SparKULee", "PKUEEG", "SEM4Lang"]):
            with self.subTest(run=run):
                value = validate_result(result_for(run), run)
                self.assertEqual(sum(map(len, value["split_unit_names"].values())),
                                 sum(SPLIT_COUNTS[run.dataset]))
                self.assertEqual(len(subject_values(value)), SPLIT_COUNTS[run.dataset][2])

    def test_effective_architecture_is_executed_baseline_not_snte_flags(self):
        config = SNTEConfig(head="concat", standardize=False,
                            neural_dilations=(1, 1, 1), speech_encoder="linear")
        architecture = effective_architecture("snte", config)
        self.assertEqual(architecture["neural_dilations"], [1, 1, 1])
        self.assertNotIn("embed_dim", architecture)
        self.assertEqual(effective_architecture("cca", config), {
            "head": "timecos", "standardize": False, "tied_encoder": False,
            "neural_encoder": "cca", "speech_encoder": "dilated",
            "neural_dilations": None, "speech_dilations": [1, 3, 9],
            "scales": [1], "shifts": [0], "stats": [],
        })
        self.assertTrue(effective_architecture("vlaai", config)["standardize"])

    def test_run_identity_status_protocol_and_all_hyperparameters(self):
        for field, value in {"name": "cca", "model": "cca", "dataset": "SEM4Lang",
                             "seed": 0, "split_seed": 102, "status": "running",
                             "integration": True, "protocol": "old", "schema_version": 2}.items():
            self.rejects(field, value)
        for field, value in get_model_config(self.run.model).to_dict().items():
            self.rejects(field, value + 1 if type(value) in (int, float) else "wrong")
        self.rejects("seed", True)
        self.rejects("integration", 0)
        self.rejects("architecture", {})
        architecture = copy.deepcopy(self.result["architecture"])
        architecture["standardize"] = 1
        self.rejects("architecture", architecture)

    def test_fixed_split_seed_none_must_be_present_and_literal(self):
        run = build_runs("3", ["PKUEEG"])[0]
        result = result_for(run)
        validate_result(result, run)
        for change in ("absent", 0, False):
            changed = copy.deepcopy(result)
            if change == "absent":
                changed.pop("split_seed")
            else:
                changed["split_seed"] = change
            with self.assertRaises(ResultError):
                validate_result(changed, run)

    def test_finite_bounded_metrics_reject_boolean_and_wrong_macro(self):
        for metric in ("loss", "segment_accuracy", "subject_macro_accuracy"):
            for value in (float("nan"), float("inf"), -0.01, True, "0.3"):
                result = copy.deepcopy(self.result)
                result["test"][metric] = value
                with self.assertRaises(ResultError, msg=f"{metric}={value}"):
                    validate_result(result, self.run)
        for metric in ("segment_accuracy", "subject_macro_accuracy"):
            result = copy.deepcopy(self.result)
            result["test"][metric] = 1.01
            with self.assertRaises(ResultError):
                validate_result(result, self.run)
        result = copy.deepcopy(self.result)
        result["test"]["subject_macro_accuracy"] += 2e-8
        with self.assertRaises(ResultError):
            validate_result(result, self.run)
        result["test"]["subject_macro_accuracy"] -= 1.5e-8
        validate_result(result, self.run)
        for value in (True, float("nan"), 1.1):
            result = copy.deepcopy(self.result)
            result["test"]["per_subject_accuracy"]["0"] = value
            with self.assertRaises(ResultError):
                validate_result(result, self.run)

    def test_positions_must_be_exact_and_segments_consistent(self):
        for positions in ({"0": 0.3}, {str(i + 1): 0.3 for i in range(5)},
                          {str(i): 0.3 for i in range(6)}, {i: 0.3 for i in range(5)}):
            result = copy.deepcopy(self.result)
            result["test"]["per_subject_accuracy"] = positions
            with self.assertRaises(ResultError):
                validate_result(result, self.run)
        for field, value in (("segments", True), ("segments", 99), ("segments", 1)):
            result = copy.deepcopy(self.result)
            result["test"][field] = value
            with self.assertRaises(ResultError):
                validate_result(result, self.run)
        self.rejects("split_segments", {"train": 1, "val": 100, "test": 100})
        self.rejects("split_files", {"train": 30, "val": 10})
        self.rejects("epochs_ran", 2)

    def test_mappings_disjoint_correct_membership_and_lexicographic_order(self):
        names = copy.deepcopy(self.result["split_unit_names"])
        self.assertNotEqual(names["train"], sorted(names["train"], key=int))
        result = copy.deepcopy(self.result)
        result["split_unit_names"]["train"] = sorted(names["train"], key=int)
        with self.assertRaises(ResultError):
            validate_result(result, self.run)
        names["test"][0], names["val"][0] = names["val"][0], names["test"][0]
        names = {split: sorted(values) for split, values in names.items()}
        self.rejects("split_unit_names", names)
        names = copy.deepcopy(self.result["split_unit_names"])
        names["test"][0] = names["val"][0]
        names["test"].sort()
        self.rejects("split_unit_names", names)
        for names in (None, {}, {"test": ["a", "b"]}):
            self.rejects("split_unit_names", names)
        with self.assertRaises(ResultError):
            subject_values({"test": self.result["test"]})
        expected = {name: 100 * self.result["test"]["per_subject_accuracy"][str(index)]
                    for index, name in enumerate(self.result["split_unit_names"]["test"])}
        self.assertEqual(subject_values(self.result), expected)

    def test_subject_universe_rejects_out_of_range_replacements(self):
        for dataset in ("SparKULee", "PKUEEG", "SEM4Lang"):
            for table in ("1", "3"):
                run = build_runs(table, [dataset])[0]
                result = result_for(run)
                highest = str(sum(SPLIT_COUNTS[dataset]))
                for names in result["split_unit_names"].values():
                    if highest in names:
                        names[names.index(highest)] = str(int(highest) + 1)
                        names.sort()
                with self.subTest(dataset=dataset, table=table), self.assertRaisesRegex(ResultError, "universe"):
                    validate_result(result, run)

    def test_present_execution_settings_cannot_contradict_formal_contract(self):
        changes = ({"device": "cpu"}, {"optimizer": "SGD"},
                   {"training": {"optimizer": "SGD"}},
                   {"training": {"selection_metric": "test subject-macro accuracy"}},
                   {"environment": {"device": "cpu", "precision": "float32"}},
                   {"environment": {"precision": "float32"}},
                   {"gpu_speech_bank": False}, {"cudnn_deterministic": True})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ResultError):
                validate_result({**self.result, **change}, self.run)
        metadata = {"device": "cuda:0", "environment": {"device": "cuda:0", "precision": "bfloat16 autocast"},
                    "training": {"optimizer": "AdamW", "selection_metric": "validation subject-macro accuracy"}}
        validate_result({**self.result, **metadata}, self.run)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, self.run, {**self.result, **metadata})
            alias = run_aliases(self.run)[0]
            write_run(root, alias, {**self.result, "training": {"optimizer": "AdamW"}})
            with self.assertRaisesRegex(ResultError, "conflicting"):
                load_run(root, self.run)

    def test_legacy_overlay_fills_unknown_but_rejects_known_conflicts(self):
        result = copy.deepcopy(self.result)
        result.pop("schema_version")
        architecture = copy.deepcopy(result["architecture"])
        result["architecture"]["head"] = "concat"
        with self.assertRaisesRegex(ResultError, "conflicts"):
            validate_result(result, self.run, {"architecture": architecture})
        result["architecture"] = architecture
        different = copy.deepcopy(result["split_unit_names"])
        different["train"] = sorted(name.zfill(2) for name in different["train"])
        with self.assertRaisesRegex(ResultError, "conflicts"):
            validate_result(result, self.run, {"split_unit_names": different})
        baseline = next(run for run in build_runs("1", ["PKUEEG"]) if run.model == "cca")
        legacy = result_for(baseline)
        legacy.pop("schema_version")
        legacy["architecture"] = effective_architecture("snte", SNTEConfig())
        correction = {"architecture": expected_architecture(baseline)}
        with self.assertRaisesRegex(ResultError, "conflicts"):
            validate_result(legacy, baseline, correction)
        correction["provenance"] = "Synthetic schema-0 baseline metadata defect correction"
        checked = validate_result(legacy, baseline, correction)
        self.assertEqual(checked["architecture"], expected_architecture(baseline))
        self.assertEqual(legacy["architecture"], effective_architecture("snte", SNTEConfig()))
        self.assertEqual(checked["legacy_metadata_correction"], correction["provenance"])

    def test_preflight_rejects_cross_run_padding_aliases(self):
        runs = build_runs("2", ["PKUEEG"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in runs:
                result = result_for(run)
                if run.split_seed == 102:
                    for split, names in result["split_unit_names"].items():
                        old = dict(zip(names, result["validation" if split == "val" else split]["per_subject_accuracy"].values()))
                        padded = sorted(name.zfill(2) for name in names)
                        result["split_unit_names"][split] = padded
                        result["validation" if split == "val" else split]["per_subject_accuracy"] = {
                            str(index): old[str(int(name))] for index, name in enumerate(padded)}
                write_run(root, run, result)
            with self.assertRaisesRegex(ResultError, "inconsistent subject ID spelling"):
                preflight_runs(root, runs)

    def test_candidate_seeds_exact_not_bool_or_missing(self):
        self.rejects("candidate_seeds", {})
        candidates = dict(FIXED_CANDIDATE_SEEDS)
        candidates["test"] = 20260804
        self.rejects("candidate_seeds", candidates)
        result = copy.deepcopy(self.result)
        result.pop("candidate_seeds")
        with self.assertRaises(ResultError):
            validate_result(result, self.run)

    def test_legacy_explicit_overlay_no_mutation_or_inferred_ids(self):
        result = copy.deepcopy(self.result)
        result.pop("schema_version")
        result.pop("candidate_seeds")
        overlay = {"split_unit_names": result.pop("split_unit_names"),
                   "architecture": result.pop("architecture"), "provenance": "fixture"}
        original = copy.deepcopy(result)
        for require in (True, False):
            with self.assertRaises(ResultError):
                validate_result(result, self.run, require_subjects=require)
        checked = validate_result(result, self.run, overlay)
        self.assertEqual(subject_values(result, subject_map=overlay), subject_values(checked))
        self.assertEqual(result, original)
        result["architecture"] = overlay["architecture"]
        validate_result(result, self.run, require_subjects=False)
        with self.assertRaises(ResultError):
            validate_result(result, self.run)
        with self.assertRaises(ResultError):
            validate_result(self.result, self.run, overlay)

    def test_alias_only_canonical_only_and_conflict(self):
        alias = run_aliases(self.run)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, alias, self.result)
            self.assertEqual(load_run(root, self.run), self.result)
            alternate = copy.deepcopy(self.result)
            alternate["elapsed_seconds"] = 999
            alternate["checkpoint"] = "/another/checkpoint.pt"
            write_run(root, self.run, alternate)
            load_run(root, alias)
            alternate["test"]["segment_accuracy"] = 0.4
            write_run(root, self.run, alternate)
            with self.assertRaisesRegex(ResultError, "conflicting"):
                load_run(root, self.run)
            alternate["test"]["segment_accuracy"] = 0.3
            write_run(root, self.run, alternate)
            invalid_alias = copy.deepcopy(self.result)
            invalid_alias["seed"] = 0
            write_run(root, alias, invalid_alias)
            with self.assertRaises(ResultError):
                load_run(root, self.run)
            run_path(root, alias).unlink()
            self.assertEqual(load_run(root, self.run), alternate)

    def test_legacy_map_loading_and_missing_invalid_distinction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_run(root, self.run)
            loaded, missing = preflight_runs(root, [self.run, canonical_run(self.run)])
            self.assertEqual(loaded, {})
            self.assertEqual(missing, [self.run])
            legacy = copy.deepcopy(self.result)
            legacy.pop("schema_version")
            entry = {"split_unit_names": legacy.pop("split_unit_names"),
                     "architecture": legacy.pop("architecture")}
            path = write_run(root, self.run, legacy)
            map_path = root / "mapping.json"
            map_path.write_text(json.dumps({path.relative_to(root).as_posix(): entry}))
            mapping = load_subject_map(map_path)
            self.assertEqual(subject_values(load_run(root, self.run, mapping)), subject_values(self.result))
            for bad in ({"../escape.json": entry}, {"/absolute.json": entry}, {"run.json": {}},
                        {"run.json": {"guessed_ids": []}}):
                map_path.write_text(json.dumps(bad))
                with self.assertRaises(ResultError):
                    load_subject_map(map_path)
            path.write_text("not JSON")
            with self.assertRaises(ResultError):
                preflight_runs(root, [self.run])

    def test_run_paths_preserve_historical_layouts(self):
        paths = run_paths(Path("/results"), run_aliases(self.run)[0])
        self.assertEqual(paths[0], Path("/results/table1/snte_split101/PKUEEG_seed2.json"))
        self.assertEqual(paths[1], Path("/results/table2/perstat_split101/PKUEEG_seed2.json"))


if __name__ == "__main__":
    unittest.main()
