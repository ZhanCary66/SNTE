"""Synthetic CLI, output-safety and one-epoch CPU training checks.

No real data or archival checkpoints are opened. Temporary NPY/JSON/PT files
belong to each test and are removed with its temporary directory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import main as training
from baselines.common import LatentCosineMatcher, Spatial64
from config import MODEL_NAMES, get_model_config
from dataset import FIXED_CANDIDATE_SEEDS, PROTOCOL_VERSION
from model import create_model
from results_io import SCHEMA_VERSION, ResultError, effective_architecture, load_run
from run_paper import Run

torch.set_num_threads(1)


def arguments(*flags: str):
    with mock.patch.object(sys, "argv", ["main.py", "--dataset", "SparKULee", *flags]):
        return training.parse_args()


def invoke(*flags: str):
    with mock.patch.object(sys, "argv", ["main.py", "--dataset", "SparKULee", *flags]):
        with contextlib.redirect_stdout(io.StringIO()):
            return training.main()


@contextlib.contextmanager
def failed_invocation(test, pattern: str):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with test.assertRaises(SystemExit) as raised:
            yield
    test.assertEqual(raised.exception.code, 1)
    test.assertRegex(stderr.getvalue(), pattern)
    test.assertNotIn("Traceback", stderr.getvalue())


def synthetic_data(root: Path) -> None:
    """One two-window recording in each fixed split, one shared stimulus."""
    neural = root / "neural_lp30_64hz" / "SparKULee"
    mel = root / "stimuli" / "SparKULee" / "mel10_64Hz"
    wav = root / "stimuli" / "SparKULee" / "wav2vec_l14_pca64_64Hz"
    for directory in (neural, mel, wav):
        directory.mkdir(parents=True)
    rng = np.random.default_rng(20261006)
    for subject in ("1", "55", "69"):
        np.save(neural / f"sub-{subject}_s0_sentence0_LP-30_64Hz.npy",
                rng.normal(size=(640, 64)).astype(np.float32))
    np.save(mel / "s0_sentence0.npy", rng.normal(size=(640, 10)).astype(np.float32))
    np.save(wav / "s0_sentence0.npy", rng.normal(size=(640, 64)).astype(np.float32))


class TrainingCLI(unittest.TestCase):
    def test_invalid_numeric_options_are_parser_errors(self):
        cases = (("--workers", "-1"), ("--split-seed", "-1"),
                 ("--split-seed", "x"), ("--seed", "3"))
        for flags in cases:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    arguments(*flags)
                self.assertEqual(raised.exception.code, 2)

    def test_dilation_choices_are_supported_positive_nonempty_triples(self):
        for value in ("", "1,,9", "0,3,9", "-1,3,9", "1,2,3", "1,3", "a,3,9"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    training._dilations(value)
        self.assertEqual(training._dilations("1,3,9"), (1, 3, 9))
        self.assertEqual(training._dilations("1,1,1"), (1, 1, 1))

    def test_baselines_reject_ineffective_snte_overrides(self):
        flags = (("--head", "concat"), ("--head", "timecos"),
                 ("--no-standardize",), ("--tied-encoder",),
                 ("--neural-encoder", "linear"), ("--speech-encoder", "linear"),
                 ("--neural-dilations", "1,1,1"), ("--speech-dilations", "1,1,1"))
        for name in MODEL_NAMES[1:]:
            for override in flags:
                with self.subTest(name=name, flags=override):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            arguments("--model", name, *override)
                    self.assertEqual(raised.exception.code, 2)

    def test_invalid_device_and_output_suffix_are_parser_errors(self):
        cases = (("--device", "mps"), ("--device", "cuda:-1"),
                 ("--device", "cpu:0"), ("--device", "cuda:banana"),
                 ("--output", "result.pt"), ("--output", "result"),
                 ("--output", "result.JSON"))
        for flags in cases:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    arguments(*flags)
                self.assertEqual(raised.exception.code, 2)

    def test_incompatible_tied_encoders_are_parser_errors(self):
        for flags in (("--tied-encoder", "--neural-encoder", "linear"),
                      ("--tied-encoder", "--speech-encoder", "linear"),
                      ("--tied-encoder", "--neural-dilations", "1,1,1")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    arguments(*flags)
                self.assertEqual(raised.exception.code, 2)

    def test_check_data_is_selected_and_does_not_need_cuda(self):
        report = {"SparKULee": {"neural": 662}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(training, "validate_shared_contract", return_value=report) as validate:
                with mock.patch.object(training, "resolve_device", side_effect=AssertionError("device used")):
                    invoke("--check-data", "--device", "cuda:99", "--data-root", str(root))
            validate.assert_called_once_with(root, datasets=("SparKULee",))

    def test_invalid_device_fails_before_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with mock.patch.object(training, "validate_shared_contract") as validate:
                    with failed_invocation(self, "CUDA.*not available|not available.*CUDA"):
                        invoke("--integration", "--device", "cuda:0", "--output", str(output))
                validate.assert_not_called()

    def test_device_validation_checks_availability_index_and_bf16(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(training.resolve_device("auto", True), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA"):
                training.resolve_device("cpu", False)
            with self.assertRaisesRegex(RuntimeError, "CUDA.*not available|not available.*CUDA"):
                training.resolve_device("cuda", True)
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(torch.cuda, "device_count", return_value=1):
                with self.assertRaisesRegex(ValueError, "index.*unavailable|index.*device"):
                    training.resolve_device("cuda:1", True)
                with mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()):
                    with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
                        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
                            training.resolve_device("cuda:0", False)
                    with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
                        self.assertEqual(training.resolve_device("cuda:0", False), torch.device("cuda:0"))

    def test_large_cuda_indices_are_rejected_before_torch_device_conversion(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
                torch.cuda, "device_count", return_value=1), mock.patch.object(
                torch.cuda, "device") as context, mock.patch.object(
                training, "validate_shared_contract") as validate:
            for index in (256, 257, 1024, 10**20):
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, str(index)):
                    training.resolve_device(f"cuda:{index}", True)
            with failed_invocation(self, "256.*unavailable"):
                invoke("--integration", "--device", "cuda:256")
        context.assert_not_called()
        validate.assert_not_called()

    def test_bf16_legacy_api_requires_native_cuda_before_probe(self):
        calls = []

        def legacy_checker():
            calls.append("probe")
            return True

        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
                torch.cuda, "device_count", return_value=1), mock.patch.object(
                torch.cuda, "device", return_value=contextlib.nullcontext()), mock.patch.object(
                torch.cuda, "is_bf16_supported", new=legacy_checker), mock.patch.object(
                torch.version, "hip", None):
            with mock.patch.object(torch.version, "cuda", "11.8"), mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=(8, 0)):
                self.assertEqual(training.resolve_device("cuda:0", False), torch.device("cuda:0"))
                self.assertEqual(calls, ["probe"])
                calls.clear()
                with mock.patch.object(training, "validate_shared_contract", side_effect=RuntimeError("preflight reached")):
                    with tempfile.TemporaryDirectory() as directory:
                        with failed_invocation(self, "preflight reached"):
                            invoke("--device", "cuda:0", "--output", str(Path(directory) / "run.json"))
                self.assertEqual(calls, ["probe"])
            for runtime, capability in ((None, (8, 0)), ("10.2", (8, 0)), ("12.1", (7, 5))):
                calls.clear()
                with self.subTest(runtime=runtime, capability=capability), mock.patch.object(
                        torch.version, "cuda", runtime), mock.patch.object(
                        torch.cuda, "get_device_capability", return_value=capability):
                    with self.assertRaisesRegex(RuntimeError, "bfloat16"):
                        training.resolve_device("cuda:0", False)
                self.assertEqual(calls, [])

    def test_bf16_legacy_rocm_preserves_checker_semantics(self):
        def legacy_checker():
            return True

        with mock.patch.object(torch.cuda, "is_bf16_supported", new=legacy_checker), mock.patch.object(
                torch.version, "hip", "6.0"), mock.patch.object(
                torch.cuda, "get_device_capability", side_effect=AssertionError("NVIDIA guard on ROCm")):
            self.assertTrue(training._native_bf16_supported(0))

    def test_bf16_checker_internal_type_error_is_not_hidden(self):
        with mock.patch.object(torch.cuda, "is_bf16_supported", side_effect=TypeError("internal failure")):
            with self.assertRaisesRegex(TypeError, "internal failure"):
                training._native_bf16_supported(0)

    def test_default_baseline_effective_metadata_matches_executed_modules(self):
        for name in MODEL_NAMES[1:]:
            with self.subTest(name=name):
                args = arguments("--model", name)
                config = training.build_config(args)
                final = get_model_config(name)
                self.assertEqual((config.embed_dim, config.dropout), (final.embed_dim, final.dropout))
                rng_before = torch.random.get_rng_state().clone()
                configuration = training.build_configuration(args, config, torch.device("cpu"))
                metadata = configuration["architecture"]
                self.assertEqual(metadata, effective_architecture(name, config))
                for key, expected in final.to_dict().items():
                    self.assertEqual(configuration[key], expected)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
                model = create_model(name, 64, 74, config)
                self.assertIsInstance(model, LatentCosineMatcher)
                self.assertIsInstance(model.temperature, nn.Parameter)
                self.assertEqual(metadata["head"], "timecos")
                self.assertEqual(metadata["neural_encoder"], name)
                self.assertEqual(metadata["speech_encoder"], "dilated")
                self.assertEqual(metadata["stats"], [])
                self.assertFalse(metadata["tied_encoder"])
                standardizers = [module.use_standardization for module in model.neural_body.modules()
                                 if isinstance(module, Spatial64)]
                self.assertEqual(metadata["standardize"], any(standardizers))
                self.assertEqual(metadata["standardize"], name != "cca")
                dilations = [module.dilation[0] for module in model.speech_encoder.modules()
                             if isinstance(module, nn.Conv1d) and module.kernel_size == (3,)]
                self.assertEqual(metadata["speech_dilations"], dilations)
                self.assertIsNone(metadata["neural_dilations"])

    def test_configuration_records_actual_limits_without_consuming_rng(self):
        args = arguments("--integration", "--workers", "6")
        config = training.build_config(args)
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        configuration = training.build_configuration(args, config, torch.device("cpu"))
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual((configuration["epochs"], configuration["batch_size"],
                          configuration["evaluation_batch_size"], configuration["workers"]), (1, 2, 2, 0))
        self.assertEqual(configuration["patience"], get_model_config("snte").patience)
        self.assertFalse(configuration["gpu_speech_bank"])
        self.assertFalse(configuration["environment"]["autocast_enabled"])
        self.assertIsNone(configuration["environment"]["autocast_dtype"])
        self.assertEqual(configuration["training"]["optimizer"], "AdamW")

    def test_cli_runtime_error_has_clear_nonzero_exit(self):
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1"}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"),
                                     "--dataset", "SparKULee", "--device", "cpu",
                                     "--output", str(Path(directory) / "run.json")],
                                    cwd=directory, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("formal training requires a CUDA", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class OutputSafety(unittest.TestCase):
    def test_default_output_distinguishes_modes_and_scientific_identity(self):
        cases = ((), ("--seed", "1"), ("--split-seed", "101"), ("--head", "concat"),
                 ("--head", "timecos"), ("--no-standardize",), ("--tied-encoder",),
                 ("--neural-encoder", "linear"), ("--speech-encoder", "linear"),
                 ("--neural-dilations", "1,1,1", "--speech-dilations", "1,1,1"),
                 ("--integration",), ("--model", "cca"))
        paths = []
        for flags in cases:
            args = arguments(*flags)
            path = training.default_output(args, training.build_config(args))
            self.assertEqual(path.suffix, ".json")
            self.assertIn("integration" if args.integration else "formal", path.parts)
            paths.append(path)
        self.assertEqual(len(set(paths)), len(paths))
        args = arguments()
        args.dataset = "PKUEEG"
        self.assertNotEqual(paths[0], training.default_output(args, training.build_config(args)))

    def test_output_paths_are_absolute_distinct_json_and_pt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments("--output", str(Path(directory) / "space in name.json"))
            output, checkpoint = training.output_paths(args, training.build_config(args))
            self.assertTrue(output.is_absolute())
            self.assertTrue(checkpoint.is_absolute())
            self.assertEqual(output.suffix, ".json")
            self.assertEqual(checkpoint.suffix, ".pt")
            self.assertNotEqual(output, checkpoint)
            self.assertEqual(output.with_suffix(".pt"), checkpoint)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_pair_precheck_after_lock_catches_earlier_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            checkpoint = output.with_suffix(".pt")
            original_open = os.open

            def earlier_writer_completed(path, flags, mode=0o777):
                descriptor = original_open(path, flags, mode)
                checkpoint.write_bytes(b"earlier synthetic writer")
                return descriptor

            with mock.patch.object(training.os, "open", side_effect=earlier_writer_completed):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    with training.output_lock(output, checkpoint):
                        self.fail("lock yielded despite an existing checkpoint")
            self.assertEqual(checkpoint.read_bytes(), b"earlier synthetic writer")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json.lock").exists())

    def test_default_output_collision_refused_before_data(self):
        args = arguments("--integration", "--device", "cpu")
        with tempfile.TemporaryDirectory() as directory:
            # Redirect just the default root into this test, not the process cwd.
            output = Path(directory) / training.default_output(args, training.build_config(args))
            checkpoint = output.with_suffix(".pt")
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic preexisting default checkpoint")
            with mock.patch.object(training, "default_output", return_value=output):
                with mock.patch.object(training, "validate_shared_contract") as validate:
                    with failed_invocation(self, "exist|overwrite"):
                        invoke("--integration", "--device", "cpu")
            validate.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"synthetic preexisting default checkpoint")
            self.assertEqual(set(checkpoint.parent.iterdir()), {checkpoint})

    def test_lock_cleanup_preserves_replacement_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            lock = output.with_suffix(".json.lock")
            with training.output_lock(output, output.with_suffix(".pt")):
                lock.unlink()
                lock.write_bytes(b"replacement owner")
            self.assertEqual(lock.read_bytes(), b"replacement owner")

    def test_preexisting_json_or_checkpoint_is_rejected_before_data(self):
        for suffix in (".json", ".pt"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run.json"
                existing = output.with_suffix(suffix)
                existing.write_bytes(b"previous output must stay unchanged")
                with mock.patch.object(training, "validate_shared_contract") as validate:
                    with failed_invocation(self, "exist|overwrite"):
                        invoke("--integration", "--device", "cpu", "--output", str(output))
                self.assertEqual(existing.read_bytes(), b"previous output must stay unchanged")
                validate.assert_not_called()
                self.assertFalse(output.with_suffix(".json.lock").exists())

    def test_preexisting_lock_is_refused_and_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            lock = output.with_suffix(".json.lock")
            lock.write_bytes(b"another writer")
            with mock.patch.object(training, "validate_shared_contract") as validate:
                with failed_invocation(self, "lock|writer|exist"):
                    invoke("--integration", "--device", "cpu", "--output", str(output))
            validate.assert_not_called()
            self.assertEqual(lock.read_bytes(), b"another writer")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".pt").exists())

    def test_symlink_outputs_do_not_overwrite_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.pt"
            target.write_bytes(b"synthetic old weights")
            output = root / "run.json"
            output.with_suffix(".pt").symlink_to(target)
            with mock.patch.object(training, "validate_shared_contract") as validate:
                with failed_invocation(self, "symlink|exist|overwrite"):
                    invoke("--integration", "--device", "cpu", "--output", str(output))
            validate.assert_not_called()
            self.assertEqual(target.read_bytes(), b"synthetic old weights")
            self.assertFalse(output.exists())

    def test_failure_cleans_only_owned_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            other = Path(directory) / "other.json.lock"
            other.write_bytes(b"not ours")
            with mock.patch.object(training, "validate_shared_contract", side_effect=RuntimeError("synthetic data failure")):
                with failed_invocation(self, "synthetic data failure"):
                    invoke("--integration", "--device", "cpu", "--output", str(output))
            self.assertFalse(output.with_suffix(".json.lock").exists())
            self.assertEqual(other.read_bytes(), b"not ours")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".pt").exists())

    def test_concurrent_writer_is_refused_for_entire_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            argv = ["main.py", "--dataset", "SparKULee", "--integration", "--device", "cpu",
                    "--output", str(output)]
            entered = threading.Event()
            release = threading.Event()
            stderr = io.StringIO()

            def hold_data(*args, **kwargs):
                entered.set()
                if not release.wait(10):
                    raise AssertionError("test did not release writer")
                raise RuntimeError("release synthetic writer")

            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
                with mock.patch.object(training, "validate_shared_contract", side_effect=hold_data) as validate:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        writer = executor.submit(training.main)
                        try:
                            self.assertTrue(entered.wait(10), "first writer did not reach data validation")
                            self.assertTrue(output.with_suffix(".json.lock").exists())
                            with failed_invocation(self, "lock|writer|exist"):
                                training.main()
                            self.assertEqual(validate.call_count, 1)
                        finally:
                            release.set()
                        with self.assertRaises(SystemExit) as raised:
                            writer.result(timeout=10)
                        self.assertEqual(raised.exception.code, 1)
            self.assertIn("release synthetic writer", stderr.getvalue())
            self.assertFalse(output.with_suffix(".json.lock").exists())
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".pt").exists())

    def test_atomic_json_rejects_nonfinite_and_removes_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            output.write_text('{"previous": true}', encoding="utf-8")
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    training.atomic_json({"loss": value}, output)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"previous": True})
                self.assertEqual(set(Path(directory).iterdir()), {output})

    def test_atomic_json_replace_failure_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            with mock.patch.object(training.os, "replace", side_effect=OSError("synthetic rename failure")):
                with self.assertRaisesRegex(OSError, "synthetic rename failure"):
                    training.atomic_json({"finite": 1.0}, output)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_atomic_checkpoint_save_failure_leaves_old_weights_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.pt"
            output.write_bytes(b"synthetic old weights")

            def partial_save(payload, destination):
                if hasattr(destination, "write"):
                    destination.write(b"partial synthetic bytes")
                else:
                    Path(destination).write_bytes(b"partial synthetic bytes")
                raise OSError("synthetic save failure")

            with mock.patch.object(torch, "save", side_effect=partial_save):
                with self.assertRaisesRegex(OSError, "synthetic save failure"):
                    training.atomic_checkpoint({}, output)
            self.assertEqual(output.read_bytes(), b"synthetic old weights")
            self.assertEqual(set(Path(directory).iterdir()), {output})


class CPUIntegration(unittest.TestCase):
    def test_one_epoch_records_actual_settings_units_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            synthetic_data(data)
            run = Run("table3", "full", "snte", "SparKULee", 0, None, ())
            output = root / "results" / run.table / run.label / "SparKULee_seed0.json"
            with mock.patch.object(training, "validate_shared_contract", return_value={"SparKULee": {}}) as validate:
                invoke("--integration", "--device", "cpu", "--workers", "7",
                       "--data-root", str(data), "--output", str(output))
            validate.assert_called_once_with(data, datasets=("SparKULee",))
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], SCHEMA_VERSION)
            self.assertEqual(result["protocol"], PROTOCOL_VERSION)
            self.assertEqual(result["status"], "complete")
            self.assertIs(result["integration"], True)
            for field, expected in {"epochs": 1, "epochs_ran": 1, "best_epoch": 1,
                                    "batch_size": 2, "evaluation_batch_size": 2, "workers": 0,
                                    "patience": get_model_config("snte").patience}.items():
                self.assertEqual(result[field], expected)
            self.assertEqual(result["split_unit_names"], {"train": ["1"], "val": ["55"], "test": ["69"]})
            self.assertEqual(result["candidate_seeds"], FIXED_CANDIDATE_SEEDS)
            self.assertEqual(result["split_files"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(result["split_segments"], {"train": 2, "val": 2, "test": 2})
            for field in ("validation", "test"):
                self.assertEqual(set(result[field]["per_subject_accuracy"]), {"0"})
                self.assertEqual(result[field]["segments"], 2)
            self.assertEqual(result["architecture"], effective_architecture("snte", training.build_config(arguments())))
            for field in ("python", "torch", "numpy", "cuda", "cudnn", "device", "precision"):
                self.assertIn(field, result["environment"])
            self.assertEqual(result["environment"]["device"], "cpu")
            checkpoint = Path(result["checkpoint"])
            self.assertNotEqual(checkpoint.resolve(), output.resolve())
            self.assertEqual(checkpoint.suffix, ".pt")
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(weights["result"], result)
            self.assertEqual(weights["configuration"]["epochs"], 1)
            model = create_model("snte", 64, 74, training.build_config(arguments()))
            model.load_state_dict(weights["model_state_dict"], strict=True)
            self.assertFalse(output.with_suffix(".json.lock").exists())
            self.assertFalse(list(output.parent.glob("*.tmp*")))
            with self.assertRaises(ResultError):
                load_run(root / "results", run)
            before = checkpoint.read_bytes()
            with mock.patch.object(training, "validate_shared_contract") as validate:
                with failed_invocation(self, "exist|overwrite"):
                    invoke("--integration", "--device", "cpu", "--data-root", str(data), "--output", str(output))
            validate.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_json_failure_does_not_leave_checkpoint_or_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            synthetic_data(data)
            output = root / "results" / "run.json"
            other = root / "results" / "other.pt"
            other.parent.mkdir()
            other.write_bytes(b"other synthetic checkpoint")
            with mock.patch.object(training, "validate_shared_contract", return_value={"SparKULee": {}}):
                with mock.patch.object(training, "atomic_json", side_effect=OSError("synthetic JSON failure")):
                    with failed_invocation(self, "synthetic JSON failure"):
                        invoke("--integration", "--device", "cpu", "--data-root", str(data), "--output", str(output))
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".pt").exists())
            self.assertFalse(output.with_suffix(".json.lock").exists())
            self.assertEqual(other.read_bytes(), b"other synthetic checkpoint")


if __name__ == "__main__":
    unittest.main()
