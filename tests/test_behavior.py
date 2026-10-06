"""BEFORE-CHANGE behavior on synthetic data, NOT paper results.

Run from any directory with PYTHONDONTWRITEBYTECODE=1:
    python -m unittest discover -s /path/to/snte_final_code/tests -v

The explicit, before-change-only recording interface is:
    python /path/to/snte_final_code/tests/test_behavior.py --record-before-change
Never regenerate the reference merely to make a changed implementation pass.
No training, real data, checkpoints, or shared-data contract are involved.
Initialization bytes are Torch-base-version-bound; numerical regressions are
bound to the full Torch and NumPy versions. Invariants always run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import dataset as data_module
from config import get_model_config
from dataset import MatchMismatchDataset, _zscore
from model import SNTEConfig, create_model

torch.set_num_threads(1)
REFERENCE_PATH = Path(__file__).with_name("fixtures") / "behavior_reference.json"
MODEL_SEED = 20261006
INPUT_SEED = 20261007
BATCH, TIME, CANDIDATES, SPEECH_CHANNELS = 2, 320, 5, 74
FINAL_MODELS = ("snte", "cca", "convconcatnet", "vlaai", "eeg2vec", "brainmagic")
FINAL_DIM_DROPOUT = {
    "snte": (256, 0.5), "cca": (32, 0.0), "convconcatnet": (32, 0.4),
    "vlaai": (32, 0.4), "eeg2vec": (64, 0.2), "brainmagic": (64, 0.2),
}
MODEL_CASES = {name: (name, {}) for name in FINAL_MODELS}
MODEL_CASES.update({
    "snte-concat": ("snte", {"head": "concat"}),
    "snte-timecos": ("snte", {"head": "timecos"}),
    "snte-no-window-norm": ("snte", {"standardize": False}),
    "snte-neural-linear": ("snte", {"neural_encoder": "linear"}),
    "snte-speech-linear": ("snte", {"speech_encoder": "linear"}),
    "snte-tied": ("snte", {"tied_encoder": True}),
    "snte-no-dilation": ("snte", {
        "neural_dilations": (1, 1, 1), "speech_dilations": (1, 1, 1),
    }),
})
TOLERANCES = {
    "outputs": {"rtol": 2e-5, "atol": 2e-6},
    "gradients": {"rtol": 2e-4, "atol": 2e-8},
    "data": {"rtol": 2e-6, "atol": 2e-7},
}
# Full speech windows plus unused tails exercise truncation and bank offsets.
STIMULI = {"s0_sentence0": 6 * TIME + 17, "s0_sentence1": 3 * TIME + 11}
DATA_CASES = {
    "sparkulee-train": ("SparKULee", "train", ("01", "02"), 64, 0),
    "sparkulee-val": ("SparKULee", "val", ("55", "56"), 64, 0),
    "sparkulee-test-trimmed": ("SparKULee", "test", ("69", "70"), 64, 4),
    "pkueeg-train": ("PKUEEG", "train", ("01", "02"), 57, 0),
    "sem4lang-train": ("SEM4Lang", "train", ("01", "02"), 306, 0),
}
ZSCORE_INPUT = np.asarray([
    [-3, 7, -3e-8, 1000.125], [-1, 7, -1e-8, 1000.25],
    [0, 7, 0, 1000.5], [1, 7, 1e-8, 1000.75], [3, 7, 3e-8, 1000.875],
], dtype=np.float32)
_MODEL_TIMINGS: dict[str, float] = {}


def versions() -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torch_base": str(torch.__version__).split("+", 1)[0], "numpy": np.__version__}


def numerical_skip_reason(reference: dict) -> str | None:
    recorded, current = reference["versions"], versions()
    if any(current[key] != recorded[key] for key in ("torch", "numpy")):
        return ("version-bound synthetic numerical regression: recorded "
                f"torch={recorded['torch']}, numpy={recorded['numpy']}; current "
                f"torch={current['torch']}, numpy={current['numpy']}; invariants still run")
    return None


def explicit_config(case_name: str) -> SNTEConfig:
    name, changes = MODEL_CASES[case_name]
    final = get_model_config(name)
    # In particular, NEVER exercise the baseline factory's omitted-config path.
    return SNTEConfig(embed_dim=final.embed_dim, dropout=final.dropout, **changes)


def make_model(case_name: str, neural_channels: int = 64):
    name, _ = MODEL_CASES[case_name]
    torch.manual_seed(MODEL_SEED)  # Seed immediately before model construction.
    return create_model(name, neural_channels, SPEECH_CHANNELS,
                        config=explicit_config(case_name)).cpu().eval()


def make_inputs(neural_channels: int = 64, gradients: bool = False):
    # Call only AFTER construction. This generator cannot consume initialization RNG.
    generator = torch.Generator(device="cpu").manual_seed(INPUT_SEED)
    neural = torch.randn(BATCH, TIME, neural_channels, generator=generator)
    neural = (neural * torch.linspace(0.5, 1.5, neural_channels)
              + torch.linspace(-0.5, 0.5, neural_channels))
    candidates = torch.randn(BATCH, CANDIDATES, TIME, SPEECH_CHANNELS, generator=generator)
    return neural.requires_grad_(gradients), candidates.requires_grad_(gradients)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def gradient_summary(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None  # An unused parameter is not a non-finite gradient.
    value = tensor.detach().cpu().to(torch.float64)
    return {"sum": value.sum().item(), "l2": value.norm().item()}


@lru_cache(maxsize=None)
def capture_model(case_name: str) -> dict:
    """Public snapshot interface: ordered state layout/bytes, scores, selected grads."""
    start = time.perf_counter()
    model = make_model(case_name)
    state = model.state_dict()
    digest = hashlib.sha256()
    layout = []
    for key, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        layout.append([key, list(value.shape), str(value.dtype)])
        digest.update(value.numpy().tobytes())
    neural, candidates = make_inputs(gradients=True)
    with torch.no_grad():
        scores = model(neural, candidates)
        repeated = model(neural, candidates)
        permutation = torch.tensor([4, 2, 0, 3, 1])
        permuted = model(neural, candidates[:, permutation])
    # One evaluation-mode backward probe only; no optimizer or training occurs.
    probe_scores = model(neural, candidates)
    loss = F.cross_entropy(probe_scores, torch.tensor([1, 3]))
    loss.backward()
    parameters = dict(model.named_parameters())
    selected = []
    for prefix in ("neural_encoder.", "neural_body.", "speech_encoder.",
                   "correlation.scorer.", "correlation.heads."):
        key = next((key for key in parameters
                    if key.startswith(prefix) and key.endswith("weight")), None)
        if key is not None:
            selected.append(key)
    if explicit_config(case_name).tied_encoder:
        # This gradient accumulates BOTH modalities; input projections stay separate.
        selected.append(next(key for key in parameters
                             if key.startswith("neural_encoder.trunk.") and key.endswith("weight")))
    selected.append("temperature")
    gradients = {key: gradient_summary(parameters[key].grad) for key in selected}
    all_finite = all(torch.isfinite(value).all().item() for value in state.values())
    all_finite = all_finite and all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in parameters.values())
    all_finite = all_finite and all(torch.isfinite(value).all().item() for value in
                                  (scores, probe_scores, loss, neural.grad, candidates.grad))
    result = {
        "model": MODEL_CASES[case_name][0],
        "config": json.loads(json.dumps(asdict(explicit_config(case_name)))),
        "neural_channels": 64, "state_layout": layout,
        "initialization_sha256": digest.hexdigest(),
        "eval_outputs": scores.tolist(), "gradient_probe_outputs": probe_scores.detach().tolist(),
        "cross_entropy": loss.item(), "parameter_gradients": gradients,
        "input_gradients": {"neural": gradient_summary(neural.grad),
                            "candidates": gradient_summary(candidates.grad)},
        "invariants": {
            "all_finite": bool(all_finite), "output_shape": list(scores.shape),
            "repeatable_eval": bool(torch.equal(scores, repeated)),
            "candidate_permutation_equivariant": bool(torch.allclose(
                permuted, scores[:, permutation], **TOLERANCES["outputs"])),
            "nonzero_neural_gradient": bool(neural.grad.norm().item() > 0),
            "nonzero_speech_gradient": bool(candidates.grad.norm().item() > 0),
        },
    }
    _MODEL_TIMINGS[case_name] = time.perf_counter() - start
    return result


def synthetic_array(length: int, channels: int, offset: int) -> np.ndarray:
    """Deterministic float32 arithmetic; every value is invented, not a recording."""
    t = np.arange(length, dtype=np.int64)[:, None]
    c = np.arange(channels, dtype=np.int64)[None, :]
    value = ((t * (c % 7 + 1) + c * 13 + offset * 17) % 101).astype(np.float32)
    value = value / np.float32(25) + (t % 5).astype(np.float32) * np.float32(0.125)
    value[:, 0] = np.float32(3)  # Constant channel.
    value[:, 1] *= np.float32(1e-8)  # Below the dataset normalization floor.
    return np.ascontiguousarray(value, dtype=np.float32)


def independent_zscore(array: np.ndarray) -> np.ndarray:
    """Population standard deviation, float32 centering, and a 1e-6 floor."""
    value = np.asarray(array, dtype=np.float32)
    double = value.astype(np.float64)
    mean = (double.sum(axis=0) / len(double)).astype(np.float32)
    variance = ((double - double.mean(axis=0)) ** 2).sum(axis=0) / len(double)
    denominator = np.maximum(np.sqrt(variance).astype(np.float32), np.float32(1e-6))
    return (value - mean) / denominator


def write_synthetic_tree(root: Path) -> None:
    """Write ONLY small temporary .npy files following the existing filename contract."""
    for dataset in {case[0] for case in DATA_CASES.values()}:
        for stem_index, (stem, length) in enumerate(STIMULI.items()):
            for folder, channels, offset in (
                ("wav2vec_l14_pca64_64Hz", 64, 10 + stem_index),
                ("mel10_64Hz", 10, 20 + stem_index),
            ):
                directory = root / "stimuli" / dataset / folder
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / f"{stem}.npy", synthetic_array(length, channels, offset))
    for _, (dataset, split, subjects, channels, _) in DATA_CASES.items():
        directory = root / "neural_lp30_64hz" / dataset
        directory.mkdir(parents=True, exist_ok=True)
        entries = [(subjects[0], "s0_sentence1", 2 * TIME + 23, False),
                   (subjects[1], "s0_sentence0", 5 * TIME + 29, True),
                   (subjects[1], "s0_sentence1", 2 * TIME + 7, False)]
        for index, (subject, stem, length, transposed) in enumerate(entries):
            value = synthetic_array(length, channels, 30 + index)
            if dataset == "SEM4Lang":
                value *= np.float32(1e-11)
            if transposed:
                value = value.T
            filename = f"sub-{subject}_{stem}_LP-30_64Hz.npy"
            np.save(directory / filename, value)


def array_summary(value: torch.Tensor | np.ndarray) -> dict:
    tensor = torch.as_tensor(value)
    flat = tensor.reshape(-1)
    indices = sorted({0, 1, 63, 64, 73, 74, len(flat) // 3, len(flat) // 2, len(flat) - 1})
    double = flat.to(torch.float64)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor), "sum": double.sum().item(),
            "l2": double.norm().item(), "probe_indices": indices,
            "probe_values": flat[indices].tolist()}


@lru_cache(maxsize=1)
def capture_synthetic_data() -> dict:
    """Public snapshot interface: normalization, bank bytes/probes, plan, labels, units."""
    snapshots = {}
    with tempfile.TemporaryDirectory(prefix="snte-synthetic-behavior-") as directory:
        root = Path(directory)
        write_synthetic_tree(root)
        # Isolate split state and forbid global contract validation, without touching sources.
        with mock.patch.object(data_module, "_RANDOM_MAPPING", {}), mock.patch.object(
                data_module, "validate_shared_contract",
                side_effect=AssertionError("synthetic tests must not validate a global contract")):
            for case_name, (dataset, split, subjects, channels, trim) in DATA_CASES.items():
                ds = MatchMismatchDataset(dataset, split, root=root, max_files=2, max_segments=trim)
                model_channels = 204 if dataset == "SEM4Lang" else channels
                expected_bank = []
                for index, (_, length) in enumerate(STIMULI.items()):
                    speech = np.concatenate([
                        independent_zscore(synthetic_array(length, 64, 10 + index)),
                        independent_zscore(synthetic_array(length, 10, 20 + index)),
                    ], axis=1)
                    expected_bank.append(speech[:len(speech) // TIME * TIME].reshape(-1, TIME, 74))
                bank_matches = np.allclose(ds.speech_bank.numpy(), np.concatenate(expected_bank),
                                           **TOLERANCES["data"])
                expected_neural = []
                for index, length in enumerate((2 * TIME + 23, 5 * TIME + 29)):
                    value = synthetic_array(length, channels, 30 + index)
                    if dataset == "SEM4Lang":
                        value *= np.float32(1e-11)
                        value = value[:, [i for i in range(306) if i % 3 in (1, 2)]]
                    expected_neural.append(independent_zscore(value))
                neural_matches = len(ds.neural) == 2 and all(
                    np.allclose(value, expected, **TOLERANCES["data"])
                    for value, expected in zip(ds.neural, expected_neural))
                aligned, pool_replacement, large_pool_unique, samples_match = True, True, True, True
                for index, segment in enumerate(ds.segments):
                    window, plan, label, unit = ds[index]
                    offset, total = (6, 3) if segment.stimulus == "s0_sentence1" else (0, 6)
                    positive = offset + segment.start // TIME
                    negatives = torch.cat((plan[:int(label)], plan[int(label) + 1:]))
                    aligned = aligned and int(plan[int(label)]) == positive
                    aligned = aligned and bool(((plan >= offset) & (plan < offset + total)).all())
                    aligned = aligned and bool((negatives != positive).all())
                    if total == 3:
                        pool_replacement = pool_replacement and len(set(negatives.tolist())) < 4
                    else:
                        large_pool_unique = large_pool_unique and len(set(negatives.tolist())) == 4
                    expected = expected_neural[segment.trial][segment.start:segment.start + TIME]
                    samples_match = samples_match and tuple(window.shape) == (TIME, model_channels)
                    samples_match = samples_match and np.allclose(window.numpy(), expected,
                                                                  **TOLERANCES["data"])
                    samples_match = samples_match and int(unit) == subjects.index(segment.subject)
                snapshots[case_name] = {
                    "dataset": dataset, "split": split, "max_files": 2, "max_segments": trim,
                    "candidate_seed": ds.candidate_seed, "unit_names": ds.unit_names,
                    "unit_indices": ds.unit_indices.tolist(), "labels": ds.labels.tolist(),
                    "candidate_plan": ds.candidate_plan.tolist(),
                    "segments": [asdict(segment) for segment in ds.segments],
                    "speech_bank": array_summary(ds.speech_bank),
                    "neural_trials": [array_summary(value) for value in ds.neural],
                    "invariants": {
                        "all_finite": bool(torch.isfinite(ds.speech_bank).all()) and all(
                            np.isfinite(value).all().item() for value in ds.neural),
                        "speech_bank_wav_then_mel": bool(bank_matches),
                        "neural_orientation_and_channel_selection": bool(neural_matches),
                        "positive_aligned_same_stimulus_negatives": bool(aligned),
                        "small_pool_uses_replacement": bool(pool_replacement),
                        "large_pool_no_replacement": bool(large_pool_unique),
                        "getitem_matches_plan_window_and_subject": bool(samples_match),
                        "max_files_applied": len(ds.neural) == 2,
                        "max_segments_applied": len(ds) == (trim or 7),
                        "unit_names_correct": ds.unit_names == list(subjects),
                    },
                }
    return {"zscore": {"input": ZSCORE_INPUT.tolist(), "output": _zscore(ZSCORE_INPUT).tolist()},
            "datasets": snapshots}


def build_reference() -> dict:
    """Recreate the compact baseline; timing information is intentionally not stored."""
    return {
        "schema_version": 1, "phase": "BEFORE-CHANGE", "synthetic": True,
        "not_paper_results": True,
        "description": "Invented inputs and temporary .npy data; computation regression only.",
        "versions": versions(),
        "version_policy": {"initialization": "torch_base equality only",
                           "numerical": "full torch and numpy equality; otherwise explicit skip",
                           "invariants": "always run, independent of recorded versions"},
        "protocol": {"model_seed": MODEL_SEED, "input_seed": INPUT_SEED,
                     "input_rng": "independent CPU generator created AFTER model construction",
                     "threads": 1, "device": "cpu", "mode": "eval", "neural_channels": 64,
                     "batch": BATCH, "time": TIME, "candidates": CANDIDATES,
                     "speech_channels": SPEECH_CHANNELS, "gradient_labels": [1, 3],
                     "gradient_objective": "cross_entropy; one backward, no training",
                     "tolerances": TOLERANCES},
        "models": {name: capture_model(name) for name in MODEL_CASES},
        "data": capture_synthetic_data(),
    }


class BehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with REFERENCE_PATH.open(encoding="utf-8") as handle:
            cls.reference = json.load(handle)

    def assert_numerical_version(self):
        reason = numerical_skip_reason(self.reference)
        if reason:
            self.skipTest(reason)

    def test_reference_scope_and_size(self):
        self.assertEqual(self.reference["phase"], "BEFORE-CHANGE")
        self.assertTrue(self.reference["synthetic"])
        self.assertTrue(self.reference["not_paper_results"])
        self.assertLess(REFERENCE_PATH.stat().st_size, 100_000)
        self.assertEqual(set(self.reference["models"]), set(MODEL_CASES))

    def test_explicit_final_dimensions_and_dropout(self):
        for name, expected in FINAL_DIM_DROPOUT.items():
            with self.subTest(model=name):
                final, explicit = get_model_config(name), explicit_config(name)
                self.assertEqual((final.embed_dim, final.dropout), expected)
                self.assertEqual((explicit.embed_dim, explicit.dropout), expected)
        for name in MODEL_CASES:
            with self.subTest(case=name):
                # Ignore future metadata fields; freeze only recorded computational fields.
                actual = json.loads(json.dumps(asdict(explicit_config(name))))
                expected = self.reference["models"][name]["config"]
                self.assertEqual({key: actual[key] for key in expected}, expected)

    def test_ordered_state_dict_keys_shapes_and_dtypes(self):
        for name in MODEL_CASES:
            with self.subTest(case=name):
                self.assertEqual(capture_model(name)["state_layout"],
                                 self.reference["models"][name]["state_layout"])

    def test_version_bound_initialization_hashes(self):
        recorded = self.reference["versions"]["torch_base"]
        if versions()["torch_base"] != recorded:
            self.skipTest(f"initialization hashes require torch base {recorded}; "
                          f"current {versions()['torch_base']}; invariants still run")
        for name in MODEL_CASES:
            with self.subTest(case=name):
                self.assertEqual(capture_model(name)["initialization_sha256"],
                                 self.reference["models"][name]["initialization_sha256"])

    def test_version_bound_eval_outputs_and_gradients(self):
        self.assert_numerical_version()
        for name in MODEL_CASES:
            with self.subTest(case=name):
                actual, expected = capture_model(name), self.reference["models"][name]
                for key in ("eval_outputs", "gradient_probe_outputs", "cross_entropy"):
                    np.testing.assert_allclose(actual[key], expected[key], **TOLERANCES["outputs"])
                for group in ("parameter_gradients", "input_gradients"):
                    self.assertEqual(set(actual[group]), set(expected[group]))
                    for key, summary in expected[group].items():
                        if summary is None:
                            self.assertIsNone(actual[group][key])
                        else:
                            for statistic in ("sum", "l2"):
                                np.testing.assert_allclose(actual[group][key][statistic],
                                                           summary[statistic], **TOLERANCES["gradients"])

    def test_model_invariants_unconditionally(self):
        for name in MODEL_CASES:
            with self.subTest(case=name):
                checks = capture_model(name)["invariants"]
                self.assertEqual(checks["output_shape"], [BATCH, CANDIDATES])
                for key, value in checks.items():
                    if key != "output_shape":
                        self.assertTrue(value, f"{name}: {key}")

    def test_tied_encoder_reuses_trunk_but_not_input_projection(self):
        model = make_model("snte-tied")
        self.assertIs(model.neural_encoder.trunk, model.speech_encoder.trunk)
        self.assertIsNot(model.neural_encoder.input_projection.weight,
                         model.speech_encoder.input_projection.weight)
        neural, candidates = make_inputs(gradients=True)
        F.cross_entropy(model(neural, candidates), torch.tensor([1, 3])).backward()
        for parameter in model.neural_encoder.trunk.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_all_final_models_accept_57_and_204_channels(self):
        for name in FINAL_MODELS:
            for channels in (57, 204):
                with self.subTest(model=name, neural_channels=channels):
                    model = make_model(name, channels)
                    neural, candidates = make_inputs(channels)
                    with torch.no_grad():
                        scores = model(neural, candidates)
                    self.assertEqual(tuple(scores.shape), (BATCH, CANDIDATES))
                    self.assertEqual(scores.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(scores).all())

    def test_version_bound_synthetic_data_snapshot(self):
        self.assert_numerical_version()
        # Exact hashes bind speech-bank/neural float32 bytes, not private data.
        self.assertEqual(capture_synthetic_data(), self.reference["data"])

    def test_synthetic_data_invariants_unconditionally(self):
        normalized = _zscore(ZSCORE_INPUT)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_allclose(normalized, independent_zscore(ZSCORE_INPUT),
                                   **TOLERANCES["data"])
        np.testing.assert_array_equal(normalized[:, 1], np.zeros(5, dtype=np.float32))
        np.testing.assert_allclose(normalized[:, 2], ZSCORE_INPUT[:, 2] / np.float32(1e-6),
                                   **TOLERANCES["data"])
        for name, snapshot in capture_synthetic_data()["datasets"].items():
            with self.subTest(case=name):
                for key, value in snapshot["invariants"].items():
                    self.assertTrue(value, f"{name}: {key}")
                self.assertEqual(snapshot["speech_bank"]["shape"], [9, TIME, SPEECH_CHANNELS])
                self.assertEqual(snapshot["speech_bank"]["dtype"], "torch.float32")

    def test_numerical_version_gate_explains_skips(self):
        for key in ("torch", "numpy"):
            modified = {"versions": dict(versions())}
            modified["versions"][key] = "deliberately-different"
            reason = numerical_skip_reason(modified)
            self.assertIn("version-bound", reason)
            self.assertIn("deliberately-different", reason)
            self.assertIn("invariants still run", reason)
        self.assertIsNone(numerical_skip_reason({"versions": versions()}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--record-before-change"]:
        started = time.perf_counter()
        reference = build_reference()
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(json.dumps(reference, separators=(",", ":"), allow_nan=False) + "\n",
                                  encoding="utf-8")
        print(json.dumps({"fixture": str(REFERENCE_PATH), "bytes": REFERENCE_PATH.stat().st_size,
                          "versions": versions(), "elapsed_seconds": time.perf_counter() - started,
                          "model_elapsed_seconds": _MODEL_TIMINGS}, indent=2))
    else:
        unittest.main(verbosity=2)
