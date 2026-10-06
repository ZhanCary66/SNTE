"""Synthetic input-boundary tests; no recordings or paper results are used."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import dataset as data
from baselines.model import BASELINE_NAMES, create_baseline
from config import get_model_config
from model import SNTEConfig, create_model

torch.set_num_threads(1)


def write_contract_tree(root: Path, datasets=data.DATASETS) -> None:
    """Empty placeholders deliberately test the documented shallow preflight."""
    for dataset in datasets:
        neural = root / "neural_lp30_64hz" / dataset
        neural.mkdir(parents=True)
        population = {"SparKULee": 85, "PKUEEG": 25, "SEM4Lang": 12}[dataset]
        stimuli = data.EXPECTED_STIMULI[dataset]
        for index in range(data.EXPECTED_NEURAL[dataset]):
            subject = index % population + 1
            stem = f"stim{index // population % stimuli}"
            # SparKULee has fewer recordings than the Cartesian product: use
            # coprime subject/stimulus cycling to cover both full universes.
            if dataset == "SparKULee":
                stem = f"stim{index % stimuli}"
            (neural / f"sub-{subject:02d}_{stem}_LP-30_64Hz.npy").touch()
        for folder in ("mel10_64Hz", "wav2vec_l14_pca64_64Hz"):
            directory = root / "stimuli" / dataset / folder
            directory.mkdir(parents=True)
            for index in range(stimuli):
                (directory / f"stim{index}.npy").touch()


def write_window_tree(root: Path, subjects=("2", "10"), lengths=None) -> None:
    lengths = lengths or {subject: 640 for subject in subjects}
    for folder, channels in (("mel10_64Hz", 10), ("wav2vec_l14_pca64_64Hz", 64)):
        directory = root / "stimuli" / "PKUEEG" / folder
        directory.mkdir(parents=True, exist_ok=True)
        value = np.arange(960 * channels, dtype=np.float32).reshape(960, channels) % 97
        np.save(directory / "stim.npy", value)
    neural = root / "neural_lp30_64hz" / "PKUEEG"
    neural.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        value = np.arange(lengths[subject] * 57, dtype=np.float32).reshape(-1, 57) % 101
        np.save(neural / f"sub-{subject}_stim_LP-30_64Hz.npy", value)


class ModelInputTests(unittest.TestCase):
    def test_config_rejects_invalid_dimensions_and_dropout(self):
        for dimension in (0, -1, True, 1.5, "32"):
            with self.subTest(embed_dim=dimension), self.assertRaises(ValueError):
                SNTEConfig(embed_dim=dimension)
        for dropout in (-0.1, 1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                SNTEConfig(dropout=dropout)
        # Small dimensions are useful for synthetic probes and baseline configs.
        self.assertEqual(SNTEConfig(embed_dim=8, dropout=0).embed_dim, 8)

    def test_config_rejects_unsupported_scorer_and_encoder_settings(self):
        changes = (
            {"head": "unknown"}, {"neural_encoder": "unknown"},
            {"speech_encoder": "unknown"}, {"scales": (1, 2)},
            {"scales": ()}, {"scales": (True,)}, {"scales": (1.0,)},
            {"shifts": (0, 1)}, {"shifts": (False,)}, {"shifts": (0.0,)}, {"stats": ("mean",)},
            {"stats": ("max", "mean", "absdiff")},
            {"neural_dilations": (1, 2, 3)}, {"speech_dilations": (1, 3)},
            {"neural_dilations": (True, 3, 9)},
            {"speech_dilations": (1, 1, 0)}, {"standardize": 1},
            {"tied_encoder": 1},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                SNTEConfig(**change)

    def test_tied_config_requires_equal_dilated_encoders(self):
        for change in ({"neural_encoder": "linear"}, {"speech_encoder": "linear"},
                       {"speech_dilations": (1, 1, 1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                SNTEConfig(tied_encoder=True, **change)
        SNTEConfig(tied_encoder=True, neural_dilations=(1, 1, 1),
                   speech_dilations=(1, 1, 1))

    def test_baseline_omitted_config_matches_explicit_final_configuration(self):
        for name in BASELINE_NAMES:
            with self.subTest(model=name):
                final = get_model_config(name)
                torch.manual_seed(17)
                implicit = create_baseline(name, 57)
                implicit_rng = torch.get_rng_state().clone()
                torch.manual_seed(17)
                explicit = create_baseline(name, 57, config=SNTEConfig(
                    embed_dim=final.embed_dim, dropout=final.dropout))
                self.assertTrue(torch.equal(implicit_rng, torch.get_rng_state()))
                self.assertEqual((implicit.neural_channels, implicit.speech_channels), (57, 74))
                self.assertEqual(list(implicit.state_dict()), list(explicit.state_dict()))
                for key, value in implicit.state_dict().items():
                    self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
                self.assertEqual(implicit.speech_encoder.network[0].out_channels, final.embed_dim)

    def test_baseline_explicit_config_preserved_and_snte_overrides_rejected(self):
        cfg = SNTEConfig(embed_dim=8, dropout=0.25)
        network = create_baseline("cca", 57, config=cfg)
        self.assertEqual(network.speech_encoder.network[0].out_channels, 8)
        self.assertEqual(network.speech_encoder.network[3].p, 0.25)
        changes = ({"head": "concat"}, {"head": "timecos"}, {"standardize": False},
                   {"tied_encoder": True}, {"neural_encoder": "linear"},
                   {"speech_encoder": "linear"}, {"neural_dilations": (1, 1, 1)},
                   {"speech_dilations": (1, 1, 1)})
        for name in BASELINE_NAMES:
            for change in changes:
                with self.subTest(model=name, change=change), self.assertRaises(ValueError):
                    create_model(name, 57, config=SNTEConfig(embed_dim=32, dropout=0, **change))

    def test_all_matchers_reject_rank_batch_time_candidate_and_channel_errors(self):
        valid_neural = torch.zeros(2, 64, 57)
        valid_speech = torch.zeros(2, 5, 64, 74)
        cases = (
            (torch.zeros(2, 57), valid_speech),
            (valid_neural, torch.zeros(2, 64, 74)),
            (torch.zeros(1, 64, 57), valid_speech),  # Would silently broadcast.
            (valid_neural, torch.zeros(1, 5, 64, 74)),
            (valid_neural, torch.zeros(2, 5, 63, 74)),  # Used to truncate.
            (valid_neural, torch.zeros(2, 4, 64, 74)),
            (valid_neural, torch.zeros(2, 0, 64, 74)),
            (torch.zeros(2, 64, 56), valid_speech),
            (valid_neural, torch.zeros(2, 5, 64, 73)),
            (torch.zeros(0, 64, 57), torch.zeros(0, 5, 64, 74)),
            (torch.zeros(2, 0, 57), torch.zeros(2, 5, 0, 74)),
        )
        for name in ("snte",) + BASELINE_NAMES:
            final = get_model_config(name)
            network = create_model(name, 57, config=SNTEConfig(
                embed_dim=final.embed_dim, dropout=final.dropout))
            for neural, speech in cases:
                with self.subTest(model=name, neural=neural.shape, speech=speech.shape):
                    with self.assertRaises(ValueError):
                        network(neural, speech)

    def test_generic_positive_time_lengths_are_allowed(self):
        for name in ("snte", "cca"):
            with self.subTest(model=name), torch.no_grad():
                network = create_model(name, 57, config=SNTEConfig(embed_dim=8, dropout=0)).eval()
                scores = network(torch.randn(2, 17, 57), torch.randn(2, 5, 17, 74))
                self.assertEqual(tuple(scores.shape), (2, 5))
                self.assertTrue(torch.isfinite(scores).all())


class LoaderInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="snte-input-loaders-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_neural_rejects_empty_wrong_rank_channels_and_nonfinite(self):
        for index, value in enumerate((np.empty((0, 57)), np.zeros((3, 4, 57)),
                                       np.zeros((10, 56)), np.full((10, 57), np.nan))):
            path = self.root / f"neural{index}.npy"
            np.save(path, value)
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                data._load_neural(path, 57)

    def test_neural_rejects_normalization_overflow_with_path(self):
        maximum = np.finfo(np.float32).max
        path = self.root / "overflow.npy"
        np.save(path, np.tile(np.asarray([-maximum, maximum, maximum])[:, None], (1, 57)))
        with self.assertRaisesRegex(ValueError, "overflow.npy"):
            data._load_neural(path, 57)

    def test_speech_rejects_empty_nonfinite_unaligned_and_wrong_shapes(self):
        mel_path, wav_path = self.root / "mel.npy", self.root / "wav.npy"
        for mel, wav in ((np.empty((0, 10)), np.empty((0, 64))),
                         (np.full((10, 10), np.inf), np.zeros((10, 64))),
                         (np.zeros((9, 10)), np.zeros((10, 64))),
                         (np.zeros((10, 11)), np.zeros((10, 64)))):
            np.save(mel_path, mel)
            np.save(wav_path, wav)
            with self.subTest(mel=mel.shape, wav=wav.shape), self.assertRaises(ValueError):
                data._load_speech(mel_path, wav_path)

    def test_speech_rejects_normalization_overflow_with_path(self):
        maximum = np.finfo(np.float32).max
        mel_path, wav_path = self.root / "mel-overflow.npy", self.root / "wav-overflow.npy"
        np.save(mel_path, np.zeros((3, 10), dtype=np.float32))
        np.save(wav_path, np.tile(np.asarray([-maximum, maximum, maximum])[:, None], (1, 64)))
        with self.assertRaisesRegex(ValueError, "overflow.npy"):
            data._load_speech(mel_path, wav_path)

    def test_loaders_never_allow_pickle(self):
        neural = self.root / "object.npy"
        np.save(neural, np.asarray([[object()]], dtype=object))
        with self.assertRaises(ValueError):
            data._load_neural(neural, 1)


class ContractInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="snte-input-contract-")
        cls.root = Path(cls.temporary.name)
        write_contract_tree(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def rename(self, source: Path, destination: Path):
        source.rename(destination)
        self.addCleanup(destination.rename, source)

    def test_default_all_and_selected_dataset_preflight(self):
        report = data.validate_shared_contract(self.root)
        self.assertEqual(set(report), set(data.DATASETS))
        for selection in (["PKUEEG"], ("SEM4Lang",)):
            selected = data.validate_shared_contract(self.root, datasets=selection)
            self.assertEqual(set(selected), set(selection))
        with tempfile.TemporaryDirectory(prefix="snte-selected-contract-") as directory:
            root = Path(directory)
            write_contract_tree(root, ("SEM4Lang",))
            self.assertEqual(set(data.validate_shared_contract(root, datasets=("SEM4Lang",))),
                             {"SEM4Lang"})
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                data.validate_shared_contract(root)

    def test_unknown_dataset_selection_rejected(self):
        with self.assertRaises(ValueError):
            data.validate_shared_contract(self.root, datasets=("unknown",))

    def test_missing_directories_are_reported_before_iteration(self):
        with tempfile.TemporaryDirectory(prefix="snte-missing-contract-") as directory:
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                data.validate_shared_contract(Path(directory), datasets=("PKUEEG",))
        directory = self.root / "stimuli" / "SEM4Lang" / "mel10_64Hz"
        self.rename(directory, directory.with_name("missing-mel"))
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            data.validate_shared_contract(self.root, datasets=("SEM4Lang",))

    def test_invalid_filename_and_subject_range_rejected(self):
        neural = self.root / "neural_lp30_64hz" / "SEM4Lang"
        source = neural / "sub-01_stim0_LP-30_64Hz.npy"
        for filename in ("subject-01_stim0_LP-30_64Hz.npy", "sub-x_stim0_LP-30_64Hz.npy",
                         "sub-00_stim0_LP-30_64Hz.npy", "sub-13_stim0_LP-30_64Hz.npy",
                         "sub-01_stim0_64Hz.npy", "sub-01__LP-30_64Hz.npy"):
            destination = neural / filename
            source.rename(destination)
            try:
                with self.subTest(filename=filename), self.assertRaises((ValueError, RuntimeError)):
                    data.validate_shared_contract(self.root, datasets=("SEM4Lang",))
            finally:
                destination.rename(source)

    def test_subject_padding_alias_and_duplicate_numeric_record_rejected(self):
        neural = self.root / "neural_lp30_64hz" / "SEM4Lang"
        # Same numeric subject with a different stimulus is still ambiguous.
        self.rename(neural / "sub-02_stim1_LP-30_64Hz.npy",
                    neural / "sub-2_stim1_LP-30_64Hz.npy")
        with self.assertRaises((ValueError, RuntimeError)):
            data.validate_shared_contract(self.root, datasets=("SEM4Lang",))

    def test_duplicate_numeric_subject_stimulus_record_rejected(self):
        neural = self.root / "neural_lp30_64hz" / "SEM4Lang"
        # Preserve total count but turn another record into an alias duplicate.
        self.rename(neural / "sub-03_stim0_LP-30_64Hz.npy",
                    neural / "sub-2_stim0_LP-30_64Hz.npy")
        with self.assertRaises((ValueError, RuntimeError)):
            data.validate_shared_contract(self.root, datasets=("SEM4Lang",))

    def test_speech_pair_identifier_and_neural_stimulus_mismatch_rejected(self):
        mel = self.root / "stimuli" / "SEM4Lang" / "mel10_64Hz"
        wav = self.root / "stimuli" / "SEM4Lang" / "wav2vec_l14_pca64_64Hz"
        self.rename(mel / "stim0.npy", mel / "unknown.npy")
        with self.assertRaises((ValueError, RuntimeError)):
            data.validate_shared_contract(self.root, datasets=("SEM4Lang",))
        self.rename(wav / "stim0.npy", wav / "unknown.npy")
        with self.assertRaises((ValueError, RuntimeError)):
            data.validate_shared_contract(self.root, datasets=("SEM4Lang",))

    def test_complete_subject_universe_required_even_when_file_count_matches(self):
        neural = self.root / "neural_lp30_64hz" / "SparKULee"
        # This sparse synthetic layout leaves free (subject, stimulus) pairs.
        # Remove subject 85 while preserving counts and speech correspondence.
        for path in list(neural.glob("sub-85_*.npy")):
            destination = path.with_name(path.name.replace("sub-85_", "sub-01_"))
            self.assertFalse(destination.exists())
            self.rename(path, destination)
        self.assertEqual(len(list(neural.glob("*.npy"))), data.EXPECTED_NEURAL["SparKULee"])
        with self.assertRaisesRegex(RuntimeError, "85"):
            data.validate_shared_contract(self.root, datasets=("SparKULee",))

    def test_missing_subject_universe_with_consistent_remaining_layout(self):
        with tempfile.TemporaryDirectory(prefix="snte-missing-subject-") as directory:
            root = Path(directory)
            write_contract_tree(root, ("SEM4Lang",))
            neural = root / "neural_lp30_64hz" / "SEM4Lang"
            for path in neural.glob("sub-12_*.npy"):
                path.unlink()
            with mock.patch.dict(data.EXPECTED_NEURAL, {"SEM4Lang": 660}):
                with self.assertRaises((ValueError, RuntimeError)):
                    data.validate_shared_contract(root, datasets=("SEM4Lang",))


class WindowCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="snte-input-windows-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.object(data, "_RANDOM_MAPPING", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_actual_unit_names_keep_lexical_order_and_small_pool_replacement(self):
        write_window_tree(self.root)
        dataset = data.MatchMismatchDataset("PKUEEG", "train", root=self.root)
        self.assertEqual(dataset.unit_names, ["10", "2"])
        self.assertEqual(data.all_subjects("PKUEEG", self.root), ["2", "10"])
        for plan, label in zip(dataset.candidate_plan, dataset.labels):
            negatives = [int(candidate) for index, candidate in enumerate(plan) if index != int(label)]
            self.assertLess(len(set(negatives)), 4)
            self.assertNotIn(int(plan[int(label)]), negatives)

    def test_formal_coverage_rejects_subject_without_full_window(self):
        write_window_tree(self.root, lengths={"2": 100, "10": 640})
        relaxed = data.MatchMismatchDataset("PKUEEG", "train", root=self.root)
        self.assertEqual(relaxed.unit_names, ["10"])
        with self.assertRaisesRegex(RuntimeError, "2"):
            data.MatchMismatchDataset("PKUEEG", "train", root=self.root, require_all_subjects=True)

    def test_formal_coverage_is_checked_after_trimming(self):
        write_window_tree(self.root)
        for trims in ({"max_files": 1}, {"max_segments": 1}):
            with self.subTest(trims=trims), self.assertRaises(RuntimeError):
                data.MatchMismatchDataset("PKUEEG", "train", root=self.root,
                                          require_all_subjects=True, **trims)
            partial = data.MatchMismatchDataset("PKUEEG", "train", root=self.root, **trims)
            self.assertEqual(partial.unit_names, ["10"])

    def test_dataset_rejects_missing_stimulus_file(self):
        write_window_tree(self.root)
        (self.root / "stimuli" / "PKUEEG" / "mel10_64Hz" / "stim.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            data.MatchMismatchDataset("PKUEEG", "train", root=self.root)

    def test_build_splits_distinguishes_formal_and_integration_coverage(self):
        with mock.patch.object(data, "configure_split_seed") as configure, mock.patch.object(
                data, "MatchMismatchDataset", return_value=object()) as constructor:
            data.build_splits("PKUEEG", root=self.root, integration=False, require_all_subjects=True)
            self.assertEqual(constructor.call_count, 3)
            for call in constructor.call_args_list:
                self.assertIs(call.kwargs["require_all_subjects"], True)
            constructor.reset_mock()
            data.build_splits("PKUEEG", root=self.root, integration=True)
            for call in constructor.call_args_list:
                self.assertIs(call.kwargs["require_all_subjects"], False)
                self.assertEqual(call.kwargs["max_files"], 2)
                self.assertEqual(call.kwargs["max_segments"], 12)
            self.assertEqual(configure.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
