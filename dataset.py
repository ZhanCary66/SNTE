#!/usr/bin/env python3
"""Data pipeline for the paper's cross-subject match--mismatch protocol.

Turns the shared data directory into the "5-second window + 5 candidates
(1 matched, 4 mismatched)" classification task. Candidate identity, order,
labels, normalization and the train/validation/test splits follow the protocol
used for every number in the paper.

Design notes:
- DataLoader workers only return a neural window view plus five global speech
  indices. The candidate features themselves live in a ``speech_bank`` moved to
  the GPU up front, so the training loop fetches them with a single
  ``index_select`` instead of re-transferring them per sample.
- Each window's five candidates are drawn once with a fixed seed, so the
  candidate plan is deterministic and reproducible across runs and splits.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _default_root() -> Path:
    """Location of the shared dataset directory.

    Override with the ``SNTE_DATA_ROOT`` environment variable; the layout
    expected under it is documented in the README.
    """
    return Path(os.environ.get("SNTE_DATA_ROOT", "./data/ICASSP_shared_v1"))


SHARED_ROOT = _default_root()
# Datasets covered by the paper.
DATASETS = ("SparKULee", "PKUEEG", "SEM4Lang")
# Expected file counts, used to validate data integrity.
EXPECTED_NEURAL = {"SparKULee": 662, "PKUEEG": 1250, "SEM4Lang": 720}
EXPECTED_STIMULI = {"SparKULee": 72, "PKUEEG": 50, "SEM4Lang": 60}
# Raw files contain 64/57/306 channels; the model receives 64/57/204 channels.
SOURCE_NEURAL_CHANNELS = {"SparKULee": 64, "PKUEEG": 57, "SEM4Lang": 306}
NEURAL_CHANNELS = {"SparKULee": 64, "PKUEEG": 57, "SEM4Lang": 204}

# Neuromag arrays contain 102 sensor triplets. In every triplet, offset 0 is
# the magnetometer and offsets 1/2 are the two planar gradiometers. SEM4Lang
# uses only the 204 planar gradiometers in every split.
SEM4LANG_CHANNEL_INDICES = np.asarray(
    [index for index in range(306) if index % 3 in (1, 2)], dtype=np.int64
)
assert len(SEM4LANG_CHANNEL_INDICES) == 204
# Sampling rate of both the neural signals and the speech features (Hz).
SAMPLE_RATE = 64
# Window length: 5 seconds at 64 Hz = 320 samples.
SEGMENT_SAMPLES = 5 * SAMPLE_RATE
# Number of candidates per trial (five-way).
N_CANDIDATES = 5
# Speech feature dimension: wav2vec-L14-PCA64 (64) + Mel10 (10) = 74.
SPEECH_DIM = 74
# Fixed seeds used to draw candidates for each split, so that the plan is
# identical across runs.
FIXED_CANDIDATE_SEEDS = {"train": 20260801, "val": 20260802, "test": 20260803}
# Protocol version, recorded in every result JSON.
PROTOCOL_VERSION = "icassp_mm_v2_5s_w2v64_mel10_fixed5way_subjectsplit_sem204grad"


def validate_shared_contract(root: Path = SHARED_ROOT) -> dict:
    """Validate the shared data layout and return per-dataset statistics.

    Checks that the directory exists, that all three datasets have both neural
    and stimulus subdirectories, that the file counts match the expected ones,
    and that the Mel and wav2vec identifiers correspond one to one.
    """
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected = set(DATASETS)
    neural_sets = {p.name for p in (root / "neural_lp30_64hz").iterdir() if p.is_dir()}
    stimulus_sets = {p.name for p in (root / "stimuli").iterdir() if p.is_dir()}
    missing_neural = expected - neural_sets
    missing_stimuli = expected - stimulus_sets
    if missing_neural or missing_stimuli:
        raise RuntimeError(
            f"paper datasets missing: neural={missing_neural}, stimuli={missing_stimuli}"
        )
    report = {}
    for dataset in DATASETS:
        neural = sorted((root / "neural_lp30_64hz" / dataset).glob("*.npy"))
        mel = sorted((root / "stimuli" / dataset / "mel10_64Hz").glob("*.npy"))
        wav = sorted((root / "stimuli" / dataset / "wav2vec_l14_pca64_64Hz").glob("*.npy"))
        if len(neural) != EXPECTED_NEURAL[dataset]:
            raise RuntimeError(
                f"{dataset}: {len(neural)} neural files, expected {EXPECTED_NEURAL[dataset]}"
            )
        if len(mel) != EXPECTED_STIMULI[dataset] or len(wav) != len(mel):
            raise RuntimeError(f"{dataset}: invalid stimulus counts mel={len(mel)} wav={len(wav)}")
        if {p.stem for p in mel} != {p.stem for p in wav}:
            raise RuntimeError(f"{dataset}: Mel/wav2vec identifiers differ")
        report[dataset] = {
            "neural": len(neural),
            "stimuli": len(mel),
            "source_channels": SOURCE_NEURAL_CHANNELS[dataset],
            "model_channels": NEURAL_CHANNELS[dataset],
        }
    return report


def subject_id(path: Path) -> str:
    """Extract the subject ID from a filename, e.g. "sub-05_xxx.npy" -> "05"."""
    return path.name.split("_", 1)[0].removeprefix("sub-")


def stimulus_id_from_neural(path: Path) -> str:
    """Recover the stimulus identifier from a neural filename.

    For example "sub-05_s0_sentence3_LP-30_64Hz.npy" -> "s0_sentence3": the
    leading "sub-<subject>_" and the trailing "_LP-30_64Hz" are removed.
    """
    stem = path.stem
    stem = re.sub(r"^sub-[^_]+_", "", stem)
    return stem.removesuffix("_LP-30_64Hz")


def _fixed_split_number(dataset: str, subject: str) -> str:
    """The paper's fixed cross-subject split, decided by subject number.

    The three datasets have different subject counts and therefore different
    thresholds:
    - SparKULee: subjects 1-54 train, 55-68 validation, 69+ test (54/14/17);
    - PKUEEG:    subjects 1-15 train, 16-20 validation, 21+ test (15/5/5);
    - SEM4Lang:  subjects 1-8  train, 9-10 validation, 11+ test (8/2/2).
    """
    number = int(subject)
    if dataset == "SparKULee":
        return "train" if number <= 54 else ("val" if number <= 68 else "test")
    if dataset == "PKUEEG":
        return "train" if number <= 15 else ("val" if number <= 20 else "test")
    if dataset == "SEM4Lang":
        return "train" if number <= 8 else ("val" if number <= 10 else "test")
    raise ValueError(dataset)


# Random cross-subject re-partitions. ``configure_split_seed`` shuffles the
# subjects and re-applies the same train/validation/test sizes as the fixed
# split, so that results can be checked against several different assignments
# of participants to the test set. Every model sees the same partition within
# one split seed.
#
# Note that these are independent random re-partitions, not disjoint folds of a
# k-fold cross-validation: the test sets of two different seeds overlap, and
# the same subject can be tested under more than one seed.
_RANDOM_MAPPING: dict[str, dict[str, str]] = {}


def random_subject_splits(
    dataset: str, split_seed: int, root: Path = SHARED_ROOT
) -> dict[str, str]:
    """Return a {subject: split} mapping for one random re-partition.

    The split sizes are identical to the fixed split.
    """
    subjects = sorted(
        {subject_id(p) for p in (root / "neural_lp30_64hz" / dataset).glob("*.npy")}, key=int
    )
    counts = {"train": 0, "val": 0, "test": 0}
    for subject in subjects:
        counts[_fixed_split_number(dataset, subject)] += 1
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(subjects))
    mapping: dict[str, str] = {}
    for rank, index in enumerate(order):
        if rank < counts["train"]:
            split = "train"
        elif rank < counts["train"] + counts["val"]:
            split = "val"
        else:
            split = "test"
        mapping[subjects[index]] = split
    return mapping


def configure_split_seed(split_seed: int | None, root: Path = SHARED_ROOT) -> None:
    """Select the split: None = the paper's fixed split, int = random re-partition."""
    global _RANDOM_MAPPING
    if split_seed is None:
        _RANDOM_MAPPING = {}
    else:
        _RANDOM_MAPPING = {
            dataset: random_subject_splits(dataset, split_seed, root) for dataset in DATASETS
        }


def split_name(dataset: str, path: Path) -> str:
    """Decide whether a file belongs to train, validation or test.

    Uses the paper's fixed split by default, or the random re-partition
    configured through :func:`configure_split_seed`.
    """
    subject = subject_id(path)
    if _RANDOM_MAPPING:
        return _RANDOM_MAPPING[dataset][subject]
    return _fixed_split_number(dataset, subject)


def _zscore(array: np.ndarray) -> np.ndarray:
    """Standardize along the feature axis: z = (x - mean) / (std + 1e-6)."""
    array = np.asarray(array, dtype=np.float32)
    mean = array.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    return ((array - mean) / np.maximum(std, np.float32(1e-6))).astype(np.float32, copy=False)


def _load_neural(
    path: Path,
    source_channels: int,
    channel_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Load one neural file and standardize it feature-wise.

    Both storage layouts (time x channel and channel x time) are accepted; the
    result is always time x channel.
    """
    value = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if value.ndim != 2:
        raise ValueError(f"{path}: expected 2-D, got {value.shape}")
    if value.shape[1] == source_channels:
        pass
    elif value.shape[0] == source_channels:
        value = value.T
    else:
        raise ValueError(
            f"{path}: neither dimension matches {source_channels} channels: {value.shape}"
        )
    if channel_indices is not None:
        value = value[:, channel_indices]
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: non-finite neural values")
    return _zscore(value)


def _load_speech(mel_path: Path, wav_path: Path) -> np.ndarray:
    """Load and concatenate the Mel and wav2vec features of one stimulus.

    The wav2vec features come first, then the Mel features, giving a
    standardized array of shape [time, 74].
    """
    mel = np.load(mel_path, allow_pickle=False).astype(np.float32, copy=False)
    wav = np.load(wav_path, allow_pickle=False).astype(np.float32, copy=False)
    if mel.ndim != 2 or mel.shape[1] != 10 or wav.ndim != 2 or wav.shape[1] != 64:
        raise ValueError(f"invalid speech shapes: {mel_path} {mel.shape}, {wav_path} {wav.shape}")
    if mel.shape[0] != wav.shape[0]:
        raise ValueError(
            f"unaligned speech features: {mel_path} {mel.shape}, {wav_path} {wav.shape}"
        )
    speech = np.concatenate([_zscore(wav), _zscore(mel)], axis=1)
    if not np.isfinite(speech).all():
        raise ValueError(f"{mel_path}: non-finite speech values")
    return speech.astype(np.float32, copy=False)


@dataclass(frozen=True)
class Segment:
    """Location of one 5-second segment.

    - trial: index of the neural recording this segment belongs to;
    - start: start sample of the segment within that trial;
    - subject: subject ID;
    - stimulus: identifier of the speech stimulus.
    """

    trial: int
    start: int
    subject: str
    stimulus: str


class MatchMismatchDataset(Dataset):
    """Preloaded dataset with a deterministic candidate plan.

    Every 5-second neural window is one sample. The candidate plan gives the
    five speech segments (global indices into ``speech_bank``) and the label
    for that window; it is drawn once at construction time with a fixed seed
    and never changes.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        root: Path = SHARED_ROOT,
        max_files: int = 0,
        max_segments: int = 0,
    ) -> None:
        # max_files / max_segments trim the data for quick smoke tests.
        if dataset not in DATASETS or split not in FIXED_CANDIDATE_SEEDS:
            raise ValueError((dataset, split))
        source_channels = SOURCE_NEURAL_CHANNELS[dataset]
        channel_indices = SEM4LANG_CHANNEL_INDICES if dataset == "SEM4Lang" else None
        neural_paths = [
            p for p in sorted((root / "neural_lp30_64hz" / dataset).glob("*.npy"))
            if split_name(dataset, p) == split
        ]
        if max_files:
            neural_paths = neural_paths[:max_files]
        if not neural_paths:
            raise RuntimeError(f"empty {dataset}/{split} split")

        mel_dir = root / "stimuli" / dataset / "mel10_64Hz"
        wav_dir = root / "stimuli" / dataset / "wav2vec_l14_pca64_64Hz"
        # Load the speech features of every stimulus this split touches.
        required = sorted({stimulus_id_from_neural(p) for p in neural_paths})
        speech = {
            stem: _load_speech(mel_dir / f"{stem}.npy", wav_dir / f"{stem}.npy")
            for stem in required
        }
        self.neural: list[np.ndarray] = []
        self.segments: list[Segment] = []
        for path in neural_paths:
            stimulus = stimulus_id_from_neural(path)
            neural = _load_neural(path, source_channels, channel_indices)
            trial = len(self.neural)
            self.neural.append(neural)
            # Number of whole windows that fit in both the neural recording and
            # the stimulus.
            usable = min(len(neural), len(speech[stimulus])) // SEGMENT_SAMPLES
            for index in range(usable):
                self.segments.append(
                    Segment(trial, index * SEGMENT_SAMPLES, subject_id(path), stimulus)
                )
        if max_segments:
            self.segments = self.segments[:max_segments]
        if not self.segments:
            raise RuntimeError(f"no five-second segments for {dataset}/{split}")
        self.dataset = dataset
        self.split = split
        # Fixed seed used to draw the candidate plan for this split.
        self.candidate_seed = FIXED_CANDIDATE_SEEDS[split]
        # Subjects are sorted and indexed, which is what per-subject
        # (subject-macro) accuracies are computed over. Note that the keys of
        # the reported per-subject accuracies are these positional indices, not
        # the subject IDs themselves.
        self.unit_names = sorted({segment.subject for segment in self.segments})
        self.unit_to_index = {name: index for index, name in enumerate(self.unit_names)}

        # The concatenated speech array is already C-contiguous: truncate each
        # stimulus to whole segments and reshape to [segments, time, features],
        # so candidates can be gathered with integer indexing.
        self.speech_segments = {}
        for stem, value in speech.items():
            count = len(value) // SEGMENT_SAMPLES
            segments = value[: count * SEGMENT_SAMPLES].reshape(count, SEGMENT_SAMPLES, SPEECH_DIM)
            self.speech_segments[stem] = torch.from_numpy(segments)

        # Build the candidate plan. This reproduces the baseline code's random
        # draw exactly, but pays the Python/NumPy cost once instead of per
        # sample, per worker and per epoch.
        count = len(self.segments)
        candidate_plan = np.empty((count, N_CANDIDATES), dtype=np.int64)
        labels = np.empty(count, dtype=np.int64)
        unit_indices = np.empty(count, dtype=np.int64)
        for index, record in enumerate(self.segments):
            # Positive: the stimulus segment aligned with this neural window.
            positive_index = record.start // SEGMENT_SAMPLES
            total = len(speech[record.stimulus]) // SEGMENT_SAMPLES
            # Negatives: other time positions of the SAME stimulus, never a
            # different stimulus.
            alternatives = np.delete(np.arange(total, dtype=np.int64), positive_index)
            if len(alternatives) == 0:
                raise RuntimeError(f"{record.stimulus}: no negative segment")
            rng = np.random.default_rng(np.random.SeedSequence([self.candidate_seed, index]))
            negatives = rng.choice(
                alternatives,
                N_CANDIDATES - 1,
                replace=len(alternatives) < N_CANDIDATES - 1,
            )
            candidate_indices = np.concatenate([[positive_index], negatives]).astype(np.int64)
            # Shuffle the five candidates and record where the positive landed.
            order = rng.permutation(N_CANDIDATES)
            labels[index] = int(np.flatnonzero(order == 0)[0])
            candidate_plan[index] = candidate_indices[order]
            unit_indices[index] = self.unit_to_index[record.subject]

        # Concatenate every stimulus into a single speech bank and record each
        # stimulus offset, so that "segment within stimulus" becomes "global
        # segment index".
        stems = sorted(self.speech_segments)
        offsets = {}
        bank = []
        offset = 0
        for stem in stems:
            offsets[stem] = offset
            value = self.speech_segments[stem]
            bank.append(value)
            offset += len(value)
        for index, record in enumerate(self.segments):
            candidate_plan[index] += offsets[record.stimulus]
        self.speech_bank = torch.cat(bank, dim=0)
        del self.speech_segments
        self.candidate_plan = torch.from_numpy(candidate_plan)
        self.labels = torch.from_numpy(labels)
        self.unit_indices = torch.from_numpy(unit_indices)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int):
        """One sample: neural window, five global candidate indices, label, subject index.

        The candidate features themselves are not returned here; the training
        loop gathers them from the GPU-resident speech bank to maximize
        throughput.
        """
        record = self.segments[index]
        neural = self.neural[record.trial][record.start : record.start + SEGMENT_SAMPLES]
        return (
            torch.from_numpy(neural),
            self.candidate_plan[index],
            self.labels[index],
            self.unit_indices[index],
        )


def build_splits(
    dataset: str,
    integration: bool = False,
    root: Path = SHARED_ROOT,
    split_seed: int | None = None,
):
    """Build the train / validation / test splits of one dataset.

    ``integration=True`` loads only a couple of files and segments for a quick
    smoke test. ``split_seed=None`` uses the paper's fixed split; an integer
    selects the corresponding random cross-subject re-partition.
    """
    configure_split_seed(split_seed, root)
    kwargs = {"max_files": 2, "max_segments": 12} if integration else {}
    return {
        split: MatchMismatchDataset(dataset, split, root=root, **kwargs)
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    print(validate_shared_contract())
