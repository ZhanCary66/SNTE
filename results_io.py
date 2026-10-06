"""Strict, raw-data-free result I/O shared by collectors and statistical tools.

Schema 1 records actual, lexicographically ordered ``split_unit_names`` and
``candidate_seeds``. Missing legacy subject IDs or architecture fields are not inferred.
An explicit legacy overlay supplies missing ``split_unit_names`` and ``architecture``.
Conflicting recorded fields are rejected, except for the known schema-0 baseline
architecture bookkeeping pattern when a nonempty ``provenance`` is provided.
Overlays affect a copied result only.
On disk the subject-map JSON is keyed by root-relative POSIX result paths, e.g.
``table1/snte_split101/PKUEEG_seed2.json``. ``load_run`` takes that whole map;
``validate_result`` and ``subject_values`` take one entry. No raw data is read.
"""

from __future__ import annotations

import copy
import json
import math
import re
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from config import get_model_config
from dataset import DATASETS, FIXED_CANDIDATE_SEEDS, PROTOCOL_VERSION
from model import SNTEConfig
from run_paper import Run, canonical_run, run_aliases

SCHEMA_VERSION = 1
SPLIT_COUNTS = {"SparKULee": (54, 14, 17), "PKUEEG": (15, 5, 5), "SEM4Lang": (8, 2, 2)}
SPLITS = ("train", "val", "test")
ARCH_FIELDS = ("head", "standardize", "tied_encoder", "neural_encoder", "speech_encoder",
               "neural_dilations", "speech_dilations", "scales", "shifts", "stats")


class ResultError(ValueError):
    """An existing result is invalid or two historical aliases conflict."""


def effective_architecture(name: str, config: SNTEConfig) -> dict:
    """Architecture actually executed; ModelConfig dimensions remain top-level."""
    get_model_config(name)
    if name == "snte":
        values = asdict(config)
        return {key: list(values[key]) if isinstance(values[key], tuple) else values[key]
                for key in ARCH_FIELDS}
    return {"head": "timecos", "standardize": name != "cca", "tied_encoder": False,
            "neural_encoder": name, "speech_encoder": "dilated", "neural_dilations": None,
            "speech_dilations": [1, 3, 9], "scales": [1], "shifts": [0], "stats": []}


def expected_architecture(run: Run) -> dict:
    """Interpret the public Run flags without importing the training entry point."""
    model = get_model_config(run.model)
    config = SNTEConfig(embed_dim=model.embed_dim, dropout=model.dropout)
    flags = iter(run.flags)
    boolean = {"--no-standardize": ("standardize", False), "--tied-encoder": ("tied_encoder", True)}
    fields = {"--head": "head", "--neural-encoder": "neural_encoder",
              "--speech-encoder": "speech_encoder", "--neural-dilations": "neural_dilations",
              "--speech-dilations": "speech_dilations"}
    try:
        for flag in flags:
            if flag in boolean:
                key, value = boolean[flag]
            elif flag in fields:
                key, value = fields[flag], next(flags)
                if key.endswith("dilations"):
                    value = tuple(int(item) for item in value.split(","))
            else:
                raise ResultError(f"unsupported run flag {flag!r}")
            config = replace(config, **{key: value})
    except (StopIteration, ValueError) as error:
        raise ResultError(f"invalid architecture flags for {run.tag}: {error}") from error
    return effective_architecture(run.model, config)


def run_path(root: Path, run: Run) -> Path:
    """Exact location of this reference, without canonicalizing it."""
    return Path(root) / run.table / run.label / f"{run.dataset}_seed{run.seed}.json"


def run_paths(root: Path, run: Run) -> tuple[Path, ...]:
    """Canonical path first, then known historical aliases."""
    run = canonical_run(run)
    return tuple(run_path(root, item) for item in (run, *run_aliases(run)))


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ResultError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise ResultError(f"{path}: expected a JSON object")
    return value


def load_subject_map(path: Path) -> dict:
    """Load explicit legacy provenance keyed by relative result path."""
    mapping = _json(Path(path))
    for key, entry in mapping.items():
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != key:
            raise ResultError(f"subject-map key must be a root-relative POSIX path: {key!r}")
        if not isinstance(entry, dict) or set(entry) - {"split_unit_names", "architecture", "provenance"}:
            raise ResultError(f"invalid subject-map entry for {key!r}")
        if not {"split_unit_names", "architecture"} & set(entry):
            raise ResultError(f"subject-map entry {key!r} supplies no known metadata")
    return mapping


def _equal(actual, expected) -> bool:
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and math.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _expect(result: dict, key: str, expected) -> None:
    if key not in result or not _equal(result[key], expected):
        raise ResultError(f"{key}: expected {expected!r}, got {result.get(key, '<missing>')!r}")


def _number(value, label: str, lower: float = 0, upper: float = math.inf) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ResultError(f"{label}: expected finite number in [{lower}, {upper}]")
    return float(value)


def _integer(value, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ResultError(f"{label}: expected integer >= {minimum}")
    return value


def _overlay(result: dict, subject_map: dict | None) -> dict:
    result = copy.deepcopy(result)
    if subject_map is not None:
        if not isinstance(subject_map, dict):
            raise ResultError("legacy overlay must be a dictionary")
        if result.get("schema_version", 0) != 0:
            raise ResultError("subject-map overlays are only allowed for legacy schema 0")
        if set(subject_map) - {"split_unit_names", "architecture", "provenance"}:
            raise ResultError("unknown legacy overlay fields")
        for field in ("split_unit_names", "architecture"):
            if field not in subject_map:
                continue
            supplied = subject_map[field]
            if field in result and not _equal(result[field], supplied):
                # Schema 0 default baseline runs recorded unused SNTE switches.
                name = result.get("name")
                correction = (field == "architecture" and name in ("cca", "convconcatnet", "vlaai", "eeg2vec", "brainmagic")
                              and _equal(result[field], effective_architecture("snte", SNTEConfig()))
                              and _equal(supplied, effective_architecture(name, SNTEConfig()))
                              and isinstance(subject_map.get("provenance"), str)
                              and bool(subject_map["provenance"].strip()))
                if not correction:
                    raise ResultError(f"legacy overlay conflicts with recorded {field}")
                result["legacy_metadata_correction"] = subject_map["provenance"]
            result[field] = copy.deepcopy(supplied)
    return result


def _unit_names(result: dict, run: Run, required: bool) -> dict | None:
    names = result.get("split_unit_names")
    if names is None:
        if required:
            raise ResultError("split_unit_names absent; supply explicit legacy subject-map metadata")
        return None
    if not isinstance(names, dict) or set(names) != set(SPLITS):
        raise ResultError("split_unit_names must contain exactly train/val/test")
    counts = SPLIT_COUNTS[run.dataset]
    for split, count in zip(SPLITS, counts):
        units = names[split]
        if (not isinstance(units, list) or len(units) != count
                or any(not isinstance(unit, str) or not unit.isascii() or not unit.isdecimal()
                       or int(unit) < 1 for unit in units)
                or units != sorted(set(units))):
            raise ResultError(f"split_unit_names.{split}: expected {count} unique lexicographically sorted IDs")
    universe = [unit for split in SPLITS for unit in names[split]]
    if len(set(universe)) != sum(counts) or len({int(unit) for unit in universe}) != sum(counts):
        raise ResultError("split_unit_names: splits overlap or numeric subject IDs repeat")
    if {int(unit) for unit in universe} != set(range(1, sum(counts) + 1)):
        raise ResultError(f"split_unit_names: expected numeric subject universe 1..{sum(counts)}")
    subjects = sorted(universe, key=int)
    train, val, _ = counts
    fixed = {unit: ("train" if int(unit) <= train else
                    "val" if int(unit) <= train + val else "test") for unit in subjects}
    if any(sum(value == split for value in fixed.values()) != count
           for split, count in zip(SPLITS, counts)):
        raise ResultError("declared subject universe does not match the formal fixed split counts")
    if run.split_seed is None:
        expected = fixed
    else:
        order = np.random.default_rng(run.split_seed).permutation(len(subjects))
        expected = {subjects[index]: ("train" if rank < train else
                                     "val" if rank < train + val else "test")
                    for rank, index in enumerate(order)}
    for split in SPLITS:
        if set(names[split]) != {unit for unit, assigned in expected.items() if assigned == split}:
            raise ResultError(f"split_unit_names.{split}: membership disagrees with split_seed={run.split_seed}")
    return names


def _metrics(metrics, label: str, count: int, segments: int) -> None:
    if not isinstance(metrics, dict):
        raise ResultError(f"{label}: missing metric object")
    _number(metrics.get("loss"), f"{label}.loss")
    _number(metrics.get("segment_accuracy"), f"{label}.segment_accuracy", upper=1)
    macro = _number(metrics.get("subject_macro_accuracy"), f"{label}.subject_macro_accuracy", upper=1)
    per_subject = metrics.get("per_subject_accuracy")
    if not isinstance(per_subject, dict) or set(per_subject) != {str(i) for i in range(count)}:
        raise ResultError(f"{label}.per_subject_accuracy: expected positional keys 0..{count - 1}")
    values = [_number(value, f"{label}.per_subject_accuracy.{key}", upper=1)
              for key, value in per_subject.items()]
    if not math.isclose(macro, statistics.mean(values), rel_tol=0, abs_tol=1e-8):
        raise ResultError(f"{label}: subject macro disagrees with per-subject accuracies")
    _integer(metrics.get("segments"), f"{label}.segments", count)
    _expect(metrics, "segments", segments)


def _execution_contract(result: dict) -> None:
    """Reject known contradictory settings without inventing absent legacy metadata."""
    training = {"optimizer": "AdamW", "gradient_clip_norm": 5.0,
                "selection_metric": "validation subject-macro accuracy",
                "selection_tolerance": 1e-8, "selection_tiebreak": "lower validation loss"}
    runtime = {"precision": "bfloat16 autocast", "autocast_enabled": True,
               "autocast_dtype": "bfloat16", "float32_matmul_precision": "high"}
    for field, values in (("training", training), ("environment", runtime)):
        if field in result:
            if not isinstance(result[field], dict):
                raise ResultError(f"{field}: expected a metadata object")
            for key, expected in values.items():
                if key in result[field]:
                    _expect(result[field], key, expected)
    for key, expected in {**training, "cudnn_deterministic": False,
                          "cudnn_benchmark": True, "gpu_speech_bank": True}.items():
        if key in result:
            _expect(result, key, expected)
    devices = [source["device"] for source in (result, result.get("environment", {})) if "device" in source]
    if any(not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device) for device in devices):
        raise ResultError("device: formal results must declare CUDA execution")
    if len(devices) == 2 and devices[0] != devices[1]:
        raise ResultError("device: top-level and environment metadata disagree")


def validate_subject_spellings(entries) -> None:
    """One numeric participant must use one explicit ID spelling across selected runs."""
    known = {}
    for run, result in entries:
        for names in result.get("split_unit_names", {}).values():
            for name in names:
                key = (run.dataset, int(name))
                if key in known and known[key] != name:
                    raise ResultError(f"inconsistent subject ID spelling for {run.dataset}/{key[1]}: "
                                      f"{known[key]!r} vs {name!r}")
                known[key] = name


def validate_result(result: dict, run: Run, subject_map: dict | None = None,
                    require_subjects: bool = True) -> dict:
    """Validate expected run metadata and return a copy with any explicit overlay."""
    if not isinstance(result, dict):
        raise ResultError("result must be a dictionary")
    if run.dataset not in DATASETS:
        raise ResultError(f"unknown dataset {run.dataset!r}")
    result = _overlay(result, subject_map)
    version = result.get("schema_version", 0)
    if type(version) is not int or version not in (0, SCHEMA_VERSION):
        raise ResultError(f"unsupported schema_version {version!r}")
    for key, expected in {**get_model_config(run.model).to_dict(), "dataset": run.dataset,
                          "seed": run.seed, "split_seed": run.split_seed,
                          "protocol": PROTOCOL_VERSION, "status": "complete", "integration": False}.items():
        _expect(result, key, expected)
    if "model" in result:
        _expect(result, "model", run.model)
    _expect(result, "architecture", expected_architecture(run))
    _execution_contract(result)
    if version == SCHEMA_VERSION:
        _expect(result, "candidate_seeds", FIXED_CANDIDATE_SEEDS)
    elif "candidate_seeds" in result:
        _expect(result, "candidate_seeds", FIXED_CANDIDATE_SEEDS)
    names = _unit_names(result, run, require_subjects or version == SCHEMA_VERSION)
    counts = dict(zip(SPLITS, SPLIT_COUNTS[run.dataset]))
    for field in ("split_files", "split_segments"):
        values = result.get(field)
        if not isinstance(values, dict) or set(values) != set(SPLITS):
            raise ResultError(f"{field}: expected train/val/test counts")
        for split in SPLITS:
            _integer(values[split], f"{field}.{split}", counts[split])
    _integer(result.get("parameters"), "parameters")
    epochs = _integer(result.get("epochs_ran"), "epochs_ran")
    best = _integer(result.get("best_epoch"), "best_epoch")
    if not best <= epochs <= get_model_config(run.model).epochs:
        raise ResultError("expected 1 <= best_epoch <= epochs_ran <= configured epochs")
    for split, field in (("val", "validation"), ("test", "test")):
        _metrics(result.get(field), field, counts[split], result["split_segments"][split])
    if "train" in result:
        _metrics(result["train"], "train", counts["train"], result["split_segments"]["train"])
    for field, expected in {"sample_rate_hz": 64, "window_seconds": 5, "candidates": 5,
                            "speech_features": ["wav2vec_l14_pca64_64Hz", "mel10_64Hz"]}.items():
        if field in result:
            _expect(result, field, expected)
    if names is not None and len(names["test"]) != len(result["test"]["per_subject_accuracy"]):
        raise ResultError("test subject mapping and metrics have different lengths")
    return result


def subject_values(result: dict, split: str = "test", subject_map: dict | None = None) -> dict[str, float]:
    """Percent accuracies keyed by actual IDs, never by inferred index IDs."""
    if split not in SPLITS:
        raise ResultError(f"unknown split {split!r}")
    if not isinstance(result, dict):
        raise ResultError("result must be a dictionary")
    result = _overlay(result, subject_map)
    mapping = result.get("split_unit_names")
    names = mapping.get(split) if isinstance(mapping, dict) else None
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or names != sorted(names)):
        raise ResultError(f"split_unit_names.{split}: explicit ordered IDs are required")
    metrics = result.get("validation" if split == "val" else split)
    if not isinstance(metrics, dict):
        raise ResultError(f"{split}: no metrics available")
    _metrics(metrics, split, len(names), _integer(metrics.get("segments"), f"{split}.segments"))
    return {name: 100 * metrics["per_subject_accuracy"][str(index)] for index, name in enumerate(names)}


def _scientific(result: dict) -> dict:
    fields = (*get_model_config(result["name"]).to_dict(), "dataset", "seed", "split_seed", "protocol",
              "architecture", "split_unit_names", "split_files", "split_segments", "candidate_seeds",
              "validation", "test", "train", "parameters", "best_epoch", "epochs_ran",
              "sample_rate_hz", "window_seconds", "candidates", "speech_features", "training",
              "optimizer", "gradient_clip_norm", "selection_metric", "selection_tolerance",
              "selection_tiebreak", "cudnn_deterministic", "cudnn_benchmark", "gpu_speech_bank")
    return {field: result[field] for field in fields if field in result}


def load_run(root: Path, run: Run, subject_map: dict | None = None,
             require_subjects: bool = True) -> dict:
    """Load canonical or alias JSON; absence is FileNotFoundError, invalidity ResultError."""
    root = Path(root)
    paths = run_paths(root, run)
    found: list[tuple[Path, dict]] = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            entry = None if subject_map is None else subject_map.get(path.relative_to(root).as_posix())
            result = validate_result(_json(path), canonical_run(run), entry, require_subjects)
        except ResultError as error:
            raise ResultError(f"{path}: {error}") from error
        found.append((path, result))
    if not found:
        raise FileNotFoundError("missing result; tried " + ", ".join(str(path) for path in paths))
    first_path, first = found[0]
    for path, result in found[1:]:
        if not _equal(_scientific(first), _scientific(result)):
            raise ResultError(f"conflicting canonical/alias results: {first_path} and {path}")
    return first


def preflight_runs(root: Path, runs: list[Run], subject_map: dict | None = None,
                   require_subjects: bool = True) -> tuple[dict[Run, dict], list[Run]]:
    """Load every selected physical run before output; missing is never invalid."""
    loaded: dict[Run, dict] = {}
    missing: list[Run] = []
    for run in dict.fromkeys(canonical_run(run) for run in runs):
        try:
            loaded[run] = load_run(root, run, subject_map, require_subjects)
        except FileNotFoundError:
            missing.append(run)
    validate_subject_spellings(loaded.items())
    return loaded, missing
