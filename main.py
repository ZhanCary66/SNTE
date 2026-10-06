#!/usr/bin/env python3
"""Training entry point for SNTE and the paper's baselines.

Pipeline:
1. parse configuration and validate the device and output paths;
2. acquire the output lock and check the selected dataset's layout;
3. set seeds, build subject splits and move speech banks to the selected device;
4. create the model and AdamW optimizer, then select checkpoints using
   validation subject-macro accuracy and validation loss;
5. reload the selected state, evaluate validation and test, and save JSON and
   a separate .pt checkpoint. Formal runs require CUDA; integration checks may
   use CPU.

Within one invocation, checkpoint selection uses validation and the final test
split is evaluated after that checkpoint is chosen. This describes this
training loop, not the historical selection of configurations or seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import MODEL_NAMES, get_model_config
from dataset import (
    DATASETS,
    NEURAL_CHANNELS,
    N_CANDIDATES,
    PROTOCOL_VERSION,
    SEGMENT_SAMPLES,
    SHARED_ROOT,
    SPEECH_DIM,
    build_splits,
    validate_shared_contract,
)
from model import (
    ENCODER_DILATED,
    ENCODER_LINEAR,
    HEAD_CONCAT,
    HEAD_PER_STATISTIC,
    HEAD_TIME_MEAN,
    SNTEConfig,
    create_model,
)
from results_io import SCHEMA_VERSION, effective_architecture

ENCODER_CHOICES = (ENCODER_DILATED, ENCODER_LINEAR)
HEAD_CHOICES = (HEAD_PER_STATISTIC, HEAD_CONCAT, HEAD_TIME_MEAN)


def _dilations(value: str) -> tuple[int, ...]:
    """Parse one of the two dilation sequences supported by the paper runs."""
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("dilations must be 1,3,9 or 1,1,1") from None
    if result not in ((1, 3, 9), (1, 1, 1)):
        raise argparse.ArgumentTypeError("dilations must be 1,3,9 or 1,1,1 (positive, nonempty)")
    return result


def _nonnegative(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from None
    if result < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The architecture flags below reproduce the paper's ablations: the match
    head (Table 2), and window normalization, encoder type, tied parameters and
    dilation rates (Table 3). Their defaults are the paper's main model.
    """
    parser = argparse.ArgumentParser(
        description="Train one cross-subject five-way match--mismatch model."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="snte")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--split-seed",
        type=_nonnegative,
        default=None,
        help="random cross-subject re-partition to use; omitted = the paper's fixed split",
    )
    parser.add_argument("--data-root", type=Path, default=SHARED_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=_nonnegative, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--check-data", action="store_true")

    # --- match head (Table 2) ---
    parser.add_argument("--head", choices=HEAD_CHOICES, default=HEAD_PER_STATISTIC)
    # --- encoder ablations (Table 3) ---
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="disable the window-level z-score applied to the neural input",
    )
    parser.add_argument("--neural-encoder", choices=ENCODER_CHOICES, default=ENCODER_DILATED)
    parser.add_argument("--speech-encoder", choices=ENCODER_CHOICES, default=ENCODER_DILATED)
    parser.add_argument(
        "--tied-encoder",
        action="store_true",
        help="share the convolutional trunk between the two encoders",
    )
    parser.add_argument("--neural-dilations", type=_dilations, default=(1, 3, 9))
    parser.add_argument("--speech-dilations", type=_dilations, default=(1, 3, 9))
    args = parser.parse_args()
    try:
        build_config(args)
        if args.device not in ("auto", "cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", args.device):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N (nonnegative index)")
        if args.output is not None and args.output.suffix != ".json":
            raise ValueError("--output must have a .json suffix; its checkpoint uses .pt")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    cuDNN determinism is disabled and benchmark mode is enabled. Fixed seeds
    do not guarantee deterministic execution or identical numerical results
    across environments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def _native_bf16_supported(index: int) -> bool:
    """Exclude emulation, including on Torch versions with a no-argument API."""
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError as error:
        if "including_emulation" not in str(error):
            raise
        if torch.version.hip is not None:
            return torch.cuda.is_bf16_supported()
        # Older APIs may probe emulation; check native CUDA hardware first.
        cuda_version = torch.version.cuda
        return (cuda_version is not None and int(cuda_version.split(".")[0]) >= 11
                and torch.cuda.get_device_capability(index)[0] >= 8
                and torch.cuda.is_bf16_supported())


def resolve_device(requested: str, integration: bool) -> torch.device:
    """Validate the requested index before Torch can narrow it to a device index."""
    if requested not in ("auto", "cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", requested):
        raise ValueError("device must be auto, cpu, cuda, or cuda:N (nonnegative index)")
    name = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if not integration and name == "cpu":
        raise RuntimeError("formal training requires a CUDA GPU; use --integration for a CPU smoke test")
    if name != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {requested}, but CUDA is not available")
        index = int(name.split(":", 1)[1]) if ":" in name else torch.cuda.current_device()
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise ValueError(f"CUDA device index {index} is unavailable; detected {count} device(s)")
        # Do not replace the existing autocast precision on unsupported hardware.
        with torch.cuda.device(index):
            if not _native_bf16_supported(index):
                raise RuntimeError(f"cuda:{index} lacks native bfloat16 support required by CUDA autocast")
    return torch.device(name)


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a DataLoader with a seeded sampler so shuffling is reproducible."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    speech_bank: Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Run one full epoch, training when an optimizer is given.

    Returns the loss, segment accuracy, subject-macro accuracy, the
    per-subject accuracies and the number of segments.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    # Per-subject hit counts, used for the subject-macro accuracy.
    unit_correct: dict[int, int] = defaultdict(int)
    unit_total: dict[int, int] = defaultdict(int)
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for neural, candidate_indices, labels, unit_indices in data_loader:
            neural = neural.to(device, non_blocking=True)
            candidate_indices = candidate_indices.to(device, non_blocking=True)
            # Gather the five candidates from the device-resident speech bank:
            # [B*N] -> [B, N, T, C].
            candidates = speech_bank.index_select(0, candidate_indices.flatten()).view(
                len(neural), N_CANDIDATES, SEGMENT_SAMPLES, SPEECH_DIM
            )
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # bfloat16 autocast on GPU for throughput.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(neural, candidates)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {float(loss)}")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            predictions = logits.argmax(dim=1)
            hits = predictions.eq(labels)
            size = labels.numel()
            total_loss += float(loss.detach()) * size
            correct += int(hits.sum())
            total += size
            for unit, hit in zip(unit_indices.tolist(), hits.detach().cpu().tolist()):
                unit_correct[int(unit)] += int(hit)
                unit_total[int(unit)] += 1

    if total == 0:
        raise RuntimeError("empty epoch")
    # Per-subject accuracy, averaged into the subject-macro accuracy. The keys
    # are positional subject indices within this split, not subject IDs.
    per_subject = {
        str(unit): unit_correct[unit] / unit_total[unit] for unit in sorted(unit_total)
    }
    return {
        "loss": total_loss / total,
        "segment_accuracy": correct / total,
        "subject_macro_accuracy": float(np.mean(list(per_subject.values()))),
        "per_subject_accuracy": per_subject,
        "segments": total,
    }


@contextmanager
def _temporary_output(path: Path):
    """Allocate a unique sibling and clean it on both success and failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict, path: Path) -> None:
    """Strict finite JSON, atomically published as the job completion marker."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    with _temporary_output(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)


def atomic_checkpoint(payload: dict, path: Path) -> None:
    """Save a PyTorch checkpoint atomically; failed saves leave no temporary."""
    with _temporary_output(path) as temporary:
        torch.save(payload, temporary)
        os.replace(temporary, path)


def default_output(args: argparse.Namespace, config: SNTEConfig) -> Path:
    """Separate modes, splits, seeds and supported architecture variants by default."""
    architecture = effective_architecture(args.model, config)
    split = "fixed" if args.split_seed is None else f"split{args.split_seed}"
    dilations = lambda values: "na" if values is None else "-".join(map(str, values))
    variant = (
        f"{split}_head-{architecture['head']}_norm-{int(architecture['standardize'])}"
        f"_ne-{architecture['neural_encoder']}_se-{architecture['speech_encoder']}"
        f"_tied-{int(architecture['tied_encoder'])}"
        f"_nd-{dilations(architecture['neural_dilations'])}"
        f"_sd-{dilations(architecture['speech_dilations'])}"
    )
    mode = "integration" if args.integration else "formal"
    return Path("results") / mode / args.model / variant / f"{args.dataset}_seed{args.seed}.json"


def output_paths(args: argparse.Namespace, config: SNTEConfig) -> tuple[Path, Path]:
    requested = args.output or default_output(args, config)
    if requested.suffix != ".json":
        raise ValueError("--output must have a .json suffix; its checkpoint uses .pt")
    if requested.is_symlink():
        raise FileExistsError(f"refusing existing result symlink: {requested}")
    output = requested.resolve()
    checkpoint = output.with_suffix(".pt")
    if checkpoint.is_symlink():
        raise FileExistsError(f"refusing existing checkpoint symlink: {checkpoint}")
    checkpoint = checkpoint.resolve()
    if output == checkpoint:
        raise ValueError("result JSON and checkpoint must have distinct paths")
    return output, checkpoint


def _reject_existing(output: Path, checkpoint: Path) -> None:
    for path in (output, checkpoint):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new --output .json path")


@contextmanager
def output_lock(output: Path, checkpoint: Path):
    """Hold an exclusive, one-writer lock for the complete training invocation."""
    _reject_existing(output, checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError(f"output is locked by another job: {lock}") from None
    owned = os.fstat(descriptor)
    try:
        # An earlier writer may have completed between precheck and acquisition.
        _reject_existing(output, checkpoint)
        yield
    finally:
        os.close(descriptor)
        try:
            current = lock.lstat()
        except FileNotFoundError:
            pass
        else:
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                lock.unlink()


def build_config(args: argparse.Namespace) -> SNTEConfig:
    """Combine the model configuration with the architecture ablation flags."""
    model_config = get_model_config(args.model)
    if args.model != "snte" and (
        args.head != HEAD_PER_STATISTIC or args.no_standardize or args.tied_encoder
        or args.neural_encoder != ENCODER_DILATED or args.speech_encoder != ENCODER_DILATED
        or tuple(args.neural_dilations) != (1, 3, 9) or tuple(args.speech_dilations) != (1, 3, 9)
    ):
        raise ValueError("SNTE architecture overrides are not supported for baseline models")
    if args.tied_encoder and (
        args.neural_encoder != ENCODER_DILATED or args.speech_encoder != ENCODER_DILATED
        or tuple(args.neural_dilations) != tuple(args.speech_dilations)
    ):
        raise ValueError("--tied-encoder requires two dilated encoders with identical dilations")
    return SNTEConfig(
        embed_dim=model_config.embed_dim,
        dropout=model_config.dropout,
        neural_dilations=tuple(args.neural_dilations),
        speech_dilations=tuple(args.speech_dilations),
        head=args.head,
        standardize=not args.no_standardize,
        tied_encoder=args.tied_encoder,
        neural_encoder=args.neural_encoder,
        speech_encoder=args.speech_encoder,
    )


def build_configuration(args: argparse.Namespace, config: SNTEConfig, device: torch.device) -> dict:
    """Record effective architecture, actual training limits and runtime environment."""
    model_config = get_model_config(args.model)
    configuration = {
        **model_config.to_dict(),
        "schema_version": SCHEMA_VERSION,
        "architecture": effective_architecture(args.model, config),
        "dataset": args.dataset,
        "seed": args.seed,
        "protocol": PROTOCOL_VERSION,
        "split_seed": args.split_seed,
        "data_root": str(args.data_root),
        "speech_features": ["wav2vec_l14_pca64_64Hz", "mel10_64Hz"],
        "sample_rate_hz": 64,
        "window_seconds": 5,
        "candidates": N_CANDIDATES,
        "workers": 0 if args.integration else args.workers,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "gpu_speech_bank": device.type == "cuda",
        "device": str(device),
        "integration": args.integration,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "precision": "bfloat16 autocast" if device.type == "cuda" else "float32",
            "autocast_enabled": device.type == "cuda",
            "autocast_dtype": "bfloat16" if device.type == "cuda" else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "training": {
            "optimizer": "AdamW",
            "gradient_clip_norm": 5.0,
            "selection_metric": "validation subject-macro accuracy",
            "selection_tolerance": 1e-8,
            "selection_tiebreak": "lower validation loss",
        },
    }
    if args.integration:
        configuration.update(epochs=1, batch_size=min(model_config.batch_size, 2),
                             evaluation_batch_size=min(model_config.evaluation_batch_size, 2))
    return configuration


def publish_result(payload: dict, result: dict, output: Path, checkpoint: Path) -> None:
    """Publish checkpoint first and JSON last; roll back this job's checkpoint on failure."""
    json.dumps(result, allow_nan=False)
    _reject_existing(output, checkpoint)
    atomic_checkpoint(payload, checkpoint)
    try:
        atomic_json(result, output)
    except BaseException:
        checkpoint.unlink(missing_ok=True)
        raise


def train(args: argparse.Namespace, snte_config: SNTEConfig, device: torch.device,
          output: Path, checkpoint: Path) -> None:
    """Train one already-validated invocation while its output lock is held."""
    model_config = get_model_config(args.model)
    set_seed(args.seed)
    configuration = build_configuration(args, snte_config, device)
    epochs = configuration["epochs"]
    batch_size = configuration["batch_size"]
    evaluation_batch_size = configuration["evaluation_batch_size"]
    patience = configuration["patience"]
    workers = configuration["workers"]
    print(json.dumps(configuration, indent=2, sort_keys=True), flush=True)

    # Build the three splits and move each speech bank to the selected device.
    datasets = build_splits(
        args.dataset,
        integration=args.integration,
        root=args.data_root,
        split_seed=args.split_seed,
        require_all_subjects=not args.integration,
    )
    speech_banks = {
        split: dataset.speech_bank.to(device) for split, dataset in datasets.items()
    }
    loaders = {
        "train": make_loader(datasets["train"], batch_size, workers, True, 1000 + args.seed),
        "val": make_loader(datasets["val"], evaluation_batch_size, workers, False, 2000),
        "test": make_loader(datasets["test"], evaluation_batch_size, workers, False, 3000),
    }
    print("segments", {key: len(value) for key, value in datasets.items()}, flush=True)

    model = create_model(
        args.model,
        NEURAL_CHANNELS[args.dataset],
        SPEECH_DIM,
        snte_config,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    print(f"parameters={parameters:,}", flush=True)

    # Early stopping / best-checkpoint state.
    best_state = None
    best_validation = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], speech_banks["train"], device, optimizer
        )
        validation_metrics = run_epoch(model, loaders["val"], speech_banks["val"], device)
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        score = validation_metrics["subject_macro_accuracy"]
        # Better means a higher subject-macro accuracy; an equal score with a
        # lower loss also counts as an improvement.
        improved = score > best_validation + 1e-8 or (
            abs(score - best_validation) <= 1e-8
            and validation_metrics["loss"] < best_validation_loss
        )
        if improved:
            best_validation = score
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train_metrics['subject_macro_accuracy']:.6f} "
            f"val={score:.6f} best={best_validation:.6f} stale={stale}",
            flush=True,
        )
        if not args.integration and stale >= patience:
            print(f"early_stop epoch={epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    # Reload the best checkpoint and evaluate validation and test once.
    model.load_state_dict(best_state)
    validation_metrics = run_epoch(model, loaders["val"], speech_banks["val"], device)
    test_metrics = run_epoch(model, loaders["test"], speech_banks["test"], device)
    for split, metrics in (("val", validation_metrics), ("test", test_metrics)):
        expected_keys = {str(index) for index in range(len(datasets[split].unit_names))}
        if set(metrics["per_subject_accuracy"]) != expected_keys:
            raise RuntimeError(f"{split}: metric subject keys disagree with actual unit_names")
    result = {
        **configuration,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": time.time() - start_time,
        "split_files": {key: len(value.neural) for key, value in datasets.items()},
        "split_segments": {key: len(value) for key, value in datasets.items()},
        "split_unit_names": {key: list(value.unit_names) for key, value in datasets.items()},
        "candidate_seeds": {key: value.candidate_seed for key, value in datasets.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "checkpoint": str(checkpoint),
        "status": "complete",
    }
    publish_result(
        {"model_state_dict": best_state, "configuration": configuration, "result": result},
        result, output, checkpoint,
    )
    print(
        f"RESULT val_subject_macro={validation_metrics['subject_macro_accuracy']:.6f} "
        f"test_subject_macro={test_metrics['subject_macro_accuracy']:.6f} output={output}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    try:
        snte_config = build_config(args)
        if args.check_data:
            contract = validate_shared_contract(args.data_root, datasets=(args.dataset,))
            print(json.dumps(contract[args.dataset], indent=2, sort_keys=True, allow_nan=False))
            return
        output, checkpoint = output_paths(args, snte_config)
        _reject_existing(output, checkpoint)
        device = resolve_device(args.device, args.integration)
        # Claim ownership before expensive data validation; invalid devices write nothing.
        with output_lock(output, checkpoint):
            validate_shared_contract(args.data_root, datasets=(args.dataset,))
            train(args, snte_config, device, output, checkpoint)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
