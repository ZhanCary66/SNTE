#!/usr/bin/env python3
"""Training entry point for SNTE and the paper's baselines.

Pipeline:
1. parse arguments, look up the model configuration, validate the data layout;
2. set random seeds and resolve the device (CUDA is required for real runs);
3. build the train/validation/test datasets and move each speech bank to the GPU;
4. create the model and an AdamW optimizer, then train with early stopping on
   validation subject-macro accuracy;
5. reload the best checkpoint, evaluate once on validation and test, and write
   the metrics to JSON (plus a .pt checkpoint).

Model selection uses the validation split only; the test split is evaluated
once, after the checkpoint has been chosen, so the reported numbers are not
selected on the test set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
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

ENCODER_CHOICES = (ENCODER_DILATED, ENCODER_LINEAR)
HEAD_CHOICES = (HEAD_PER_STATISTIC, HEAD_CONCAT, HEAD_TIME_MEAN)


def _dilations(value: str) -> tuple[int, ...]:
    """Parse a comma-separated dilation sequence, e.g. "1,3,9"."""
    return tuple(int(x) for x in value.split(","))


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
        type=int,
        default=None,
        help="random cross-subject re-partition to use; omitted = the paper's fixed split",
    )
    parser.add_argument("--data-root", type=Path, default=SHARED_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    This matches the throughput settings used for the reported experiments:
    the seeds are fixed, but cuDNN determinism is disabled and benchmark mode
    is on so that the fastest convolution algorithms are selected. Results are
    reproducible from a given seed on a given machine.
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


def resolve_device(requested: str, integration: bool) -> torch.device:
    """Resolve the torch device; real training requires a CUDA GPU."""
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if not integration and device.type != "cuda":
        raise RuntimeError("formal training requires a CUDA GPU")
    return device


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
            # Gather the five candidates from the GPU speech bank:
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


def atomic_json(payload: dict, path: Path) -> None:
    """Write JSON atomically: temporary file, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(payload: dict, path: Path) -> None:
    """Save a PyTorch checkpoint atomically (weights + config + result)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_config(args: argparse.Namespace) -> SNTEConfig:
    """Combine the model configuration with the architecture ablation flags."""
    model_config = get_model_config(args.model)
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


def main() -> None:
    args = parse_args()
    model_config = get_model_config(args.model)
    # Validate the data layout; --check-data prints the report and exits.
    contract = validate_shared_contract(args.data_root)
    if args.check_data:
        print(json.dumps(contract[args.dataset], indent=2, sort_keys=True))
        return

    set_seed(args.seed)
    device = resolve_device(args.device, args.integration)
    epochs = model_config.epochs
    batch_size = model_config.batch_size
    evaluation_batch_size = model_config.evaluation_batch_size
    patience = model_config.patience
    workers = args.workers
    # Smoke-test mode: shrink the run so the pipeline can be exercised quickly.
    if args.integration:
        epochs = 1
        batch_size = min(batch_size, 2)
        evaluation_batch_size = min(evaluation_batch_size, 2)
        workers = 0

    output = args.output or Path(
        f"results/{args.model}/{args.dataset}_seed{args.seed}.json"
    )
    checkpoint = output.with_suffix(".pt")
    snte_config = build_config(args)
    # Full run configuration, written into the result JSON for traceability.
    configuration = {
        **model_config.to_dict(),
        "architecture": {
            "head": snte_config.head,
            "standardize": snte_config.standardize,
            "tied_encoder": snte_config.tied_encoder,
            "neural_encoder": snte_config.neural_encoder,
            "speech_encoder": snte_config.speech_encoder,
            "neural_dilations": list(snte_config.neural_dilations),
            "speech_dilations": list(snte_config.speech_dilations),
            "scales": list(snte_config.scales),
            "shifts": list(snte_config.shifts),
            "stats": list(snte_config.stats),
        },
        "dataset": args.dataset,
        "seed": args.seed,
        "protocol": PROTOCOL_VERSION,
        "split_seed": args.split_seed,
        "data_root": str(args.data_root),
        "speech_features": ["wav2vec_l14_pca64_64Hz", "mel10_64Hz"],
        "sample_rate_hz": 64,
        "window_seconds": 5,
        "candidates": N_CANDIDATES,
        "workers": workers,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "gpu_speech_bank": True,
        "device": str(device),
        "integration": args.integration,
    }
    print(json.dumps(configuration, indent=2, sort_keys=True), flush=True)

    # Build the three splits and move each speech bank to the GPU.
    datasets = build_splits(
        args.dataset,
        integration=args.integration,
        root=args.data_root,
        split_seed=args.split_seed,
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
    result = {
        **configuration,
        "parameters": parameters,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": time.time() - start_time,
        "split_files": {key: len(value.neural) for key, value in datasets.items()},
        "split_segments": {key: len(value) for key, value in datasets.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "checkpoint": str(checkpoint),
        "status": "complete",
    }
    atomic_checkpoint(
        {"model_state_dict": best_state, "configuration": configuration, "result": result},
        checkpoint,
    )
    atomic_json(result, output)
    print(
        f"RESULT val_subject_macro={validation_metrics['subject_macro_accuracy']:.6f} "
        f"test_subject_macro={test_metrics['subject_macro_accuracy']:.6f} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
