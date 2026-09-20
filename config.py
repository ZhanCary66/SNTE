"""Final training configuration for SNTE and the five baselines.

Every value here is the configuration that produced the numbers reported in
the paper. The baselines are deliberately tuned toward their natural operating
point: they share SNTE's speech features and, except for CCA, the same neural
encoder search space, so the comparison isolates the match head and encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Training configuration for one model.

    - name: value accepted by ``--model``;
    - display_name: name used in the paper's tables;
    - family: ``proposed`` or ``baseline``;
    - embed_dim: latent dimension of the shared representation space;
    - dropout: dropout rate;
    - learning_rate / weight_decay: AdamW hyperparameters;
    - epochs: maximum number of epochs;
    - batch_size / evaluation_batch_size: training / evaluation batch size;
    - patience: early-stopping patience in epochs without validation improvement.
    """

    name: str
    display_name: str
    family: str
    embed_dim: int
    dropout: float
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 50
    batch_size: int = 64
    evaluation_batch_size: int = 1024
    patience: int = 10

    def to_dict(self) -> dict:
        """Plain dictionary, written into the result JSON for traceability."""
        return asdict(self)


MODEL_CONFIGS = {
    "snte": ModelConfig(
        "snte", "SNTE (proposed)", "proposed", embed_dim=256, dropout=0.5
    ),
    "cca": ModelConfig(
        "cca", "CCA", "baseline", embed_dim=32, dropout=0.0
    ),
    "convconcatnet": ModelConfig(
        "convconcatnet", "ConvConcatNet", "baseline",
        embed_dim=32, dropout=0.4, learning_rate=4e-4,
    ),
    "vlaai": ModelConfig(
        "vlaai", "VLAAI", "baseline",
        embed_dim=32, dropout=0.4, weight_decay=0.1,
    ),
    "eeg2vec": ModelConfig(
        "eeg2vec", "Eeg2Vec", "baseline",
        embed_dim=64, dropout=0.2, learning_rate=3e-4,
    ),
    "brainmagic": ModelConfig(
        "brainmagic", "BrainMagic", "baseline", embed_dim=64, dropout=0.2
    ),
}

# Model names accepted by --model.
MODEL_NAMES = tuple(MODEL_CONFIGS)


def get_model_config(name: str) -> ModelConfig:
    """Look up a model configuration by name."""
    try:
        return MODEL_CONFIGS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}") from None
