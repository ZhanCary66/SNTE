"""Registry of the baseline models compared in the paper.

``create_model`` in :mod:`model` forwards any name other than ``snte`` here.
All baselines share :class:`~baselines.common.LatentCosineMatcher`. Their
neural encoders and training configurations differ; CCA also omits the window
standardization used by the other four neural adapters.
"""

from __future__ import annotations

from torch import nn

from . import brainmagic, cca, convconcatnet, eeg2vec, vlaai

# Baseline names accepted by --model.
BASELINE_NAMES = (
    "cca",
    "convconcatnet",
    "vlaai",
    "eeg2vec",
    "brainmagic",
)


def create_baseline(
    name: str,
    neural_channels: int,
    speech_channels: int = 74,
    config=None,
) -> nn.Module:
    """Build one baseline by name.

    Every baseline matches through the shared learned-temperature, time-mean
    cosine head. Neural adapters and model-specific configurations differ:

    - ``cca``: linear 33-tap temporal filter;
    - ``convconcatnet``: spatial projection, repeated concat-convolutions and
      refinement iterations;
    - ``vlaai``: VLAAI extractor plus context convolutions;
    - ``eeg2vec``: window-standardized Transformer/Conformer neural encoder;
    - ``brainmagic``: BrainMagic convolutional backbone.
    """
    from dataclasses import fields

    from config import get_model_config
    from model import SNTEConfig

    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
    if config is None:
        final = get_model_config(name)
        config = SNTEConfig(embed_dim=final.embed_dim, dropout=final.dropout)
    defaults = SNTEConfig()
    unsupported = [field.name for field in fields(SNTEConfig)
                   if field.name not in ("embed_dim", "dropout")
                   and getattr(config, field.name) != getattr(defaults, field.name)]
    if unsupported:
        raise ValueError(f"{name}: SNTE-only overrides are unsupported: {', '.join(unsupported)}")
    builders = {
        "cca": cca.create_model,
        "convconcatnet": convconcatnet.create_model,
        "vlaai": vlaai.create_model,
        "eeg2vec": eeg2vec.create_model,
        "brainmagic": brainmagic.create_model,
    }
    try:
        builder = builders[name]
    except KeyError:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}") from None
    return builder(neural_channels, speech_channels, config.embed_dim, config.dropout)
