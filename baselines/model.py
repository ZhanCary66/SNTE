"""Registry of the baseline models compared in the paper.

``create_model`` in :mod:`model` forwards any name other than ``snte`` here.
All baselines share :class:`~baselines.common.LatentCosineMatcher` and differ
only in their neural encoder, so the comparison isolates the neural encoding
architecture.
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

    Every baseline matches through the shared latent cosine head; only the
    neural encoder differs:

    - ``cca``: linear 33-tap temporal filter;
    - ``convconcatnet``: spatial projection, repeated concat-convolutions and
      refinement iterations;
    - ``vlaai``: VLAAI extractor plus context convolutions;
    - ``eeg2vec``: convolutional EEG backbone with window standardization;
    - ``brainmagic``: BrainMagic convolutional backbone.
    """
    from model import SNTEConfig

    config = config or SNTEConfig()
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
