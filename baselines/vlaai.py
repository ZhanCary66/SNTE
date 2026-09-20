"""VLAAI baseline: convolutional extractor plus context convolutions.

Based on the VLAAIDecoder, adapted to serve as the neural encoder of the
shared latent cosine matcher: four rounds of "extractor (five dilated
convolutions) + context convolution", with a residual connection to the base
branch and GroupNorm for stable training.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .common import LatentCosineMatcher, Spatial64


class VlaaiExtractor(nn.Module):
    """VLAAI extractor: five 8-tap convolutions, widths 256/256/256/128/128."""

    def __init__(self, dim: int, width_scale: float = 1.0) -> None:
        super().__init__()
        widths = tuple(int(w * width_scale) for w in (256, 256, 256, 128, 128))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        previous = dim
        for width in widths:
            self.convs.append(nn.Conv1d(previous, width, 8))
            self.norms.append(nn.GroupNorm(1, width))
            previous = width

    def forward(self, value: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            value = F.pad(conv(value), (0, 7))
            value = F.leaky_relu(norm(value), 0.1)
        return value


class VLAAISpatial64Encoder(nn.Module):
    """Map to 64 channels, then encode through the VLAAI extractor.

    Outputs per-time-step latent features of shape [B, T, output_dim].
    """

    def __init__(
        self, input_channels: int, output_dim: int, rounds: int = 4, width_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.rounds = rounds
        # Window-level z-score plus a linear projection to 64 channels, which
        # is what makes the tiny-scale MEG recordings usable.
        self.spatial64 = Spatial64(input_channels, standardize=True)
        self.input = nn.Conv1d(64, output_dim, 1)
        self.extractor = VlaaiExtractor(output_dim, width_scale)
        self.reduce = nn.Conv1d(int(128 * width_scale), output_dim, 1)
        # Context convolution: a 32-step receptive field with causal padding.
        self.context = nn.Conv1d(output_dim, output_dim, 32)
        self.norm = nn.GroupNorm(1, output_dim)

    def forward(self, signal: Tensor) -> Tensor:
        base = self.input(self.spatial64(signal).transpose(1, 2))
        value = base
        for _ in range(self.rounds):
            value = self.reduce(self.extractor(value + base))
            value = F.leaky_relu(self.norm(self.context(F.pad(value, (31, 0)))), 0.1)
        return value.transpose(1, 2)


def create_model(
    neural_channels: int,
    speech_channels: int,
    embed_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    body = VLAAISpatial64Encoder(neural_channels, embed_dim, rounds=4, width_scale=1.0)
    return LatentCosineMatcher(body, speech_channels, embed_dim, dropout)
