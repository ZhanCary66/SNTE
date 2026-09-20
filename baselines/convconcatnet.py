"""ConvConcatNet baseline, adapted to the frozen five-way task.

Keeps the shared body of IEEEtrans ConvCatNetSameLarge and drops the
subject-specific matrix: the input is first mapped to 64 channels, then passed
through the shared body, and finally matched against the candidates with the
same latent cosine matcher as every other baseline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import LatentCosineMatcher, Spatial64


class ConvConcatExtractor(nn.Module):
    """The ExtractorSame topology of IEEEtrans ConvCatNetSameLarge.

    Four rounds of "spatial 1x1 convolution -> concatenate with the original
    input -> grouped depthwise convolution", followed by a final convolution to
    256 channels, alternating spatial and temporal feature extraction.
    """

    def __init__(self, input_dim: int = 384, kernel_size: int = 25, width: int = 128) -> None:
        super().__init__()
        self.width = width
        self.spatial = nn.ModuleList()
        self.temporal = nn.ModuleList()
        for _ in range(4):
            # The spatial branch takes input_dim on the first round and the
            # concatenated input afterwards.
            spatial_input = input_dim if not self.spatial else input_dim + width
            self.spatial.append(nn.Conv1d(spatial_input, width, 1))
            # The temporal branch is a grouped depthwise convolution over the
            # concatenated features.
            self.temporal.append(
                nn.Conv1d(
                    input_dim + width,
                    input_dim + width,
                    kernel_size,
                    groups=input_dim + width,
                    padding="same",
                )
            )
        self.final = nn.Conv1d(input_dim + width, 2 * width, kernel_size, padding="same")
        self.norm_w = nn.LayerNorm(width)
        self.norm_wide = nn.LayerNorm(input_dim + width)
        self.norm_out = nn.LayerNorm(2 * width)

    def forward(self, value: Tensor) -> Tensor:
        original = value
        for spatial, temporal in zip(self.spatial, self.temporal):
            value = F.leaky_relu(
                self.norm_w(spatial(value.transpose(1, 2)).transpose(1, 2))
            )
            value = torch.cat((original, value), dim=-1)
            value = F.leaky_relu(
                self.norm_wide(temporal(value.transpose(1, 2)).transpose(1, 2))
            )
        return F.leaky_relu(
            self.norm_out(self.final(value.transpose(1, 2)).transpose(1, 2))
        )


class ConvConcatNetDecoder(nn.Module):
    """The full ConvCatNetSameLarge shared body (without the subject matrix).

    Six refinement iterations: the base features, the current value and the
    attention-weighted value are concatenated and passed through the extractor,
    then refined by a context convolution and a self-attention gate.
    """

    def __init__(
        self, input_dim: int, output_dim: int, iterations: int = 6, width: int = 128
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.pre_linear = nn.Linear(input_dim, width)
        self.extractor = ConvConcatExtractor(3 * width, 25, width)
        self.reduce = nn.Linear(2 * width, width)
        # Context convolution: a 49-step receptive field (~0.77 s).
        self.context = nn.Conv1d(width, width, 49, padding="same")
        self.context_norm = nn.LayerNorm(width)
        # Self-attention gate: value * attention(value).
        self.attention = nn.Sequential(
            nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, width)
        )
        self.final = nn.Linear(width, output_dim)

    def forward(self, signal: Tensor) -> Tensor:
        base = self.pre_linear(signal)
        value = torch.zeros_like(base)
        attended = torch.zeros_like(base)
        for _ in range(self.iterations):
            value = self.reduce(
                self.extractor(torch.cat((base, value, attended), dim=-1))
            )
            value = self.context(value.transpose(1, 2)).transpose(1, 2)
            value = F.leaky_relu(self.context_norm(value))
            attended = self.attention(value) * value
        return self.final(value)


def create_model(
    neural_channels: int,
    speech_channels: int,
    embed_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    # Spatial64(standardize=True) handles window standardization and adapts the
    # native sensors; the shared body takes 64 channels and outputs embed_dim.
    neural_body = nn.Sequential(
        Spatial64(neural_channels, standardize=True),
        ConvConcatNetDecoder(64, embed_dim, iterations=6, width=128),
    )
    return LatentCosineMatcher(neural_body, speech_channels, embed_dim, dropout)
