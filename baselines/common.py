"""Adapters and the shared match head used by every baseline.

All baselines share the same matching mechanism, :class:`LatentCosineMatcher`:

- ``neural_body``: the baseline's own neural encoder, mapping the neural
  window to an ``embed_dim`` latent space;
- ``speech_encoder``: the same :class:`~model.DilatedEncoder` that SNTE uses,
  encoding the 74-dimensional speech features into the same space;
- the cosine similarity between the two modalities is computed at every time
  step and averaged over time to produce the score.

This keeps the comparison fair: the matching mechanism is identical across all
baselines, so the only difference is the neural encoding architecture.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model import DilatedEncoder


class Spatial64(nn.Module):
    """Map the dataset's native sensors onto 64 channels.

    ``standardize=True``: window-level z-score, then a linear projection to 64
    dimensions, then LayerNorm. ``standardize=False``: identity when the input
    already has 64 channels, otherwise a 1x1 convolution.
    """

    def __init__(self, input_channels: int, standardize: bool = False) -> None:
        super().__init__()
        self.use_standardization = standardize
        if standardize:
            self.projection: nn.Module = nn.Linear(input_channels, 64)
            self.normalization: nn.Module = nn.LayerNorm(64)
            self.linear_layout = True
        else:
            self.projection = (
                nn.Identity() if input_channels == 64 else nn.Conv1d(input_channels, 64, 1)
            )
            self.normalization = nn.Identity()
            self.linear_layout = False

    @staticmethod
    def window_standardize(signal: Tensor) -> Tensor:
        """Window-level z-score along the time axis."""
        value = signal.float()
        mean = value.mean(dim=1, keepdim=True)
        std = value.std(dim=1, keepdim=True, unbiased=False)
        return (value - mean) / std.clamp_min(1e-8)

    def forward(self, signal: Tensor) -> Tensor:
        if self.use_standardization:
            return self.normalization(self.projection(self.window_standardize(signal)))
        if self.linear_layout:
            return self.normalization(self.projection(signal))
        return self.projection(signal.transpose(1, 2)).transpose(1, 2)


class LatentCosineMatcher(nn.Module):
    """Shared baseline match head: learned latent space + per-step cosine.

    - ``neural_body`` maps [B, T, C_neural] to [B, T, E];
    - ``speech_encoder`` maps the 74-dimensional candidates to [B, N, T, E];
    - both are L2 normalized along the feature dimension at every time step,
      and their dot product (the cosine similarity) is averaged over time and
      scaled by a learned temperature.
    """

    def __init__(
        self,
        neural_body: nn.Module,
        speech_channels: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.neural_body = neural_body
        # The speech encoder matches SNTE's architecture to keep the
        # comparison fair.
        self.speech_encoder = DilatedEncoder(speech_channels, embed_dim, dropout)
        # Learned temperature (initialized to e^ln10 ~ 10) scaling the cosine.
        self.temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, neural: Tensor, candidates: Tensor) -> Tensor:
        """Inputs:
        - neural: [B, T, C_neural] neural windows;
        - candidates: [B, N, T, 74] features of the N candidates.
        Output: matching scores [B, N].
        """
        batch, count, time, channels = candidates.shape
        neural_feat = self.neural_body(neural)                        # [B, T, E]
        # All candidates share the speech encoder.
        speech_feat = self.speech_encoder(
            candidates.reshape(batch * count, time, channels)
        )                                                             # [B*N, T, E]
        speech_feat = speech_feat.reshape(batch, count, time, -1)     # [B, N, T, E]
        neural_feat = F.normalize(neural_feat, dim=-1).unsqueeze(1)   # [B, 1, T, E]
        speech_feat = F.normalize(speech_feat, dim=-1)                # [B, N, T, E]
        cosine = (neural_feat * speech_feat).sum(dim=-1)              # [B, N, T]
        return cosine.mean(dim=-1) * self.temperature.exp().clamp(max=100)
