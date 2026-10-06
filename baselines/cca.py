"""CCA-style temporal-filtering adapter for the five-way matching task.

The neural encoder is a 33-tap linear convolution. It is paired with the
learned speech encoder and learned-temperature, time-mean cosine matcher used
by the other adapted baselines. Only the neural encoder is linear; this module
does not solve the classical CCA optimization problem.
"""

from __future__ import annotations

from torch import Tensor, nn

from .common import LatentCosineMatcher


class CCATemporalEncoder(nn.Module):
    """CCA-style linear temporal encoder: a 33-tap Conv1d.

    Produces a per-time-step linear map of shape [B, T, output_dim].
    """

    def __init__(self, neural_dim: int, output_dim: int) -> None:
        super().__init__()
        # 33 time steps (~0.5 s) of temporal filtering; padding=16 preserves length.
        self.filter = nn.Conv1d(neural_dim, output_dim, 33, padding=16, bias=False)

    def forward(self, signal: Tensor) -> Tensor:
        return self.filter(signal.transpose(1, 2)).transpose(1, 2)


def create_model(
    neural_channels: int,
    speech_channels: int,
    embed_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    body = CCATemporalEncoder(neural_channels, embed_dim)
    return LatentCosineMatcher(body, speech_channels, embed_dim, dropout)
