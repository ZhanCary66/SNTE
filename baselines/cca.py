"""Linear temporal-filtering baseline (CCA).

CCA is essentially a linear projection. Adapted to the shared matching
mechanism, it keeps its 33-tap linear temporal filter as the neural encoder:
the filter maps the neural signal into the latent space, and the resulting
features are matched against the candidates with the same latent cosine
matcher as every other baseline. The baseline thus stays linear (no
non-linearity), while the matching mechanism is identical across baselines.
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
