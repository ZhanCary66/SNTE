"""Eeg2Vec + Conformer baseline.

Based on Eeg2VecConformerMatcher. The original model couples masked-
reconstruction self-supervised pretraining with a speech Conformer; here only
its EEG encoding backbone is kept (linear projection -> two Transformer layers
-> two Conformer blocks) as the neural encoder. The reconstruction head is
dropped, the speech side uses the shared speech encoder, and matching is the
shared per-time-step cosine.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .common import LatentCosineMatcher, Spatial64


class ConformerBlock(nn.Module):
    """Conformer block: FFN + self-attention + convolution module + FFN."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, 15, padding=7, groups=dim)
        self.pointwise2 = nn.Conv1d(dim, dim, 1)
        self.norm4 = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, value: Tensor) -> Tensor:
        value = value + 0.5 * self.ffn1(self.norm1(value))
        normalized = self.norm2(value)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = value + attended
        convolution = self.pointwise1(self.norm3(value).transpose(1, 2))
        convolution = F.glu(convolution, dim=1)
        convolution = self.pointwise2(F.silu(self.depthwise(convolution))).transpose(1, 2)
        value = value + convolution
        value = value + 0.5 * self.ffn2(self.norm4(value))
        return self.final_norm(value)


class Eeg2VecSpatial64Encoder(nn.Module):
    """Map to 64 channels, then encode through the Eeg2Vec EEG backbone.

    Outputs per-time-step latent features of shape [B, T, output_dim].
    """

    def __init__(self, input_channels: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Window-level per-channel z-score + Linear + LayerNorm, the same input
        # stage used by BrainMagic and ConvConcatNet. The MEG recordings are
        # stored at ~1e-11 and can only be used by a deep model after window
        # standardization.
        self.spatial64 = Spatial64(input_channels, standardize=True)
        self.eeg_feature = nn.Sequential(
            nn.Linear(64, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )
        # Two Transformer layers: self-attention over the whole window.
        encoder_layer = nn.TransformerEncoderLayer(
            output_dim, 4, 4 * output_dim, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.eeg_context = nn.TransformerEncoder(encoder_layer, 2)
        # Two Conformer blocks refine the features further.
        self.backbone = nn.Sequential(
            ConformerBlock(output_dim, dropout), ConformerBlock(output_dim, dropout)
        )

    def forward(self, signal: Tensor) -> Tensor:
        value = self.eeg_feature(self.spatial64(signal))
        return self.backbone(self.eeg_context(value))


def create_model(
    neural_channels: int,
    speech_channels: int,
    embed_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    body = Eeg2VecSpatial64Encoder(neural_channels, embed_dim, dropout)
    return LatentCosineMatcher(body, speech_channels, embed_dim, dropout)
