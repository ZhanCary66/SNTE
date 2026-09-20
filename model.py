"""SNTE: Symmetric Neural--speech Temporal Encoding.

Two structurally identical, independently parameterized dilated-convolution
encoders map the neural window and the speech candidates into a shared latent
space. A correlation scorer summarizes each candidate with multi-statistic
descriptors and produces one matching score per candidate (five-way
match--mismatch classification).

The configuration in :class:`SNTEConfig` defaults to the exact setting used for
the paper's main results. The remaining switches reproduce the paper's
ablations (Table 2: match head; Table 3: window normalization, encoder type,
tied parameters, dilation rates); they are plain arguments rather than
environment variables so that every run is fully described by its config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Match heads (Table 2).
HEAD_PER_STATISTIC = "perstat"  # per-statistic projection (SNTE, paper default)
HEAD_CONCAT = "concat"          # concatenate the descriptors, no projection
HEAD_TIME_MEAN = "timecos"      # single scalar time-mean cosine

# Encoder types (Table 3).
ENCODER_DILATED = "dilated"
ENCODER_LINEAR = "linear"


@dataclass(frozen=True)
class SNTEConfig:
    """Full configuration of the SNTE model.

    Defaults are the paper's main configuration. Only ``head``,
    ``standardize``, ``tied_encoder``, ``neural_encoder``, ``speech_encoder``
    and the dilation tuples are varied by the ablations.
    """

    embed_dim: int = 256
    dropout: float = 0.5
    # Symmetric encoders: identical dilation sequences on both modalities.
    neural_dilations: tuple[int, ...] = (1, 3, 9)
    speech_dilations: tuple[int, ...] = (1, 3, 9)
    # Temporal scales and neural-vs-speech lags considered by the scorer. The
    # paper uses a single scale and zero lag; wider settings were explored and
    # gave no gain, so the defaults keep the scorer minimal.
    scales: tuple[int, ...] = (1,)
    shifts: tuple[int, ...] = (0,)
    stats: tuple[str, ...] = ("mean", "max", "absdiff")
    head: str = HEAD_PER_STATISTIC
    standardize: bool = True
    tied_encoder: bool = False
    neural_encoder: str = ENCODER_DILATED
    speech_encoder: str = ENCODER_DILATED


class DilatedEncoder(nn.Module):
    """One-dimensional temporal encoder with increasing dilation rates.

    A 1x1 convolution first lifts the input to ``dim`` channels; three
    convolutions with kernel size 3 and dilations 1/3/9 then widen the
    receptive field without reducing the temporal sampling rate, so the same
    output captures both local and long-range structure.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        dropout: float = 0.0,
        dilations: tuple[int, ...] = (1, 3, 9),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(input_dim, dim, 1)]
        for dilation in dilations:
            layers.extend(
                [
                    nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, signal: Tensor) -> Tensor:
        # [B, T, C] -> [B, C, T] for Conv1d, then back to [B, T, dim].
        return self.network(signal.transpose(1, 2)).transpose(1, 2)


class DilatedTrunk(nn.Module):
    """Dilated stack without the 1x1 input projection, so it can be shared.

    Used only by the tied-parameter ablation: the two modalities have different
    input channel counts, so their 1x1 input projections must stay separate
    while the convolutional trunk is shared.
    """

    def __init__(self, dim: int, dropout: float, dilations: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for dilation in dilations:
            layers.extend(
                [
                    nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, signal: Tensor) -> Tensor:
        """Input and output are both [B, dim, T] (the trunk preserves length)."""
        return self.network(signal)


class TiedDilatedEncoder(nn.Module):
    """Encoder that keeps its own input projection but shares the trunk.

    Structurally identical to :class:`DilatedEncoder`; the only difference is
    that the convolutional kernels are shared with the other modality.
    """

    def __init__(self, input_dim: int, dim: int, trunk: DilatedTrunk) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, dim, 1)
        self.trunk = trunk

    def forward(self, signal: Tensor) -> Tensor:
        projected = self.input_projection(signal.transpose(1, 2))
        return self.trunk(projected).transpose(1, 2)


class MultiStatisticScorer(nn.Module):
    """Correlation scorer: normalize, summarize over time, project, score.

    For every candidate the encoded neural and speech features are L2
    normalized along the feature dimension. Three descriptors are extracted
    over time -- the mean product (correlation strength), the maximum product
    (peak alignment) and the mean absolute difference (matching residual) --
    optionally weighted over a set of temporal scales and lags. The descriptors
    are combined by the match head and mapped to a single score by an MLP.
    """

    def __init__(
        self,
        dim: int,
        config: SNTEConfig,
    ) -> None:
        super().__init__()
        self.scales = config.scales
        self.shifts = config.shifts
        self.stats = config.stats
        self.head = config.head
        # One descriptor group per scale, each of size len(stats) * dim.
        stat_dim = len(self.stats) * dim
        # Learned weighting over the temporal lags, one linear layer per scale.
        self.shift_attention = nn.ModuleList(
            [nn.Linear(stat_dim, 1) for _ in self.scales]
        )
        # Final scorer: concatenated descriptors -> MLP -> one score.
        self.scorer = nn.Sequential(
            nn.Linear(len(self.scales) * stat_dim, max(dim, 64)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(max(dim, 64), 1),
        )
        # Per-statistic projection (the paper's head): each descriptor is
        # projected by its own linear layer before concatenation. "concat"
        # omits the projection; "timecos" bypasses the scorer entirely.
        if self.head == HEAD_PER_STATISTIC:
            self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in self.stats])

    @staticmethod
    def _pool(sequence: Tensor, factor: int) -> Tensor:
        """Average-pool the time axis by ``factor`` (channels unchanged)."""
        if factor == 1:
            return sequence
        shape = sequence.shape
        flat = sequence.reshape(-1, shape[-2], shape[-1]).transpose(1, 2)
        pooled = F.avg_pool1d(flat, factor, factor).transpose(1, 2)
        return pooled.reshape(*shape[:-2], pooled.shape[-2], pooled.shape[-1])

    def _statistics(self, neural: Tensor, speech: Tensor, shift: int) -> Tensor:
        """Descriptors for one temporal lag between the two modalities.

        ``shift > 0`` lags the neural signal behind the speech by that many
        samples, capturing the latency of the neural response to the stimulus.
        """
        length = min(neural.shape[1], speech.shape[2])
        if shift > 0:
            neural = neural[:, shift:length]
            speech = speech[:, :, : length - shift]
        elif shift < 0:
            neural = neural[:, : length + shift]
            speech = speech[:, :, -shift : length]
        else:
            neural = neural[:, :length]
            speech = speech[:, :, :length]
        # Normalize along features, then take the element-wise product.
        neural = F.normalize(neural, dim=-1).unsqueeze(1)
        speech = F.normalize(speech, dim=-1)
        product = neural * speech
        stats = []
        for stat in self.stats:
            if stat == "mean":
                stats.append(product.mean(dim=2))
            elif stat == "max":
                stats.append(product.amax(dim=2))
            elif stat == "absdiff":
                stats.append((neural - speech).abs().mean(dim=2))
            else:
                raise ValueError(f"unknown statistic {stat!r}")
        return torch.cat(stats, dim=-1)

    def forward(self, neural: Tensor, speech: Tensor) -> Tensor:
        """Inputs:
        - neural: [B, T, dim] encoded neural features;
        - speech: [B, N, T, dim] encoded features of the N candidates.
        Output: one score per candidate, [B, N].
        """
        features = []
        for scale_index, factor in enumerate(self.scales):
            neural_scale = self._pool(neural, factor)
            speech_scale = self._pool(speech, factor)
            # Lags are rescaled to the pooled resolution (e.g. shift 24 at
            # scale 4 becomes 6).
            scale_shifts = sorted({round(shift / factor) for shift in self.shifts})
            statistics = torch.stack(
                [self._statistics(neural_scale, speech_scale, shift) for shift in scale_shifts],
                dim=2,
            )
            # Softmax-normalized weighting over the lag axis.
            weights = self.shift_attention[scale_index](statistics).softmax(dim=2)
            features.append((weights * statistics).sum(dim=2))
        features = torch.cat(features, dim=-1)  # [B, N, len(scales) * stat_dim]

        if self.head == HEAD_PER_STATISTIC:
            chunks = torch.chunk(features, len(self.stats), dim=-1)
            fused = torch.cat(
                [self.heads[i](chunk) for i, chunk in enumerate(chunks)], dim=-1
            )
        elif self.head == HEAD_CONCAT:
            fused = features
        else:
            raise ValueError(f"unknown head {self.head!r}")
        return self.scorer(fused).squeeze(-1)


class SNTE(nn.Module):
    """The proposed model: window-standardized dual encoders + scorer.

    Input: a 5-second neural window and N=5 candidate speech features.
    Output: five matching scores, trained with cross-entropy.
    """

    objective = "cross_entropy"

    def __init__(
        self,
        neural_channels: int,
        speech_channels: int = 74,
        config: SNTEConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SNTEConfig()
        config = self.config

        # The two encoders are structurally identical but independently
        # parameterized, unless the tied-parameter ablation is enabled.
        if config.tied_encoder:
            if config.neural_encoder == ENCODER_LINEAR or config.speech_encoder == ENCODER_LINEAR:
                raise ValueError("tied_encoder is incompatible with linear encoders")
            if config.neural_dilations != config.speech_dilations:
                raise ValueError("tied_encoder requires equal dilation sequences")
            trunk = DilatedTrunk(config.embed_dim, config.dropout, config.neural_dilations)
            self.neural_encoder: nn.Module = TiedDilatedEncoder(
                neural_channels, config.embed_dim, trunk
            )
            self.speech_encoder: nn.Module = TiedDilatedEncoder(
                speech_channels, config.embed_dim, trunk
            )
        else:
            if config.neural_encoder == ENCODER_LINEAR:
                self.neural_encoder = nn.Linear(neural_channels, config.embed_dim)
            else:
                self.neural_encoder = DilatedEncoder(
                    neural_channels,
                    config.embed_dim,
                    config.dropout,
                    config.neural_dilations,
                )
            if config.speech_encoder == ENCODER_LINEAR:
                self.speech_encoder = nn.Linear(speech_channels, config.embed_dim)
            else:
                self.speech_encoder = DilatedEncoder(
                    speech_channels,
                    config.embed_dim,
                    config.dropout,
                    config.speech_dilations,
                )

        # Scoring in the shared latent space.
        self.correlation = MultiStatisticScorer(config.embed_dim, config)
        # Learned temperature for the scalar time-mean head, matching the
        # baseline matcher (initialized to e^ln10 ~ 10, clamped at 100).
        self.temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def standardize(self, neural: Tensor) -> Tensor:
        """Window-level z-score along time.

        Each 5-second window is normalized by its own mean and standard
        deviation. This is what makes the MEG dataset usable: SEM4Lang stores
        values around 1e-11, and without normalization accuracy collapses to
        chance level. Its effect on the two EEG datasets is small.
        """
        value = neural.float()
        mean = value.mean(dim=1, keepdim=True)
        std = value.std(dim=1, keepdim=True, unbiased=False)
        return (value - mean) / std.clamp_min(1e-8)

    def forward(self, neural: Tensor, candidates: Tensor) -> Tensor:
        """Inputs:
        - neural: [B, T, C_neural] neural windows;
        - candidates: [B, N, T, C_speech] features of the N candidates.
        Output: matching scores [B, N].
        """
        batch, count, time, channels = candidates.shape
        if self.config.standardize:
            neural = self.standardize(neural)
        neural_features = self.neural_encoder(neural)
        # All candidates share the speech encoder, so flatten the candidate
        # axis into the batch for the convolution.
        speech_features = self.speech_encoder(
            candidates.reshape(batch * count, time, channels)
        )
        speech_features = speech_features.reshape(batch, count, time, -1)

        if self.config.head == HEAD_TIME_MEAN:
            # Scalar time-mean cosine: normalize per time step, average over
            # time, scale by the learned temperature.
            neural_norm = F.normalize(neural_features, dim=-1).unsqueeze(1)
            speech_norm = F.normalize(speech_features, dim=-1)
            cosine = (neural_norm * speech_norm).sum(dim=-1)
            return cosine.mean(dim=-1) * self.temperature.exp().clamp(max=100)
        return self.correlation(neural_features, speech_features)


def create_model(
    name: str,
    neural_channels: int,
    speech_channels: int = 74,
    config: SNTEConfig | None = None,
) -> nn.Module:
    """Build SNTE or one of the paper's baseline models."""
    if name == "snte":
        return SNTE(neural_channels, speech_channels, config)

    from baselines.model import create_baseline

    return create_baseline(name, neural_channels, speech_channels, config)


if __name__ == "__main__":
    # Smoke test: 204-channel neural input (SEM4Lang) with 5 candidates.
    network = SNTE(204, 74, SNTEConfig())
    scores = network(torch.randn(2, 320, 204), torch.randn(2, 5, 320, 74))
    total = sum(parameter.numel() for parameter in network.parameters())
    print(tuple(scores.shape), f"{total:,}")
