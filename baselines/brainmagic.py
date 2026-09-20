"""BrainMagic baseline, adapted to the frozen five-way task.

Because the paper evaluates unseen participants (cross-subject), only the
shared BrainMagic body is kept and the subject-specific layers are dropped:
the input is window-standardized and mapped to 64 channels, passed through the
shared body, and matched against the candidates with the same latent cosine
matcher as every other baseline.
"""

from __future__ import annotations

from torch import Tensor, nn

from .common import LatentCosineMatcher, Spatial64


class BrainMagicConvSequence(nn.Module):
    """The convolutional sequence from the reference implementation.

    Residual connections, GLU gating and dilated convolutions.
    """

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        self.sequence = nn.ModuleList()
        self.gates = nn.ModuleList()
        dilation = 1
        for index, (input_channels, output_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            # The dilation rate resets to 1 every five layers and doubles in
            # between.
            if index % 5 == 0:
                dilation = 1
            self.sequence.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(output_channels),
                    nn.GELU(),
                )
            )
            dilation *= 2
            # Every second layer (1-indexed) is followed by a GLU gate.
            if (index + 1) % 2 == 0:
                self.gates.append(
                    nn.Sequential(
                        nn.Conv1d(output_channels, 2 * output_channels, 3, padding=1),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.gates.append(nn.Identity())

    def forward(self, value: Tensor) -> Tensor:
        for layer, gate in zip(self.sequence, self.gates):
            residual = value
            value = layer(value)
            # Add the residual whenever the shape is unchanged.
            if value.shape == residual.shape:
                value = value + residual
            value = gate(value)
        return value


class BrainMagicDecoder(nn.Module):
    """The full BrainMagic shared body (subject-specific layers disabled)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        # Lift to 270 channels, run the [270, 320 x 10] convolution sequence,
        # then project down to the output dimension.
        self.initial_linear = nn.Conv1d(input_dim, 270, 1)
        self.encoder = BrainMagicConvSequence([270] + [320] * 10)
        self.final = nn.Sequential(
            nn.Conv1d(320, 640, 1),
            nn.GELU(),
            nn.Conv1d(640, output_dim, 1),
        )

    def forward(self, signal: Tensor) -> Tensor:
        value = self.initial_linear(signal.transpose(1, 2))
        return self.final(self.encoder(value)).transpose(1, 2)


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
        BrainMagicDecoder(64, embed_dim),
    )
    return LatentCosineMatcher(neural_body, speech_channels, embed_dim, dropout)
