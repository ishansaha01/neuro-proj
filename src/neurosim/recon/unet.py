"""Small time-conditioned U-Net for single-channel images (the denoiser of the prior).

Layout after Ho et al. (2020), "Denoising diffusion probabilistic models",
appendix B: residual blocks with GroupNorm and SiLU, a sinusoidal timestep
embedding added inside every block, stride-2 downsampling, nearest-neighbour
upsampling, skip connections by concatenation, and one self-attention block at
the lowest resolution.

Parameter counts (`count_parameters`):
    configs/prior_default.yaml (base 16, multipliers 1-2-4-8, 1 block per level): 1,947,633
    configs/prior_small.yaml (base 8, multipliers 1-2, 1 block per level): 32,617
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from neurosim.common.config import field

# Longest period of the sinusoidal timestep embedding, in timesteps (Vaswani et al.
# 2017, "Attention is all you need", section 3.5; reused by Ho et al. 2020).
_MAX_PERIOD = 10_000.0
# Width of the timestep embedding relative to the base channel count (Ho et al. 2020).
_TIME_WIDTH_FACTOR = 4


@dataclass(frozen=True)
class UNetConfig:
    """Size of the U-Net. The image side must be divisible by ``2 ** (levels - 1)``."""

    base_channels: int = field(16, ge=2, doc="Channels at full resolution (even).")
    channel_multipliers: tuple[int, ...] = field(
        (1, 2, 4, 8),
        ge=1,
        doc="Channel count of each resolution level as a multiple of base_channels; "
        "the resolution halves between levels.",
    )
    blocks_per_level: int = field(1, ge=1, doc="Residual blocks per level on each path.")
    norm_groups: int = field(8, ge=1, doc="Groups of every GroupNorm.")
    attention: bool = field(
        True, doc="Self-attention between the two residual blocks at the lowest resolution."
    )

    def __post_init__(self) -> None:
        if not self.channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        if self.base_channels % 2:
            raise ValueError("base_channels must be even (sinusoidal embedding)")
        for multiplier in self.channel_multipliers:
            if (self.base_channels * multiplier) % self.norm_groups:
                raise ValueError(
                    f"norm_groups {self.norm_groups} must divide every channel count, "
                    f"got {self.base_channels * multiplier}"
                )

    @property
    def downsampling(self) -> int:
        """Factor between the full and the lowest resolution."""
        return 2 ** (len(self.channel_multipliers) - 1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of timesteps.

    t: timestep indices, shape (batch,).
    dim: embedding size (even).

    Returns float32, shape (batch, dim): ``sin(t f_k)`` then ``cos(t f_k)`` with
    frequencies ``f_k`` spaced geometrically from 1 down to 1 / 10000.
    """
    half = dim // 2
    exponent = torch.arange(half, dtype=torch.float32) / half
    angles = t.to(torch.float32)[:, None] * torch.exp(-math.log(_MAX_PERIOD) * exponent)[None]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class ResidualBlock(nn.Module):
    """GroupNorm-SiLU-conv twice, with the time embedding added in between."""

    def __init__(self, in_channels: int, out_channels: int, time_width: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_width, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time(time_embedding)[:, :, None, None]
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return self.skip(x) + h


class SelfAttention(nn.Module):
    """Single-head self-attention over the pixels of a feature map, with a residual."""

    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.query_key_value = nn.Conv2d(channels, 3 * channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        query, key, value = (
            self.query_key_value(self.norm(x)).reshape(batch, 3, channels, height * width).unbind(1)
        )
        weights = torch.softmax(query.transpose(1, 2) @ key / math.sqrt(channels), dim=-1)
        attended = (value @ weights.transpose(1, 2)).reshape(batch, channels, height, width)
        return x + self.out(attended)


class UNet(nn.Module):
    """Denoiser network: ``(x_t, t) -> prediction`` with the shape of ``x_t``.

    forward(x, t):
        x: noisy images in model space, float32, shape (batch, 1, n, n), with n
            divisible by `UNetConfig.downsampling`.
        t: timestep indices, integer, shape (batch,).
        Returns float32, shape (batch, 1, n, n). What the output means (here the
        v-prediction) is set by the training loss in `recon.diffusion`.

    Build it with `build_unet`, which initialises the weights from an explicit
    generator.
    """

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.config = cfg
        base, groups = cfg.base_channels, cfg.norm_groups
        widths = [base * multiplier for multiplier in cfg.channel_multipliers]
        time_width = _TIME_WIDTH_FACTOR * base
        self.time_mlp = nn.Sequential(
            nn.Linear(base, time_width), nn.SiLU(), nn.Linear(time_width, time_width)
        )
        self.conv_in = nn.Conv2d(1, base, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_widths = []
        channels = base
        for level, width in enumerate(widths):
            for _ in range(cfg.blocks_per_level):
                self.down_blocks.append(ResidualBlock(channels, width, time_width, groups))
                channels = width
                skip_widths.append(channels)
            if level < len(widths) - 1:
                self.downsamples.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))

        self.middle1 = ResidualBlock(channels, channels, time_width, groups)
        self.attention = SelfAttention(channels, groups) if cfg.attention else nn.Identity()
        self.middle2 = ResidualBlock(channels, channels, time_width, groups)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level, width in reversed(list(enumerate(widths))):
            for _ in range(cfg.blocks_per_level):
                self.up_blocks.append(
                    ResidualBlock(channels + skip_widths.pop(), width, time_width, groups)
                )
                channels = width
            if level > 0:
                self.upsamples.append(nn.Conv2d(channels, channels, 3, padding=1))

        self.norm_out = nn.GroupNorm(groups, channels)
        self.conv_out = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if x.shape[-1] % cfg.downsampling or x.shape[-2] % cfg.downsampling:
            raise ValueError(
                f"image size {tuple(x.shape[-2:])} is not divisible by {cfg.downsampling}"
            )
        time_embedding = self.time_mlp(timestep_embedding(t, cfg.base_channels))
        n_levels = len(cfg.channel_multipliers)

        h = self.conv_in(x)
        skips = []
        down_blocks = iter(self.down_blocks)
        for level in range(n_levels):
            for _ in range(cfg.blocks_per_level):
                h = next(down_blocks)(h, time_embedding)
                skips.append(h)
            if level < n_levels - 1:
                h = self.downsamples[level](h)

        h = self.middle2(self.attention(self.middle1(h, time_embedding)), time_embedding)

        up_blocks = iter(self.up_blocks)
        for level in range(n_levels):
            for _ in range(cfg.blocks_per_level):
                h = next(up_blocks)(torch.cat((h, skips.pop()), dim=1), time_embedding)
            if level < n_levels - 1:
                h = self.upsamples[level](nn.functional.interpolate(h, scale_factor=2.0))
        return self.conv_out(nn.functional.silu(self.norm_out(h)))


def _initialise(model: UNet, generator: torch.Generator) -> None:
    """Draw every weight from `generator`; nothing is left from torch's global generator.

    Convolutions and linear layers: Xavier-uniform weights, zero biases. The
    second convolution of every residual block and the output convolution start
    at zero, so the network starts as the zero function (Ho et al. 2020 use the
    same zero initialisation).
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, generator=generator)
            nn.init.zeros_(module.bias)
    for module in model.modules():
        if isinstance(module, ResidualBlock):
            nn.init.zeros_(module.conv2.weight)
    nn.init.zeros_(model.conv_out.weight)


def build_unet(cfg: UNetConfig, generator: torch.Generator) -> UNet:
    """Create a U-Net whose initial weights depend only on `cfg` and `generator`."""
    model = UNet(cfg)
    _initialise(model, generator)
    return model


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
