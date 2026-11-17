"""Map between physical CBV and the space the diffusion model works in."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

from neurosim.common.config import field

Array = TypeVar("Array", np.ndarray, torch.Tensor)


def _namespace(values: np.ndarray | torch.Tensor):
    """The module (`torch` or `numpy`) whose functions apply to `values`."""
    return torch if isinstance(values, torch.Tensor) else np


@dataclass(frozen=True)
class CbvTransform:
    """Log map ``u = 2 log(1 + cbv / cbv_knee) / log(1 + cbv_max / cbv_knee) - 1``.

    Why a log with a knee:

    - CBV is heavy-tailed (tissue 0.02-0.05, vessels up to about 0.3). The log
      spreads tissue and vessels evenly over the model range instead of
      squeezing all tissue into a few percent of it: with the defaults zero CBV
      maps to -1, white matter (0.02) to -0.57, grey matter (0.04) to -0.33,
      a vessel (0.3) to 0.70 and `cbv_max` to +1.
    - The task response is multiplicative, ``cbv * (1 + a * activation)``; well
      above the knee it becomes an additive shift of ``log(1 + a * activation)``,
      independent of the local blood volume.
    - The knee keeps the map finite and smooth at zero CBV (outside the brain),
      unlike a plain log or a square root, whose slopes diverge there.
    - Both directions are smooth, strictly increasing and closed-form, so data
      guidance can differentiate through `to_physical`. Its slope,
      ``(cbv + cbv_knee) * log(1 + cbv_max / cbv_knee) / 2``, is positive for every
      real model value.

    `to_physical` accepts any real model value and returns CBV above
    ``-cbv_knee``; model values below `model_floor` (-1) give slightly negative
    CBV. `to_model` needs ``cbv > -cbv_knee``.

    Both methods take a numpy array or a torch tensor and return the same kind,
    computed by the same expression.
    """

    cbv_knee: float = field(
        0.02,
        gt=0.0,
        doc="CBV fraction at which the map turns from linear to logarithmic: the white-"
        "matter median of head.config.CbvConfig, so that all tissue lies on the log part.",
    )
    cbv_max: float = field(
        0.5,
        gt=0.0,
        doc="CBV fraction mapped to +1: above the brightest vessel pixel with the "
        "largest activation of the prior data (about 0.3 * 1.5); values above it are "
        "still valid and map above +1.",
    )

    def __post_init__(self) -> None:
        if self.cbv_max <= self.cbv_knee:
            raise ValueError("cbv_max must be larger than cbv_knee")

    @property
    def model_floor(self) -> float:
        """Model-space value of zero CBV; physical images never go below it."""
        return -1.0

    @property
    def _log_span(self) -> float:
        return math.log1p(self.cbv_max / self.cbv_knee)

    def to_model(self, cbv: Array) -> Array:
        """Physical CBV fraction (any shape) -> model space (same shape, dimensionless)."""
        return 2.0 * _namespace(cbv).log1p(cbv / self.cbv_knee) / self._log_span - 1.0

    def to_physical(self, model_values: Array) -> Array:
        """Model space (any shape) -> physical CBV fraction (same shape). Inverse of `to_model`."""
        scaled = 0.5 * (model_values + 1.0) * self._log_span
        return self.cbv_knee * _namespace(model_values).expm1(scaled)
