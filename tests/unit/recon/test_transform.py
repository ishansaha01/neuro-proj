import math

import numpy as np
import pytest
import torch

from neurosim.common.config import ConfigError, from_dict
from neurosim.recon.transform import CbvTransform

TRANSFORM = CbvTransform()
CBV = np.array([0.0, 0.005, 0.02, 0.04, 0.3, 0.5, 0.8], dtype=np.float32)


def test_anchor_values():
    model = TRANSFORM.to_model(np.array([0.0, TRANSFORM.cbv_max]))
    np.testing.assert_allclose(model, [TRANSFORM.model_floor, 1.0], atol=1e-12)
    # The values quoted in the class docstring.
    quoted = TRANSFORM.to_model(np.array([0.02, 0.04, 0.3]))
    np.testing.assert_allclose(quoted, [-0.57, -0.33, 0.70], atol=0.01)


def test_round_trip_numpy_and_torch():
    np.testing.assert_allclose(TRANSFORM.to_physical(TRANSFORM.to_model(CBV)), CBV, atol=1e-6)
    tensor = torch.from_numpy(CBV)
    round_trip = TRANSFORM.to_physical(TRANSFORM.to_model(tensor))
    assert torch.allclose(round_trip, tensor, atol=1e-6)


def test_numpy_and_torch_agree_and_keep_type_and_dtype():
    from_numpy = TRANSFORM.to_model(CBV)
    from_torch = TRANSFORM.to_model(torch.from_numpy(CBV))
    assert isinstance(from_numpy, np.ndarray) and from_numpy.dtype == np.float32
    assert isinstance(from_torch, torch.Tensor) and from_torch.dtype == torch.float32
    np.testing.assert_allclose(from_torch.numpy(), from_numpy, rtol=1e-6)
    back_numpy = TRANSFORM.to_physical(from_numpy)
    back_torch = TRANSFORM.to_physical(from_torch)
    assert back_numpy.dtype == np.float32 and back_torch.dtype == torch.float32
    np.testing.assert_allclose(back_torch.numpy(), back_numpy, rtol=1e-5)


def test_map_is_strictly_increasing_and_compresses_the_tail():
    cbv = np.linspace(0.0, 0.6, 200)
    model = TRANSFORM.to_model(cbv)
    assert np.all(np.diff(model) > 0.0)
    assert np.all(np.diff(model, 2) < 0.0)


def test_to_physical_is_differentiable_with_the_documented_slope():
    model = torch.linspace(-1.5, 1.2, 12, dtype=torch.float64, requires_grad=True)
    cbv = TRANSFORM.to_physical(model)
    (slope,) = torch.autograd.grad(cbv.sum(), model)
    span = math.log1p(TRANSFORM.cbv_max / TRANSFORM.cbv_knee)
    expected = (cbv.detach() + TRANSFORM.cbv_knee) * span / 2.0
    assert torch.allclose(slope, expected, rtol=1e-10)
    assert torch.all(slope > 0.0)
    assert torch.all(cbv > -TRANSFORM.cbv_knee)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="cbv_max"):
        CbvTransform(cbv_knee=0.1, cbv_max=0.05)
    with pytest.raises(ConfigError, match="cbv_knee"):
        from_dict(CbvTransform, {"cbv_knee": 0.0})
