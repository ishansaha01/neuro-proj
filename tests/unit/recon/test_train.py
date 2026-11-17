import dataclasses

import numpy as np
import pytest
import torch

from neurosim.common.seeding import SeedTree
from neurosim.recon.diffusion import GaussianDiffusion
from neurosim.recon.seeding import TRAINING_STREAM, torch_generator
from neurosim.recon.train import (
    PriorConfig,
    heldout_losses,
    measure_step_seconds,
    to_model_tensor,
    train_prior,
)
from neurosim.recon.unet import UNetConfig

TRAIN_SEED = 21
N_STEPS = 8


@pytest.fixture(scope="module")
def tiny_cfg(small_cfg) -> PriorConfig:
    """A handful of steps of the small configuration."""
    training = dataclasses.replace(
        small_cfg.training, n_steps=N_STEPS, eval_every=4, warmup_steps=2
    )
    return dataclasses.replace(small_cfg, training=training)


def _train(cfg, data, seed=TRAIN_SEED, **kwargs):
    heldout = data["heldout"]
    return train_prior(
        cfg, data["images"][~heldout], data["images"][heldout], SeedTree(seed), **kwargs
    )


@pytest.fixture(scope="module")
def tiny_result(tiny_cfg, small_data):
    return _train(tiny_cfg, small_data)


def test_to_model_tensor(small_cfg, small_data):
    tensor = to_model_tensor(small_data["images"][:3], small_cfg.transform)
    n = small_cfg.data.pixel_grid.n
    assert tensor.shape == (3, 1, n, n) and tensor.dtype == torch.float32
    assert tensor.min() == pytest.approx(small_cfg.transform.model_floor)


def test_training_is_bit_reproducible_for_a_fixed_seed(tiny_cfg, small_data, tiny_result):
    torch.rand(3)  # the global generator must not matter
    again = _train(tiny_cfg, small_data)
    for first, second in (
        (tiny_result.model, again.model),
        (tiny_result.ema_model, again.ema_model),
    ):
        for name, values in first.state_dict().items():
            assert torch.equal(values, second.state_dict()[name]), name
    assert again.history == tiny_result.history


def test_another_seed_gives_other_weights(tiny_cfg, small_data, tiny_result):
    other = _train(tiny_cfg, small_data, seed=TRAIN_SEED + 1)
    assert not torch.equal(other.model.conv_in.weight, tiny_result.model.conv_in.weight)


def test_result_and_history_records(tiny_cfg, tiny_result):
    assert tiny_result.steps_completed == N_STEPS and tiny_result.wall_seconds > 0.0
    assert [record["step"] for record in tiny_result.history] == [4, 8]
    for record in tiny_result.history:
        assert np.isfinite(record["train_loss"])
        assert list(record["heldout"]) == ["0.9", "0.5", "0.1"]
        for values in record["heldout"].values():
            assert set(values) == {"timestep", "v_loss", "x0_mse"}
            assert 0.0 < values["x0_mse"] < values["v_loss"]
    for model in (tiny_result.model, tiny_result.ema_model):
        assert not model.training
        assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not torch.equal(tiny_result.model.conv_in.weight, tiny_result.ema_model.conv_in.weight)


def test_callback_receives_every_record(tiny_cfg, small_data, tiny_result):
    records = []
    _train(tiny_cfg, small_data, on_eval=records.append)
    assert records == tiny_result.history


def test_wall_time_limit_stops_early_with_a_final_record(tiny_cfg, small_data):
    limited = dataclasses.replace(
        tiny_cfg,
        training=dataclasses.replace(tiny_cfg.training, n_steps=10_000, max_wall_minutes=1e-9),
    )
    result = _train(limited, small_data)
    assert result.steps_completed == 1
    assert [record["step"] for record in result.history] == [1]


def test_heldout_x0_error_matches_the_tweedie_estimate(small_cfg, small_data, tiny_result):
    diffusion = GaussianDiffusion(small_cfg.diffusion)
    x0 = to_model_tensor(small_data["images"][:6], small_cfg.transform)
    noise = torch.randn(x0.shape, generator=torch_generator(SeedTree(0), TRAINING_STREAM, 99))
    losses = heldout_losses(tiny_result.model, diffusion, x0, noise, (0.5,))
    timestep = losses["0.5"]["timestep"]
    assert timestep == diffusion.timestep_for_alpha_bar(0.5)
    x_t = diffusion.q_sample(x0, timestep, noise)
    x0_hat = diffusion.predict_x0(tiny_result.model, x_t, timestep)
    assert losses["0.5"]["x0_mse"] == pytest.approx(float((x0_hat - x0).square().mean()), rel=1e-4)


def test_step_time_measurement(tiny_cfg):
    assert 0.0 < measure_step_seconds(tiny_cfg, SeedTree(TRAIN_SEED), n_steps=1) < 5.0


def test_image_size_must_fit_the_network(small_cfg):
    deeper = UNetConfig(base_channels=8, channel_multipliers=(1, 2, 2, 2, 2, 2), norm_groups=4)
    with pytest.raises(ValueError, match="divisible by the U-Net downsampling"):
        dataclasses.replace(small_cfg, unet=deeper)
