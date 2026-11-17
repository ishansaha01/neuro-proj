import pytest
import torch

from neurosim.recon.unet import (
    UNetConfig,
    build_unet,
    count_parameters,
    timestep_embedding,
)

SMALL = UNetConfig(base_channels=8, channel_multipliers=(1, 2), norm_groups=4)


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randomised(cfg: UNetConfig) -> torch.nn.Module:
    """A U-Net whose zero-initialised layers are filled too, so the output is not zero."""
    model = build_unet(cfg, _generator())
    generator = _generator(1)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.1, generator=generator)
    return model.eval()


def test_default_network_has_the_stated_size():
    model = build_unet(UNetConfig(), _generator())
    assert count_parameters(model) == 1_947_633
    assert UNetConfig().downsampling == 8


def test_output_shape_and_zero_start():
    model = build_unet(SMALL, _generator())
    x = torch.randn(3, 1, 16, 16, generator=_generator(2))
    out = model(x, torch.tensor([0, 10, 199]))
    assert out.shape == x.shape and out.dtype == torch.float32
    assert torch.count_nonzero(out) == 0


def test_weights_depend_only_on_the_generator():
    first = build_unet(SMALL, _generator(7))
    torch.rand(5)  # advance the global generator between the two builds
    second = build_unet(SMALL, _generator(7))
    other = build_unet(SMALL, _generator(8))
    for name, values in first.state_dict().items():
        assert torch.equal(values, second.state_dict()[name]), name
    assert not torch.equal(first.conv_in.weight, other.conv_in.weight)


def test_output_depends_on_the_timestep_and_the_whole_image():
    model = _randomised(SMALL)
    x = torch.randn(1, 1, 16, 16, generator=_generator(3), requires_grad=True)
    early = model(x, torch.tensor([5]))
    late = model(x, torch.tensor([150]))
    assert not torch.allclose(early, late)
    (gradient,) = torch.autograd.grad(early[0, 0, 0, 0], x)
    # Attention at the lowest resolution couples the corner pixel to every input pixel.
    assert torch.count_nonzero(gradient) == gradient.numel()


def test_attention_can_be_switched_off():
    without = build_unet(
        UNetConfig(base_channels=8, channel_multipliers=(1, 2), norm_groups=4, attention=False),
        _generator(),
    )
    assert count_parameters(without) < count_parameters(build_unet(SMALL, _generator()))


def test_two_blocks_per_level_run():
    cfg = UNetConfig(
        base_channels=8, channel_multipliers=(1, 2, 2), blocks_per_level=2, norm_groups=4
    )
    out = _randomised(cfg)(torch.zeros(2, 1, 8, 8), torch.tensor([1, 2]))
    assert out.shape == (2, 1, 8, 8) and torch.all(torch.isfinite(out))


def test_image_size_must_match_the_downsampling():
    model = build_unet(SMALL, _generator())
    with pytest.raises(ValueError, match="not divisible"):
        model(torch.zeros(1, 1, 15, 16), torch.tensor([0]))


def test_timestep_embedding():
    embedding = timestep_embedding(torch.tensor([0, 1, 500]), 16)
    assert embedding.shape == (3, 16)
    assert torch.allclose(embedding[0], torch.cat((torch.zeros(8), torch.ones(8))))
    assert embedding[1, 0] == pytest.approx(float(torch.sin(torch.tensor(1.0))))
    assert not torch.allclose(embedding[1], embedding[2])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"base_channels": 9}, "even"),
        ({"base_channels": 12, "norm_groups": 8}, "norm_groups"),
        ({"channel_multipliers": ()}, "empty"),
    ],
)
def test_invalid_configs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        UNetConfig(**kwargs)
