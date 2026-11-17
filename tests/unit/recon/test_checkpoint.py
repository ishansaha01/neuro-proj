"""End-to-end test of the two scripts on configs/prior_small.yaml, and of the checkpoint."""

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import torch

from neurosim.common.config import config_hash, from_dict, load_yaml
from neurosim.common.io import load_bundle
from neurosim.common.seeding import SeedTree
from neurosim.recon.cbv_images import pixel_grid
from neurosim.recon.checkpoint import (
    DEFAULT_DATASET_PATH,
    DEFAULT_PRIOR_PATH,
    Prior,
    load_prior,
    load_prior_with_data,
)
from neurosim.recon.seeding import SAMPLING_STREAM, torch_generator
from neurosim.recon.train import PriorConfig
from scripts import make_prior_dataset, train_prior

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "prior_small.yaml"
DATA_SEED, TRAIN_SEED = 1, 2


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    """Run both scripts once; returns the paths and what they printed."""
    folder = tmp_path_factory.mktemp("prior")
    data_path, prior_path = folder / "dataset.h5", folder / "prior.pt"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        data_code = make_prior_dataset.main(
            ["--config", str(CONFIG), "--seed", str(DATA_SEED), "--out", str(data_path)]
        )
        train_code = train_prior.main(
            [
                "--config",
                str(CONFIG),
                "--data",
                str(data_path),
                "--seed",
                str(TRAIN_SEED),
                "--out",
                str(prior_path),
            ]
        )
    return {
        "data": data_path,
        "prior": prior_path,
        "codes": (data_code, train_code),
        "printed": printed.getvalue().splitlines(),
        "config": from_dict(PriorConfig, load_yaml(CONFIG)),
    }


@pytest.fixture(scope="module")
def prior(run) -> Prior:
    return load_prior(run["prior"])


def test_scripts_succeed_and_print_estimates_first(run):
    assert run["codes"] == (0, 0)
    printed = run["printed"]
    assert "GiB peak memory" in printed[0] and "160 images" in printed[0]
    assert printed[1].startswith("saved")
    assert "s per step" in printed[2] and "M parameters" in printed[2]
    assert printed[-1].startswith("saved") and "300 of 300 steps" in printed[-1]


def test_dataset_bundle_has_images_split_and_provenance(run):
    arrays, attrs = load_bundle(run["data"])
    cfg = run["config"].data
    assert arrays["images"].shape == (160, 16, 16)
    assert int(arrays["heldout"].sum()) == 32
    assert attrs["root_seed"] == DATA_SEED
    assert attrs["config_hash"] == config_hash(cfg)
    assert attrs["pixel_grid"]["n"] == 16


def test_training_reduced_the_heldout_error(prior):
    first, last = prior.history[0], prior.history[-1]
    assert last["step"] == 300 and prior.steps_completed == 300
    assert last["train_loss"] < first["train_loss"]
    for level in first["heldout"]:
        assert last["heldout"][level]["x0_mse"] < first["heldout"][level]["x0_mse"]


def test_loaded_prior_exposes_model_schedule_transform_and_grid(prior, run):
    cfg = run["config"]
    assert prior.config == cfg
    assert not prior.model.training
    assert not any(parameter.requires_grad for parameter in prior.model.parameters())
    assert prior.diffusion.n_timesteps == cfg.diffusion.n_timesteps
    assert prior.transform == cfg.transform
    assert prior.pixel_grid == pixel_grid(cfg.data.pixel_grid)
    assert prior.provenance["root_seed"] == TRAIN_SEED
    assert prior.provenance["config_hash"] == config_hash(cfg)
    assert prior.provenance["dataset"]["root_seed"] == DATA_SEED
    assert prior.provenance["dataset"]["config_hash"] == config_hash(cfg.data)


def test_raw_weights_can_be_loaded_instead_of_the_average(prior, run):
    raw = load_prior(run["prior"], ema=False)
    assert not torch.equal(raw.model.conv_in.weight, prior.model.conv_in.weight)


def test_loaded_prior_supports_guided_sampling_in_physical_units(prior):
    n = prior.pixel_grid.shape[0]
    target = torch.full((2, 1, n, n), 0.03)

    def guidance(x_t: torch.Tensor, x0_hat: torch.Tensor, t: int) -> torch.Tensor:
        misfit = (prior.transform.to_physical(x0_hat) - target).square().sum()
        (gradient,) = torch.autograd.grad(misfit, x_t)
        assert torch.all(torch.isfinite(gradient)) and gradient.abs().max() > 0.0
        return -0.1 * gradient

    samples = prior.diffusion.ancestral_sample(
        prior.model,
        (2, 1, n, n),
        torch_generator(SeedTree(0), SAMPLING_STREAM),
        n_steps=10,
        guidance=guidance,
        x0_clip=(prior.transform.model_floor, float("inf")),
    )
    cbv = prior.transform.to_physical(samples).numpy()
    assert cbv.shape == (2, 1, n, n) and np.all(np.isfinite(cbv)) and cbv.min() >= -1e-6


def test_training_refuses_a_dataset_from_another_config(run, tmp_path, capsys):
    other_config = tmp_path / "other.yaml"
    other_config.write_text(CONFIG.read_text().replace("n_heads: 20", "n_heads: 21"))
    code = train_prior.main(
        [
            "--config",
            str(other_config),
            "--data",
            str(run["data"]),
            "--seed",
            "0",
            "--out",
            str(tmp_path / "p.pt"),
        ]
    )
    assert code == 2
    assert "config hashes differ" in capsys.readouterr().err
    assert not (tmp_path / "p.pt").exists()


def test_unknown_checkpoint_format_is_rejected(run, tmp_path):
    checkpoint = torch.load(run["prior"], weights_only=True)
    checkpoint["format_version"] = 99
    torch.save(checkpoint, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="format 99"):
        load_prior(tmp_path / "future.pt")


def test_prior_and_its_data_set_load_together(run, prior):
    loaded, arrays = load_prior_with_data(run["prior"], run["data"])
    assert loaded.config == prior.config
    assert arrays["images"].shape == (160, 16, 16) and int(arrays["heldout"].sum()) == 32


def test_missing_files_are_reported_with_the_commands_to_run(run, tmp_path):
    for prior_path, data_path in (
        (tmp_path / "no.pt", run["data"]),
        (run["prior"], tmp_path / "no.h5"),
    ):
        with pytest.raises(FileNotFoundError) as error:
            load_prior_with_data(prior_path, data_path)
        message = str(error.value)
        assert "no." in message
        assert "scripts/make_prior_dataset.py" in message and "scripts/train_prior.py" in message


def test_a_data_set_the_prior_was_not_trained_on_is_rejected(run, tmp_path):
    other_data = tmp_path / "other.h5"
    make_prior_dataset.main(
        ["--config", str(CONFIG), "--seed", str(DATA_SEED + 1), "--out", str(other_data)]
    )
    with pytest.raises(ValueError, match="is not the data set"):
        load_prior_with_data(run["prior"], other_data)


def test_default_paths_are_in_the_ignored_outputs_folder():
    repo = Path(__file__).resolve().parents[3]
    assert DEFAULT_PRIOR_PATH == repo / "outputs" / "prior" / "prior.pt"
    assert DEFAULT_DATASET_PATH == repo / "outputs" / "prior" / "dataset.h5"
