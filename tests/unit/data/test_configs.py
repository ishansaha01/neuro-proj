"""The run and sweep config files: they load, validate, estimate, and stay consistent."""

from __future__ import annotations

from dataclasses import replace

import pytest

from neurosim.common.config import from_dict, load_yaml
from neurosim.data.config import RunConfig, trial_fidelity
from neurosim.data.scene import estimate_cost
from neurosim.data.sweep import SweepConfig
from neurosim.fus.config import pixel_grid_from_config, roi_grid_from_config
from neurosim.head.types import SkullEffects
from neurosim.recon.cbv_images import PixelGridConfig, pixel_grid

from .conftest import CONFIG_DIR

RUN_FILES = ("run_small.yaml", "run_demo.yaml", "run_default.yaml")
GIB = 2**30


def load_run_config(name: str) -> RunConfig:
    return from_dict(RunConfig, load_yaml(CONFIG_DIR / name))


@pytest.mark.parametrize("name", RUN_FILES)
def test_run_configs_validate_and_give_a_cost_estimate(name):
    cost = estimate_cost(load_run_config(name))
    for stage in ("head", "ct_prior", "bank", "bank_ray", "operator", "trial_image"):
        assert cost[stage]["memory_bytes"] > 0
    assert cost["time_s"] > 0.0 and cost["memory_bytes"] > 0 and cost["bundle_bytes"] > 0
    assert cost["n_backend_banks"] == 2 and cost["n_ray_banks"] == 1


def test_small_and_demo_fit_the_development_machine():
    small, demo = (estimate_cost(load_run_config(name)) for name in RUN_FILES[:2])
    assert small["time_s"] < 30.0 and small["memory_bytes"] < 0.5 * GIB
    assert demo["time_s"] < 600.0 and demo["memory_bytes"] < 1.5 * GIB


def test_default_run_has_the_roadmap_settings():
    cfg = load_run_config("run_default.yaml")
    assert cfg.imaging.transducer.n_elements == 128
    assert cfg.imaging.transducer.center_frequency == 2.0e6
    angles = cfg.imaging.sequence.angles_deg
    assert len(angles) == 9 and (angles[0], angles[-1]) == (-10.0, 10.0)
    assert cfg.head.grid.spacing == 0.5e-3 and cfg.head.grid.region == "box"
    assert cfg.backend.backend == "bent_ray" and trial_fidelity(cfg) == "statistical"
    assert cfg.effects == SkullEffects()
    # Apart from the choices the file names, it is the package defaults.
    defaults = RunConfig()
    assert cfg.head == defaults.head and cfg.ct == defaults.ct
    assert cfg.backend == replace(defaults.backend, backend="bent_ray")
    assert cfg.imaging == replace(defaults.imaging, fidelity="statistical")


def test_demo_pixel_grid_is_the_grid_of_the_diffusion_prior():
    demo = load_run_config("run_demo.yaml")
    # Only the pixel-grid section of the prior config is read, so that this test does
    # not import the training code (and torch).
    prior_grid = load_yaml(CONFIG_DIR / "prior_default.yaml")["data"]["pixel_grid"]
    grid = pixel_grid_from_config(demo.imaging)
    expected = pixel_grid(from_dict(PixelGridConfig, prior_grid))
    assert grid.shape == expected.shape == (32, 32)
    assert grid.spacing == expected.spacing == 1.0e-3
    assert grid.origin == expected.origin == (12.0e-3, -15.5e-3)
    assert grid == expected == pixel_grid(PixelGridConfig())
    # The ROI holds every pixel cell (half a pixel beyond the outer pixel centres).
    roi_z, roi_x = roi_grid_from_config(demo.imaging).axes()
    z, x = grid.axes()
    half = 0.5 * grid.spacing
    assert roi_z[0] <= z[0] - half and roi_z[-1] >= z[-1] + half
    assert roi_x[0] <= x[0] - half and roi_x[-1] >= x[-1] + half


@pytest.mark.parametrize(
    ("sweep_name", "run_name"),
    [("sweep_small.yaml", "run_small.yaml"), ("sweep_demo.yaml", "run_demo.yaml")],
)
def test_sweep_base_is_the_matching_run_config(sweep_name, run_name):
    sweep = from_dict(SweepConfig, load_yaml(CONFIG_DIR / sweep_name))
    run = load_run_config(run_name)
    # Only the image series may differ (the demo sweep uses fewer images and block jitter).
    assert replace(sweep.base, trials=run.trials) == run
    assert trial_fidelity(sweep.base) == "statistical"


def test_demo_sweep_is_on_the_prior_grid_with_rest_and_task_images():
    sweep = from_dict(SweepConfig, load_yaml(CONFIG_DIR / "sweep_demo.yaml"))
    assert sweep.n_scenes == 8
    assert pixel_grid_from_config(sweep.base.imaging) == pixel_grid(PixelGridConfig())
    assert 20 <= sweep.base.trials.n_images <= 120
    # The centre frequency is not swept, so every scene keeps that grid and array.
    assert sweep.center_frequencies == ()
