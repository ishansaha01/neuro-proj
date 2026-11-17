import dataclasses

import numpy as np
import yaml

from neurosim.common.config import from_dict
from neurosim.common.io import load_bundle
from neurosim.fus.backends.phantoms import slab_medium
from neurosim.fus.config import RoiConfig
from neurosim.fus.io import load_bank
from neurosim.head.io import load_plane_medium, save_plane_medium
from scripts.compute_bank import BankJob, main

JOB = {
    "transducer": {
        "n_elements": 3,
        "pitch": 1.2e-3,
        "element_width": 0.3e-3,
        "center_frequency": 0.5e6,
        "fractional_bandwidth": 0.8,
    },
    "roi": {"x_range": [-3.0e-3, 3.0e-3], "z_range": [6.0e-3, 9.0e-3], "spacing": 1.5e-3},
    "frequencies": {"first": 0.4e6, "last": 0.6e6, "count": 3},
    "backend": {"backend": "ray"},
}


def write_inputs(tmp_path, medium):
    save_plane_medium(tmp_path / "plane.h5", medium, {"root_seed": 7})
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(JOB), encoding="utf-8")
    return str(tmp_path / "plane.h5"), str(tmp_path / "job.yaml")


def test_plane_medium_round_trip(tmp_path, grid, soft, bone, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.3)
    medium_path, _ = write_inputs(tmp_path, medium)
    loaded = load_plane_medium(medium_path)
    assert load_bundle(medium_path)[1]["root_seed"] == 7
    assert loaded.grid == medium.grid and loaded.plane == medium.plane
    assert loaded.alpha_power == 1.3
    for field in dataclasses.fields(medium):
        if field.name not in ("grid", "plane", "alpha_power"):
            np.testing.assert_array_equal(getattr(loaded, field.name), getattr(medium, field.name))


def test_job_roi_is_the_imaging_roi_config():
    job = from_dict(BankJob, JOB)
    assert job.roi == RoiConfig(x_range=(-3.0e-3, 3.0e-3), z_range=(6.0e-3, 9.0e-3), spacing=1.5e-3)
    # One ROI schema for the bank job and the imaging config: omitted keys take its defaults.
    assert from_dict(BankJob, {**JOB, "roi": {}}).roi == RoiConfig()


def test_script_prints_the_estimate_and_saves_the_bank(tmp_path, grid, soft, bone, spacing, capsys):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.3)
    medium_path, job_path = write_inputs(tmp_path, medium)
    out = tmp_path / "bank.h5"

    assert main(["--medium", medium_path, "--config", job_path, "--estimate-only"]) == 0
    assert "Green's bank (ray): 0 simulations" in capsys.readouterr().out
    assert not out.exists()

    assert main(["--medium", medium_path, "--config", job_path, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.index("Green's bank (ray)") < printed.index("saved")
    arrays, attrs = load_bundle(out)
    assert arrays["B"].shape == (3, 3, 3, 5) and arrays["B"].dtype == np.complex64
    np.testing.assert_allclose(arrays["freqs"], [0.4e6, 0.5e6, 0.6e6])
    assert "static_fmc" not in arrays
    assert attrs["backend"] == "ray" and attrs["root_seed"] == 7 and attrs["roi_shape"] == [3, 5]
    assert "git_commit" in attrs and "backend: ray" in attrs["config_yaml"]
    bank = load_bank(out)
    assert bank.transducer.n_elements == 3 and bank.transducer.pitch == 1.2e-3
    np.testing.assert_array_equal(bank.B, arrays["B"])
