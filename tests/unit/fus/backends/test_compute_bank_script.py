import dataclasses

import numpy as np
import yaml

from neurosim.common.io import load_bundle, save_bundle
from neurosim.fus.backends.phantoms import slab_medium
from scripts.compute_bank import _ARRAY_FIELDS, load_plane_medium, main

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
    attrs = {
        "grid_shape": list(medium.grid.shape),
        "grid_spacing": medium.grid.spacing,
        "grid_origin": list(medium.grid.origin),
        "alpha_power": medium.alpha_power,
        "plane_origin": list(medium.plane.origin),
        "plane_u_lateral": list(medium.plane.u_lateral),
        "plane_u_depth": list(medium.plane.u_depth),
        "root_seed": 7,
    }
    arrays = {name: getattr(medium, name) for name in _ARRAY_FIELDS}
    save_bundle(tmp_path / "plane.h5", arrays, attrs)
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(JOB), encoding="utf-8")
    return str(tmp_path / "plane.h5"), str(tmp_path / "job.yaml")


def test_plane_medium_round_trip(tmp_path, grid, soft, bone, spacing):
    medium = slab_medium(grid, soft, bone, 9.5 * spacing, 6 * spacing, alpha_power=1.3)
    medium_path, _ = write_inputs(tmp_path, medium)
    loaded, root_seed = load_plane_medium(medium_path)
    assert root_seed == 7 and loaded.grid == medium.grid and loaded.plane == medium.plane
    assert loaded.alpha_power == 1.3
    for field in dataclasses.fields(medium):
        if field.name in _ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(loaded, field.name), getattr(medium, field.name))


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
