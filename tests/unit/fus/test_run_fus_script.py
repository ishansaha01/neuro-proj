"""scripts/run_fus.py: end-to-end smoke test on the tiny setup."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neurosim.common.config import config_hash, dump_yaml
from neurosim.common.io import load_bundle, save_bundle
from scripts.run_fus import main


@pytest.fixture()
def inputs(tmp_path, tiny, tiny_cfg):
    """A tiny config file and a CBV bundle on the ROI grid."""
    quiet = replace(tiny_cfg, noise=replace(tiny_cfg.noise, tissue_motion=False))
    config_path = tmp_path / "fus.yaml"
    config_path.write_text(dump_yaml(quiet), encoding="utf-8")
    cbv_path = tmp_path / "cbv.h5"
    rng = np.random.default_rng(0)
    save_bundle(
        cbv_path,
        {
            "cbv_baseline": rng.uniform(0.01, 0.05, tiny.roi.shape).astype(np.float32),
            "blood_velocity": np.full((2, *tiny.roi.shape), 0.02, dtype=np.float32),
        },
        {
            "grid_shape": list(tiny.roi.shape),
            "grid_spacing": tiny.roi.spacing,
            "grid_origin": list(tiny.roi.origin),
        },
    )
    return quiet, config_path, cbv_path


def _run(config_path, cbv_path, out_path, *extra: str) -> int:
    return main(
        ["--config", str(config_path), "--water", "--cbv", str(cbv_path), "--seed", "3", *extra]
        + (["--out", str(out_path)] if out_path else [])
    )


def test_scatterer_run_saves_frames_and_provenance(tmp_path, inputs, capsys):
    cfg, config_path, cbv_path = inputs
    out_path = tmp_path / "fus.h5"
    assert _run(config_path, cbv_path, out_path) == 0
    assert capsys.readouterr().out.startswith("fUS (scatterer): 6 frames")
    arrays, attrs = load_bundle(out_path)
    assert arrays["frames"].shape == (6, 8, 8) and arrays["frames"].dtype == np.complex64
    assert arrays["power_doppler"].shape == (8, 8) and arrays["power_doppler"].dtype == np.float32
    assert attrs["root_seed"] == 3 and attrs["config_hash"] == config_hash(cfg)
    assert attrs["backend"] == "analytic" and attrs["n_discard"] == 1

    again = tmp_path / "again.h5"
    _run(config_path, cbv_path, again)
    assert again.read_bytes() == out_path.read_bytes()


def test_statistical_run(tmp_path, inputs):
    cfg, config_path, cbv_path = inputs
    config_path.write_text(dump_yaml(replace(cfg, fidelity="statistical")), encoding="utf-8")
    out_path = tmp_path / "statistical.h5"
    assert _run(config_path, cbv_path, out_path) == 0
    arrays, _ = load_bundle(out_path)
    assert set(arrays) == {"power_doppler"}
    assert np.all(arrays["power_doppler"] > 0.0)


def test_estimate_only_writes_nothing(tmp_path, inputs, capsys):
    _, config_path, cbv_path = inputs
    assert _run(config_path, cbv_path, None, "--estimate-only") == 0
    assert "flops" in capsys.readouterr().out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cbv.h5", "fus.yaml"]


def test_bank_files_are_read(tmp_path, inputs, tiny, water):
    """A bank bundle in the format of scripts/compute_bank.py, used as medium and as model."""
    _, config_path, cbv_path = inputs
    bank_path = tmp_path / "bank.h5"
    save_bundle(
        bank_path,
        {"B": 0.5 * water.B, "freqs": water.freqs},
        {
            "roi_shape": list(water.roi.shape),
            "roi_spacing": water.roi.spacing,
            "roi_origin": list(water.roi.origin),
            "c_ref": water.c_ref,
            "backend": "ray",
        },
    )
    common = ["--config", str(config_path), "--cbv", str(cbv_path), "--seed", "3"]
    conventional, corrected, in_water = (tmp_path / name for name in ("a.h5", "b.h5", "c.h5"))
    assert main([*common, "--bank", str(bank_path), "--out", str(conventional)]) == 0
    assert (
        main(
            [
                *common,
                "--bank",
                str(bank_path),
                "--model-bank",
                str(bank_path),
                "--out",
                str(corrected),
            ]
        )
        == 0
    )
    assert main([*common, "--water", "--out", str(in_water)]) == 0
    images = [load_bundle(path)[0]["power_doppler"] for path in (conventional, corrected, in_water)]
    assert load_bundle(conventional)[1]["backend"] == "ray"
    # B = 0.5 on transmit and receive: the conventional image loses a factor 16 in power;
    # the true model restores the blood signal (the unchanged noise is amplified instead).
    assert images[0].mean() < images[2].mean()
    assert images[1].mean() > images[0].mean()
