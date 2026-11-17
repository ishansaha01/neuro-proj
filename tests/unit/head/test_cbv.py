import numpy as np
import pytest

from neurosim.head.cbv import cbv_at, hrf, task_labels, task_timecourse
from neurosim.head.config import TaskConfig


def test_hrf_has_unit_peak_near_five_seconds_and_an_undershoot():
    t = np.linspace(-5.0, 40.0, 45001)
    response = hrf(t)
    assert response.shape == t.shape
    assert response.max() == pytest.approx(1.0, abs=1e-6)
    assert t[np.argmax(response)] == pytest.approx(5.0, abs=0.3)
    assert np.all(response[t < 0.0] == 0.0)
    assert response.min() < -0.05
    assert abs(response[-1]) < 1e-3


def test_task_labels_mark_the_blocks():
    task = TaskConfig(onsets=(10.0, 50.0), block_duration=20.0)
    t = np.array([0.0, 9.99, 10.0, 29.99, 30.0, 49.0, 55.0, 70.0])
    labels = task_labels(task, t)
    assert labels.dtype == np.int8
    assert labels.tolist() == [0, 0, 1, 1, 0, 0, 1, 0]
    assert task_labels(task, 12.0).shape == ()


def test_timecourse_matches_a_numerical_convolution():
    task = TaskConfig(onsets=(5.0, 60.0), block_duration=12.0, peak_fractional_change=0.3)
    dt = 0.005
    t = np.arange(0.0, 120.0, dt)
    numerical = np.convolve(task_labels(task, t).astype(float), hrf(t))[: t.size] * dt
    exact = task_timecourse(task, t)
    scale = exact.max() / numerical.max()
    np.testing.assert_allclose(scale * numerical, exact, atol=2e-3 * exact.max())


def test_long_block_peaks_at_the_configured_change_and_short_blocks_lower():
    long_block = TaskConfig(onsets=(0.0,), block_duration=40.0, peak_fractional_change=0.2)
    short_block = TaskConfig(onsets=(0.0,), block_duration=2.0, peak_fractional_change=0.2)
    t = np.linspace(0.0, 80.0, 16001)
    assert task_timecourse(long_block, t).max() == pytest.approx(0.2, rel=1e-4)
    assert 0.0 < task_timecourse(short_block, t).max() < 0.1
    assert task_timecourse(long_block, np.array([-3.0, 0.0])) == pytest.approx(0.0)


def test_cbv_at_broadcasts_over_time():
    baseline = np.full((3, 4), 0.04, dtype=np.float32)
    activation = np.zeros((3, 4), dtype=np.float32)
    activation[1, 2] = 0.5
    single = cbv_at(baseline, activation, 0.2)
    assert single.shape == (3, 4) and single.dtype == np.float32
    assert single[1, 2] == pytest.approx(0.04 * 1.1)
    assert single[0, 0] == pytest.approx(0.04)
    series = cbv_at(baseline, activation, np.array([0.0, 0.2, 0.4]))
    assert series.shape == (3, 3, 4)
    assert series[:, 1, 2] == pytest.approx([0.04, 0.044, 0.048])
