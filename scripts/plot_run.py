"""Plot one saved run: the skull, the CBV truth, the four power Doppler images and the response.

Usage: ``python scripts/plot_run.py --run outputs/demo/run.h5 --out outputs/demo/run.png``.
The run must have been saved with ``scripts/run_pipeline.py --save-media``.

Top row: true and CT-derived sound speed of the imaging plane (the pixel grid is outlined),
the resting CBV truth with the half-maximum contour of the activation pattern, and the
fractional change of the activated region over time. Rows two and three: the power Doppler
image of every beamformer for the image with the smallest and the largest response.

Power Doppler is shown after subtracting the scene's noise floor of that beamformer
(negative values are drawn as zero), each beamformer on its own scale: the largest value of
its two images. The regional time course of a series is the activation-weighted sum of
(power Doppler - noise floor), relative to its mean over the images whose true response is
below 5 % of the largest; the truth is the same statistic of the CBV ground truth.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from neurosim.common.grid import Grid2D  # noqa: E402
from neurosim.data.io import load_run, media_paths  # noqa: E402
from neurosim.data.types import CLEAN_DAS, SKULL_CT, SKULL_DAS, SKULL_TRUE  # noqa: E402
from neurosim.head.io import load_plane_medium  # noqa: E402

_MM_PER_M = 1.0e3
# Single-hue ramp, light to dark, for every magnitude map (as in scripts/plot_head.py).
_MAGNITUDE_COLORS = ("#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
_INK = "#0b0b0b"
_MUTED_INK = "#52514e"
_TASK_BAND = "#e1e0d9"
# Line colours of the time courses: neutral ink for the truth, then two categorical hues.
_SERIES = {
    "truth": ("CBV ground truth", _INK),
    CLEAN_DAS: ("no skull", "#2a78d6"),
    SKULL_DAS: ("skull, uncorrected", "#eb6834"),
}
_COLUMNS = (
    (CLEAN_DAS, "no skull (clean)"),
    (SKULL_DAS, "skull, uncorrected"),
    (SKULL_CT, "skull, CT-corrected"),
    (SKULL_TRUE, "skull, true-model corrected"),
)
# Images whose true response is below this fraction of the largest count as rest.
_REST_FRACTION = 0.05


def _extent(grid: Grid2D) -> tuple[float, float, float, float]:
    """imshow extent (left, right, bottom, top) of a plane grid [mm], depth downwards."""
    z, x = grid.axes()
    half = 0.5 * grid.spacing
    return (
        (x[0] - half) * _MM_PER_M,
        (x[-1] + half) * _MM_PER_M,
        (z[-1] + half) * _MM_PER_M,
        (z[0] - half) * _MM_PER_M,
    )


def regional_change(series: np.ndarray, weights: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Fractional change of the weighted regional sum of an image series.

    series: images, shape (n, nz, nx).
    weights: regional weights, shape (nz, nx).
    rest: True for the images that define the baseline, shape (n,).

    Returns the weighted sum per image divided by its mean over the rest images, minus 1.
    """
    regional = np.tensordot(series.astype(np.float64), weights.astype(np.float64), axes=2)
    return regional / regional[rest].mean() - 1.0


def task_blocks(time: np.ndarray, label: np.ndarray) -> list[tuple[float, float]]:
    """Time spans [s] of the runs of task-labelled images, as (first, last) image times."""
    steps = np.diff(np.concatenate(([0], label, [0])))
    first, last = np.flatnonzero(steps == 1), np.flatnonzero(steps == -1) - 1
    return [(float(time[a]), float(time[b])) for a, b in zip(first, last)]


def main(argv: list[str]) -> int:
    """Plot a saved run from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run bundle written by run_pipeline.py")
    parser.add_argument("--out", required=True, help="output PNG file")
    args = parser.parse_args(argv)

    scene, trials = load_run(args.run)
    medium_true, medium_ct = (load_plane_medium(path) for path in media_paths(args.run))
    shape = scene.pixel_grid.shape
    floors = {key: scene.noise_floor[key].reshape(shape) for key, _ in _COLUMNS}
    blood = {key: trials.power_doppler[key] - floors[key] for key, _ in _COLUMNS}
    rest = np.abs(trials.amplitude) < _REST_FRACTION * np.abs(trials.amplitude).max()
    shown = {
        "rest": int(np.argmin(np.abs(trials.amplitude))),
        "task": int(np.argmax(trials.amplitude)),
    }

    ramp = LinearSegmentedColormap.from_list("magnitude", _MAGNITUDE_COLORS)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "text.color": _INK})
    figure, axes = plt.subplots(3, 4, figsize=(15.5, 11.5), constrained_layout=True)

    speed_range = (
        float(min(medium_true.sound_speed.min(), medium_ct.sound_speed.min())),
        float(max(medium_true.sound_speed.max(), medium_ct.sound_speed.max())),
    )
    pixel_box = _extent(scene.pixel_grid)
    r2 = scene.meta["ct_sound_speed_r2"]
    for axis, medium, title in (
        (axes[0, 0], medium_true, "true sound speed"),
        (axes[0, 1], medium_ct, f"CT-derived sound speed (skull R$^2$ = {r2:.2f})"),
    ):
        image = axis.imshow(
            medium.sound_speed,
            cmap=ramp,
            vmin=speed_range[0],
            vmax=speed_range[1],
            extent=_extent(medium.grid),
            interpolation="nearest",
        )
        axis.add_patch(
            Rectangle(
                (pixel_box[0], pixel_box[3]),
                pixel_box[1] - pixel_box[0],
                pixel_box[2] - pixel_box[3],
                fill=False,
                edgecolor=_INK,
                linewidth=1.0,
                linestyle="--",
            )
        )
        axis.set_title(title)
    figure.colorbar(image, ax=axes[0, :2], label="m/s", shrink=0.8, location="bottom")

    cbv_axis = axes[0, 2]
    cbv_image = cbv_axis.imshow(
        scene.cbv_baseline, cmap=ramp, vmin=0.0, extent=pixel_box, interpolation="nearest"
    )
    z, x = scene.pixel_grid.axes()
    cbv_axis.contour(
        x * _MM_PER_M,
        z * _MM_PER_M,
        scene.activation,
        levels=[0.5],
        colors=_SERIES[SKULL_DAS][1],
        linewidths=1.5,
    )
    cbv_axis.set_title("resting CBV truth; line: activation at half maximum")
    figure.colorbar(
        cbv_image, ax=cbv_axis, label="blood volume fraction", shrink=0.8, location="bottom"
    )

    course_axis = axes[0, 3]
    for start, stop in task_blocks(trials.time, trials.label):
        course_axis.axvspan(start, stop, color=_TASK_BAND, linewidth=0)
    courses = {
        "truth": regional_change(trials.cbv, scene.activation, rest),
        CLEAN_DAS: regional_change(blood[CLEAN_DAS], scene.activation, rest),
        SKULL_DAS: regional_change(blood[SKULL_DAS], scene.activation, rest),
    }
    for key, (label, colour) in _SERIES.items():
        course_axis.plot(trials.time, courses[key], color=colour, linewidth=2.0, label=label)
    course_axis.axhline(0.0, color=_MUTED_INK, linewidth=0.5)
    course_axis.set_xlabel("time [s] (shaded: task blocks)")
    course_axis.set_ylabel("fractional change of the activated region")
    course_axis.set_title("response of the activated region")
    course_axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncols=3,
        fontsize=8,
        columnspacing=1.0,
        handlelength=1.5,
    )
    for side in ("top", "right"):
        course_axis.spines[side].set_visible(False)

    for column, (key, title) in enumerate(_COLUMNS):
        scale = max(float(blood[key][index].max()) for index in shown.values())
        for row, (state, index) in enumerate(shown.items(), start=1):
            axis = axes[row, column]
            pd_image = axis.imshow(
                np.clip(blood[key][index] / scale, 0.0, None),
                cmap=ramp,
                vmin=0.0,
                vmax=1.0,
                extent=pixel_box,
                interpolation="nearest",
            )
            axis.set_title(
                f"{title}\n{state} image, t = {trials.time[index]:.0f} s, "
                f"a = {trials.amplitude[index]:+.2f}"
            )
    figure.colorbar(
        pd_image,
        ax=axes[1:, :],
        label="power Doppler minus noise floor, relative to the largest value of the column",
        shrink=0.5,
        location="bottom",
    )
    for axis in axes.flat:
        if axis is not course_axis:
            axis.set_xlabel("lateral x [mm]")
            axis.set_ylabel("depth z [mm]")

    figure.savefig(args.out, dpi=110)
    plt.close(figure)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
