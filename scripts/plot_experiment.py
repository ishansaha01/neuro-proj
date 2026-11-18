"""Plot the results of the reconstruction experiment as one figure.

Usage: ``python scripts/plot_experiment.py --results outputs/experiment_v2 --out
outputs/experiment_v2/figure.png``. Reads ``results.json`` and ``examples.npz`` written by
``scripts/run_experiment.py`` and follows the design recorded there.

Medium design. Top: the true rest CBV map and task-minus-rest map of the first scene and
every method's two maps (each image has its own colour scale; negative values of a
difference map are shown as zero). Bottom: three metric panels (mean over the scenes a
method was scored on, bars = standard deviation over scenes) and the skull-model
travel-time error of every scene before and after calibration.

Screen design (the first design). Top: the true CBV image and every method's image for
one rest and one task trial of the first scene. Bottom: one panel per image metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from neurosim.recon import experiment as screen_design
from neurosim.recon import medium_experiment as medium_design

# Colours: a three-hue categorical palette for the method families (validated for
# colour-vision deficiency on all pairs against the light surface), neutral grey for the
# control, and a one-hue light-to-dark ramp for image magnitude.
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e4e3df"
_BLUE, _ORANGE, _GREEN = "#2a78d6", "#eb6834", "#1baf7a"
_MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", [_SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
# Upper percentile of an image that maps to the darkest colour.
_COLOUR_PERCENTILE = 99.5


@dataclass(frozen=True)
class _Layout:
    """What the figure of one experiment design shows."""

    methods: tuple[str, ...]
    families: tuple[tuple[str, str, tuple[str, ...]], ...]  # (legend label, colour, methods)
    metrics: dict[str, str]  # metric -> panel title
    chance_metric: str  # the metric whose chance level 0.5 is marked
    example_rows: tuple[str, str]
    shared_scale: bool  # the two example images of a column share one colour scale
    caption: str


_SCREEN = _Layout(
    methods=screen_design.METHODS,
    families=(
        ("raw power Doppler", _BLUE, screen_design.RAW_METHODS),
        ("deconvolution (no learned prior)", _ORANGE, screen_design.DECONVOLUTION_METHODS),
        ("diffusion posterior sampling", _GREEN, screen_design.DPS_METHODS),
        ("prior only (control)", _MUTED, (screen_design.PRIOR_ONLY,)),
    ),
    metrics={
        "decoding_accuracy": "Decoding accuracy (chance 0.5)",
        "activation_dice": "Activation overlap (Dice)",
        "cbv_correlation": "CBV correlation in the brain",
        "activation_cnr": "Activation contrast-to-noise",
    },
    chance_metric="decoding_accuracy",
    example_rows=("rest", "task"),
    shared_scale=True,
    caption="Top: one rest and one task trial of scene 0; every column has its own colour "
    "scale (light = 0, dark = its 99.5th percentile). Bottom: mean over scenes, bars = "
    "standard deviation over scenes.",
)
_MEDIUM = _Layout(
    methods=medium_design.METHODS,
    families=(
        ("beamformed power Doppler", _BLUE, medium_design.BEAMFORMED_METHODS),
        ("deconvolution (no learned prior)", _ORANGE, medium_design.DECONVOLUTION_METHODS),
        ("diffusion posterior sampling", _GREEN, medium_design.DPS_METHODS),
        ("prior only (control)", _MUTED, (medium_design.PRIOR_ONLY,)),
    ),
    metrics={
        "baseline_correlation": "Baseline CBV correlation in the brain",
        "activation_correlation": "Activation-map correlation",
        "decoding_matched_filter": "Matched-filter decoding (chance 0.5)",
    },
    chance_metric="decoding_matched_filter",
    example_rows=("rest map", "task - rest"),
    shared_scale=False,
    caption="Top: scene 0, rest map and task-minus-rest map (each image has its own colour "
    "scale: light = 0 or below, dark = its 99.5th percentile).\nBottom: mean over the scenes "
    "a method was scored on, bars = standard deviation over scenes; right: one-way "
    "travel-time error of the skull model per scene.",
)


def _method_colour(layout: _Layout, method: str) -> str:
    return next(colour for _, colour, members in layout.families if method in members)


def _plot_examples(
    figure: plt.Figure, grid, examples: dict[str, np.ndarray], layout: _Layout
) -> None:
    for column, name in enumerate(("truth", *layout.methods)):
        for row, label in enumerate(layout.example_rows):
            axis = figure.add_subplot(grid[row, column])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(_GRID)
            if row == 0:
                axis.set_title(name.replace("_", "\n", 1), fontsize=6.5, color=_TEXT)
            if column == 0:
                axis.set_ylabel(label, fontsize=9, color=_TEXT_SECONDARY)
            if name not in examples:  # the method failed on this scene: an empty grey image
                blank = np.zeros(examples["truth"].shape[1:])
                axis.imshow(blank, cmap=LinearSegmentedColormap.from_list("blank", [_GRID] * 2))
                continue
            pair = examples[name]
            scaled = pair if layout.shared_scale else pair[row]
            upper = max(float(np.percentile(scaled, _COLOUR_PERCENTILE)), 1.0e-30)
            axis.imshow(pair[row], cmap=_MAGNITUDE, vmin=0.0, vmax=upper, interpolation="nearest")


def _style_panel(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=9.5, color=_TEXT, loc="left")
    axis.tick_params(axis="x", labelsize=8, colors=_TEXT_SECONDARY, length=0)
    axis.tick_params(axis="y", length=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_facecolor(_SURFACE)


def _plot_metrics(figure: plt.Figure, grid, summary: dict, layout: _Layout) -> None:
    positions = np.arange(len(layout.methods))[::-1]
    colours = [_method_colour(layout, method) for method in layout.methods]
    for panel, (metric, title) in enumerate(layout.metrics.items()):
        axis = figure.add_subplot(grid[0, panel])
        entries = [summary[method].get(metric) for method in layout.methods]
        # A method without this metric (not computed, or failed on every scene) has no bar.
        means = np.array([np.nan if entry is None else entry["mean"] for entry in entries])
        spreads = np.array([0.0 if entry is None else entry["std"] for entry in entries])
        shown = ~np.isnan(means)
        axis.barh(
            positions[shown],
            means[shown],
            height=0.62,
            color=[colour for colour, keep in zip(colours, shown) if keep],
            edgecolor=_SURFACE,
            linewidth=1.0,
        )
        axis.errorbar(
            means[shown],
            positions[shown],
            xerr=spreads[shown],
            fmt="none",
            ecolor=_TEXT_SECONDARY,
            elinewidth=1.0,
            capsize=2,
        )
        if metric == layout.chance_metric:
            axis.axvline(0.5, color=_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        axis.axvline(0.0, color=_MUTED, linewidth=0.8)
        low = min(0.0, float((means[shown] - spreads[shown]).min()))
        high = max(float((means[shown] + spreads[shown]).max()), 1.0e-9)
        span = high - low
        axis.set_xlim(low - 0.04 * span, high + 0.22 * span)
        for position, mean, spread, keep in zip(positions, means, spreads, shown):
            axis.text(
                (max(mean, 0.0) + spread if keep else 0.0) + 0.02 * span,
                position,
                f"{mean:.2f}" if keep else "not scored",
                va="center",
                ha="left",
                fontsize=7.5,
                color=_TEXT if keep else _MUTED,
            )
        axis.set_yticks(positions)
        axis.set_yticklabels(layout.methods if panel == 0 else [], fontsize=8.5, color=_TEXT)
        axis.grid(axis="x", color=_GRID, linewidth=0.8)
        _style_panel(axis, title)


def _plot_skull_errors(axis: plt.Axes, scenes: list[dict]) -> None:
    """Travel-time error of the nominal and the calibrated skull model, one row per scene."""
    rows = np.arange(len(scenes))[::-1]
    nominal = np.array([scene["skull"]["travel_time_error_nominal_common_ns"] for scene in scenes])
    calibrated = np.array(
        [scene["skull"]["travel_time_error_calibrated_common_ns"] for scene in scenes]
    )
    limit = np.array([scene["skull"]["family_limit_lateral_ns"] for scene in scenes])
    axis.hlines(rows, np.minimum(nominal, calibrated), np.maximum(nominal, calibrated),
                color=_GRID, linewidth=2.0, zorder=1)
    axis.scatter(nominal, rows, s=42, color=_MUTED, zorder=3, label="nominal CT model")
    axis.scatter(calibrated, rows, s=42, color=_BLUE, zorder=3, label="calibrated")
    axis.scatter(limit, rows, s=70, marker="|", color=_TEXT, zorder=2,
                 label="best fit of the family")
    axis.set_yticks(rows)
    axis.set_yticklabels([f"scene {scene['index']}" for scene in scenes], fontsize=8.5, color=_TEXT)
    axis.set_xlim(0.0, 1.12 * float(max(nominal.max(), calibrated.max())))
    axis.set_xlabel(
        "ns (250 ns = half a period at 2 MHz)", fontsize=8, color=_TEXT_SECONDARY, loc="left"
    )
    axis.grid(axis="x", color=_GRID, linewidth=0.8)
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, -0.09),
        ncol=3,
        frameon=False,
        fontsize=7.5,
        labelcolor=_TEXT,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    _style_panel(axis, "Skull-model travel-time error (rms)")


def plot_experiment(results: dict, examples: dict[str, np.ndarray]) -> plt.Figure:
    """Build the figure from the contents of ``results.json`` and ``examples.npz``."""
    medium = results.get("design", "screen") == "medium"
    layout = _MEDIUM if medium else _SCREEN
    figure = plt.figure(figsize=(17.0 if medium else 15.0, 8.8), facecolor=_SURFACE)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.75 if medium else 1.55),
        hspace=0.2,
        left=0.085,
        right=0.985,
        top=0.875,
        bottom=0.085,
    )
    example_grid = outer[0].subgridspec(2, len(layout.methods) + 1, wspace=0.06, hspace=0.06)
    _plot_examples(figure, example_grid, examples, layout)
    n_panels = len(layout.metrics) + (1 if medium else 0)
    metric_grid = outer[1].subgridspec(
        1, n_panels, wspace=0.08, width_ratios=[1.0] * len(layout.metrics) + [1.25] * medium
    )
    _plot_metrics(figure, metric_grid, results["summary"], layout)
    if medium:
        axis = figure.add_subplot(metric_grid[0, n_panels - 1])
        axis.set_position(_shifted_right(axis.get_position(), 0.035))
        _plot_skull_errors(axis, results["scenes"])
    n_scenes = len(results["scenes"])
    label = "simulated dataset"
    if results["synthetic_dataset"]:
        label = "SYNTHETIC phase-screen dataset"
    figure.suptitle(
        f"Reconstruction experiment, {layout_name(medium)}, {label} "
        f"({n_scenes} scenes, seed {results['root_seed']})",
        fontsize=12.5,
        color=_TEXT,
        x=0.085,
        y=0.985,
        ha="left",
    )
    figure.text(
        0.085, 0.955, layout.caption, fontsize=8.5, color=_TEXT_SECONDARY, va="top", linespacing=1.4
    )
    figure.legend(
        handles=[Patch(facecolor=colour, label=name) for name, colour, _ in layout.families],
        loc="lower left",
        bbox_to_anchor=(0.085, 0.005),
        ncol=len(layout.families),
        frameon=False,
        fontsize=8.5,
        labelcolor=_TEXT,
    )
    return figure


def layout_name(medium: bool) -> str:
    """Name of the experiment design in the figure title."""
    return "CT skull model calibrated by re-beamforming" if medium else "blind phase screen"


def _shifted_right(box, shift: float):
    """An axes position moved and narrowed from the left, to make room for row labels."""
    return [box.x0 + shift, box.y0, box.width - shift, box.height]


def main(argv: list[str]) -> int:
    """Plot from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="directory written by run_experiment.py")
    parser.add_argument("--out", required=True, help="output image file (.png)")
    args = parser.parse_args(argv)
    directory = Path(args.results)
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    with np.load(directory / "examples.npz") as archive:
        examples = {name: archive[name] for name in archive.files}
    figure = plot_experiment(results, examples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
