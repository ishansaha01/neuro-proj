"""Plot the results of the reconstruction experiment as one figure.

Usage: ``python scripts/plot_experiment.py --results outputs/experiment --out
outputs/experiment/figure.png``. Reads ``results.json`` and ``examples.npz`` written by
``scripts/run_experiment.py``.

Top: the true CBV image and every method's image for one rest and one task trial of the
first scene (each column has its own colour scale, because raw power Doppler images are
not in CBV units). Bottom: one panel per image metric, mean over scenes with the standard
deviation over scenes as a bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from neurosim.recon.experiment import (
    DECONVOLUTION_METHODS,
    DPS_METHODS,
    IMAGE_METRICS,
    METHODS,
    PRIOR_ONLY,
    RAW_METHODS,
)

# Colours: a three-hue categorical palette for the method families (validated for
# colour-vision deficiency on all pairs against the light surface), neutral grey for the
# control, and a one-hue light-to-dark ramp for image magnitude.
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e4e3df"
_FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("raw power Doppler", "#2a78d6", RAW_METHODS),
    ("deconvolution (no learned prior)", "#eb6834", DECONVOLUTION_METHODS),
    ("diffusion posterior sampling", "#1baf7a", DPS_METHODS),
    ("prior only (control)", _MUTED, (PRIOR_ONLY,)),
)
_MAGNITUDE = LinearSegmentedColormap.from_list(
    "magnitude", [_SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
_METRIC_TITLES = {
    "decoding_accuracy": "Decoding accuracy (chance 0.5)",
    "activation_dice": "Activation overlap (Dice)",
    "cbv_correlation": "CBV correlation in the brain",
    "activation_cnr": "Activation contrast-to-noise",
}
# Upper percentile of a column's two images that maps to the darkest colour.
_COLOUR_PERCENTILE = 99.5


def _method_colour(method: str) -> str:
    return next(colour for _, colour, members in _FAMILIES if method in members)


def _plot_examples(figure: plt.Figure, grid, examples: dict[str, np.ndarray]) -> None:
    columns = ("truth", *METHODS)
    for column, name in enumerate(columns):
        pair = examples[name]
        upper = float(np.percentile(pair, _COLOUR_PERCENTILE))
        for row, state in enumerate(("rest", "task")):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(pair[row], cmap=_MAGNITUDE, vmin=0.0, vmax=upper, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(_GRID)
            if row == 0:
                axis.set_title(name.replace("_", "\n", 1), fontsize=7.5, color=_TEXT)
            if column == 0:
                axis.set_ylabel(state, fontsize=9, color=_TEXT_SECONDARY)


def _plot_metrics(figure: plt.Figure, grid, summary: dict) -> None:
    positions = np.arange(len(METHODS))[::-1]
    colours = [_method_colour(method) for method in METHODS]
    for panel, metric in enumerate(IMAGE_METRICS):
        axis = figure.add_subplot(grid[0, panel])
        means = np.array([summary[method][metric]["mean"] for method in METHODS])
        spreads = np.array([summary[method][metric]["std"] for method in METHODS])
        axis.barh(positions, means, height=0.62, color=colours, edgecolor=_SURFACE, linewidth=1.0)
        axis.errorbar(
            means,
            positions,
            xerr=spreads,
            fmt="none",
            ecolor=_TEXT_SECONDARY,
            elinewidth=1.0,
            capsize=2,
        )
        if metric == "decoding_accuracy":
            axis.axvline(0.5, color=_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        axis.axvline(0.0, color=_MUTED, linewidth=0.8)
        low = min(0.0, float((means - spreads).min()))
        high = max(float((means + spreads).max()), 1.0e-9)
        span = high - low
        axis.set_xlim(low - 0.04 * span, high + 0.22 * span)
        for position, mean, spread in zip(positions, means, spreads):
            axis.text(
                max(mean, 0.0) + spread + 0.02 * span,
                position,
                f"{mean:.2f}",
                va="center",
                ha="left",
                fontsize=7.5,
                color=_TEXT,
            )
        axis.set_yticks(positions)
        axis.set_yticklabels(METHODS if panel == 0 else [], fontsize=8.5, color=_TEXT)
        axis.set_title(_METRIC_TITLES[metric], fontsize=9.5, color=_TEXT, loc="left")
        axis.tick_params(axis="x", labelsize=8, colors=_TEXT_SECONDARY, length=0)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color=_GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_facecolor(_SURFACE)


def plot_experiment(results: dict, examples: dict[str, np.ndarray]) -> plt.Figure:
    """Build the figure from the contents of ``results.json`` and ``examples.npz``."""
    figure = plt.figure(figsize=(15.0, 8.6), facecolor=_SURFACE)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.55),
        hspace=0.2,
        left=0.085,
        right=0.985,
        top=0.875,
        bottom=0.085,
    )
    example_grid = outer[0].subgridspec(2, len(METHODS) + 1, wspace=0.06, hspace=0.06)
    _plot_examples(figure, example_grid, examples)
    metric_grid = outer[1].subgridspec(1, len(IMAGE_METRICS), wspace=0.08)
    _plot_metrics(figure, metric_grid, results["summary"])
    n_scenes = len(results["scenes"])
    label = "simulated dataset"
    if results["synthetic_dataset"]:
        label = "SYNTHETIC phase-screen dataset"
    figure.suptitle(
        f"Reconstruction experiment, {label} ({n_scenes} scenes, seed {results['root_seed']})",
        fontsize=12.5,
        color=_TEXT,
        x=0.085,
        y=0.985,
        ha="left",
    )
    figure.text(
        0.085,
        0.94,
        "Top: one rest and one task trial of scene 0; every column has its own colour scale "
        "(light = 0, dark = its 99.5th percentile). Bottom: mean over scenes, bars = standard "
        "deviation over scenes.",
        fontsize=8.5,
        color=_TEXT_SECONDARY,
    )
    figure.legend(
        handles=[Patch(facecolor=colour, label=name) for name, colour, _ in _FAMILIES],
        loc="lower left",
        bbox_to_anchor=(0.085, 0.005),
        ncol=len(_FAMILIES),
        frameon=False,
        fontsize=8.5,
        labelcolor=_TEXT,
    )
    return figure


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
