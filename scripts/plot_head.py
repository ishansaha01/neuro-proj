"""Plot orthogonal slices of a saved head.

Usage: ``python scripts/plot_head.py --head head.h5 --out head.png``. One row
per field (labels, sound speed, porosity, CBV, activation) and one column per
slice orientation, each through the voxel where the activation is largest.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

from neurosim.head.io import load_head  # noqa: E402
from neurosim.head.types import Tissue  # noqa: E402

_MM_PER_M = 1.0e3
# One colour per `Tissue` value: a neutral for the coupling medium, then seven
# categorical hues in a fixed order (checked for colour-vision-deficiency
# separation; the colour bar names every tissue, so colour is never the only cue).
_TISSUE_COLORS = (
    "#e1e0d9",
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
)
# Single-hue ramp, light to dark, for every magnitude map.
_MAGNITUDE_COLORS = ("#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
# (title, colour-bar label) per plotted magnitude field of `HeadVolume`.
_MAGNITUDES = {
    "sound_speed": ("sound speed", "m/s"),
    "porosity": ("skull porosity", "pore volume fraction"),
    "cbv_baseline": ("baseline CBV", "blood volume fraction"),
    "activation": ("activation pattern", "task response weight"),
}
# Per column: the head axis normal to the slice.
_COLUMNS = (("x", 0), ("y", 1), ("z", 2))


def main(argv: list[str]) -> int:
    """Plot a saved head from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--head", required=True, help="head bundle written by generate_head.py")
    parser.add_argument("--out", required=True, help="output PNG file")
    args = parser.parse_args(argv)

    head = load_head(args.head)
    axes_mm = [axis * _MM_PER_M for axis in head.grid.axes()]
    center = np.unravel_index(np.argmax(head.activation), head.activation.shape)

    tissue_map = ListedColormap(_TISSUE_COLORS)
    tissue_norm = BoundaryNorm(np.arange(len(Tissue) + 1) - 0.5, len(Tissue))
    magnitude_map = LinearSegmentedColormap.from_list("magnitude", _MAGNITUDE_COLORS)
    rows = [("labels", "tissue", "")] + [(name, *text) for name, text in _MAGNITUDES.items()]

    figure, panels = plt.subplots(
        len(rows), len(_COLUMNS), figsize=(4.6 * len(_COLUMNS), 3.6 * len(rows)), squeeze=False
    )
    for row, (name, title, unit) in enumerate(rows):
        volume = getattr(head, name)
        is_labels = name == "labels"
        for column, (normal, axis) in enumerate(_COLUMNS):
            in_plane = [a for a in range(3) if a != axis]
            panel = panels[row, column]
            drawn = panel.imshow(
                np.take(volume, center[axis], axis=axis).T,
                origin="lower",
                cmap=tissue_map if is_labels else magnitude_map,
                norm=tissue_norm if is_labels else None,
                vmin=None if is_labels else float(volume.min()),
                vmax=None if is_labels else float(volume.max()),
                extent=(
                    axes_mm[in_plane[0]][0],
                    axes_mm[in_plane[0]][-1],
                    axes_mm[in_plane[1]][0],
                    axes_mm[in_plane[1]][-1],
                ),
                interpolation="nearest",
            )
            panel.set_title(f"{title}, {normal} = {axes_mm[axis][center[axis]]:.1f} mm")
            panel.set_xlabel(f"{'xyz'[in_plane[0]]} [mm]")
            panel.set_ylabel(f"{'xyz'[in_plane[1]]} [mm]")
            bar = figure.colorbar(drawn, ax=panel, label=unit)
            if is_labels:
                bar.set_ticks(list(Tissue), labels=[t.name.lower() for t in Tissue])
    figure.tight_layout()
    figure.savefig(args.out, dpi=110)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
