"""Draw a measured patch and the original and trained output weights it connects to."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def plot_connection(features, output, photo, channel, token, connection):
    record = json.loads(features.read_text())
    width, height = record["token_grid"]
    if not (0 <= token < width * height and 0 <= connection < 128):
        raise ValueError(
            "Choose an image patch and output connection within the measured grid."
        )
    sample = record["photos"][str(photo)]
    input_weights = record["channels"][str(channel)]
    values = np.asarray(sample["by_channel"][str(channel)]).reshape(height, width)
    coverage = np.asarray(sample["garment_fraction"]).reshape(height, width)
    weight = np.asarray(input_weights["weight"]).reshape(8, 16)
    change = np.asarray(input_weights["learned_delta"]).reshape(8, 16)
    cmap = LinearSegmentedColormap.from_list(
        "signed_feature", ["#a64f19", "#faf8ef", "#205e87"]
    )
    fig = plt.figure(figsize=(10.0, 4.7), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.1, 1.6])
    left = fig.add_subplot(grid[:, 0])
    original = fig.add_subplot(grid[0, 1])
    learned = fig.add_subplot(grid[1, 1])
    for ax, arr, title in (
        (left, values, f"Response at each image patch · input {channel}"),
        (original, weight, "Frozen connections to the 128 outputs"),
        (learned, change, "Correction learned for those connections"),
    ):
        limit = max(float(np.abs(arr).max()), 1e-12)
        im = ax.imshow(arr, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(title, fontsize=11, color="#16353f", loc="left", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        bar = fig.colorbar(
            im, ax=ax, shrink=0.8, ticks=[-limit, 0, limit], fraction=0.055
        )
        bar.ax.set_yticklabels([f"−{limit:.3g}", "0", f"+{limit:.3g}"])
        bar.ax.tick_params(length=0, labelsize=8)
        bar.outline.set_visible(False)
    left.contour(coverage, levels=[0.5], colors=["#1f3031"], linewidths=[1.2])
    for ax, index, cols in (
        (left, token, width),
        (original, connection, 16),
        (learned, connection, 16),
    ):
        for color, linewidth in (("white", 2.8), ("#172b35", 1.1)):
            ax.add_patch(
                Rectangle(
                    (index % cols - 0.5, index // cols - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                )
            )
    activation = values.flat[token]
    base = weight.flat[connection]
    delta = change.flat[connection]
    fig.suptitle(
        f"Photograph {photo} · patch ({token % width + 1}, {token // width + 1}) → output {connection}",
        fontsize=15,
        fontweight="bold",
        color="#16353f",
    )
    fig.supxlabel(
        f"Activation {activation:.3f} × frozen weight {base:.5f} gives {activation * base:.6f}.  After training the contribution is {activation * (base + delta):.6f}.",
        fontsize=10,
        color="#314e55",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("features", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--photo", type=int, default=10471)
    p.add_argument("--channel", type=int, default=89)
    p.add_argument("--token", type=int, default=152)
    p.add_argument("--connection", type=int, default=17)
    a = p.parse_args()
    plot_connection(a.features, a.output, a.photo, a.channel, a.token, a.connection)


if __name__ == "__main__":
    main()
