"""Measure local garment reconstruction error in the study's already rendered photographs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_pixels(path, mode):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode), dtype=np.int16)


def patch_totals(values, edge):
    height, width = values.shape
    return values.reshape(height // edge, edge, width // edge, edge).sum(axis=(1, 3))


def measure(pairs, evidence):
    manifest = json.loads((pairs / "manifest.json").read_text())
    width, height = manifest["canvas"]
    edge = 16
    if width % edge or height % edge:
        raise ValueError("Garment maps require complete 16-pixel patches.")
    with (evidence / "ablation.csv").open() as file:
        runs = list(csv.DictReader(file))
    photos = {}
    for photo in manifest["records"]:
        if photo["split"] != "test":
            continue
        key = str(photo["id"])
        folder = pairs / key
        target = read_pixels(folder / "target.png", "RGB")
        garment = read_pixels(folder / "mask.png", "L") > 127
        count = patch_totals(garment, edge).ravel()
        methods = {}
        for run in (row for row in runs if row["id"] == key):
            method = run["method"]
            photo_rgb = read_pixels(folder / (method + ".png"), "RGB")
            error_sum = np.abs(photo_rgb - target).sum(axis=2) * garment
            totals = patch_totals(error_sum, edge).ravel()
            measured = totals.sum() / (3 * count.sum())
            if not np.isclose(
                measured, float(run["garment_rgb_mae"]), atol=1e-4, rtol=0
            ):
                raise ValueError(
                    f"Image and recorded garment score disagree for {key}, {method}."
                )
            methods[method] = totals.tolist()
        photos[key] = {"garment_pixels": count.tolist(), "rgb_error_sums": methods}
    return {
        "patch_pixels": edge,
        "grid": [width // edge, height // edge],
        "photos": photos,
        "unit": "Sum of absolute R, G and B errors over annotated garment pixels in each patch. Divide by three times its garment pixel count to obtain mean RGB error.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    (args.evidence / "output-maps.json").write_text(
        json.dumps(measure(args.pairs, args.evidence), separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
