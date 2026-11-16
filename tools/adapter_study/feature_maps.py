"""Publish small numeric views of real activations without publishing any training photos."""

import csv
import json

import numpy as np


def feature_maps(pairs, out):
    manifest = json.loads((pairs / "manifest.json").read_text())
    scope = json.loads((out / "activity.json").read_text())
    with (out / "head-channels.csv").open() as f:
        ranked = [int(r["channel"]) for r in csv.DictReader(f)]
    indices = [0, 1023, 2047, 3071, 4095]
    base = np.load(pairs / "base-output-head.npy", allow_pickle=False)
    with np.load(out / "upper-final-r4.npz", allow_pickle=False) as z:
        delta = (z["down"] @ z["up"]).T
    channels = {}
    for idx in indices:
        channel = ranked[idx]
        channels[str(channel)] = {
            "rank": idx + 1,
            "weight": base[:, channel].tolist(),
            "learned_delta": delta[:, channel].tolist(),
        }
    width, height = manifest["canvas"]
    photos = {}
    for image_id in scope["calibration_images"]:
        with np.load(pairs / str(image_id) / "features.npz", allow_pickle=False) as z:
            hidden = z["hidden"].astype(np.float32)
            image = {
                "garment_fraction": np.round(z["fraction"], 2).tolist(),
                "overall_rms": np.round(
                    np.sqrt(np.mean(hidden**2, axis=1)), 3
                ).tolist(),
                "by_channel": {
                    k: hidden[:, int(k)].tolist() for k in channels
                },
            }
        photos[str(image_id)] = image
    packet = {
        "token_grid": [width // 16, height // 16],
        "pixel_patch": [16, 16],
        "channels": channels,
        "photos": photos,
        "site": "Immediately before the real final output projection, last rendering step; signed features before any adapter.",
    }
    (out / "feature-maps.json").write_text(
        json.dumps(packet, separators=(",", ":")) + "\n"
    )
