"""Render all held-out garments through the real model and score paired RGB and latent errors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from gpu_service.klein_adapter import Adapter, final_projection
from gpu_service.klein_model import CachedInstructionKlein
from gpu_service.klein_prompts import INSTRUCTIONS

from .extract import CaptureFinalLatents, checkpoint_hashes


def masked_rgb(prediction, target, mask):
    error = np.abs(
        np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    ).mean(axis=2)
    return float(error[mask].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--weights", type=Path, default=Path("model_src/klein-kv/weights")
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path("model_src/klein-kv/prompt-cache")
    )
    args = parser.parse_args()
    adapter = Adapter.load(args.output / "upper-final-r4.npz")
    compact = Adapter.load(args.output / "upper-compact.npz")
    training = json.loads((args.output / "training.json").read_text())
    if (
        hashlib.sha256((args.output / "upper-final-r4.npz").read_bytes()).hexdigest()
        != training["adapter_sha256"]
    ):
        raise ValueError("Trained adapter checksum mismatch")
    manifest = json.loads((args.pairs / "manifest.json").read_text())
    model = CachedInstructionKlein(args.weights, args.prompts)
    model.freeze()
    mx.eval(model.parameters())
    final = CaptureFinalLatents()
    model.callbacks.register(final)
    width, height = manifest["canvas"]
    results = []
    zero_verified = False
    for item in manifest["records"]:
        if item["split"] != "test":
            continue
        folder = args.pairs / str(item["id"])
        params = {
            "seed": 42,
            "prompt": INSTRUCTIONS["upper"],
            "num_inference_steps": 3,
            "height": height,
            "width": width,
            "guidance": 1.0,
            "use_kv_cache": True,
            "image_paths": [folder / "person.png", folder / "item.png"],
        }
        with Image.open(folder / "baseline.png") as im:
            baseline = im.convert("RGB")
        if not zero_verified:
            zero = Adapter(adapter.down, np.zeros_like(adapter.up))
            with final_projection(model, 3, adapter=zero):
                control = model.generate_image(**params).image.convert("RGB")
            if control.tobytes() != baseline.tobytes():
                raise RuntimeError(
                    "Zero adapter differs from frozen baseline for the same seed."
                )
            zero_verified = True
        with final_projection(model, 3, adapter=adapter):
            output = model.generate_image(**params).image.convert("RGB")
        output.save(folder / "adapted.png")
        adapted_latents = final.base.copy()
        with final_projection(model, 3, adapter=compact):
            compressed_image = model.generate_image(**params).image.convert("RGB")
        compressed_image.save(folder / "compact.png")
        with Image.open(folder / "target.png") as im:
            target = im.convert("RGB")
        with Image.open(folder / "mask.png") as im:
            mask = np.asarray(im.convert("L")) > 127
        with np.load(folder / "features.npz", allow_pickle=False) as z:
            w = z["fraction"]
            old = float(
                np.average(np.square(z["base"] - z["target"]).mean(axis=1), weights=w)
            )
            new = float(
                np.average(
                    np.square(adapted_latents - z["target"]).mean(axis=1), weights=w
                )
            )
        row = {
            "id": item["id"],
            "category": item["category"],
            "latent_mse_base": old,
            "latent_mse_adapter": new,
            "garment_mae_base": masked_rgb(baseline, target, mask),
            "garment_mae_adapter": masked_rgb(output, target, mask),
            "outside_mae_base": masked_rgb(baseline, target, ~mask),
            "outside_mae_adapter": masked_rgb(output, target, ~mask),
            "outside_mae_change": masked_rgb(output, baseline, ~mask),
            "garment_mae_compact": masked_rgb(compressed_image, target, mask),
            "outside_mae_compact": masked_rgb(compressed_image, target, ~mask),
            "compact_vs_trained_mae": float(
                np.abs(
                    np.asarray(output, dtype=np.float32)
                    - np.asarray(compressed_image, dtype=np.float32)
                ).mean()
            ),
            "compact_exact_pixel_fraction": float(
                np.mean(
                    np.all(np.asarray(output) == np.asarray(compressed_image), axis=2)
                )
            ),
        }
        results.append(row)
        print(
            row["id"], row["garment_mae_base"], row["garment_mae_adapter"], flush=True
        )
        mx.synchronize()
        mx.clear_cache()
    with (args.output / "held-out.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    means = {
        k: float(np.mean([row[k] for row in results]))
        for k in results[0]
        if k not in ("id", "category")
    }
    weights_ok = checkpoint_hashes(args.weights) == training["base_checkpoint_sha256"]
    summary = {
        "n": len(results),
        "canvas": manifest["canvas"],
        "per_image_means": means,
        "garment_rgb_wins": sum(
            r["garment_mae_adapter"] < r["garment_mae_base"] for r in results
        ),
        "zero_adapter_pixel_identical": zero_verified,
        "unchanged_base_checkpoint": weights_ok,
        "scope": "Same-photo garment reconstruction on held-out images. No physical fit or cross-outfit accuracy measured.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not weights_ok:
        raise RuntimeError("Original model files changed")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
