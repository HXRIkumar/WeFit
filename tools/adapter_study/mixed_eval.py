"""Render the same compact adapter on a real mixed Q3/Q4 checkpoint and its Q4 control."""

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from statistics import mean, median

import mlx.core as mx
import numpy as np
from PIL import Image

from gpu_service.klein_adapter import Adapter, final_projection
from gpu_service.klein_model import CachedInstructionKlein
from gpu_service.klein_prompts import INSTRUCTIONS

from .evaluate import masked_rgb
from .extract import checkpoint_hashes
from .mixed_checkpoint import build
from .replay import capture, save_state
from .replay import study as replay_study


def render_group(
    weights, prompts, adapter, samples, pairs, method, expected, warmup, size
):
    model = CachedInstructionKlein(weights, prompts)
    model.freeze()
    mx.eval(model.parameters())
    selected = []
    for i, entry in enumerate(model.transformer.single_transformer_blocks):
        block = getattr(entry, "block", entry)
        precision = [block.attn.to_qkv_mlp_proj.bits, block.attn.to_out.bits]
        if precision == [3, 3]:
            selected.append(i)
        elif precision != [4, 4]:
            raise ValueError(f"Unexpected stored precision {precision} on block {i}")
    if sorted(selected) != sorted(expected):
        raise ValueError("Loaded model precision differs from checkpoint manifest.")
    width, height = size
    entries = []
    for index, item in enumerate([warmup, *samples]):
        folder = pairs / str(item["id"])
        with capture(model) as last_step, final_projection(model, 3, adapter=adapter):
            t = time.perf_counter()
            picture = model.generate_image(
                seed=42,
                prompt=INSTRUCTIONS["upper"],
                num_inference_steps=3,
                guidance=1.0,
                width=width,
                height=height,
                image_paths=[folder / "person.png", folder / "item.png"],
                use_kv_cache=True,
            ).image.convert("RGB")
            mx.synchronize()
            elapsed = time.perf_counter() - t
        if index:
            if method == "q4":
                save_state(last_step, folder, size, item["id"])
            with Image.open(folder / "target.png") as im:
                target = im.convert("RGB")
            with Image.open(folder / "mask.png") as im:
                mask = np.asarray(im.convert("L")) > 127
            with Image.open(folder / "compact.png") as im:
                control = im.convert("RGB")
            if method == "q4" and picture.tobytes() != control.tobytes():
                raise RuntimeError("Q4 control is not reproducible.")
            entries.append(
                {
                    "id": item["id"],
                    "method": method,
                    "wall_s": round(elapsed, 3),
                    "garment_mae": masked_rgb(picture, target, mask),
                    "outside_mae": masked_rgb(picture, target, ~mask),
                    "absolute_change_from_q4": float(
                        np.abs(
                            np.asarray(picture, dtype=np.float32)
                            - np.asarray(control, dtype=np.float32)
                        ).mean()
                    ),
                }
            )
            picture.save(folder / (method + "-compact.png"))
            print(method, item["id"], round(elapsed, 2), flush=True)
        mx.clear_cache()
    del model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return entries


def compare(pairs, out, weights, prompts, candidate):
    training = json.loads((out / "training.json").read_text())
    if checkpoint_hashes(weights) != training["base_checkpoint_sha256"]:
        raise ValueError("Original checkpoint differs from the training baseline.")
    activity = json.loads((out / "activity.json").read_text())
    if candidate.exists():
        meta = json.loads((candidate / "wefit-mixed.json").read_text())
        if (
            meta["source_sha256"] != training["base_checkpoint_sha256"]
            or meta["blocks"] != activity["residual_rms_order_low_to_high"][:6]
        ):
            raise ValueError(
                "Existing mixed checkpoint has different calibration or base files. Choose another folder."
            )
        if checkpoint_hashes(candidate) != meta["candidate_sha256"]:
            raise ValueError("Stored mixed file has changed.")
    else:
        meta = build(weights, candidate, out / "activity.json")
    adapter = Adapter.load(out / "upper-compact.npz")
    manifest = json.loads((pairs / "manifest.json").read_text())
    test = [r for r in manifest["records"] if r["split"] == "test"]
    warmup = next(r for r in manifest["records"] if r["split"] == "train")
    results = []
    for group, order in [(test[:3], ("q4", "mixed")), (test[3:], ("mixed", "q4"))]:
        for method in order:
            new = method == "mixed"
            results += render_group(
                candidate if new else weights,
                prompts,
                adapter,
                group,
                pairs,
                method,
                meta["blocks"] if new else [],
                warmup,
                manifest["canvas"],
            )
    with (out / "mixed-runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    totals = {}
    for method in ("q4", "mixed"):
        r = [x for x in results if x["method"] == method]
        totals[method] = {
            "median_s": median(x["wall_s"] for x in r),
            "garment_mae": mean(x["garment_mae"] for x in r),
            "outside_mae": mean(x["outside_mae"] for x in r),
        }
    output = {
        **meta,
        "loaded_precision_verified": True,
        "held_out_images": len(test),
        "canvas": manifest["canvas"],
        "methods": totals,
        "scope": "Identical compact adapter on both backbones. Mixed model stores Q3 in two projections of six selected single blocks; all other tensors retain their original precision. Independent one-image warm-up per group.",
    }
    (out / "mixed-checkpoint.json").write_text(json.dumps(output, indent=2) + "\n")
    replay_study(pairs, out, weights, prompts)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--weights", type=Path, default=Path("model_src/klein-kv/weights"))
    p.add_argument(
        "--prompts", type=Path, default=Path("model_src/klein-kv/prompt-cache")
    )
    a = p.parse_args()
    compare(a.pairs, a.output, a.weights, a.prompts, a.candidate)
