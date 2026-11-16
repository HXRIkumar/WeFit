"""Evaluate an actual all-single-stream Q3 checkpoint against the preregistered Q4 renders."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from gpu_service.klein_runtime import KleinRuntime, garment_rgb, RenderRejected
from .capture import render
from .quality import compare, summarize
from .run import save


def evaluate(weights, out, cases):
    started = time.perf_counter()
    rt = KleinRuntime(
        weights,
        Path("model_src/klein-kv/prompt-cache"),
        Path("model_src/fashn-vton/weights/human-parser"),
    )
    mx.synchronize()
    load_s = time.perf_counter() - started
    if rt.weight_profile != "wefit-q3-24-blocks":
        raise ValueError("Expected the 24-block mixed checkpoint")
    resident = mx.get_active_memory()
    directory = out / "quantization"
    directory.mkdir(exist_ok=True)
    manifest = json.loads((out / "pairs/manifest.json").read_text())
    first = json.loads((out / "learned/training.json").read_text())["training_images"][
        0
    ]
    render(rt.model, out / "pairs" / str(first))
    mx.clear_cache()
    rows = []
    for entry in manifest["records"]:
        if entry["split"] == "train":
            continue
        photo = entry["id"]
        folder = out / "pairs" / str(photo)
        mx.synchronize()
        mx.reset_peak_memory()
        start = time.perf_counter()
        image = render(rt.model, folder)
        mx.synchronize()
        wall = time.perf_counter() - start
        peak = mx.get_peak_memory()
        with (
            Image.open(out / "renders" / str(photo) / "frozen.png") as base,
            Image.open(folder / "target.png") as target,
            Image.open(folder / "mask.png") as mask,
        ):
            metrics = compare(image, base, target, mask)
        image.save(directory / (str(photo) + ".png"))
        rows.append(
            dict(
                id=photo,
                method="q3-all-single",
                status="success",
                split=entry["split"],
                wall_seconds=round(wall, 5),
                peak_mlx_bytes=peak,
                rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                **metrics,
            )
        )
        mx.clear_cache()
        print(photo, wall, metrics["garment_ssim"], flush=True)
    native = []
    for e in cases:
        photo = e["case"]
        with Image.open(e["person"]) as source, Image.open(e["garment"]) as garment:
            person = ImageOps.exif_transpose(source).convert("RGB")
            item = garment_rgb(garment)
        start = time.perf_counter()
        try:
            image, t = rt.run(
                person, item, "upper", e["photo_type"], "balanced", "auto", 3, "neutral"
            )
            wall = time.perf_counter() - start
            with (
                Image.open(out / "native" / photo / "frozen.png") as base,
                Image.open(out / "native" / photo / "mask.png") as mask,
            ):
                metrics = compare(image, base, base, mask)
            image.save(directory / (photo + ".png"))
            native.append(
                dict(
                    id=photo,
                    status="success",
                    fallback=t["region_fallback"],
                    wall_seconds=round(wall, 5),
                    peak_mlx_bytes=round(t["peak_mlx_gb"] * 1e9),
                    rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    **{
                        k: v
                        for k, v in metrics.items()
                        if k not in ("garment_mae", "outside_mae")
                    },
                )
            )
        except RenderRejected as err:
            native.append(
                dict(
                    id=photo,
                    status="rejected",
                    error=str(err),
                    wall_seconds=round(time.perf_counter() - start, 5),
                    fallback=err.retried,
                )
            )
        print("studio", native[-1], flush=True)
    result = dict(
        method=rt.weight_profile,
        load_seconds=round(load_s, 4),
        resident_mlx_after_load_bytes=resident,
        measurements=rows,
        native=native,
        summary={
            s: summarize([r for r in rows if r["split"] == s])["q3-all-single"]
            for s in ("val", "test")
        },
    )
    save(directory / "proof.json", result)
    encoder_plan = out / "encoder-protocol.json"
    if cases and encoder_plan.is_file():
        from .unused_variance import inspect

        plan = json.loads(encoder_plan.read_text())
        if (
            plan.get("experiment") != "vae-posterior-mean"
            or plan.get("profile") != rt.weight_profile
        ):
            raise ValueError(
                "The projection experiment must name its reference checkpoint."
            )
        held_out = [r["id"] for r in manifest["records"] if r["split"] == "test"]
        chosen = next(
            (c for c in cases if (directory / (c["case"] + ".png")).is_file()), cases[0]
        )
        with Image.open(chosen["person"]) as src, Image.open(chosen["garment"]) as item:
            inspect(
                rt,
                out,
                chosen,
                ImageOps.exif_transpose(src).convert("RGB"),
                garment_rgb(item),
                held_out[0],
                directory,
            )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--cases", type=Path)
    a = p.parse_args()
    evaluate(a.weights, a.output, json.loads(a.cases.read_text()) if a.cases else [])
