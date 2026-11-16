"""Load one physical checkpoint in a new interpreter and compare its output with the saved intervention."""

import argparse
import hashlib
import os
from contextlib import nullcontext
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from .capture import render
from .run import save
from gpu_service.klein_runtime import KleinRuntime, garment_rgb, RenderRejected


def prove(weights, out, method, case=None):
    temporal = method.startswith(("last", "later", "reuse"))
    if temporal:
        os.environ["WEFIT_KLEIN_DEPTH_POLICY"] = str(out / "selected-policy.json")
    started = time.perf_counter()
    rt = KleinRuntime(
        weights,
        Path("model_src/klein-kv/prompt-cache"),
        Path("model_src/fashn-vton/weights/human-parser"),
    )
    mx.synchronize()
    load_s = time.perf_counter() - started
    expected_profile = "q4" if method in ("frozen", "archive") or temporal else method
    if rt.weight_profile != expected_profile:
        raise RuntimeError(f"Loaded {rt.weight_profile}, expected {expected_profile}")
    if temporal and rt.temporal_policy["method"] != method:
        raise ValueError(
            "The selected policy does not match the requested proof method."
        )
    resident = mx.get_active_memory()
    ids = json.loads((out / "protocol.json").read_text())["test_ids"]
    first = json.loads((out / "learned/training.json").read_text())["training_images"][
        0
    ]
    directory = out / "physical" / method
    directory.mkdir(parents=True, exist_ok=True)
    rows = []

    def run_one(photo, native=False):
        if native:
            with Image.open(case["person"]) as im:
                person = ImageOps.exif_transpose(im).convert("RGB")
            with Image.open(case["garment"]) as im:
                item = garment_rgb(im)
        mx.synchronize()
        mx.reset_peak_memory()
        timer = time.perf_counter()
        if native:
            image, details = rt.run(
                person,
                item,
                "upper",
                case["photo_type"],
                "balanced",
                "auto",
                3,
                "neutral",
            )
        else:
            from gpu_service.klein_depth import apply_schedule

            policy_context = (
                apply_schedule(rt.model, rt.temporal_policy)
                if temporal
                else nullcontext()
            )
            with policy_context:
                image = render(rt.model, out / "pairs" / str(photo))
        mx.synchronize()
        wall = time.perf_counter() - timer
        peak = mx.get_peak_memory()
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        compare = (
            out
            / ("native" if native else "renders")
            / str(photo)
            / (("frozen" if method == "archive" else method) + ".png")
        )
        info = dict(
            id=photo,
            method=method,
            native=native,
            wall_seconds=round(wall, 5),
            peak_mlx_bytes=peak,
            width=image.width,
            height=image.height,
            rgb_sha256=digest,
        )
        if compare.is_file():
            with Image.open(compare) as other:
                delta = np.abs(
                    np.array(image, dtype=np.int16) - np.array(other, dtype=np.int16)
                )
                info.update(
                    matching_rgb_channels=int((delta == 0).sum()),
                    rgb_channels=delta.size,
                    max_abs_channel_change=int(delta.max()),
                    comparison_rgb_sha256=hashlib.sha256(other.tobytes()).hexdigest(),
                )
        if native:
            info["fallback"] = details["region_fallback"]
            info["served_profile"] = details["weight_profile"]
        image.save(directory / (str(photo) + ".png"))
        mx.clear_cache()
        return info

    warm = run_one(first)
    for photo in ids:
        rows.append(run_one(photo))
        print(
            method,
            photo,
            rows[-1].get("max_abs_channel_change"),
            rows[-1]["wall_seconds"],
            flush=True,
        )
    if case:
        try:
            rows.append(run_one(case["case"], True))
        except RenderRejected as exc:
            rows.append(
                dict(
                    id=case["case"],
                    native=True,
                    status="rejected",
                    error=str(exc),
                    fallback=exc.retried,
                )
            )
    report = dict(
        method=method,
        profile=rt.weight_profile,
        storage=rt.weight_storage,
        load_seconds=round(load_s, 4),
        resident_mlx_after_load_bytes=resident,
        warmup=warm,
        measurements=rows,
    )
    save(directory / "proof.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("study_output", type=Path)
    p.add_argument("method")
    p.add_argument("--case", type=Path, help="private JSON for one source and garment")
    a = p.parse_args()
    prove(
        a.weights,
        a.study_output,
        a.method,
        json.loads(a.case.read_text()) if a.case else None,
    )
