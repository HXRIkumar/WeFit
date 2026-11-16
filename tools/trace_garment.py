"""Trace one upper-body edit through the actual Mac model and open its numeric trace in the inspector."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from gpu_service.klein_adapter import final_projection
from gpu_service.klein_runtime import KleinRuntime, garment_rgb
from tools.adapter_study.importance import block_activity


def trace_on_runtime(runtime, person, garment, photo_type, study):
    # A full-frame request has one generation pass, unlike an automatic crop which may retry.
    if runtime.weight_profile != "q4" or runtime.upper_adapter is not None:
        raise ValueError(
            "Trace the ordinary frozen Q4 preset with its upper-body adapter unset."
        )
    start = time.perf_counter()
    with (
        block_activity(runtime.model, True) as (single, joint),
        final_projection(runtime.model, 3, capture=True) as tap,
    ):
        image, timing = runtime.run(
            person, garment, "upper", photo_type, "balanced", "full", 3, "neutral"
        )
    elapsed = time.perf_counter() - start
    width, height = image.size
    _, labels, _ = runtime._person(person, (width, height))
    mask = Image.fromarray(np.isin(labels, [3, 4, 10]).astype(np.uint8) * 255)
    footprint = (
        np.asarray(
            mask.resize((width // 16, height // 16), Image.Resampling.BOX),
            dtype=np.float32,
        ).reshape(-1)
        / 255
    )
    if len(footprint) != len(tap.last_hidden):
        raise ValueError("Captured token grid does not match the source canvas.")
    exemplars = json.loads((study / "feature-maps.json").read_text())["channels"]
    sent = tap.last_hidden.astype(np.float32)
    sample = {
        "garment_fraction": np.round(footprint, 2).tolist(),
        "overall_rms": np.round(np.sqrt(np.mean(sent**2, axis=1)), 3).tolist(),
        "by_channel": {k: sent[:, int(k)].tolist() for k in exemplars},
    }
    trace = {
        "format": "wefit-trace-v1",
        "single": [b.step_rms for b in single],
        "joint": [b.step_rms for b in joint],
        "grid": [width // 16, height // 16],
        "sample": sample,
        "timing_s": round(elapsed, 3),
        "model_render_s": round(timing["sampling_seconds"], 3),
        "steps": 3,
        "region": "full",
        "scope": "One instrumented, frozen upper-body render. Probe synchronization affects its speed.",
    }
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("person", type=Path)
    parser.add_argument("garment", type=Path)
    parser.add_argument(
        "--photo-type", choices=["flat-lay", "model"], default="flat-lay"
    )
    parser.add_argument("--study", type=Path, default=Path("docs/study-v2/evidence"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.getenv("WEFIT_KLEIN_UPPER_ADAPTER"):
        parser.error("Unset WEFIT_KLEIN_UPPER_ADAPTER to trace the baseline.")
    if args.output.exists():
        parser.error("Choose a fresh output file.")
    runtime = KleinRuntime(
        Path("model_src/klein-kv/weights"),
        Path("model_src/klein-kv/prompt-cache"),
        Path("model_src/fashn-vton/weights/human-parser"),
    )
    with Image.open(args.person) as person, Image.open(args.garment) as garment:
        trace = trace_on_runtime(
            runtime,
            ImageOps.exif_transpose(person).convert("RGB"),
            garment_rgb(garment),
            args.photo_type,
            args.study,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(trace, separators=(",", ":")) + "\n")
    print(f"Open {args.output} with the Open a local trace control in the inspector.")


if __name__ == "__main__":
    main()
