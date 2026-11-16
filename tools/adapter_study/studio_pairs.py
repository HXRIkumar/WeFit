"""Run full-size source and product photos through the studio runtime, with and without the trained adapter."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from gpu_service.klein_adapter import Adapter
from gpu_service.klein_runtime import (
    KleinRuntime,
    canvas_size,
    garment_reference,
    garment_rgb,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases", type=Path)
    p.add_argument("trained", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--trace-first",
        action="store_true",
        help="Capture one additional full-frame baseline trace on the first source.",
    )
    p.add_argument("--weights", type=Path, default=Path("model_src/klein-kv/weights"))
    p.add_argument(
        "--prompts", type=Path, default=Path("model_src/klein-kv/prompt-cache")
    )
    p.add_argument(
        "--parser", type=Path, default=Path("model_src/fashn-vton/weights/human-parser")
    )
    args = p.parse_args()
    if os.environ.get("WEFIT_KLEIN_UPPER_ADAPTER"):
        raise RuntimeError(
            "Start this paired study with the default model configuration."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rt = KleinRuntime(args.weights, args.prompts, args.parser)
    path = args.trained / "upper-compact.npz"
    adapter = Adapter.load(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = json.loads(args.cases.read_text())
    rows = []
    for index, entry in enumerate(inputs):
        with Image.open(args.cases.parent / entry["person"]) as im:
            person = ImageOps.exif_transpose(im).convert("RGB")
        with Image.open(args.cases.parent / entry["garment"]) as im:
            garment = garment_rgb(im)
        source = person.resize(canvas_size(person.size, 768), Image.Resampling.LANCZOS)
        source.save(args.output / (entry["case"] + "-source.png"))
        reference = garment_reference(
            garment, "upper", entry["photo_type"], rt.parser.predict
        )
        reference.convert("RGB").save(args.output / (entry["case"] + "-product.png"))
        order = ["frozen", "lora"] if index % 2 == 0 else ["lora", "frozen"]
        for name in order:
            rt.upper_adapter = adapter if name == "lora" else None
            rt.upper_adapter_sha256 = digest if name == "lora" else ""
            start = time.perf_counter()
            row = {"case": entry["case"], "method": name}
            try:
                image, timing = rt.run(
                    person,
                    garment,
                    "upper",
                    entry["photo_type"],
                    "balanced",
                    "auto",
                    3,
                    "neutral",
                )
                row["wall_s"] = round(time.perf_counter() - start, 3)
                row.update(
                    status="success",
                    width=image.width,
                    height=image.height,
                    fallback=timing["region_fallback"],
                    region=timing["render_region"],
                    sample_s=round(timing["sampling_seconds"], 3),
                    adapter_sha256=timing["adapter_sha256"],
                    output_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                )
                image.save(args.output / (entry["case"] + "-" + name + ".png"))
            except ValueError as exc:
                row.update(
                    status="rejected",
                    error=str(exc),
                    wall_s=round(time.perf_counter() - start, 3),
                )
            rows.append(row)
            print(entry["case"], name, row["status"], row["wall_s"], flush=True)
            (args.output / "pairs.json").write_text(json.dumps(rows, indent=2) + "\n")
    comparisons = []
    for entry in inputs:
        paths = [
            args.output / (entry["case"] + "-" + k + ".png") for k in ("frozen", "lora")
        ]
        if not all(f.is_file() for f in paths):
            continue
        with Image.open(paths[0]) as im:
            a = np.asarray(im.convert("RGB"))
        with Image.open(paths[1]) as im:
            b = np.asarray(im.convert("RGB"))
        labels = rt.parser.predict(a)
        garment_mask = np.isin(labels, [3, 4, 10])
        if not garment_mask.any() or garment_mask.all():
            continue
        difference = np.abs(a.astype(np.float32) - b).mean(axis=2)
        comparisons.append(
            {
                "case": entry["case"],
                "garment_rgb_change": float(difference[garment_mask].mean()),
                "outside_rgb_change": float(difference[~garment_mask].mean()),
                "pixel_exact_fraction": float((a == b).all(axis=2).mean()),
            }
        )
    (args.output / "differences.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )

    if args.trace_first:
        from tools.trace_garment import trace_on_runtime

        entry = inputs[0]
        rt.upper_adapter = None
        rt.upper_adapter_sha256 = ""
        with (
            Image.open(args.cases.parent / entry["person"]) as source,
            Image.open(args.cases.parent / entry["garment"]) as product,
        ):
            trace = trace_on_runtime(
                rt,
                ImageOps.exif_transpose(source).convert("RGB"),
                garment_rgb(product),
                entry["photo_type"],
                args.trained,
            )
        (args.output / "native-trace.json").write_text(
            json.dumps(trace, separators=(",", ":")) + "\n"
        )


if __name__ == "__main__":
    main()
