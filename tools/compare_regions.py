"""Repeat the fixed-seed, three-step blue-shirt experiment against a local WeFit service."""
import argparse
import csv
import hashlib
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("person", type=Path)
    parser.add_argument("garment", type=Path, help="A garment-only image of a top")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/region-runs.csv"))
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    person, garment = args.person.read_bytes(), args.garment.read_bytes()
    hashes = {name + "_sha256": hashlib.sha256(data).hexdigest()
              for name, data in (("person", person), ("garment", garment))}
    fields = ["pair", "requested_region", "actual_region", "steps", "pixels", "fallback",
              "queue_s", "service_s", "sample_s", "wall_s", "mlx_peak_gb", "revision", *hashes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=900, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("model") != "flux2-klein-9b-kv-q4":
            raise RuntimeError("Start the WeFit Klein service before running the comparison.")
        with args.output.open("x", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for pair in range(1, args.pairs + 1):
                order = ("full", "auto") if pair % 2 else ("auto", "full")
                for region in order:
                    start = time.perf_counter()
                    response = client.post("/tryon", files={"person": ("person", person),
                        "garment": ("garment", garment)}, data={"cloth_type": "upper",
                        "garment_photo_type": "flat-lay", "steps": 3, "quality": "balanced",
                        "conditioning": "neutral", "render_region": region})
                    wall = round(time.perf_counter() - start, 2)
                    response.raise_for_status()
                    h = response.headers
                    row = dict(pair=pair, requested_region=region, actual_region=h["X-TryOn-Region"],
                        steps=h["X-TryOn-Steps"], pixels=h["X-TryOn-Render-Pixels"],
                        fallback=h["X-TryOn-Region-Fallback"], queue_s=h["X-TryOn-Queue-Seconds"],
                        service_s=h["X-TryOn-Inference-Seconds"],
                        sample_s=h["X-TryOn-Sampling-Seconds"], wall_s=wall,
                        mlx_peak_gb=h["X-TryOn-Peak-MLX-GB"],
                        revision=h["X-TryOn-Model-Revision"], **hashes)
                    writer.writerow(row)
                    output.flush()
                    print(f"Pair {pair} | {region} | {wall:.2f} s", flush=True)


if __name__ == "__main__":
    main()
