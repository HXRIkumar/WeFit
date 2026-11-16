"""Profile a local WeFit worker with real requests and a fixed case manifest."""

import argparse
import io
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median

import httpx
from PIL import Image


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median_s": round(median(ordered), 3),
        "p95_s": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_s": ordered[0],
        "max_s": ordered[-1],
    }


def summarize(rows):
    warm = [row for row in rows if row["phase"] == "warm"]
    ok = [row for row in warm if row["status"] == 200]
    eligible = [row for row in warm if row.get("initial_region") == "clothing"]
    return {
        "first_request": next((r for r in rows if r["phase"] == "first"), None),
        "attempts": len(warm),
        "completed": len(ok),
        "status_counts": dict(Counter(r["status"] for r in warm)),
        "case_counts": dict(Counter(r["case"] for r in warm)),
        "wall": distribution([r["wall_s"] for r in ok]),
        "service": distribution([r["service_s"] for r in ok]),
        "fallbacks": sum(r["fallback"] is True for r in eligible),
        "crop_attempts": len(eligible),
        "unobserved_initial_region": sum(r.get("initial_region") is None for r in warm),
    }


def prepare(path):
    cases = []
    for entry in json.loads(path.read_text()):
        source = (path.parent / entry["person"]).read_bytes()
        item = (path.parent / entry["garment"]).read_bytes()
        cases.append((entry, source, item))
    if not cases:
        raise ValueError("Add at least one case to the JSON manifest.")
    return cases


def request(client, case, index, phase):
    entry, source, garment = case
    row = {
        "index": index,
        "phase": phase,
        "case": entry["case"],
        "started_utc": datetime.now(UTC).isoformat(),
        "person_sha256": sha256(source).hexdigest(),
        "garment_sha256": sha256(garment).hexdigest(),
    }
    data = {
        "cloth_type": entry.get("kind", "upper"),
        "quality": "balanced",
        "garment_photo_type": entry["photo_type"],
        "conditioning": "neutral",
        "render_region": "auto",
    }
    start = time.perf_counter()
    try:
        response = client.post(
            "/tryon",
            data=data,
            files={"person": ("person", source), "garment": ("item", garment)},
        )
    except httpx.TransportError as error:
        row.update(
            status=0,
            error=type(error).__name__,
            wall_s=round(time.perf_counter() - start, 3),
        )
        return row
    row.update(
        wall_s=round(time.perf_counter() - start, 3), status=response.status_code
    )
    h = response.headers
    if "x-tryon-initial-region" in h:
        row.update(
            initial_region=h["x-tryon-initial-region"],
            fallback=h["x-tryon-region-fallback"] == "true",
        )
    if response.status_code != 200:
        row["error"] = response.text[:300]
        return row
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        row["output_size"] = list(image.size)
    row.update(
        result_sha256=sha256(response.content).hexdigest(),
        region=h["x-tryon-region"],
        revision=h["x-tryon-model-revision"],
        steps=int(h["x-tryon-steps"]),
        pixels=int(h["x-tryon-render-pixels"]),
        mlx_peak_gb=float(h["x-tryon-peak-mlx-gb"]),
    )
    for dest, header in [
        ("service", "inference"),
        ("queue", "queue"),
        ("sample", "sampling"),
        ("prepare", "preprocess"),
        ("finish", "postprocess"),
    ]:
        row[dest + "_s"] = float(h["x-tryon-" + header + "-seconds"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        type=Path,
        help="JSON array of case, person, garment and photo_type fields",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("Use at least one round.")
    cases = prepare(args.cases)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", timeout=900, trust_env=False
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("model") != "flux2-klein-9b-kv-q4":
            raise ValueError("Expected the local Klein renderer.")
        with (args.output / "requests.jsonl").open("x") as log:
            for index in range(1 + len(cases) * args.rounds):
                phase = "warm" if index else "first"
                row = request(
                    client,
                    cases[(index - 1) % len(cases)] if index else cases[0],
                    index,
                    phase,
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                rows.append(row)
                print(
                    index, phase, row["case"], row["status"], row["wall_s"], flush=True
                )
    (args.output / "summary.json").write_text(
        json.dumps(summarize(rows), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
