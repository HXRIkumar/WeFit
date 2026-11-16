"""Time the same full Q4 requests with and without exact reusable VAE encodings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from gpu_service.klein_runtime import KleinRuntime, canvas_size, garment_rgb
from .run import save


def measure(weights, cases, destination, frozen_results=None):
    if len(cases) < 2:
        raise ValueError(
            "Supply a priming case and at least one changed person/garment pair."
        )
    os.environ["WEFIT_KLEIN_REFERENCE_CACHE"] = "1"
    runtime = KleinRuntime(
        weights,
        Path("model_src/klein-kv/prompt-cache"),
        Path("model_src/fashn-vton/weights/human-parser"),
    )
    if (
        runtime.weight_profile != "q4"
        or runtime.upper_adapter
        or runtime.temporal_policy
    ):
        raise ValueError("Use full Q4 without an adapter or a step schedule.")
    cache = runtime.model.vae.reference_latents

    def photos(case):
        with Image.open(case["person"]) as image:
            person = ImageOps.exif_transpose(image).convert("RGB")
        with Image.open(case["garment"]) as image:
            item = garment_rgb(image)
        return person, item

    def render(case, person, item, enabled):
        cache.enabled = enabled
        hits, misses = cache.hits, cache.misses
        mx.synchronize()
        start = time.perf_counter()
        result, details = runtime.run(
            person, item, "upper", case["photo_type"], "balanced", "auto", 3, "neutral"
        )
        mx.synchronize()
        return result, dict(
            cache_enabled=enabled,
            wall_seconds=round(time.perf_counter() - start, 5),
            encoder_hits=cache.hits - hits,
            encoder_misses=cache.misses - misses,
            retained_latent_bytes=cache.bytes,
            fallback=details["region_fallback"],
            person_parser_cached=details["person_cached"],
            rgb_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        )

    priming = cases[0]
    _, warm = render(priming, *photos(priming), True)
    print("Primed two references", priming["case"], warm["wall_seconds"], flush=True)
    rows = []
    for i, case in enumerate(cases[1:]):
        person, item = photos(case)
        runtime._person(person, canvas_size(person.size, 768))
        # Counterbalance pair order. The bypassed encoder never populates the cache.
        order = (True, False) if i % 2 == 0 else (False, True)
        images, observations = {}, {}
        for enabled in order:
            images[enabled], observations[enabled] = render(case, person, item, enabled)
        actual, expected = (
            np.asarray(images[v], dtype=np.int16) for v in (True, False)
        )
        delta = np.abs(actual - expected)
        row = dict(
            case=case["case"],
            order=["reuse" if v else "original" for v in order],
            reuse=observations[True],
            original=observations[False],
            seconds_saved=observations[False]["wall_seconds"]
            - observations[True]["wall_seconds"],
            matching_rgb_channels=int((delta == 0).sum()),
            channels=delta.size,
            max_rgb_delta=int(delta.max()),
        )
        if frozen_results:
            with Image.open(frozen_results / case["case"] / "frozen.png") as source:
                previous = source.convert("RGB")
                row["previous_frozen_rgb_sha256"] = hashlib.sha256(
                    previous.tobytes()
                ).hexdigest()
                prior_delta = np.abs(np.asarray(previous, dtype=np.int16) - expected)
                row["previous_max_rgb_delta"] = int(prior_delta.max())
                row["previous_matching_channels"] = int((prior_delta == 0).sum())
        rows.append(row)
        print(case["case"], row["seconds_saved"], row["max_rgb_delta"], flush=True)
    data = dict(
        profile=runtime.weight_profile,
        capacity_bytes=cache.max_bytes,
        ttl_seconds=cache.ttl_seconds,
        prime=warm,
        pairs=rows,
        mean_seconds_saved=statistics.mean(r["seconds_saved"] for r in rows),
        median_seconds_saved=statistics.median(r["seconds_saved"] for r in rows),
    )
    save(destination, data)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("cases", type=Path, help="A case list with the priming pair first")
    p.add_argument("output", type=Path)
    p.add_argument("--frozen-results", type=Path)
    args = p.parse_args()
    measure(
        args.weights,
        json.loads(args.cases.read_text()),
        args.output,
        args.frozen_results,
    )
