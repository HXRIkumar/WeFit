"""Time a real model pass with the chosen research intervention."""

import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from gpu_service.klein_runtime import garment_rgb, RenderRejected
from .capture import render
from .policies import intervention


def measure(runtime, out, job):
    trained = out / "learned"
    training = json.loads((trained / "training.json").read_text())
    method = job["method"]
    native = job["stage"] == "native"
    if native:
        e = job["case"]
        photo = e["case"]
        with Image.open(e["person"]) as im:
            person = ImageOps.exif_transpose(im).convert("RGB")
        with Image.open(e["garment"]) as im:
            garment = garment_rgb(im)
    else:
        photo = job["id"]
    dest = out / ("native" if native else "renders") / str(photo)
    dest.mkdir(parents=True, exist_ok=True)
    with (
        np.load(trained / "students.npz", allow_pickle=False) as weights,
        intervention(runtime.model, method, training, weights) as schedule,
    ):
        mx.synchronize()
        mx.reset_peak_memory()
        start = time.perf_counter()
        extra = {}
        try:
            if native:
                image, timing = runtime.run(
                    person,
                    garment,
                    "upper",
                    e["photo_type"],
                    "balanced",
                    "auto",
                    3,
                    "neutral",
                )
                extra = dict(
                    fallback=timing["region_fallback"],
                    region=timing["render_region"],
                    person_cached=timing["person_cached"],
                )
            else:
                image = render(runtime.model, out / "pairs" / str(photo))
            mx.synchronize()
            seconds = time.perf_counter() - start
            peak = mx.get_peak_memory()
            row = dict(
                id=photo,
                method=method,
                status="success",
                wall_seconds=round(seconds, 5),
                peak_mlx_bytes=peak,
                changed_visits=sum(map(len, schedule)),
                width=image.width,
                height=image.height,
                rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                **extra,
            )
            image.save(dest / (method + ".png"))
            if native and method == "frozen":
                labels = runtime.parser.predict(np.asarray(image))
                Image.fromarray(
                    np.isin(labels, [3, 4, 10]).astype(np.uint8) * 255
                ).save(dest / "mask.png")
        except RenderRejected as exc:
            row = dict(
                id=photo,
                method=method,
                status="rejected",
                wall_seconds=round(time.perf_counter() - start, 5),
                peak_mlx_bytes=mx.get_peak_memory(),
                fallback=exc.retried,
                error=str(exc),
            )
        finally:
            mx.clear_cache()
    return row
