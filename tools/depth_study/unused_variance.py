"""Measure the unused half of the pinned VAE's posterior projection."""

import hashlib
from pathlib import Path
import time

import mlx.core as mx
from mlx import nn
import numpy as np
from PIL import Image
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
    _Flux2KleinEditHelpers,
)
from mflux.utils.image_util import ImageUtil

from .capture import render
from .run import save


class PosteriorMean(nn.Module):
    """Compute only the 32 outputs consumed by Flux2VAE.encode's split."""

    def __init__(self, original):
        super().__init__()
        if original.weight.shape != (64, 1, 1, 64) or original.bias.shape != (64,):
            raise ValueError("Expected the pinned 64-channel posterior projection")
        self.mean = nn.Conv2d(64, 32, kernel_size=1)
        self.mean.weight = mx.contiguous(original.weight[:32])
        self.mean.bias = mx.contiguous(original.bias[:32])
        mx.eval(self.mean.parameters())

    def __call__(self, x):
        mean = self.mean(x)
        # The upstream encoder splits this result in half and uses only the first.
        return mx.concatenate((mean, mx.zeros_like(mean)), axis=-1)


def compare_bytes(a, b):
    delta = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
    return dict(
        channels=delta.size,
        matching_channels=int((delta == 0).sum()),
        max_delta=int(delta.max()),
    )


def inspect(runtime, study, case, person, garment, photo, reference_dir):
    source = runtime.model.vae.quant_conv
    half = PosteriorMean(source)
    directory = study / "mean-projection"
    directory.mkdir(exist_ok=True)
    files = {}
    for name, conv in (("original", source), ("mean", half.mean)):
        file = directory / (name + ".safetensors")
        mx.save_safetensors(str(file), {"weight": conv.weight, "bias": conv.bias})
        files[name] = dict(
            parameters=conv.weight.size + conv.bias.size,
            payload_bytes=conv.weight.nbytes + conv.bias.nbytes,
            file_bytes=file.stat().st_size,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        )
    features = []
    with Image.open(study / "pairs" / str(photo) / "person.png") as known:
        for label, img in (
            ("test-photo", known),
            ("camera-photo", person),
            ("blue-product", garment),
        ):
            rgb = img.convert("RGB")
            rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
            ref = _Flux2KleinEditHelpers.prepare_reference_image(rgb)
            encoded = runtime.model.vae.encoder(ImageUtil.to_array(ref)).transpose(
                0, 2, 3, 1
            )
            full = source(encoded)
            limited = half.mean(encoded)
            before = full[..., :32]
            mx.eval(full, limited)
            features.append(
                dict(
                    input=label,
                    reference_size=list(ref.size),
                    mean_values=before.size,
                    exact_mean_values=int(mx.sum(before == limited).item()),
                    max_mean_difference=float(mx.max(mx.abs(before - limited)).item()),
                    nonzero_discarded_values=int(mx.sum(full[..., 32:] != 0).item()),
                )
            )
            del encoded, full, limited, before
            mx.clear_cache()
    comparisons = []
    runtime.model.vae.quant_conv = half
    try:
        keys = [photo]
        if (reference_dir / (case["case"] + ".png")).is_file():
            keys.append(case["case"])
        for key in keys:
            started = time.perf_counter()
            if key == photo:
                actual = render(runtime.model, study / "pairs" / str(photo))
            else:
                actual, _ = runtime.run(
                    person,
                    garment,
                    "upper",
                    case["photo_type"],
                    "balanced",
                    "auto",
                    3,
                    "neutral",
                )
            mx.synchronize()
            saved = reference_dir / (str(key) + ".png")
            with Image.open(saved) as previous:
                expected = previous.convert("RGB")
                comparisons.append(
                    dict(
                        id=key,
                        wall_seconds=round(time.perf_counter() - started, 5),
                        rgb_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                        reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                        **compare_bytes(actual, expected),
                    )
                )
            mx.clear_cache()
    finally:
        runtime.model.vae.quant_conv = source
    result = dict(
        layer="vae.quant_conv",
        transformer_profile=runtime.weight_profile,
        source_weight_shape=list(source.weight.shape),
        source_dtype=str(source.weight.dtype),
        kept_output_rows=list(range(32)),
        projection_files=files,
        encoder_inputs=features,
        complete_renders=comparisons,
    )
    save(study / "unused-variance.json", result)
    print("VAE posterior projection", result, flush=True)
