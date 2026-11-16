"""Replay a last-step output adapter from private native tensors using only the image decoder."""

import argparse
import csv
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median

import mlx.core as mx
import numpy as np
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler,
    _step,
)
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_definition import (
    Flux2KleinNoTextEncoderWeightDefinition,
)
from mflux.utils.image_util import ImageUtil
from mlx import nn
from PIL import Image

from gpu_service.klein_adapter import Adapter
from gpu_service.klein_prompts import INSTRUCTIONS, MODEL_REVISION

from .evaluate import masked_rgb


class LastStep:
    def __init__(self):
        self.tensors = {}

    def call_in_loop(self, t, latents, config, **kwargs):
        if t == 1:
            if not isinstance(config.scheduler, FlowMatchEulerDiscreteScheduler):
                raise ValueError(
                    "Replay requires the pinned three-step flow-matching schedule."
                )
            self.tensors["prior"] = latents
            self.tensors["sigma_before"] = config.scheduler.sigmas[2]
            self.tensors["sigma_after"] = config.scheduler.sigmas[3]


class TapProjection(nn.Module):
    def __init__(self, base, recording):
        super().__init__()
        self.base = base
        self.recording = recording
        self.calls = 0

    def __call__(self, hidden):
        prediction = self.base(hidden)
        self.calls += 1
        if self.calls == 3:
            self.recording.tensors["hidden"] = hidden
            self.recording.tensors["noise"] = prediction
        return prediction


@contextmanager
def capture(model):
    """Hold native MLX references. Serializing them is outside the full-render stopwatch."""
    state = LastStep()
    old = model.transformer.proj_out
    tap = TapProjection(old, state)
    model.callbacks.register(state)
    model.transformer.proj_out = tap
    try:
        yield state
        if tap.calls != 3 or len(state.tensors) != 5:
            raise ValueError("A replay state needs all three frozen denoising steps.")
    finally:
        model.transformer.proj_out = old
        model.callbacks.in_loop_callbacks().remove(state)


def save_state(state, directory, size, item_id):
    final = directory / "last-step.safetensors"
    partial = directory / "last-step.partial.safetensors"
    photo_hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("person.png", "item.png")
    }
    meta = {
        "format": "wefit-output-replay-v1",
        "revision": MODEL_REVISION,
        "size": json.dumps(size),
        "sample": str(item_id),
        "inputs": json.dumps(photo_hashes, sort_keys=True),
        "profile": "q4.upper.seed42.guidance1.steps3",
    }
    mx.save_safetensors(str(partial), state.tensors, meta)
    partial.replace(final)


class DecoderWeights(Flux2KleinNoTextEncoderWeightDefinition):
    @staticmethod
    def get_components():
        return [
            c
            for c in Flux2KleinNoTextEncoderWeightDefinition.get_components()
            if c.name == "vae"
        ]

    @staticmethod
    def get_download_patterns():
        return ["vae/*.safetensors", "vae/*.json"]


def load_decoder(weights):
    vae = Flux2VAE()
    vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
    loaded = WeightLoader.load(
        weight_definition=DecoderWeights, model_path=str(weights.resolve())
    )
    WeightApplier.apply_and_quantize(
        weights=loaded,
        quantize_arg=4,
        weight_definition=DecoderWeights,
        models={"vae": vae},
    )
    vae.freeze()
    mx.eval(vae.parameters())
    return vae


def replay(vae, path, adapter=None, strength=1.0):
    state, metadata = mx.load(str(path), return_metadata=True)
    if (
        metadata.get("format") != "wefit-output-replay-v1"
        or metadata.get("revision") != MODEL_REVISION
        or metadata.get("profile") != "q4.upper.seed42.guidance1.steps3"
    ):
        raise ValueError("Choose a state captured with the pinned three-step Q4 study.")
    w, h = json.loads(metadata["size"])
    prior = state["prior"]
    if (
        w <= 0
        or h <= 0
        or w % 16
        or h % 16
        or prior.shape != (1, (w // 16) * (h // 16), 128)
        or state["noise"].shape != prior.shape
        or state["hidden"].shape != (*prior.shape[:2], 4096)
    ):
        raise ValueError(
            "The stored final-layer feature grid has incompatible dimensions."
        )
    if not 0 <= strength <= 1:
        raise ValueError("Adapter strength must be between zero and one.")
    noise = state["noise"]
    if adapter is not None and strength:
        delta = (
            state["hidden"].astype(mx.float32) @ mx.array(adapter.down)
        ) @ mx.array(adapter.up)
        if strength != 1:
            delta = delta * strength
        noise = (noise.astype(mx.float32) + delta).astype(noise.dtype)
    latent = _step(noise, prior, state["sigma_after"], state["sigma_before"])
    grid = latent.reshape(1, h // 16, w // 16, 128).transpose(0, 3, 1, 2)
    return ImageUtil.to_pil(vae.decode_packed_latents(grid)).convert("RGB")


def capture_validation(pairs, weights, prompts, records, size):
    from gpu_service.klein_model import CachedInstructionKlein

    model = CachedInstructionKlein(weights, prompts)
    model.freeze()
    mx.eval(model.parameters())
    for row in records:
        folder = pairs / str(row["id"])
        with capture(model) as state:
            image = model.generate_image(
                seed=42,
                prompt=INSTRUCTIONS["upper"],
                num_inference_steps=3,
                guidance=1.0,
                width=size[0],
                height=size[1],
                image_paths=[folder / "person.png", folder / "item.png"],
                use_kv_cache=True,
            ).image.convert("RGB")
        with Image.open(folder / "baseline.png") as control:
            if image.tobytes() != control.convert("RGB").tobytes():
                raise ValueError(
                    "New frozen render differed from its original validation control."
                )
        save_state(state, folder, size, row["id"])
        print("Recorded validation state", row["id"], flush=True)
        mx.clear_cache()


def study(pairs, out, weights, prompts):
    import gc

    manifest = json.loads((pairs / "manifest.json").read_text())
    validation = [r for r in manifest["records"] if r["split"] == "val"]
    test = [r for r in manifest["records"] if r["split"] == "test"]
    # Existing test states were saved after the Q4 side of the mixed-precision comparison.
    for row in test:
        if not (pairs / str(row["id"]) / "last-step.safetensors").is_file():
            raise ValueError(
                "Run the Q4/mixed paired comparison to collect held-out states first."
            )
    capture_validation(pairs, weights, prompts, validation, manifest["canvas"])
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    start = time.perf_counter()
    vae = load_decoder(weights)
    decoder_load = time.perf_counter() - start
    compact = Adapter.load(out / "upper-compact.npz")
    full = Adapter.load(out / "upper-final-r4.npz")
    choices = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []

    def run(row, variant, adapter, strength, control=None):
        folder = pairs / str(row["id"])
        state_path = folder / "last-step.safetensors"
        with Image.open(folder / "target.png") as img:
            target = img.convert("RGB")
        with Image.open(folder / "mask.png") as img:
            mask = np.asarray(img.convert("L")) > 127
        start = time.perf_counter()
        pred = replay(vae, state_path, adapter, strength)
        mx.synchronize()
        seconds = time.perf_counter() - start
        exact = None
        max_rgb_difference = None
        if control:
            with Image.open(folder / control) as img:
                control_image = img.convert("RGB")
                difference = np.abs(
                    np.asarray(control_image, dtype=np.int16) - np.asarray(pred)
                )
                max_rgb_difference = int(difference.max())
                exact = max_rgb_difference == 0
            if not exact:
                print(
                    f"Replay differs on {row['id']} {variant}: maximum RGB difference {max_rgb_difference}",
                    flush=True,
                )
        rows.append(
            {
                "id": row["id"],
                "split": row["split"],
                "method": variant,
                "strength": strength,
                "wall_s": round(seconds, 4),
                "state_bytes": state_path.stat().st_size,
                "garment_error": masked_rgb(pred, target, mask),
                "outside_error": masked_rgb(pred, target, ~mask),
                "pixel_identical": exact,
                "max_rgb_difference": max_rgb_difference,
            }
        )

    for row in validation:
        for strength in choices:
            run(
                row,
                "strength-sweep",
                compact,
                strength,
                "baseline.png" if strength == 0 else None,
            )
    chosen = min(
        choices,
        key=lambda s: (mean(r["garment_error"] for r in rows if r["strength"] == s), s),
    )
    for row in test:
        for name, adapter, strength, path in [
            ("frozen", None, 0, "baseline.png"),
            ("trained", full, 1, "adapted.png"),
            ("compact", compact, 1, "compact.png"),
            ("validation-choice", compact, chosen, None),
        ]:
            run(row, name, adapter, strength, path)
        print("Verified replay", row["id"], flush=True)
    with (out / "replay-runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    held = [r for r in rows if r["split"] == "test"]
    with (out / "mixed-runs.csv").open() as file:
        original = [r for r in csv.DictReader(file) if r["method"] == "q4"]
    compact_runs = [r for r in held if r["method"] == "compact"]
    result = {
        "canvas": manifest["canvas"],
        "validation_images": len(validation),
        "test_images": len(test),
        "validation_strengths": choices,
        "selected_strength": chosen,
        "decoder_ready_s": round(decoder_load, 4),
        "matched_full_median_s": median(float(r["wall_s"]) for r in original),
        "matched_compact_replay_median_s": median(r["wall_s"] for r in compact_runs),
        "exact_heldout_controls": sum(r["pixel_identical"] is True for r in held),
        "compared_heldout_controls": sum(
            r["pixel_identical"] is not None for r in held
        ),
        "state_bytes_per_image": sorted({r["state_bytes"] for r in held}),
        "test_garment_error": {
            k: mean(r["garment_error"] for r in held if r["method"] == k)
            for k in ["frozen", "trained", "compact", "validation-choice"]
        },
        "scope": "One full render stores the final projection input, original prediction, previous latent and schedule. The replay timer includes reading the state, correction and full native decoding with a resident VAE. New photos, garments, seeds and earlier-layer changes require the full transformer.",
    }
    (out / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("state", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--strength", type=float, default=1)
    p.add_argument("--weights", type=Path, default=Path("model_src/klein-kv/weights"))
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a fresh output image.")
    decoder = load_decoder(args.weights)
    adapter = Adapter.load(args.adapter) if args.adapter else None
    image = replay(decoder, args.state, adapter, args.strength)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")


if __name__ == "__main__":
    main()
