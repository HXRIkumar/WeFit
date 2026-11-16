"""Cache final-step training activations from the actual frozen Klein renderer."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
    _Flux2KleinEditHelpers,
)
from PIL import Image

from gpu_service.klein_adapter import final_projection
from gpu_service.klein_model import CachedInstructionKlein
from gpu_service.klein_prompts import INSTRUCTIONS, MODEL_REVISION

from .importance import block_activity, save_head_weights


class CaptureFinalLatents:
    def call_after_loop(self, latents, config, **kwargs):
        self.base = np.array(latents[0].astype(mx.float32), copy=True)
        self.step = float(config.scheduler.sigmas[-1] - config.scheduler.sigmas[-2])


def checkpoint_hashes(weights):
    result = {}
    for path in sorted(weights.rglob("*.safetensors")):
        with path.open("rb") as data:
            result[str(path.relative_to(weights))] = hashlib.file_digest(
                data, "sha256"
            ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument(
        "--weights", type=Path, default=Path("model_src/klein-kv/weights")
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path("model_src/klein-kv/prompt-cache")
    )
    args = parser.parse_args()
    manifest = json.loads((args.pairs / "manifest.json").read_text())
    width, height = manifest["canvas"]
    weights = checkpoint_hashes(args.weights)
    model = CachedInstructionKlein(args.weights, args.prompts)
    model.freeze()
    mx.eval(model.parameters())
    final = CaptureFinalLatents()
    model.callbacks.register(final)
    scope = {
        "revision": MODEL_REVISION,
        "sha256": weights,
        "canvas": manifest["canvas"],
        "steps": 3,
        "seed": 42,
        "target": "upper.final_step.proj_out",
        "guidance": 1,
        "frozen": True,
        "manifest_sha256": hashlib.sha256(
            (args.pairs / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    cache = args.pairs / "cache-info.json"
    if cache.exists() and json.loads(cache.read_text()) != scope:
        raise ValueError(
            "Existing features have different inputs or model settings. Use a new output folder."
        )
    cache.write_text(json.dumps(scope, indent=2) + "\n")
    save_head_weights(model, args.pairs)
    calibration = [r["id"] for r in manifest["records"] if r["split"] == "train"][:3]
    for row in manifest["records"]:
        folder = args.pairs / str(row["id"])
        path = folder / "features.npz"
        if path.exists():
            continue
        started = time.perf_counter()
        with (
            block_activity(model, row["id"] in calibration) as (probes, joint),
            final_projection(model, 3, capture=True) as tap,
        ):
            image = model.generate_image(
                seed=42,
                prompt=INSTRUCTIONS["upper"],
                guidance=1.0,
                height=height,
                width=width,
                num_inference_steps=3,
                use_kv_cache=True,
                image_paths=[folder / "person.png", folder / "item.png"],
            ).image
        if probes:
            (folder / "joint-activity.json").write_text(
                json.dumps([p.step_rms for p in joint]) + "\n"
            )
            (folder / "block-activity.json").write_text(
                json.dumps([p.step_rms for p in probes]) + "\n"
            )
        target, _ = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=model.vae, tiling_config=None, image_paths=[folder / "target.png"]
        )
        target = np.array(target[0].astype(mx.float32), copy=True)
        with Image.open(folder / "mask.png") as mask:
            fraction = (
                np.asarray(
                    mask.resize((width // 16, height // 16), Image.Resampling.BOX),
                    dtype=np.float32,
                ).reshape(-1)
                / 255
            )
        if target.shape != final.base.shape or tap.last_hidden.shape != (
            len(fraction),
            4096,
        ):
            raise ValueError("Unexpected token ordering or projection dimensions.")
        temp = path.with_name("features.partial.npz")
        np.savez_compressed(
            temp,
            hidden=tap.last_hidden,
            base=final.base,
            target=target,
            fraction=fraction,
            delta_sigma=np.float32(final.step),
        )
        image.convert("RGB").save(folder / "baseline.png")
        temp.replace(path)
        mx.synchronize()
        mx.clear_cache()
        print(
            row["split"], row["id"], f"{time.perf_counter() - started:.2f}s", flush=True
        )
    if checkpoint_hashes(args.weights) != weights:
        raise RuntimeError("Base checkpoint bytes changed.")


if __name__ == "__main__":
    main()
