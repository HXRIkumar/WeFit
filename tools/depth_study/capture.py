"""Collect teacher residual targets and every block's response at three denoising steps."""

import argparse
import hashlib
import os
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx import nn
import numpy as np
from PIL import Image

from gpu_service.klein_prompts import INSTRUCTIONS, MODEL_REVISION
from tools.adapter_study.importance import block_activity


def render(model, folder):
    return model.generate_image(
        seed=42,
        prompt=INSTRUCTIONS["upper"],
        guidance=1.0,
        height=320,
        width=256,
        num_inference_steps=3,
        use_kv_cache=True,
        image_paths=[folder / "person.png", folder / "item.png"],
    ).image.convert("RGB")


def token_indices(total, refs, fraction, seed):
    count = len(fraction)
    text = total - refs - count
    if text < 1:
        raise ValueError("Unexpected text and image token layout.")
    rng = np.random.default_rng(seed)

    def sample(indices, n):
        return rng.choice(indices, n, replace=len(indices) < n)

    inside = np.flatnonzero(fraction >= 0.5)
    outside = np.flatnonzero(fraction < 0.5)
    if not len(inside):
        inside = np.arange(count)
    if not len(outside):
        outside = np.arange(count)
    groups = [
        sample(np.arange(text), 12),
        text + sample(inside, 24),
        text + sample(outside, 12),
    ]
    last = np.arange(text + count, total) if refs else np.arange(text, total)
    groups.append(sample(last, 16))
    return np.concatenate(groups).astype(np.int32)


class TeacherTap(nn.Module):
    def __init__(self, block, fraction, photo):
        super().__init__()
        self.inner = block
        self.fraction = fraction
        self.photo = photo
        self.saved = []

    def __call__(self, hidden_states, **kwargs):
        out = self.inner(hidden_states=hidden_states, **kwargs)
        cache = kwargs["kv_cache"]
        refs = cache.num_ref_tokens if cache.is_extracting else 0
        idx = mx.array(
            token_indices(
                hidden_states.shape[1],
                refs,
                self.fraction,
                self.photo * 10 + len(self.saved),
            )
        )
        x = hidden_states[0, idx, :].astype(mx.float32)
        delta = out[0, idx, :].astype(mx.float32) - x
        self.saved.append(np.stack([np.array(x), np.array(delta)]))
        return out


def capture_one(model, pairs, photo, candidates):
    folder = pairs / str(photo)
    final = folder / "teacher-samples.npz"
    signature = {
        "revision": MODEL_REVISION,
        "sample_format": "float32-v1",
        "blocks": list(candidates),
        "precision": os.getenv("KLEIN_COMPUTE_PRECISION", "native"),
        "seed": 42,
        "steps": 3,
        "inputs": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("person.png", "item.png", "mask.png")
        },
    }
    meta_path = folder / "teacher-blocks.json"
    if final.exists() and meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("signature") == signature:
            return {"id": photo, "previously_recorded": True}
    with Image.open(folder / "mask.png") as im:
        fraction = (
            np.asarray(im.convert("L").resize((16, 20), Image.Resampling.BOX)).ravel()
            / 255
        )
    started = time.perf_counter()
    with block_activity(model, True) as (single, joint):
        original = model.transformer.single_transformer_blocks
        taps = {b: TeacherTap(original[b], fraction, photo) for b in candidates}
        model.transformer.single_transformer_blocks = [
            taps.get(b, block) for b, block in enumerate(original)
        ]
        try:
            image = render(model, folder)
        finally:
            model.transformer.single_transformer_blocks = original
    if not all(len(t.saved) == 3 for t in taps.values()):
        raise ValueError("Expected three teacher passes at every selected block.")
    arrays = {f"block_{b}": np.stack(t.saved) for b, t in taps.items()}
    for key, data in arrays.items():
        if data.shape != (3, 2, 64, 4096) or not np.isfinite(data).all():
            raise ValueError(
                f"Invalid {key} shape {data.shape}, nonfinite {int((~np.isfinite(data)).sum())}."
            )
    image.save(folder / "teacher.png")
    tmp = folder / "teacher-samples.partial.npz"
    np.savez(tmp, **arrays)
    meta = {
        "id": photo,
        "seconds": round(time.perf_counter() - started, 3),
        "samples_per_block_step": 64,
        "signature": signature,
        "single": [p.step_rms for p in single],
        "dual": [p.step_rms for p in joint],
        "captured_blocks": candidates,
    }
    (folder / "teacher-blocks.json").write_text(
        json.dumps(meta, separators=(",", ":")) + "\n"
    )
    tmp.replace(final)
    mx.clear_cache()
    return {"id": photo, "seconds": meta["seconds"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", type=Path)
    p.add_argument("activity", type=Path)
    args = p.parse_args()
    from gpu_service.klein_model import CachedInstructionKlein

    model = CachedInstructionKlein(
        Path("model_src/klein-kv/weights"), Path("model_src/klein-kv/prompt-cache")
    )
    model.freeze()
    mx.eval(model.parameters())
    indices = json.loads(args.activity.read_text())["residual_rms_order_low_to_high"][
        :12
    ]
    for row in json.loads((args.pairs / "manifest.json").read_text())["records"]:
        if row["split"] == "train":
            print(capture_one(model, args.pairs, row["id"], indices), flush=True)


if __name__ == "__main__":
    main()
