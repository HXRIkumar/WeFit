"""Run counterbalanced, paired block and channel ablations on held-out garments."""

import argparse
import csv
import json
import struct
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median

import mlx.core as mx
import numpy as np
from PIL import Image

from gpu_service.klein_adapter import Adapter, FinalProjection, final_projection
from gpu_service.klein_model import CachedInstructionKlein
from gpu_service.klein_prompts import INSTRUCTIONS

from .evaluate import masked_rgb


class MaskFinalInputs(FinalProjection):
    """Zero the selected columns only on the last denoise step. The dense kernel is retained."""

    def __init__(self, base, steps, adapter, channel_mask):
        super().__init__(base, steps, adapter, False)
        self.channel_mask = mx.array(channel_mask)

    def __call__(self, hidden):
        if self.calls + 1 == self.steps:
            hidden = hidden * self.channel_mask.astype(hidden.dtype)
        return super().__call__(hidden)


@contextmanager
def variant(model, adapter, skip, channel_mask):
    blocks = model.transformer.single_transformer_blocks
    original_head = model.transformer.proj_out
    try:
        if skip:
            model.transformer.single_transformer_blocks = [
                b for i, b in enumerate(blocks) if i not in skip
            ]
        if channel_mask is not None:
            mask = MaskFinalInputs(original_head, 3, adapter, channel_mask)
            model.transformer.proj_out = mask
            yield
            if mask.calls != 3:
                raise RuntimeError("Unexpected number of projection calls")
        elif adapter is not None:
            with final_projection(model, 3, adapter=adapter):
                yield
        else:
            yield
    finally:
        model.transformer.single_transformer_blocks = blocks
        model.transformer.proj_out = original_head


def block_bytes(weights):
    sizes = {}
    for path in (weights / "transformer").glob("*.safetensors"):
        with path.open("rb") as data:
            meta = json.loads(data.read(struct.unpack("<Q", data.read(8))[0]))
        for name, tensor in meta.items():
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "single_transformer_blocks":
                index = int(parts[1])
                start, end = tensor["data_offsets"]
                sizes[index] = sizes.get(index, 0) + end - start
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--weights", type=Path, default=Path("model_src/klein-kv/weights")
    )
    parser.add_argument(
        "--prompts", type=Path, default=Path("model_src/klein-kv/prompt-cache")
    )
    parser.add_argument(
        "--skip-mixed", action="store_true", help="Only test block and channel removal."
    )
    parser.add_argument(
        "--mixed-weights",
        type=Path,
        help="Destination for the separate experimental Q3/Q4 checkpoint.",
    )
    args = parser.parse_args()
    selection = json.loads((args.output / "activity.json").read_text())[
        "residual_rms_order_low_to_high"
    ]
    with (args.output / "head-channels.csv").open() as f:
        channels = [int(r["channel"]) for r in csv.DictReader(f)]
    channel_mask = np.ones(4096, np.float32)
    channel_mask[channels[: len(channels) // 4]] = 0
    adapter = Adapter.load(args.output / "upper-final-r4.npz")
    settings = {
        "frozen": (None, [], None),
        "lora": (adapter, [], None),
        "lora-skip-one": (adapter, selection[:1], None),
        "lora-skip-two": (adapter, selection[:2], None),
        "lora-mask-quarter-head": (adapter, [], channel_mask),
    }
    manifest = json.loads((args.pairs / "manifest.json").read_text())
    samples = [r for r in manifest["records"] if r["split"] == "test"]
    width, height = manifest["canvas"]
    model = CachedInstructionKlein(args.weights, args.prompts)
    model.freeze()
    mx.eval(model.parameters())

    def render(folder):
        return model.generate_image(
            seed=42,
            prompt=INSTRUCTIONS["upper"],
            num_inference_steps=3,
            height=height,
            width=width,
            guidance=1,
            use_kv_cache=True,
            image_paths=[folder / "person.png", folder / "item.png"],
        ).image.convert("RGB")

    warmup = next(r for r in manifest["records"] if r["split"] == "train")
    render(args.pairs / str(warmup["id"]))
    mx.synchronize()
    mx.clear_cache()
    results = []
    names = list(settings)
    for i, row in enumerate(samples):
        folder = args.pairs / str(row["id"])
        with Image.open(folder / "target.png") as im:
            target = im.convert("RGB")
        with Image.open(folder / "mask.png") as im:
            mask = np.asarray(im.convert("L")) > 127
        for order, name in enumerate(names[i:] + names[:i]):
            ad, skip, channels_mask = settings[name]
            mx.reset_peak_memory()
            with variant(model, ad, skip, channels_mask):
                start = time.perf_counter()
                pred = render(folder)
                mx.synchronize()
                elapsed = time.perf_counter() - start
            rgb = masked_rgb(pred, target, mask)
            outside = masked_rgb(pred, target, ~mask)
            if name in ("frozen", "lora"):
                with Image.open(
                    folder / ("baseline.png" if name == "frozen" else "adapted.png")
                ) as prev:
                    if prev.convert("RGB").tobytes() != pred.tobytes():
                        raise RuntimeError(
                            "The paired control did not reproduce its previous pixels"
                        )
            pred.save(folder / (name + ".png"))
            results.append(
                {
                    "id": row["id"],
                    "method": name,
                    "order": order + 1,
                    "wall_s": round(elapsed, 3),
                    "garment_rgb_mae": rgb,
                    "outside_rgb_mae": outside,
                    "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 3),
                }
            )
            print(row["id"], name, round(elapsed, 1), round(rgb, 3), flush=True)
            mx.clear_cache()
    with (args.output / "ablation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    size = block_bytes(args.weights)
    sums = {}
    for name, (_, skip, _) in settings.items():
        runs = [r for r in results if r["method"] == name]
        sums[name] = {
            "images": len(runs),
            "median_s": median(r["wall_s"] for r in runs),
            "garment_rgb_mae": float(np.mean([r["garment_rgb_mae"] for r in runs])),
            "outside_rgb_mae": float(np.mean([r["outside_rgb_mae"] for r in runs])),
            "skipped_blocks": skip,
            "skipped_q4_tensor_bytes": sum(size[i] for i in skip),
        }
    (args.output / "ablation.json").write_text(
        json.dumps(
            {
                "canvas": [width, height],
                "methods": sums,
                "head_columns_masked": int((channel_mask == 0).sum()),
                "base_file_bytes_removed": 0,
                "design": "Five methods, five images; rotate method order for each image after a train-image warm-up.",
            },
            indent=2,
        )
        + "\n"
    )

    if not args.skip_mixed:
        import gc

        from .mixed_eval import compare

        del model
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        candidate = args.mixed_weights or args.pairs.parent / "mixed-weights"
        compare(args.pairs, args.output, args.weights, args.prompts, candidate)


if __name__ == "__main__":
    main()
