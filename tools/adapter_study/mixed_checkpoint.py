"""Build a separate Q3/Q4 candidate by requantizing six training-selected blocks."""

import argparse
import json
import os
import shutil
import struct
import tempfile
from contextlib import ExitStack
from pathlib import Path

import mlx.core as mx
import numpy as np

from gpu_service.klein_prompts import MODEL_REVISION


def read_header(path):
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length)), length + 8


def copy_span(src, dest, count):
    while count:
        data = src.read(min(count, 4 * 1024 * 1024))
        if not data:
            raise EOFError("Truncated safetensors payload")
        dest.write(data)
        count -= len(data)


def link_or_copy(src, dest):
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def convert_projection(name, locations, scratch):
    suffixes = ("weight", "scales", "biases")
    views = []
    ends = []
    for suffix in suffixes:
        key = name + "." + suffix
        path, header, start = locations[key]
        dtype = np.uint32 if suffix == "weight" else np.uint16
        if header["dtype"] != ("U32" if suffix == "weight" else "BF16"):
            raise ValueError(f"Unsupported stored tensor type for {key}")
        views.append(
            np.memmap(
                path,
                mode="r",
                dtype=dtype,
                offset=start + header["data_offsets"][0],
                shape=tuple(header["shape"]),
            )
        )
        ends.append(scratch / key)
    norm = error = 0.0
    with mx.stream(mx.cpu), ExitStack() as stack:
        writers = [stack.enter_context(dest.open("wb")) for dest in ends]
        for start in range(0, views[0].shape[0], 128):
            arrays = [mx.array(np.asarray(v[start : start + 128])) for v in views]
            w = mx.dequantize(
                arrays[0],
                arrays[1].view(mx.bfloat16),
                arrays[2].view(mx.bfloat16),
                group_size=64,
                bits=4,
            )
            q, scale, bias = mx.quantize(w, group_size=64, bits=3)
            approx = mx.dequantize(q, scale, bias, group_size=64, bits=3)
            diff = approx.astype(mx.float32) - w.astype(mx.float32)
            norm += float(mx.sum(mx.square(w.astype(mx.float32))).item())
            error += float(mx.sum(mx.square(diff)).item())
            for data, writer in zip(
                (q, scale.view(mx.uint16), bias.view(mx.uint16)), writers, strict=True
            ):
                writer.write(np.array(data).tobytes())
    outputs = {}
    for suffix, view, path in zip(suffixes, views, ends, strict=True):
        dims = list(view.shape)
        if suffix == "weight":
            dims[1] = dims[1] * 3 // 4
        outputs[name + "." + suffix] = (path, dims)
    return outputs, (error / max(norm, 1e-20)) ** 0.5


def rewrite_shard(source, dest, replacements):
    header, start = read_header(source)
    keys = sorted(
        (k for k in header if k != "__metadata__"),
        key=lambda k: header[k]["data_offsets"][0],
    )
    new = {}
    if "__metadata__" in header:
        new["__metadata__"] = header["__metadata__"]
    offset = 0
    for key in keys:
        value = header[key].copy()
        begin, end = value["data_offsets"]
        size = end - begin
        if key in replacements:
            file, shape = replacements[key]
            size = file.stat().st_size
            value["shape"] = shape
        value["data_offsets"] = [offset, offset + size]
        offset += size
        new[key] = value
    raw = json.dumps(new, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with source.open("rb") as src, dest.open("xb") as writer:
        writer.write(struct.pack("<Q", len(raw)))
        writer.write(raw)
        for key in keys:
            if key in replacements:
                with replacements[key][0].open("rb") as data:
                    shutil.copyfileobj(data, writer, 4 * 1024 * 1024)
            else:
                begin, end = header[key]["data_offsets"]
                src.seek(start + begin)
                copy_span(src, writer, end - begin)


def build(source, out, activity_path):
    source = source.resolve()
    out = out.resolve()
    if out.exists() or source in out.parents:
        raise ValueError("Choose a new checkpoint folder outside the original weights.")
    split = json.loads(activity_path.read_text())
    indices = split["residual_rms_order_low_to_high"][:6]
    tensor_files = list(source.rglob("*.safetensors"))
    original_bytes = sum(p.stat().st_size for p in tensor_files)
    locations = {}
    for f in (source / "transformer").glob("*.safetensors"):
        header, start = read_header(f)
        locations.update(
            {
                name: (f, data, start)
                for name, data in header.items()
                if name != "__metadata__"
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".wefit-quant-") as temp:
        temp = Path(temp)
        scratch = temp / "tensors"
        scratch.mkdir()
        candidate = temp / "weights"
        candidate.mkdir()
        updates = {}
        errors = {}
        for idx in indices:
            for layer in ("to_qkv_mlp_proj", "to_out"):
                key = f"single_transformer_blocks.{idx}.attn.{layer}"
                changed, errors[key] = convert_projection(key, locations, scratch)
                updates.update(changed)
        modified = {locations[k][0] for k in updates}
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            dest = candidate / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path in modified:
                rewrite_shard(path, dest, updates)
            else:
                link_or_copy(path, dest)
        packed_bytes = sum(p.stat().st_size for p in candidate.rglob("*.safetensors"))
        result = {
            "base_revision": MODEL_REVISION,
            "profile": "wefit-q3-six-blocks",
            "blocks": indices,
            "projections_requantized": len(errors),
            "source_quantization": 4,
            "target_quantization": 3,
            "stored_bytes_before": original_bytes,
            "stored_bytes_after": packed_bytes,
            "storage_saving_bytes": original_bytes - packed_bytes,
            "projection_relative_rms_weight_error": errors,
            "unit": "Physical file length of all transformer and VAE safetensors shards. Shared files count at their full logical length.",
        }
        from .extract import checkpoint_hashes

        result["source_sha256"] = checkpoint_hashes(source)
        result["candidate_sha256"] = checkpoint_hashes(candidate)
        for name in ("LICENSE", "NOTICE"):
            terms = source.parent / name
            if terms.is_file():
                shutil.copy2(terms, candidate / name)
        with (candidate / "NOTICE").open("a") as notice:
            notice.write(
                "\nWeFit V2 research modification. Twelve projections in six "
                "single-stream transformer blocks were requantized from four "
                "to three bits. All transformer blocks and the original VAE "
                "are retained. See wefit-mixed.json for the source revision, "
                "changed tensors and checkpoint hashes.\n"
            )
        (candidate / "wefit-mixed.json").write_text(json.dumps(result, indent=2) + "\n")
        candidate.rename(out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("activity", type=Path)
    a = p.parse_args()
    r = build(a.source, a.output, a.activity)
    print(json.dumps(r, indent=2))
