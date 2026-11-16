"""Measure whether omitting literal zeros saves space in the actual packed output layer."""

import argparse
import json
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np

from .mixed_checkpoint import read_header


def measure(weights, pairs):
    folder = weights / "transformer"
    index = json.loads((folder / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    for suffix in ("weight", "scales", "biases"):
        key = "proj_out." + suffix
        path = folder / index[key]
        header, start = read_header(path)
        meta = header[key]
        dtype = np.uint32 if suffix == "weight" else np.uint16
        tensors[suffix] = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=start + meta["data_offsets"][0],
            shape=tuple(meta["shape"]),
        )
    w = np.load(pairs / "base-output-head.npy", allow_pickle=False)
    keep = w.ravel() != 0
    dense = tensors["weight"]
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    codes = ((dense[..., None] >> shifts) & 15).reshape(-1).astype(np.uint8)
    mask = np.packbits(keep, bitorder="little")
    nonzero = codes[keep]
    if len(nonzero) % 2:
        nonzero = np.pad(nonzero, (0, 1))
    payload = nonzero[::2] | (nonzero[1::2] << 4)
    # Round trip the bitmap and nibbles. The consumer substitutes zero directly at omitted locations.
    recovered_keep = np.unpackbits(mask, bitorder="little", count=codes.size).astype(
        bool
    )
    recovered_codes = np.zeros_like(codes)
    recovered_codes[recovered_keep] = np.column_stack(
        (payload & 15, payload >> 4)
    ).ravel()[: int(recovered_keep.sum())]
    packed = np.bitwise_or.reduce(
        recovered_codes.reshape(*dense.shape, 8).astype(np.uint32) << shifts, axis=-1
    )
    with mx.stream(mx.gpu):
        reconstructed = mx.dequantize(
            mx.array(packed),
            mx.array(tensors["scales"]).view(mx.bfloat16),
            mx.array(tensors["biases"]).view(mx.bfloat16),
            group_size=64,
            bits=4,
        )
        decoded = np.array(reconstructed.astype(mx.float32))
    decoded[~recovered_keep.reshape(w.shape)] = 0
    offsets = tensors["scales"].nbytes + tensors["biases"].nbytes
    affine = tensors["scales"].tobytes() + tensors["biases"].tobytes()
    originals = {
        "dense": dense.tobytes() + affine,
        "bitmap": mask.tobytes() + payload.tobytes() + affine,
    }
    deflated = {key: zlib.compress(raw, level=9) for key, raw in originals.items()}
    return {
        "head_connections": codes.size,
        "zero_weights": int((~keep).sum()),
        "original_q4_bytes": dense.nbytes + offsets,
        "zero_bitmap_bytes": mask.nbytes,
        "nonzero_nibbles_bytes": payload.nbytes,
        "scale_and_offset_bytes": offsets,
        "bitmap_representation_bytes": mask.nbytes + payload.nbytes + offsets,
        "deflate_dense_bytes": len(deflated["dense"]),
        "deflate_bitmap_bytes": len(deflated["bitmap"]),
        "deflate_byte_exact": all(
            zlib.decompress(blob) == originals[key] for key, blob in deflated.items()
        ),
        "decoded_values_exact": bool(np.array_equal(decoded, w)),
        "nonzero_codes_exact": bool(np.array_equal(recovered_codes[keep], codes[keep])),
        "scope": "Final Q4 output projection. The bitmap uses one presence bit per weight, four bits per retained code and original scales and offsets. DEFLATE level 9 separately compares archiving both layouts. Inference uses the dense kernel.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("pairs", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    stats = measure(args.weights, args.pairs)
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)
