"""Count logical parameters and stored tensor bytes directly from sharded safetensors headers."""

import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


def inventory(root):
    result = {}
    mixed_manifest = root / "wefit-mixed.json"
    changed = (
        set(
            json.loads(mixed_manifest.read_text())[
                "projection_relative_rms_weight_error"
            ]
        )
        if mixed_manifest.is_file()
        else set()
    )
    for component in ("transformer", "vae"):
        tensors = {}
        for path in sorted((root / component).glob("*.safetensors")):
            with path.open("rb") as data:
                header = json.loads(data.read(struct.unpack("<Q", data.read(8))[0]))
            for key, meta in header.items():
                if key != "__metadata__":
                    if key in tensors:
                        raise ValueError(f"Repeated tensor {key}")
                    tensors[key] = meta
        if not tensors:
            raise ValueError(f"No {component} checkpoint tensors under {root}")
        groups = defaultdict(
            lambda: {"logical_weights": 0, "stored_tensor_bytes": 0, "tensors": 0}
        )
        for key, meta in tensors.items():
            parts = key.split(".")
            group = (
                ".".join(parts[:2])
                if parts[0] in ("transformer_blocks", "single_transformer_blocks")
                else parts[0]
            )
            info = groups[group]
            begin, end = meta["data_offsets"]
            info["stored_tensor_bytes"] += end - begin
            info["tensors"] += 1
            if key.endswith(
                (
                    ".scales",
                    ".biases",
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                )
            ):
                continue
            if key.endswith(".weight") and meta["dtype"] == "U32":
                if key[:-6] + "scales" not in tensors:
                    raise ValueError(f"Packed weight missing affine scales: {key}")
                bits = (
                    3
                    if component == "transformer"
                    and key.removesuffix(".weight") in changed
                    else 4
                )
                packed_bits = math.prod(meta["shape"]) * 32
                if packed_bits % bits:
                    raise ValueError(f"Invalid packed tensor length for {key}")
                count = packed_bits // bits
            else:
                count = math.prod(meta["shape"])
            info["logical_weights"] += count
        result[component] = {
            "groups": dict(groups),
            "logical_weights": sum(v["logical_weights"] for v in groups.values()),
            "stored_tensor_bytes": sum(
                v["stored_tensor_bytes"] for v in groups.values()
            ),
            "tensors": len(tensors),
        }
    result["definition"] = (
        "Logical weight and bias entries in the pinned checkpoint. Affine quantization scales, offsets and running statistics count only toward stored bytes. Safetensors headers and index files are excluded from byte totals."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = inventory(args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    for key in ["transformer", "vae"]:
        print(key, data[key]["logical_weights"], data[key]["stored_tensor_bytes"])
