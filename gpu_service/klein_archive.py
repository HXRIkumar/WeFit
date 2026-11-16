"""Read the pinned Q4 checkpoint one tensor at a time from its lossless ZIP."""

import hashlib
import json
import math
import struct
import zipfile

import numpy as np

# Safetensors member sizes and SHA256 of the pinned community conversion.
SHARDS = {
    "transformer/0.safetensors": (
        2143173694,
        "d2d305be17d11b98f2353a4318495904a3c74259e9a1905b2c93fb5f88232f9e",
    ),
    "transformer/1.safetensors": (
        2085643520,
        "68804583c986f32493718e9bae3f8239db98a0526362fa9262a6342a6a59994f",
    ),
    "transformer/2.safetensors": (
        877964193,
        "20e903436caed8d9e048117ae28aef4425bcd367c7dc280618376073782cb337",
    ),
    "vae/0.safetensors": (
        165107888,
        "ab26bf1613e6b51a955bfaf05409e2a16138a072ba0bb7f4fc768236fb4a990b",
    ),
}


def tensors(reader, total_size, expected_sha):
    digest = hashlib.sha256()

    def take(n):
        block = reader.read(n)
        if len(block) != n:
            raise ValueError("Truncated archive member")
        digest.update(block)
        return block

    header_len = struct.unpack("<Q", take(8))[0]
    if not 2 <= header_len <= min(4 * 1024**2, total_size - 8):
        raise ValueError("Invalid tensor header length")
    header = json.loads(take(header_len))
    if header.pop("__metadata__", {}) != {
        "mflux_version": "0.18.0",
        "quantization_level": "4",
    }:
        raise ValueError("The archive must contain the saved Q4 conversion")
    offset = 0
    for name, desc in sorted(
        header.items(), key=lambda item: item[1]["data_offsets"][0]
    ):
        start, stop = desc["data_offsets"]
        dtype = {"U32": "<u4", "BF16": "<u2"}.get(desc["dtype"])
        dims = desc["shape"]
        if (
            not dtype
            or not isinstance(dims, list)
            or any(type(i) is not int or i < 0 for i in dims)
            or start != offset
            or stop - start != math.prod(dims) * np.dtype(dtype).itemsize
            or stop > total_size - header_len - 8
            or stop - start > 400 * 1024**2
        ):
            raise ValueError("Invalid tensor in checkpoint archive")
        yield (
            name,
            desc["dtype"],
            np.frombuffer(take(stop - start), dtype=dtype).reshape(dims),
        )
        offset = stop
    if (
        offset + header_len + 8 != total_size
        or reader.read(1)
        or digest.hexdigest() != expected_sha
    ):
        raise ValueError("The archived model differs from the pinned Q4 checkpoint")


def load(path):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from mflux.models.common.weights.loading.loaded_weights import (
        LoadedWeights,
        MetaData,
    )

    components = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        for member, (size, sha) in SHARDS.items():
            if entries.count(member) != 1 or archive.getinfo(member).file_size != size:
                raise ValueError(f"Missing or incorrect archived shard {member}")
            component = components.setdefault(member.split("/")[0], {})
            with archive.open(member) as stream:
                for name, kind, raw in tensors(stream, size, sha):
                    value = mx.array(raw)
                    if kind == "BF16":
                        value = value.view(mx.bfloat16)
                    if name in component:
                        raise ValueError("Duplicate checkpoint tensor in archive")
                    mx.eval(value)
                    component[name] = value
    trees = {k: tree_unflatten(list(v.items())) for k, v in components.items()}
    return LoadedWeights(
        components=trees,
        meta_data=MetaData(quantization_level=4, mflux_version="0.18.0"),
    )
