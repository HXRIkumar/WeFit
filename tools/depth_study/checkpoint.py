"""Write a runnable structural checkpoint, preserving original indices in a manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile

import numpy as np

from gpu_service.klein_prompts import MODEL_REVISION
from gpu_service.klein_archive import SHARDS
from tools.adapter_study.inventory import inventory
from tools.adapter_study.mixed_checkpoint import read_header, copy_span, link_or_copy
from tools.adapter_study.extract import checkpoint_hashes

SINGLE = re.compile(r"^single_transformer_blocks\.(\d+)\.(.+)$")


def rename(key, mapping):
    m = SINGLE.match(key)
    if not m:
        return key
    idx = int(m.group(1))
    return (
        f"single_transformer_blocks.{mapping[idx]}.{m.group(2)}"
        if idx in mapping
        else None
    )


def rewrite(source, dest, mapping):
    old, base = read_header(source)
    keys = sorted(
        (k for k in old if k != "__metadata__"), key=lambda k: old[k]["data_offsets"][0]
    )
    names = [(k, rename(k, mapping)) for k in keys]
    if all(k == new for k, new in names):
        link_or_copy(source, dest)
        return
    new_header = {"__metadata__": old["__metadata__"]} if "__metadata__" in old else {}
    offset = 0
    retained = [(k, n) for k, n in names if n is not None]
    if not retained:
        return
    for key, new in retained:
        first, last = old[key]["data_offsets"]
        new_header[new] = {**old[key], "data_offsets": [offset, offset + last - first]}
        offset += last - first
    raw = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with source.open("rb") as inp, dest.open("xb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for key, _ in retained:
            first, last = old[key]["data_offsets"]
            inp.seek(base + first)
            copy_span(inp, output, last - first)


def size(root):
    return sum(p.stat().st_size for p in root.rglob("*.safetensors"))


def build(source, learned, output, n):
    source, output = source.resolve(), output.resolve()
    if (
        output.exists()
        or source in output.parents
        or (source / "wefit-depth.json").exists()
        or (source / "wefit-mixed.json").exists()
    ):
        raise ValueError("Use the original Q4 weights and a new separate destination")
    data = json.loads((learned / "training.json").read_text())
    if data["base_revision"] != MODEL_REVISION or n not in (2, 6, 12):
        raise ValueError("Unknown source revision or depth")
    source_hashes = checkpoint_hashes(source)
    if source_hashes != {name: digest for name, (_, digest) in SHARDS.items()}:
        raise ValueError("The student requires the exact Q4 teacher used for training.")
    removed = sorted(data["structural_order"][:n])
    kept = [i for i in range(24) if i not in removed]
    mapping = {index: j for j, index in enumerate(kept)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        folder = Path(temp) / "weights"
        folder.mkdir()
        for src in source.rglob("*"):
            if not src.is_file():
                continue
            dst = folder / src.relative_to(source)
            dst.parent.mkdir(exist_ok=True, parents=True)
            if src.parent.name == "transformer" and src.suffix == ".safetensors":
                rewrite(src, dst, mapping)
            elif (
                src.parent.name == "transformer"
                and src.name == "model.safetensors.index.json"
            ):
                index = json.loads(src.read_text())
                index["weight_map"] = {
                    rename(k, mapping): v
                    for k, v in index["weight_map"].items()
                    if rename(k, mapping) is not None
                }
                dst.write_text(json.dumps(index, indent=2) + "\n")
            else:
                link_or_copy(src, dst)
        with np.load(learned / "students.npz", allow_pickle=False) as f:
            subset = {
                f"b{i}_{k}": f[f"b{i}_{k}"]
                for i in removed
                for k in ("down", "up", "gain", "bias")
            }
        np.savez(folder / "students.npz", **subset)
        actual = {}
        for shard in (folder / "transformer").glob("*.safetensors"):
            header, _ = read_header(shard)
            actual.update({k: shard.name for k in header if k != "__metadata__"})
        expected = json.loads(
            (folder / "transformer/model.safetensors.index.json").read_text()
        )["weight_map"]
        if actual != expected:
            raise ValueError("Written shards do not match the checkpoint index")
        for name in ("LICENSE", "NOTICE"):
            if (source.parent / name).is_file():
                shutil.copy2(source.parent / name, folder / name)
        with (folder / "NOTICE").open("a") as notice:
            notice.write(
                f"\nWeFit research modification. {n} single-stream blocks are replaced with trained time-specific residual students. See wefit-depth.json for the original indices and revision.\n"
            )
        inv = inventory(folder)
        base = inventory(source)
        params = sum(a.size for a in subset.values())
        report = dict(
            format=1,
            base_revision=MODEL_REVISION,
            profile=f"student{n}",
            removed=removed,
            retained=kept,
            rank=data["rank"],
            steps=3,
            student_parameters=params,
            dense_transformer_parameters=inv["transformer"]["logical_weights"],
            original_transformer_parameters=base["transformer"]["logical_weights"],
            vae_parameters=inv["vae"]["logical_weights"],
            student_sha256=hashlib.sha256(
                (folder / "students.npz").read_bytes()
            ).hexdigest(),
            student_bytes=(folder / "students.npz").stat().st_size,
            original_weight_bytes=size(source),
            compact_weight_bytes=size(folder)
            + (folder / "students.npz").stat().st_size,
            compact_shard_sha256=checkpoint_hashes(folder),
            source_shard_sha256=source_hashes,
        )
        (folder / "wefit-depth.json").write_text(json.dumps(report, indent=2) + "\n")
        folder.rename(output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("learned", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("blocks", type=int, choices=[2, 6, 12])
    a = p.parse_args()
    print(json.dumps(build(a.source, a.learned, a.output, a.blocks), indent=2))
