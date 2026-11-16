"""Repeat discrepant full-size in-memory interventions with deterministic GrabCut."""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from gpu_service.klein_runtime import KleinRuntime, garment_rgb
from .policies import intervention
from .run import save


def audit(weights, study, case):
    rt = KleinRuntime(
        weights,
        Path("model_src/klein-kv/prompt-cache"),
        Path("model_src/fashn-vton/weights/human-parser"),
    )
    with Image.open(case["person"]) as img:
        source = ImageOps.exif_transpose(img).convert("RGB")
    with Image.open(case["garment"]) as img:
        product = garment_rgb(img)
    training = json.loads((study / "learned/training.json").read_text())
    rows = []
    with np.load(study / "learned/students.npz", allow_pickle=False) as fitted:
        for proof_file in sorted((study / "physical").glob("*/proof.json")):
            proof = json.loads(proof_file.read_text())
            native = next(row for row in proof["measurements"] if row["native"])
            if native.get("max_abs_channel_change", 0) == 0:
                continue
            method = proof["method"]
            with intervention(
                rt.model, "frozen" if method == "archive" else method, training, fitted
            ):
                actual, _ = rt.run(
                    source,
                    product,
                    "upper",
                    case["photo_type"],
                    "balanced",
                    "auto",
                    3,
                    "neutral",
                )
            with Image.open(proof_file.parent / (case["case"] + ".png")) as img:
                physical = np.array(img, dtype=np.int16)
            observed = np.asarray(actual, dtype=np.int16)
            delta = np.abs(physical - observed)
            row = dict(
                method=method,
                case=case["case"],
                channels=delta.size,
                historical_different_channels=native["rgb_channels"]
                - native["matching_rgb_channels"],
                historical_max=native["max_abs_channel_change"],
                repeated_matching_channels=int((delta == 0).sum()),
                repeated_max=int(delta.max()),
                repeated_rgb_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                physical_rgb_sha256=native["rgb_sha256"],
            )
            rows.append(row)
            print(row, flush=True)
            mx.clear_cache()
    save(study / "native-reload-audit.json", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("study", type=Path)
    p.add_argument("--case", type=Path, required=True)
    args = p.parse_args()
    audit(args.weights, args.study, json.loads(args.case.read_text()))
