"""Choose a smaller LoRA file using the validation images only."""

import json

import numpy as np

from gpu_service.klein_prompts import MODEL_REVISION


def truncate(down, up, rank):
    if rank == down.shape[1]:
        return down, up
    left, ra = np.linalg.qr(down)
    right, rb = np.linalg.qr(up.T)
    u, sigma, vt = np.linalg.svd(ra @ rb.T, full_matrices=False)
    root = np.sqrt(sigma[:rank])
    return (left @ u[:, :rank]) * root, root[:, None] * (vt[:rank] @ right.T)


def losses(records, down, up):
    result = []
    for row in records:
        corrected = row["base"] + row["delta_sigma"] * (
            (row["hidden"].astype(np.float32) @ down) @ up
        )
        error = np.square(corrected - row["target"]).mean(axis=1)
        result.append(float(np.average(error, weights=row["fraction"])))
    return np.array(result)


def compress(pairs, output):
    rows = json.loads((pairs / "manifest.json").read_text())["records"]
    val = []
    for row in rows:
        if row["split"] == "val":
            with np.load(
                pairs / str(row["id"]) / "features.npz", allow_pickle=False
            ) as data:
                val.append({key: data[key] for key in data.files})
    original = output / "upper-final-r4.npz"
    with np.load(original, allow_pickle=False) as data:
        down, up = data["down"], data["up"]
    reference = losses(val, down, up)
    candidates = []
    best = None
    for rank in range(1, 5):
        a, b = truncate(down, up, rank)
        a, b = a.astype(np.float16), b.astype(np.float16)
        error = losses(val, a.astype(np.float32), b.astype(np.float32))
        relative = error / reference - 1
        permitted = bool(
            error.mean() <= reference.mean() * 1.001 and relative.max() <= 0.005
        )
        candidates.append(
            {
                "rank": rank,
                "float": "fp16",
                "mean_mse": float(error.mean()),
                "worst_relative_change": float(relative.max()),
                "eligible": permitted,
            }
        )
        if best is None and permitted:
            best = (rank, a, b)
    if best is None:
        raise ValueError("Half precision exceeded the declared validation tolerance.")
    rank, a, b = best
    target = output / "upper-compact.npz"
    np.savez(
        target,
        down=a,
        up=b,
        revision=MODEL_REVISION,
        target="upper.final_step.proj_out",
    )
    result = {
        "learned_rank": 4,
        "learned_bytes": original.stat().st_size,
        "learned_fp32_val_mse": float(reference.mean()),
        "compact_rank": rank,
        "compact_bytes": target.stat().st_size,
        "validation_images": len(val),
        "selection": "Smallest FP16 rank within +0.1% mean and +0.5% worst image latent MSE of the full FP32 adapter on validation.",
        "candidates": candidates,
    }
    (output / "compression.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
