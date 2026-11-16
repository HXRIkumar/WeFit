"""Fit tiny time-specific residual students from a frozen Klein teacher's activations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve

from gpu_service.klein_prompts import MODEL_REVISION


def fit_residual(x, y, rank):
    """Fit a per-channel correction plus a reduced-rank ridge map in standardized units."""
    xm, ym = x.mean(0), y.mean(0)
    xs, ys = np.maximum(x.std(0), 1.0), np.maximum(y.std(0), 1.0)
    xn, yn = (x - xm) / xs, (y - ym) / ys
    diag = (xn * yn).sum(0) / ((xn * xn).sum(0) + 0.05 * len(x))
    residual = yn - xn * diag
    gram = residual @ residual.T
    values, vectors = eigh(gram, subset_by_index=[len(x) - rank, len(x) - 1])
    basis = (residual.T @ vectors) / np.sqrt(np.maximum(values, 1e-8))
    gram = xn @ xn.T
    ridge = 0.05 * np.trace(gram) / len(x)
    gram.flat[:: len(gram) + 1] += max(ridge, 1e-4)
    down = (xn.T @ solve(gram, residual @ basis, assume_a="pos")) / xs[:, None]
    up = basis.T * ys
    gain = diag * ys / xs
    bias = ym - xm * gain - (xm @ down) @ up
    fitted = x * gain + bias + (x @ down) @ up
    weights = dict(down=down, up=up, gain=gain, bias=bias)
    if not all(np.isfinite(v).all() for v in weights.values()):
        raise ValueError("Nonfinite student fit")
    return weights, dict(
        samples=len(x),
        ridge=float(ridge),
        teacher_relative_squared_error_before=1.0,
        teacher_relative_squared_error_after=float(
            np.sum((fitted - y) ** 2, dtype=np.float64) / np.sum(y**2, dtype=np.float64)
        ),
        standardized_mse_before=float(np.mean((y / ys) ** 2)),
        standardized_mse_after=float(np.mean(((fitted - y) / ys) ** 2)),
    )


def fit(pairs, output, rank=8):
    start = time.perf_counter()
    training = [
        r
        for r in json.loads((pairs / "manifest.json").read_text())["records"]
        if r["split"] == "train"
    ]
    paths = [pairs / str(r["id"]) / "teacher-samples.npz" for r in training]
    metadata = [
        json.loads((p.parent / "teacher-blocks.json").read_text()) for p in paths
    ]
    blocks = metadata[0]["captured_blocks"]
    singles = np.asarray([m["single"] for m in metadata])
    doubles = np.asarray([m["dual"] for m in metadata])
    if singles.shape != (len(paths), 24, 3) or doubles.shape != (len(paths), 8, 3, 3):
        raise ValueError("Incomplete calibration")
    mean = singles.mean(0)
    order = sorted(blocks, key=lambda b: mean[b].mean())
    output.mkdir(parents=True, exist_ok=True)
    weights = {}
    losses = []
    for block in blocks:
        arrays = []
        for path in paths:
            with np.load(path, allow_pickle=False) as source:
                arrays.append(source[f"block_{block}"])
        by_step = []
        for step in range(3):
            xy = np.concatenate([a[step] for a in arrays], axis=1)
            params, loss = fit_residual(xy[0], xy[1], rank)
            by_step.append(params)
            losses.append(dict(block=block, step=step + 1, **loss))
        weights.update(
            {f"b{block}_{k}": np.stack([p[k] for p in by_step]) for k in by_step[0]}
        )
        print(f"Trained block {block}", flush=True)
    np.savez(output / "students.npz", **weights)
    study = dict(
        base_revision=MODEL_REVISION,
        rank=rank,
        training_images=[r["id"] for r in training],
        sampled_blocks=blocks,
        structural_order=order,
        step_orders=np.argsort(mean, axis=0).T.tolist(),
        single_mean=mean.tolist(),
        dual_joint_mean=doubles[:, :, :, 2].mean(0).tolist(),
        calibration_block_visits=int(len(paths) * 32 * 3),
        tokens_per_block_step=64 * len(paths),
        learned_parameters=sum(a.size for a in weights.values()),
        precision="float32",
        losses=losses,
        train_seconds=round(time.perf_counter() - start, 3),
    )
    (output / "training.json").write_text(json.dumps(study, indent=2) + "\n")
    return study


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--rank", type=int, default=8, choices=range(1, 33))
    a = p.parse_args()
    fit(a.pairs, a.output, a.rank)
