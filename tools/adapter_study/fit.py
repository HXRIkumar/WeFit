"""Fit only two rank-four matrices to cached final-step features. Runs on CPU."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from gpu_service.klein_prompts import MODEL_REVISION

from .compress import compress
from .feature_maps import feature_maps


class GarmentHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down = torch.nn.Parameter(torch.randn(4096, 4) / 64)
        self.up = torch.nn.Parameter(torch.zeros(4, 128))

    def forward(self, x):
        return (x @ self.down) @ self.up


def load_records(pairs, rows, split):
    records = []
    for row in rows:
        if row["split"] != split:
            continue
        with np.load(pairs / str(row["id"]) / "features.npz", allow_pickle=False) as z:
            item = {key: torch.from_numpy(z[key].copy()).float() for key in z.files}
        records.append(item)
    return records


def score(head, item):
    # Last Euler step converts a change in the predicted velocity into a latent correction.
    delta = head(item["hidden"]) * item["delta_sigma"]
    fraction = item["fraction"]
    mse = (item["base"] + delta - item["target"]).square().mean(dim=1)
    garment = (mse * fraction).sum() / fraction.sum().clamp_min(1)
    outside = (delta.square().mean(dim=1) * (1 - fraction)).sum() / (
        1 - fraction
    ).sum().clamp_min(1)
    return garment, outside


@torch.no_grad()
def evaluate(head, records):
    scores = [float(score(head, item)[0]) for item in records]
    return sum(scores) / len(scores)


def fit(pairs, out, train_only=False):
    torch.set_num_threads(2)
    torch.manual_seed(20261116)
    manifest = json.loads((pairs / "manifest.json").read_text())
    info = json.loads((pairs / "cache-info.json").read_text())
    if (
        info["revision"] != MODEL_REVISION
        or info["manifest_sha256"]
        != hashlib.sha256((pairs / "manifest.json").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Model revision or data manifest does not match cached activations."
        )
    rows = manifest["records"]
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Repeated source photographs in split.")
    train = load_records(pairs, rows, "train")
    head = GarmentHead()
    out.mkdir(parents=True, exist_ok=True)
    # Checkpoint the small optimizer trajectory independently of validation or test labels.
    digest = hashlib.sha256((pairs / "cache-info.json").read_bytes())
    for row in rows:
        if row["split"] == "train":
            with (pairs / str(row["id"]) / "features.npz").open("rb") as data:
                digest.update(hashlib.file_digest(data, "sha256").digest())
    fingerprint = digest.hexdigest()
    trajectory = pairs / "rank4-adamw-v1.npz"
    if not trajectory.exists():
        start = time.perf_counter()
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
        down, up, losses = [], [], []
        for epoch in range(31):
            if epoch:
                for index in torch.randperm(len(train)):
                    garment, outside = score(head, train[int(index)])
                    loss = garment + 0.25 * outside
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
            down.append(head.down.detach().numpy().copy())
            up.append(head.up.detach().numpy().copy())
            losses.append(evaluate(head, train))
        duration = time.perf_counter() - start
        tmp = trajectory.with_name("rank4-adamw-v1.partial.npz")
        np.savez(
            tmp,
            down=np.stack(down),
            up=np.stack(up),
            losses=losses,
            fingerprint=fingerprint,
            duration=duration,
            torch_version=str(torch.__version__),
        )
        tmp.replace(trajectory)
    with np.load(trajectory, allow_pickle=False) as stored:
        if str(stored["fingerprint"]) != fingerprint or str(
            stored["torch_version"]
        ) != str(torch.__version__):
            raise ValueError(
                "The saved trajectory belongs to another training dataset or PyTorch version."
            )
        downs, ups, losses = stored["down"], stored["up"], stored["losses"]
        duration = float(stored["duration"])
    if train_only:
        print(
            f"Completed {30 * len(train)} AdamW updates on {len(train)} photos in {duration:.2f}s. Training MSE {losses[0]:.5f} to {losses[-1]:.5f}."
        )
        return
    val = load_records(pairs, rows, "val")
    history = []
    for epoch, (down, up, train_loss) in enumerate(
        zip(downs, ups, losses, strict=True)
    ):
        head.load_state_dict(
            {"down": torch.from_numpy(down), "up": torch.from_numpy(up)}
        )
        history.append(
            {
                "epoch": epoch,
                "train_garment_mse": float(train_loss),
                "val_garment_mse": evaluate(head, val),
            }
        )
    best = min(history[1:], key=lambda row: row["val_garment_mse"])
    best_epoch, best_loss = best["epoch"], best["val_garment_mse"]
    best_state = {"down": downs[best_epoch], "up": ups[best_epoch]}
    output = out / "upper-final-r4.npz"
    np.savez(
        output,
        **best_state,
        revision=MODEL_REVISION,
        target="upper.final_step.proj_out",
    )
    with (out / "training.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    settings = {
        "seed": 20261116,
        "rank": 4,
        "steps": 3,
        "epochs": 30,
        "optimizer_updates": 30 * len(train),
        "optimizer_and_train_scoring_seconds": duration,
        "selected_epoch": best_epoch,
        "selection": "Lowest mean validation garment latent MSE among trained epochs. Epoch zero is the baseline.",
        "optimizer": "CPU AdamW, lr 0.001, weight decay 0.01, gradient norm limit 1.0; one photograph per update",
        "objective": "Mean squared error against target VAE tokens in the annotated garment; add 0.25 times squared change outside.",
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "backbone_trainable_parameters": 0,
        "training_images": len(train),
        "validation_images": len(val),
        "baseline_train_mse": history[0]["train_garment_mse"],
        "selected_train_mse": history[best_epoch]["train_garment_mse"],
        "baseline_val_mse": history[0]["val_garment_mse"],
        "selected_val_mse": best_loss,
        "adapter_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "adapter_bytes": output.stat().st_size,
        "base_checkpoint_sha256": info["sha256"],
        "revision": MODEL_REVISION,
        "torch": torch.__version__,
    }
    (out / "training.json").write_text(json.dumps(settings, indent=2) + "\n")
    compress(pairs, out)
    if (out / "activity.json").exists():
        feature_maps(pairs, out)
    print(
        json.dumps(
            {
                k: settings[k]
                for k in [
                    "trainable_parameters",
                    "selected_epoch",
                    "baseline_val_mse",
                    "selected_val_mse",
                ]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Fit and cache all candidate epochs using training photos only.",
    )
    args = parser.parse_args()
    fit(args.pairs, args.output, args.train_only)
