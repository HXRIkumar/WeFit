"""Measure residual block use during calibration and output-layer use over labelled garment tokens."""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn


class ResidualProbe(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.step_rms = []

    def __call__(self, hidden_states, **kwargs):
        result = self.block(hidden_states=hidden_states, **kwargs)
        before = hidden_states.astype(mx.float32)
        change = result.astype(mx.float32) - before
        relative = mx.sqrt(
            mx.mean(mx.square(change)) / mx.maximum(mx.mean(mx.square(before)), 1e-12)
        )
        self.step_rms.append(float(relative.item()))
        return result


class JointResidualProbe(nn.Module):
    """Measure the image and text streams separately before they are joined."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.step_rms = []

    def __call__(self, hidden_states, encoder_hidden_states, **kwargs):
        text, image = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        stats = []
        for before, after in [(hidden_states, image), (encoder_hidden_states, text)]:
            before = before.astype(mx.float32)
            difference = after.astype(mx.float32) - before
            stats.append((mx.sum(mx.square(difference)), mx.sum(mx.square(before))))
        ratios = [mx.sqrt(num / mx.maximum(den, 1e-12)) for num, den in stats]
        joint = mx.sqrt(
            (stats[0][0] + stats[1][0]) / mx.maximum(stats[0][1] + stats[1][1], 1e-12)
        )
        self.step_rms.append(np.array(mx.stack([*ratios, joint])).tolist())
        return text, image


@contextmanager
def block_activity(model, active):
    singles = model.transformer.single_transformer_blocks
    doubles = model.transformer.transformer_blocks
    single_probes = [ResidualProbe(block) for block in singles] if active else []
    joint_probes = [JointResidualProbe(block) for block in doubles] if active else []
    if active:
        model.transformer.single_transformer_blocks = single_probes
        model.transformer.transformer_blocks = joint_probes
    try:
        yield single_probes, joint_probes
    finally:
        if active:
            model.transformer.single_transformer_blocks = singles
            model.transformer.transformer_blocks = doubles


def save_head_weights(model, directory):
    projection = model.transformer.proj_out
    weight = mx.dequantize(
        projection.weight,
        projection.scales,
        projection.biases,
        group_size=projection.group_size,
        bits=projection.bits,
    )
    matrix = np.asarray(weight.astype(mx.float32))
    if matrix.shape != (128, 4096):
        raise ValueError("Unexpected output projection shape.")
    np.save(directory / "base-output-head.npy", matrix)


def summarize(pairs, out):
    records = json.loads((pairs / "manifest.json").read_text())["records"]
    train = [r for r in records if r["split"] == "train"]
    activity = []
    for record in train[:3]:
        data = json.loads(
            (pairs / str(record["id"]) / "block-activity.json").read_text()
        )
        activity.append(np.asarray(data))
    values = np.asarray(activity)
    if values.shape != (3, 24, 3):
        raise ValueError(
            f"Expected 3 calibration photographs, 24 blocks and 3 steps; got {values.shape}"
        )
    mean = values.mean(axis=(0, 2))
    rank = np.argsort(mean).tolist()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "block-activity.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "single_block", "step", "relative_residual_rms"])
        for ri, record in enumerate(train[:3]):
            for block in range(24):
                for step in range(3):
                    writer.writerow(
                        [record["id"], block, step + 1, values[ri, block, step]]
                    )
    with (out / "double-block-activity.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "double_block", "step", "image_rms", "text_rms", "joint_rms"]
        )
        for record in train[:3]:
            double = np.array(
                json.loads(
                    (pairs / str(record["id"]) / "joint-activity.json").read_text()
                )
            )
            if double.shape != (8, 3, 3):
                raise ValueError(f"Unexpected dual-stream probe shape {double.shape}")
            for block in range(8):
                for step in range(3):
                    writer.writerow(
                        [record["id"], block, step + 1, *double[block, step]]
                    )
    total = 0.0
    squares = np.zeros(4096)
    maximum = np.zeros(4096)
    for record in train:
        with np.load(
            pairs / str(record["id"]) / "features.npz", allow_pickle=False
        ) as z:
            x = z["hidden"].astype(np.float32)
            w = z["fraction"]
            squares += np.sum(x * x * w[:, None], axis=0, dtype=np.float64)
            maximum = np.maximum(maximum, np.max(np.abs(x[w > 0]), axis=0))
            total += float(w.sum())
    rms = np.sqrt(squares / total)
    weight = np.load(pairs / "base-output-head.npy", allow_pickle=False)
    influence = np.linalg.norm(weight, axis=0) * rms
    sorted_ch = np.argsort(influence)
    with (out / "head-channels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "channel",
                "garment_activation_rms",
                "max_abs",
                "weight_column_l2",
                "importance",
            ]
        )
        for channel in sorted_ch:
            writer.writerow(
                [
                    channel,
                    rms[channel],
                    maximum[channel],
                    np.linalg.norm(weight[:, channel]),
                    influence[channel],
                ]
            )
    result = {
        "calibration_images": [r["id"] for r in train[:3]],
        "channel_calibration_images": len(train),
        "double_blocks": 8,
        "single_blocks": len(rank),
        "residual_rms_order_low_to_high": rank,
        "residual_rms_by_block": mean.tolist(),
        "head_input_channels": 4096,
        "head_output_channels": 128,
        "head_connections": int(weight.size),
        "exactly_zero_dequantized_head_weights": int(np.count_nonzero(weight == 0)),
        "zero_products_on_all_observed_garment_tokens": int(
            np.count_nonzero((weight == 0) | (maximum == 0)[None, :])
        ),
        "individual_connection_score_quantiles": dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                np.quantile(
                    np.abs(weight) * rms[None, :], [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
                strict=True,
            )
        ),
        "channels_zero_on_every_observed_garment_token": int((maximum == 0).sum()),
        "channel_score": "RMS of training garment activations times L2 norm of its dequantized Q4 weight column",
        "blocks_score": "Mean over three steps and three training photos of RMS(output-input)/RMS(input); all tokens",
    }
    (out / "activity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pairs", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    summarize(a.pairs, a.output)
