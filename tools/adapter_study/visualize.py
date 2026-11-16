"""Build an offline model inspector directly from the experiment's CSV and JSON records."""

import argparse
import base64
import csv
import json
from pathlib import Path


def build(evidence, output, trace=None):
    with (evidence / "block-activity.csv").open() as f:
        activity = [
            (
                int(r["image_id"]),
                int(r["single_block"]),
                int(r["step"]),
                float(r["relative_residual_rms"]),
            )
            for r in csv.DictReader(f)
        ]
    with (evidence / "double-block-activity.csv").open() as f:
        joint = [
            (
                int(r["image_id"]),
                int(r["double_block"]),
                int(r["step"]),
                float(r["joint_rms"]),
                float(r["image_rms"]),
                float(r["text_rms"]),
            )
            for r in csv.DictReader(f)
        ]
    with (evidence / "head-channels.csv").open() as f:
        channels = [
            (
                int(r["channel"]),
                float(r["importance"]),
                float(r["garment_activation_rms"]),
            )
            for r in csv.DictReader(f)
        ]
    data = {
        "zeros": json.loads((evidence / "zero-weights.json").read_text()),
        "features": json.loads((evidence / "feature-maps.json").read_text()),
        "activity": activity,
        "joint": joint,
        "inventory": json.loads((evidence / "inventory.json").read_text()),
        "channels": channels,
        "selection": json.loads((evidence / "activity.json").read_text()),
        "training": json.loads((evidence / "training.json").read_text()),
        "ablation": json.loads((evidence / "ablation.json").read_text())
        if (evidence / "ablation.json").exists()
        else None,
        "evaluation": json.loads((evidence / "evaluation.json").read_text())
        if (evidence / "evaluation.json").exists()
        else None,
        "replay": json.loads((evidence / "replay.json").read_text())
        if (evidence / "replay.json").exists()
        else None,
        "mixed": json.loads((evidence / "mixed-checkpoint.json").read_text())
        if (evidence / "mixed-checkpoint.json").exists()
        else None,
        "compression": json.loads((evidence / "compression.json").read_text()),
    }
    maps = evidence / "output-maps.json"
    if maps.is_file():
        data["output_maps"] = json.loads(maps.read_text())
    if trace is not None:
        if trace.stat().st_size > 2_000_000:
            raise ValueError("Choose a local numeric trace smaller than 2 MB.")
        personal = json.loads(trace.read_text())
        if personal.get("format") != "wefit-trace-v1" or personal.get("steps") != 3:
            raise ValueError("Choose a three-step WeFit trace.")
        data["local_trace"] = {
            field: personal[field]
            for field in ("format", "steps", "grid", "single", "joint", "sample")
        }
    diagram = (
        Path(__file__).resolve().parents[2]
        / "docs/study-v2/figures/adapter-training.png"
    )
    data["training_diagram"] = "data:image/png;base64," + base64.b64encode(
        diagram.read_bytes()
    ).decode("ascii")
    template = Path(__file__).with_name("inspector.html").read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        template.replace(
            "__STUDY__",
            json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
                "<", "\\u003c"
            ),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--trace",
        type=Path,
        help="Embed a personal numeric trace in a local inspector copy.",
    )
    args = parser.parse_args()
    build(args.evidence, args.output, args.trace)
