"""Build the offline inspector from numeric evidence and the redacted publication figure."""

import argparse
import base64
import json
from pathlib import Path


def build(study, destination):
    ev = study / "evidence"

    def get(name):
        return json.loads((ev / (name + ".json")).read_text())

    train, protocol, selection = get("training"), get("protocol"), get("selection")
    checkpoints = get("checkpoints")
    reuse = get("reference-reuse")
    physical = get("physical")
    q = get("quantization")
    val, test = get("validation"), get("test")
    val += [r for r in q["measurements"] if r["split"] == "val"]
    test += [r for r in q["measurements"] if r["split"] == "test"]
    # Keep the lab and report on the same fresh-process wall clock whenever
    # a physical model or a deployed schedule was independently reloaded.
    fresh_seconds = {
        (name, row["id"]): row["wall_seconds"]
        for name, proof in physical.items()
        for row in proof["measurements"]
        if not row["native"]
    }
    for row in test:
        row["wall_seconds"] = fresh_seconds.get(
            (row["method"], row["id"]), row["wall_seconds"]
        )
    frozen = {r["id"]: r for r in test if r["method"] == "frozen"}
    for row in physical["archive"]["measurements"]:
        if not row["native"] and row["max_abs_channel_change"] == 0:
            test.append(
                {
                    **frozen[row["id"]],
                    "method": "archive",
                    "wall_seconds": row["wall_seconds"],
                }
            )
    base = checkpoints["original_weight_bytes"]
    sizes = {
        k: base for k in protocol["methods"] if not k.startswith(("student", "drop"))
    }
    sizes.update(
        {k: v["compact_weight_bytes"] for k, v in checkpoints["students"].items()}
    )
    sizes.update(
        {
            "q3-all-single": checkpoints["mixed"]["stored_bytes_after"],
            "archive": checkpoints["archive_bytes"],
        }
    )
    data = dict(
        validation=val,
        test=test,
        selection=selection,
        single=train["single_mean"],
        dual=train["dual_joint_mean"],
        schedule=protocol["schedule"],
        sizes=sizes,
        independently_timed=list(physical) + ["q3-all-single"],
        reuse=reuse,
        variance=get("unused-variance"),
        cameraCaption="Each row shows the source, frozen Q4, the selected last-step skip, the selected trained student and the 24-block three-bit checkpoint. Measurements use the original outputs before publication redaction.",
    )
    template = Path(__file__).with_name("lab.html").read_text()
    encoded = base64.b64encode(
        (study / "figures/camera-comparison.jpg").read_bytes()
    ).decode()
    stress = base64.b64encode(
        (study / "figures/blue-shirt-stress-test.jpg").read_bytes()
    ).decode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        template.replace(
            "__EVIDENCE__",
            json.dumps(data, separators=(",", ":")).replace("<", "\\u003c"),
        )
        .replace("__CAMERA__", "data:image/jpeg;base64," + encoded)
        .replace("__STRESS__", "data:image/jpeg;base64," + stress)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    build(a.study, a.destination)
