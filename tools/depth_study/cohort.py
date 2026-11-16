"""Re-use the V2 training photographs and reserve entirely new photographs for selection and testing."""

import argparse
import json
from pathlib import Path

from tools.adapter_study.prepare import prepare


def build(annotations, images, previous, destination):
    old = json.loads(previous.read_text())
    # The original deterministic sort and near-duplicate checks run over all 32 images.
    source = prepare(annotations, images, destination, (14, 10, 8))
    rows = source["records"]
    if [r["sha256"] for r in rows[:24]] != [r["sha256"] for r in old["records"]]:
        raise ValueError("The input collection does not recreate the V2 cohort.")
    keep = rows[:14] + rows[24:]
    for i, row in enumerate(keep):
        row["split"] = "train" if i < 14 else "val" if i < 17 else "test"
    source.update(
        records=keep,
        scope="14 original training photos, three fresh validation and five fresh test photos. The ten images previously used for V2 selection or testing are excluded.",
    )
    (destination / "manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("annotations", "images", "previous", "destination"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    result = build(a.annotations, a.images, a.previous, a.destination)
    print([(r["split"], r["id"], r["category"]) for r in result["records"]])
