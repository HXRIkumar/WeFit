"""Verify the distributed figures and recompute the reported median reduction using only the standard library."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1] / "docs/portfolio"


def main():
    study = json.loads((ROOT / "evidence/measurements.json").read_text())
    for name, expected in study["figures"].items():
        if hashlib.sha256((ROOT / "figures" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Figure checksum mismatch: {name}")
    with (ROOT / "evidence/region-runs.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    pairs = {key: [r for r in rows if r["pair"] == key] for key in {r["pair"] for r in rows}}
    if any(sorted(r["actual_region"] for r in p) != ["clothing", "full"] for p in pairs.values()):
        raise ValueError("Every pair must have one full and one clothing-region run.")
    for key in ("revision", "person_sha256", "garment_sha256", "steps"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"Matched runs disagree on {key}.")
    claim = study["region_experiment"]
    if len(pairs) != claim["pairs"]:
        raise ValueError("Pair count mismatch.")
    med = {region: median(float(r["service_s"]) for r in rows if r["actual_region"] == region)
           for region in ("full", "clothing")}
    reduction = round(100 * (1 - med["clothing"] / med["full"]), 1)
    if (med["full"], med["clothing"], reduction) != (
        claim["full_median_s"], claim["clothing_median_s"], claim["relative_difference_of_medians_pct"]
    ):
        raise ValueError("The published summary differs from the raw records.")
    print(f"{len(rows)} requests, {len(pairs)} pairs, {reduction}% median service-time reduction. All figure hashes match.")


if __name__ == "__main__":
    main()
