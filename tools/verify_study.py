"""Check V2's published tables and claims using only the Python standard library."""

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

PUBLIC = Path(__file__).resolve().parents[1] / "docs/study-v2"
EVIDENCE = PUBLIC / "evidence"


def read_json(name):
    return json.loads((EVIDENCE / name).read_text())


def read_csv(name):
    with (EVIDENCE / name).open() as f:
        return list(csv.DictReader(f))


def main():
    hashes = read_json("SHA256.json")
    evidence_files = {
        f.name for f in EVIDENCE.iterdir() if f.is_file() and f.name != "README.md"
    }
    assert set(hashes) == evidence_files - {"SHA256.json"}
    for name, expected in hashes.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    runs = [
        json.loads(x) for x in (EVIDENCE / "requests.jsonl").read_text().splitlines()
    ]
    summary = read_json("summary.json")
    warm = [r for r in runs if r["phase"] == "warm"]
    assert len(warm) == summary["attempts"] == 20
    ok = [r for r in warm if r["status"] == 200]
    assert summary["completed"] == len(ok)
    for clock in ("wall", "service"):
        ordered = sorted(r[clock + "_s"] for r in ok)
        assert summary[clock]["median_s"] == round(median(ordered), 3)
        assert summary[clock]["p95_s"] == ordered[math.ceil(0.95 * len(ok)) - 1]
    eligible = [r for r in warm if r.get("initial_region") == "clothing"]
    assert (summary["fallbacks"], summary["crop_attempts"]) == (
        sum(r["fallback"] for r in eligible),
        len(eligible),
    )
    train = read_json("training.json")
    archive = read_json("archive.json")
    assert archive["all_files_hash_match"]
    assert (
        sum(file["bytes"] for file in archive["files"].values())
        == archive["source_bytes"]
    )
    assert archive["source_bytes"] - archive["archive_bytes"] == archive["saved_bytes"]
    for shard, digest in train["base_checkpoint_sha256"].items():
        assert archive["files"][shard]["sha256"] == digest
    learned = PUBLIC / "adapters/upper-final-r4.npz"
    assert hashlib.sha256(learned.read_bytes()).hexdigest() == train["adapter_sha256"]
    assert (
        train["trainable_parameters"] == 4096 * 4 + 4 * 128
        and train["backbone_trainable_parameters"] == 0
    )
    compressed = read_json("compression.json")
    assert (PUBLIC / "adapters/upper-compact.npz").stat().st_size == compressed[
        "compact_bytes"
    ]
    assert compressed["compact_bytes"] < train["adapter_bytes"]
    history = read_csv("training.csv")
    assert len(history) == train["epochs"] + 1
    assert train["optimizer_updates"] == train["epochs"] * train["training_images"]
    selected = min(history[1:], key=lambda row: float(row["val_garment_mse"]))
    assert int(selected["epoch"]) == train["selected_epoch"]
    manifest = read_json("manifest.json")["records"]
    extracted = read_csv("feature-extraction.csv")
    assert len(extracted) == len(manifest) and {
        (r["split"], int(r["photo_id"])) for r in extracted
    } == {(r["split"], r["id"]) for r in manifest}
    assert all(float(r["per_photo_wall_s"]) > 0 for r in extracted)
    assert len({r["sha256"] for r in manifest}) == len(manifest) == 24
    assert [
        sum(r["split"] == s for r in manifest) for s in ["train", "val", "test"]
    ] == [14, 5, 5]
    ev = read_json("evaluation.json")
    hold = read_csv("held-out.csv")
    assert {int(r["id"]) for r in hold} == {
        r["id"] for r in manifest if r["split"] == "test"
    }
    assert (
        ev["n"] == len(hold) == 5
        and ev["unchanged_base_checkpoint"]
        and ev["zero_adapter_pixel_identical"]
    )
    for k, value in ev["per_image_means"].items():
        assert math.isclose(
            mean(float(row[k]) for row in hold), value, rel_tol=1e-9, abs_tol=1e-9
        )
    ab = read_json("ablation.json")
    timing = read_csv("ablation.csv")
    assert len(timing) == 25
    for m, stats in ab["methods"].items():
        rows = [r for r in timing if r["method"] == m]
        assert (
            len(rows) == stats["images"] == 5
            and median(float(r["wall_s"]) for r in rows) == stats["median_s"]
        )
    maps = read_json("output-maps.json")
    assert maps["patch_pixels"] == 16 and maps["grid"] == [16, 20]
    for row in timing:
        photo = maps["photos"][row["id"]]
        pixels = photo["garment_pixels"]
        errors = photo["rgb_error_sums"][row["method"]]
        assert len(pixels) == len(errors) == math.prod(maps["grid"])
        assert all(0 <= count <= maps["patch_pixels"] ** 2 for count in pixels)
        assert all(
            0 <= error <= 3 * 255 * count for error, count in zip(errors, pixels)
        )
        assert math.isclose(
            sum(errors) / (3 * sum(pixels)), float(row["garment_rgb_mae"]), abs_tol=1e-4
        )
    assert len(read_csv("block-activity.csv")) == 3 * 24 * 3
    assert len(read_csv("double-block-activity.csv")) == 3 * 8 * 3
    assert len(read_csv("head-channels.csv")) == 4096
    zero = read_json("zero-weights.json")
    assert zero["decoded_values_exact"] and zero["nonzero_codes_exact"]
    assert zero["deflate_byte_exact"]
    assert 0 < zero["deflate_dense_bytes"] < zero["original_q4_bytes"]
    assert 0 < zero["deflate_bitmap_bytes"] < zero["bitmap_representation_bytes"]
    assert (
        zero["bitmap_representation_bytes"]
        == zero["zero_bitmap_bytes"]
        + zero["nonzero_nibbles_bytes"]
        + zero["scale_and_offset_bytes"]
    )
    mixed = read_json("mixed-checkpoint.json")
    assert (
        mixed["storage_saving_bytes"]
        == mixed["stored_bytes_before"] - mixed["stored_bytes_after"]
        > 0
    )
    assert (
        mixed["loaded_precision_verified"]
        and len(mixed["blocks"]) == 6
        and mixed["projections_requantized"] == 12
    )
    mixed_rows = read_csv("mixed-runs.csv")
    for method, values in mixed["methods"].items():
        rows = [r for r in mixed_rows if r["method"] == method]
        assert (
            len(rows) == 5
            and median(float(r["wall_s"]) for r in rows) == values["median_s"]
        )
    native = read_json("native-pairs.json")
    replay = read_json("replay.json")
    replays = read_csv("replay-runs.csv")
    assert len(replays) == 45
    controls = [r for r in replays if r["split"] == "test" and r["pixel_identical"]]
    assert len(controls) == replay["compared_heldout_controls"] == 15
    assert (
        sum(int(r["max_rgb_difference"]) == 0 for r in controls)
        == replay["exact_heldout_controls"]
    )
    validation = [r for r in replays if r["split"] == "val"]
    strengths = sorted({float(r["strength"]) for r in validation})
    best = min(
        strengths,
        key=lambda s: mean(
            float(r["garment_error"]) for r in validation if float(r["strength"]) == s
        ),
    )
    assert best == replay["selected_strength"]
    matched_test = [r for r in controls if r["method"] == "compact"]
    assert (
        median(float(r["wall_s"]) for r in matched_test)
        == replay["matched_compact_replay_median_s"]
    )
    assert len(native) == 8 and len({r["case"] for r in native}) == 4
    browser = read_json("studio-trace-check.json")
    assert (
        browser["same_numeric_trace_as_independent_cli"]
        and browser["browser_errors"] == 0
    )
    assert browser["blocks"] == 32 and browser["steps"] == 3
    blind = read_json("blind-status.json")
    if blind["completed_reviewer_packets"] == 0:
        assert (
            blind["rated_pairs"] == 0
            and blind["preference_agreement"]["fraction"] is None
        )
        assert all(
            row["n"] == 0 and row["mean"] is None
            for fields in blind["ratings"].values()
            for row in fields.values()
        )
    print(
        f"{len(warm)} warm requests, {len(hold)} held-out photos, {len(timing)} ablation renders and {len(native)} native runs verified."
    )


if __name__ == "__main__":
    main()
