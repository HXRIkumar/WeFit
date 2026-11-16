"""Verify splits, decisions, metrics, parameter arithmetic and independent model reloads using only the public record."""

import csv
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

from tools.depth_study.protocol import METHODS, TEMPORAL, STRUCTURAL, choose

ROOT = Path(__file__).resolve().parents[1] / "docs/study-v3"


def verify(root=ROOT):
    ev = root / "evidence"

    def get(key):
        return json.loads((ev / (key + ".json")).read_text())

    def close(a, b):
        assert math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-9), (a, b)

    checks = get("SHA256")
    actual = {
        p.name
        for p in ev.iterdir()
        if p.is_file() and p.name not in {"SHA256.json", "README.md"}
    }
    assert set(checks) == actual
    for name, h in checks.items():
        assert hashlib.sha256((ev / name).read_bytes()).hexdigest() == h, name
    cohort = get("manifest")["records"]
    assert len({r["id"] for r in cohort}) == len({r["sha256"] for r in cohort}) == 22
    splits = {
        s: {r["id"] for r in cohort if r["split"] == s}
        for s in ("train", "val", "test")
    }
    assert [len(splits[s]) for s in splits] == [14, 3, 5]
    v2 = json.loads((root.parent / "study-v2/evidence/manifest.json").read_text())[
        "records"
    ]
    assert splits["train"] == {r["id"] for r in v2 if r["split"] == "train"}
    assert (splits["val"] | splits["test"]).isdisjoint({r["id"] for r in v2})
    protocol, train = get("protocol"), get("training")
    traces = list(csv.DictReader((ev / "block-trace.csv").open()))
    assert len(traces) == train["calibration_block_visits"] == 1344
    for family, n, expected in (
        ("single", 24, train["single_mean"]),
        ("dual", 8, train["dual_joint_mean"]),
    ):
        for b in range(n):
            for t in range(3):
                subset = [
                    r
                    for r in traces
                    if r["family"] == family
                    and int(r["block"]) == b
                    and int(r["step"]) == t + 1
                ]
                assert {int(r["training_image"]) for r in subset} == splits["train"]
                close(
                    statistics.mean(float(r["relative_rms"]) for r in subset),
                    expected[b][t],
                )
    assert train["rank"] == 8 and train["learned_parameters"] == 12 * 3 * (
        4096 * 8 * 2 + 4096 * 2
    )
    assert (
        set(protocol["validation_ids"]) == splits["val"]
        and set(protocol["test_ids"]) == splits["test"]
    )
    summary = get("summary")
    for name, split in (("validation", "val"), ("test", "test")):
        rows = get(name)
        assert len({(r["method"], r["id"]) for r in rows}) == len(rows)
        for method, m in summary[name].items():
            sample = [r for r in rows if r["method"] == method]
            assert {r["id"] for r in sample} == splits[split]
            close(
                statistics.median(r["wall_seconds"] for r in sample),
                m["median_seconds"],
            )
            for field in (
                "garment_mae",
                "outside_mae",
                "garment_within8",
                "garment_exact",
            ):
                close(statistics.mean(r[field] for r in sample), m[field])
            close(statistics.mean(r["garment_ssim"] for r in sample), m["mean_ssim"])
    selected = get("selection")
    assert tuple(METHODS) == tuple(protocol["methods"])
    for key, methods, structural in (
        ("temporal", TEMPORAL, False),
        ("structural", STRUCTURAL, True),
    ):
        assert choose(summary["validation"], methods, structural) == (
            selected[key],
            selected[key + "_tier"],
        )
    assert (
        get("selected-policy")["blocks_by_step"]
        == protocol["schedule"][selected["temporal"]]
    )
    stores = get("checkpoints")
    deployment = get("deployment-files")
    extras = sum(
        entry["bytes"]
        for group in deployment["parts"].values()
        for entry in group.values()
    )
    assert (
        extras == deployment["additional_bytes"]
        and extras + stores["archive_bytes"] == deployment["archive_plus_assets_bytes"]
    )
    for name, ck in stores["students"].items():
        n = len(ck["removed"])
        assert ck["retained"] == [i for i in range(24) if i not in ck["removed"]]
        assert (
            ck["original_transformer_parameters"] - ck["dense_transformer_parameters"]
            == n * 218104064
        )
        assert ck["student_parameters"] == n * 3 * (4096 * 8 * 2 + 4096 * 2)
        assert 0 < ck["compact_weight_bytes"] < stores["original_weight_bytes"]
    physical = get("physical")
    assert {
        "archive",
        "frozen",
        selected["structural"],
        selected["temporal"],
        "student12",
    } == set(physical)
    rechecks = {row["method"]: row for row in get("native-reload-audit")}
    discrepant_native = set()
    for name, p in physical.items():
        small = [r for r in p["measurements"] if not r["native"]]
        assert {r["id"] for r in small} == splits["test"]
        assert (
            len(p["measurements"]) == 6
            and sum(r["native"] for r in p["measurements"]) == 1
        )
        for r in p["measurements"]:
            assert r.get("status") != "rejected"
            if (
                r["native"]
                and r["max_abs_channel_change"]
                and name not in {"archive", "frozen"}
            ):
                discrepant_native.add(name)
                repeated = rechecks[name]
                assert (
                    repeated["case"] == r["id"]
                    and repeated["channels"] == r["rgb_channels"]
                )
                assert (
                    repeated["historical_different_channels"]
                    == r["rgb_channels"] - r["matching_rgb_channels"]
                )
                assert repeated["historical_max"] == r["max_abs_channel_change"]
                assert (
                    repeated["repeated_rgb_sha256"]
                    == repeated["physical_rgb_sha256"]
                    == r["rgb_sha256"]
                )
                assert (
                    repeated["repeated_max"] == 0
                    and repeated["repeated_matching_channels"] == r["rgb_channels"]
                )
                continue
            assert r["rgb_sha256"] == r["comparison_rgb_sha256"], (name, r["id"])
            assert (
                r["matching_rgb_channels"] == r["rgb_channels"]
                and r["max_abs_channel_change"] == 0
            )
    assert set(rechecks) == discrepant_native
    diagnostic = get("matte-diagnostic")
    assert (
        diagnostic["fresh_vs_next"]["different"] > 0
        and diagnostic["zero_vs_other"]["different"] > 0
    )
    assert (
        diagnostic["reset_zero_twice"]["different"]
        == diagnostic["fresh_vs_zero"]["different"]
        == 0
    )
    quant = get("quantization")
    assert (
        len(stores["mixed"]["blocks"]) == 24
        and stores["mixed"]["projections_requantized"] == 48
    )
    for s in ("val", "test"):
        by_split = [r for r in quant["measurements"] if r["split"] == s]
        assert {r["id"] for r in by_split} == splits[s]
        close(
            statistics.mean(r["garment_ssim"] for r in by_split),
            quant["summary"][s]["mean_ssim"],
        )
    projection = get("unused-variance")
    assert (
        projection["transformer_profile"]
        == quant["method"]
        == get("encoder-protocol")["profile"]
    )
    original, kept = (
        projection["projection_files"][name] for name in ("original", "mean")
    )
    assert projection["source_weight_shape"] == [64, 1, 1, 64]
    assert original["parameters"] - kept["parameters"] == 32 * 65
    assert (
        original["payload_bytes"] - kept["payload_bytes"] == 4160
        and original["file_bytes"] > kept["file_bytes"]
    )
    assert len(projection["encoder_inputs"]) == 3
    for row in projection["encoder_inputs"]:
        assert 0 < row["nonzero_discarded_values"] <= row["mean_values"]
        assert (row["max_mean_difference"] == 0) == (
            row["exact_mean_values"] == row["mean_values"]
        )
    for row in projection["complete_renders"]:
        ref = next(
            r for r in quant["measurements"] + quant["native"] if r["id"] == row["id"]
        )
        assert ref["rgb_sha256"] == row["reference_sha256"]
        assert (row["max_delta"] == 0) == (row["rgb_sha256"] == row["reference_sha256"])
        assert (row["matching_channels"] == row["channels"]) == (row["max_delta"] == 0)
    fitted = root / "students/students.npz"
    assert (
        hashlib.sha256(fitted.read_bytes()).hexdigest()
        == stores["fitted_student_file_sha256"]
    )
    with zipfile.ZipFile(fitted) as bundle:
        entries = stores["fitted_student_entries_sha256"]
        assert len(entries) == 48 and set(bundle.namelist()) == set(entries)
        for name, h in entries.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == h
    refit = get("refit")
    assert (
        refit["fit_parameters"]
        == refit["equal_parameters"]
        == train["learned_parameters"]
    )
    assert (
        refit["max_abs_delta"] == 0
        and refit["file_sha256"]
        == refit["first_file_sha256"]
        == stores["fitted_student_file_sha256"]
    )
    browser = get("archive-browser")
    assert browser["storage"] == "zip" and browser["profile"] == "q4"
    assert browser["browser_errors"] == 0 and browser["viewport"] == [390, 844]
    assert browser["width"] == 576 and browser["height"] == 768
    assert len(browser["jpeg_sha256"]) == 64 and browser["click_to_result_s"] > 0
    assert (
        browser["exact_file"]
        and browser["jpeg_sha256"] == browser["independently_encoded_reference_sha256"]
    )
    assert browser["matching_rgb_channels"] == browser["channels"] == 576 * 768 * 3
    assert browser["max_rgb_delta"] == 0
    reuse = get("reference-reuse")
    assert len(reuse["pairs"]) == 3 and reuse["profile"] == "q4"
    for case in reuse["pairs"]:
        original, cached = case["original"], case["reuse"]
        prior = next(
            row
            for row in get("native")
            if row["id"] == case["case"] and row["method"] == "frozen"
        )
        assert case["previous_frozen_rgb_sha256"] == prior["rgb_sha256"]
        assert (case["previous_max_rgb_delta"] == 0) == (
            prior["rgb_sha256"] == original["rgb_sha256"]
        )
        assert original["encoder_hits"] == 0 and not original["cache_enabled"]
        assert (
            cached["cache_enabled"]
            and cached["retained_latent_bytes"] <= reuse["capacity_bytes"]
        )
        assert cached["encoder_hits"] + cached["encoder_misses"] >= 2
        assert (case["max_rgb_delta"] == 0) == (
            original["rgb_sha256"] == cached["rgb_sha256"]
        )
        close(original["wall_seconds"] - cached["wall_seconds"], case["seconds_saved"])
    close(
        statistics.median(r["seconds_saved"] for r in reuse["pairs"]),
        reuse["median_seconds_saved"],
    )
    stream = get("archive-stream")
    assert {r["member"]: r["sha256"] for r in stream} == stores["students"][
        "student12"
    ]["source_shard_sha256"]
    assert sum(r["matched_payload_bytes"] for r in stream) > 5_200_000_000
    return get("counts")


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
