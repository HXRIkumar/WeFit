import hashlib
import io
import json
import struct

import numpy as np
import pytest

from gpu_service.klein_archive import tensors
from tools.depth_study.protocol import choose, plan


def test_nested_step_schedules_keep_full_reference_pass():
    calibration = {
        "structural_order": list(range(24)),
        "step_orders": [list(range(24)), list(reversed(range(24))), list(range(24))],
    }
    assert plan("last2", calibration) == [[], [], [0, 1]]
    assert plan("later8", calibration) == [[], list(range(23, 15, -1)), list(range(8))]
    assert plan("student2", calibration) == [[0, 1], [0, 1], [0, 1]]


def test_selection_uses_first_eligible_quality_tier():
    def row(sim, seconds):
        return dict(
            mean_ssim=sim,
            min_ssim=sim,
            garment_mae=12.0,
            outside_mae=4.0,
            median_seconds=seconds,
        )

    scores = {
        "frozen": row(1.0, 100),
        "last1": row(0.98, 95),
        "last12": row(0.84, 60),
        "student2": row(0.96, 94),
        "student6": row(0.95, 96),
        "student12": row(0.65, 45),
    }
    assert choose(scores, ("last1", "last12"), False) == ("last1", "strict")
    assert choose(scores, ("student2", "student6", "student12"), True) == (
        "student6",
        "strict",
    )


def test_lossless_shard_read_preserves_bfloat_bits_and_rejects_changed_bytes():
    values = np.array([0x3F80, 0x4000, 0xC010], dtype="<u2")
    header = {
        "__metadata__": {"mflux_version": "0.18.0", "quantization_level": "4"},
        "linear.weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]},
    }
    raw = json.dumps(header).encode()
    blob = struct.pack("<Q", len(raw)) + raw + values.tobytes()
    sha = hashlib.sha256(blob).hexdigest()
    [(name, kind, arr)] = list(tensors(io.BytesIO(blob), len(blob), sha))
    assert (name, kind) == ("linear.weight", "BF16")
    np.testing.assert_array_equal(arr, values)
    with pytest.raises(ValueError, match="pinned"):
        list(tensors(io.BytesIO(blob[:-1] + bytes([blob[-1] ^ 1])), len(blob), sha))
    with pytest.raises(ValueError, match="Truncated"):
        list(tensors(io.BytesIO(blob[:-2]), len(blob), sha))


def test_image_metrics_count_all_three_channels():
    pytest.importorskip("scipy")
    from tools.depth_study.quality import compare

    ref = np.full((16, 16, 3), 80, np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    mask[:8, :] = 255
    image = ref.copy()
    image[:4, :, 0] += 9
    actual = compare(image, ref, ref, mask)
    assert actual["garment_within8"] == 0.5 and actual["garment_exact"] == 0.5
    assert actual["garment_mae"] == 1.5 and actual["outside_mae"] == 0
    same = compare(ref, ref, ref, mask)
    assert same["garment_ssim"] == 1.0 and same["garment_within8"] == 1.0


def test_low_rank_student_learns_a_teacher_update_on_distinct_tokens():
    pytest.importorskip("scipy")
    from tools.depth_study.train import fit_residual

    rng = np.random.default_rng(44)
    x = rng.standard_normal((256, 32)).astype(np.float32)
    down = rng.normal(0, 0.2, (32, 3)).astype(np.float32)
    up = rng.normal(0, 0.3, (3, 32)).astype(np.float32)
    gain = rng.normal(0, 0.1, 32).astype(np.float32)
    bias = rng.normal(0, 0.1, 32).astype(np.float32)
    y = x * gain + bias + (x @ down) @ up
    w, _ = fit_residual(x[:200], y[:200], 3)
    test = x[200:]
    pred = test * w["gain"] + w["bias"] + (test @ w["down"]) @ w["up"]
    assert np.mean((pred - y[200:]) ** 2) < 0.15 * np.mean(y[200:] ** 2)


def test_step_policy_rejects_ambiguous_or_invalid_blocks(tmp_path):
    pytest.importorskip("mlx.core")
    from gpu_service.klein_depth import load_schedule
    from gpu_service.klein_prompts import MODEL_REVISION

    path = tmp_path / "policy.json"
    base = dict(
        format=1,
        method="last2",
        base_revision=MODEL_REVISION,
        blocks_by_step=[[], [], [8, 7]],
    )
    path.write_text(json.dumps(base))
    assert load_schedule(path) == base
    for groups in ([[1], [], [8, 7]], [[], [], [8, 8]], [[], [], [24, 8]]):
        path.write_text(json.dumps({**base, "blocks_by_step": groups}))
        with pytest.raises(ValueError):
            load_schedule(path)


def test_reference_cache_uses_input_values_shape_dtype_and_bounded_lifetime():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    from gpu_service.klein_reference import ReferenceLatents

    now = [0.0]
    cache = ReferenceLatents(max_bytes=32, ttl_seconds=10, clock=lambda: now[0])

    def compute(image):
        return image + 1

    a = mx.array([1, 2, 3, 4], dtype=mx.float32)
    for image in (a, a + 0, a.reshape(2, 2), a.astype(mx.bfloat16)):
        result = cache.encode(image, compute)
        assert (result == image + 1).all().item()
    assert (cache.hits, cache.misses) == (1, 3)
    cache.encode(a, compute)
    assert cache.misses == 4
    now[0] = 11
    cache.encode(a, compute)
    assert (cache.bytes, cache.misses) == (16, 5)
    cache.enabled = False
    cache.encode(a, compute)
    assert (cache.hits, cache.misses) == (1, 5)


def test_composite_is_reproducible_after_other_jobs_change_opencv_rng():
    import cv2
    from PIL import Image
    from gpu_service.photo_preserve import preserve_reference_edit

    rng = np.random.default_rng(71)
    y, x = np.mgrid[:100, :96]
    labels = np.zeros(x.shape, np.uint8)
    labels[((x - 45) / 25) ** 2 + ((y - 49) / 55) ** 2 < 1] = 3
    labels[((x - 48) / 15) ** 2 + ((y - 24) / 18) ** 2 < 1] = 11
    source = rng.integers(0, 255, (*x.shape, 3), dtype=np.uint8)
    edited = source.copy()
    edited[:, 11:81] = rng.integers(0, 255, (100, 70, 3), dtype=np.uint8)
    camera, shirt = Image.fromarray(source), Image.fromarray(edited)
    cv2.setRNGSeed(42)
    first = preserve_reference_edit(camera, shirt, labels, "upper", labels)
    cv2.setRNGSeed(73)
    second = preserve_reference_edit(camera, shirt, labels, "upper", labels)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
