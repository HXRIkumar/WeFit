"""Check image composition and the single-owner service, without model weights.

These are runtime contract tests. Photographic quality is evaluated separately
using real model outputs in the private benchmark directory.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import io
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from gpu_service.klein_runtime import canvas_size, garment_reference, neutral_clothing_condition


def png():
    buffer = io.BytesIO(); Image.new("RGB", (48, 64), "blue").save(buffer, "PNG")
    return buffer.getvalue()


def test_canvas_keeps_portrait_and_landscape_composition():
    assert canvas_size((1280, 1707)) == (768, 1024)
    assert canvas_size((1600, 1200)) == (1024, 768)
    assert canvas_size((900, 1600)) == (576, 1024)


def test_model_worn_reference_excludes_face_and_other_clothing():
    pixels = np.full((120, 80, 3), 255, np.uint8)
    pixels[5:25, 30:50] = [210, 130, 90]
    pixels[30:70, 10:70] = [100, 160, 220]
    pixels[75:115, 25:55] = [40, 40, 40]
    labels = np.zeros((120, 80), np.uint8)
    labels[5:25, 30:50] = 1; labels[30:70, 10:70] = 3; labels[75:115, 25:55] = 6
    top = garment_reference(Image.fromarray(pixels), "upper", "model", lambda _: labels)
    values = np.asarray(top)
    assert (values == [100, 160, 220]).all(axis=-1).sum() > 1000
    assert not (values == [210, 130, 90]).all(axis=-1).any()
    assert not (values == [40, 40, 40]).all(axis=-1).any()
    with pytest.raises(ValueError, match="Could not find"):
        garment_reference(Image.fromarray(pixels), "lower", "model", lambda _: np.zeros_like(labels))


def test_flat_garment_does_not_run_a_person_parser():
    original = Image.new("RGB", (544, 612), (100, 160, 220))
    output = garment_reference(original, "upper", "flat-lay", lambda _: pytest.fail("Not a person reference"))
    assert output.size == original.size
    assert output.tobytes() == original.tobytes()


def test_neutral_condition_removes_a_print_but_preserves_an_overlapping_hand():
    pixels = np.full((100, 80, 3), (110, 180, 150), np.uint8)
    pixels[20:90, 15:65] = [20, 30, 90]
    pixels[40:70, 30:50] = [210, 100, 30]
    pixels[65:78, 40:60] = [130, 90, 70]
    labels = np.zeros((100, 80), np.uint8); labels[20:90, 15:65] = 3
    labels[65:78, 40:60] = 13
    labels[45:55, 35:45] = 1  # a printed badge misclassified as a face
    output = neutral_clothing_condition(Image.fromarray(pixels), labels, "upper")
    assert output.getpixel((40, 50)) == (140, 140, 140)
    assert output.getpixel((50, 70)) == (130, 90, 70)
    assert output.getpixel((3, 3)) == (110, 180, 150)


def test_model_lives_on_one_thread_and_requests_are_serialized(monkeypatch):
    import gpu_service.klein_runtime as runtime_module
    calls = []; active = 0

    class FakeRuntime:
        def __init__(self, *args):
            self.owner = threading.get_ident(); calls.append(("load", self.owner))

        def run(self, *args):
            nonlocal active
            assert threading.get_ident() == self.owner
            assert active == 0
            active += 1; calls.append(("run", threading.get_ident()))
            time.sleep(.03); active -= 1
            return Image.new("RGB", (48, 64), "green"), {
                "height": 64, "preprocess_seconds": .01, "sampling_seconds": .01, "postprocess_seconds": .01}

    monkeypatch.setattr(runtime_module, "KleinRuntime", FakeRuntime)
    service = importlib.import_module("gpu_service.klein_server")
    monkeypatch.setattr(service, "KEY", "local-test-key")
    monkeypatch.setattr(service, "del_runtime", lambda app: calls.append(("release", threading.get_ident())))
    with TestClient(service.app) as client:
        assert client.get("/health").json()["trained_reference_kv_cache"] is True
        upload = {"person": ("p.png", png(), "image/png"), "garment": ("g.png", png(), "image/png")}
        assert client.post("/tryon", files=upload).status_code == 401
        headers = {"Authorization": "Bearer local-test-key"}
        assert client.post("/tryon", files=upload, headers=headers, data={"cloth_type": "overall"}).status_code == 422
        with ThreadPoolExecutor(max_workers=2) as requests:
            futures = [requests.submit(client.post, "/tryon", files=upload, headers=headers) for _ in range(2)]
            for future in futures:
                response = future.result()
                assert response.status_code == 200
                assert response.headers["X-TryOn-Model"] == "flux2-klein-9b-kv-q4"
                assert Image.open(io.BytesIO(response.content)).size == (48, 64)
    assert [name for name, _ in calls] == ["load", "run", "run", "release"]
    assert len({owner for _, owner in calls}) == 1
