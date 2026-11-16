import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.blind_review import CRITERIA, aggregate
from tools.profile_latency import summarize


def test_latency_ranks_and_failed_crop_retries():
    rows = [{"phase": "first", "case": "shirt", "status": 200, "wall_s": 999}]
    rows += [
        {
            "phase": "warm",
            "case": "shirt",
            "status": 200,
            "wall_s": i + 1,
            "service_s": i,
            "initial_region": "clothing" if i % 2 else "full",
            "fallback": False,
        }
        for i in range(1, 21)
    ]
    rows.append(
        {
            "phase": "warm",
            "case": "shirt",
            "status": 422,
            "initial_region": "clothing",
            "fallback": True,
        }
    )
    rows.append({"phase": "warm", "case": "shirt", "status": 0})
    report = summarize(rows)
    assert report["service"]["median_s"] == 10.5
    assert report["wall"]["p95_s"] == 20
    assert report["attempts"] == 22 and report["completed"] == 20
    assert report["fallbacks"] == 1 and report["crop_attempts"] == 11
    assert report["unobserved_initial_region"] == 1


def test_blind_scores_decode_different_side_assignments(tmp_path):
    keys, answers = [], []
    for i, sides in enumerate([("original", "adapter"), ("adapter", "original")]):
        pid = str(i)
        key = tmp_path / f"key{i}.json"
        key.write_text(
            json.dumps(
                {
                    "packet": pid,
                    "study": "demo",
                    "methods": ["original", "adapter"],
                    "cases": {"x": {"id": "shirt", "A": sides[0], "B": sides[1]}},
                }
            )
        )
        keys.append(key)
        r = {side: dict.fromkeys(CRITERIA, None) for side in ("A", "B")}
        which = "A" if sides[0] == "adapter" else "B"
        r[which]["neckline"] = 4
        r["preference"] = which
        response = tmp_path / f"score{i}.json"
        response.write_text(json.dumps({"packet": pid, "ratings": {"x": r}}))
        answers.append(response)
    assert aggregate(keys, [])["completed_reviewer_packets"] == 0
    result = aggregate(keys, answers)
    assert result["preferences"] == {"adapter": 2}
    assert result["preference_agreement"]["fraction"] == 1.0
    assert result["ratings"]["adapter"]["neckline"] == {"n": 2, "mean": 4.0}
    assert result["ratings"]["original"]["neckline"]["mean"] is None
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(keys, [*answers, answers[0]])


def test_projection_scope_restores_the_original_even_after_error():
    mx = pytest.importorskip("mlx.core")
    from mlx import nn

    from gpu_service.klein_adapter import Adapter, final_projection

    with mx.stream(mx.cpu):
        layer = nn.Linear(8, 2)
        model = SimpleNamespace(transformer=SimpleNamespace(proj_out=layer))
        x = mx.ones((1, 3, 8))
        base = layer(x)
        adapter = Adapter(np.ones((8, 1), np.float32), np.ones((1, 2), np.float32))
        with final_projection(model, 3, adapter=adapter, capture=True) as tap:
            for i in range(3):
                change = model.transformer.proj_out(x) - base
                assert np.allclose(np.array(change), 8 if i == 2 else 0, atol=1e-5)
        assert model.transformer.proj_out is layer
        assert tap.last_hidden.shape == (3, 8)
        with (
            pytest.raises(RuntimeError, match="projection calls"),
            final_projection(model, 3, adapter=adapter),
        ):
            model.transformer.proj_out(x)
        assert model.transformer.proj_out is layer


def test_compact_adapter_and_checkpoint_validation(tmp_path):
    pytest.importorskip("mlx.core")
    from gpu_service.klein_adapter import Adapter
    from gpu_service.klein_prompts import MODEL_REVISION

    out = tmp_path / "compact.npz"
    arrays = {
        "down": np.zeros((4096, 2), np.float16),
        "up": np.zeros((2, 128), np.float16),
        "target": "upper.final_step.proj_out",
    }
    np.savez(out, **arrays, revision=MODEL_REVISION)
    assert Adapter.load(out).down.dtype == np.float32
    np.savez(out, **arrays, revision="unrelated")
    with pytest.raises(ValueError, match="revision"):
        Adapter.load(out)


def test_replay_preserves_native_state_and_updates_only_the_final_prediction(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
        FlowMatchEulerDiscreteScheduler,
    )
    from mlx import nn

    from gpu_service.klein_adapter import Adapter
    from tools.adapter_study.replay import capture, replay, save_state

    class Head(nn.Module):
        def __call__(self, x):
            return x[..., :128]

    class Decoder:
        def decode_packed_latents(self, grid):
            return grid[:, :3]

    with mx.stream(mx.cpu):
        model = SimpleNamespace(
            transformer=SimpleNamespace(proj_out=Head()), callbacks=CallbackRegistry()
        )
        scheduler = object.__new__(FlowMatchEulerDiscreteScheduler)
        scheduler._sigmas = mx.array([1.0, 0.8, 0.5, 0.0])
        hidden = mx.zeros((1, 1, 4096)).at[:, :, 4095].add(1).astype(mx.bfloat16)
        prior = mx.zeros((1, 1, 128), dtype=mx.bfloat16)
        original = model.transformer.proj_out
        with capture(model) as state:
            for step in range(3):
                model.transformer.proj_out(hidden)
                state.call_in_loop(step, prior, SimpleNamespace(scheduler=scheduler))
        assert model.transformer.proj_out is original and not model.callbacks.in_loop
        for name in ("person.png", "item.png"):
            (tmp_path / name).write_bytes(b"test inputs")
        save_state(state, tmp_path, [16, 16], 3)
        file = tmp_path / "last-step.safetensors"
        down = np.zeros((4096, 1), np.float32)
        down[-1] = 1
        adapter = Adapter(down, np.ones((1, 128), np.float32))
        assert replay(Decoder(), file).getpixel((0, 0)) == (128, 128, 128)
        assert replay(Decoder(), file, adapter).getpixel((0, 0)) == (64, 64, 64)
        with pytest.raises(ValueError, match="between zero and one"):
            replay(Decoder(), file, adapter, 2)


def test_trace_processes_transparent_products_on_the_resident_worker(monkeypatch):
    import io
    import sys
    import threading

    from fastapi.testclient import TestClient
    from PIL import Image

    from gpu_service import klein_runtime, klein_server

    class Worker:
        def __init__(self, *paths):
            self.owner = threading.get_ident()
            self.weight_profile = "q4"
            self.upper_adapter = None

    def trace(worker, person, product, photo_type, evidence):
        assert worker.owner == threading.get_ident()
        assert person.size == (3, 1)
        assert list(product.get_flattened_data()) == [
            (255, 255, 255),
            (255, 127, 127),
            (5, 9, 13),
        ]
        return {"format": "wefit-trace-v1", "steps": 3}

    monkeypatch.setattr(klein_runtime, "KleinRuntime", Worker)
    monkeypatch.setattr(klein_server, "KEY", "unit-test")
    monkeypatch.setattr(klein_server, "del_runtime", lambda app: None)
    monkeypatch.setitem(
        sys.modules, "tools.trace_garment", SimpleNamespace(trace_on_runtime=trace)
    )
    image = Image.new("RGBA", (3, 1))
    image.putdata([(0, 0, 0, 0), (255, 0, 0, 128), (5, 9, 13, 255)])
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    files = {
        name: (name + ".png", buf.getvalue(), "image/png")
        for name in ("person", "garment")
    }
    with TestClient(klein_server.app) as client:
        assert client.post("/trace", files=files).status_code == 401
        auth = {"Authorization": "Bearer unit-test"}
        assert (
            client.post(
                "/trace",
                files=files,
                headers=auth,
                data={"garment_photo_type": "unknown"},
            ).status_code
            == 422
        )
        assert client.post("/trace", files=files, headers=auth).json()["steps"] == 3
        klein_server.app.state.runtime.weight_profile = "wefit-q3-six-blocks"
        assert client.post("/trace", files=files, headers=auth).status_code == 422
