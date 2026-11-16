"""Run and resume the predeclared validation, untouched test, and optional real-photo trials."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from .protocol import METHODS, TEMPORAL, STRUCTURAL, GATES, choose, plan
from .quality import compare, summarize


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(out, do, cases=None):
    manifest = json.loads((out / "pairs" / "manifest.json").read_text())
    training = json.loads((out / "learned" / "training.json").read_text())
    plan_file = out / "protocol.json"
    protocol = dict(
        seed=42,
        width=256,
        height=320,
        steps=3,
        rank=8,
        methods=list(METHODS),
        validation_ids=[r["id"] for r in manifest["records"] if r["split"] == "val"],
        test_ids=[r["id"] for r in manifest["records"] if r["split"] == "test"],
        selection="First qualifying quality tier. Minimum median time for temporal policies, most removed blocks for students. If neither tier passes, highest mean SSIM.",
        fixed_stress_test="student12 and drop12 also run on all five test photographs independently of selection.",
        gates=GATES,
        schedule={m: plan(m, training) for m in METHODS},
    )
    if plan_file.exists() and json.loads(plan_file.read_text()) != protocol:
        raise ValueError("Refusing to mix studies from different protocols")
    save(plan_file, protocol)

    def phase(name, ids, methods):
        path = out / (name + "-measurements.json")
        rows = json.loads(path.read_text()) if path.exists() else []
        for i, photo in enumerate(ids):
            # Rotate methods across photos to distribute thermal and order effects.
            offset = i * max(1, len(methods) // len(ids)) % len(methods)
            for method in methods[offset:] + methods[:offset]:
                folder = out / "renders" / str(photo)
                previous = next(
                    (r for r in rows if r["id"] == photo and r["method"] == method),
                    None,
                )
                saved = folder / (method + ".png")
                if previous is not None:
                    if saved.is_file():
                        with Image.open(saved) as im:
                            if (
                                hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
                                == previous["rgb_sha256"]
                            ):
                                continue
                    rows.remove(previous)
                row = do(dict(stage="render", id=photo, method=method))
                if row["status"] != "success":
                    raise ValueError(row)
                rows.append(row)
                save(path, rows)
                print(name, photo, method, row["wall_seconds"], flush=True)
        for row in rows:
            folder = out / "pairs" / str(row["id"])
            renders = out / "renders" / str(row["id"])

            def arr(path):
                with Image.open(path) as im:
                    return np.asarray(im)

            image = arr(renders / (row["method"] + ".png"))
            if hashlib.sha256(image.tobytes()).hexdigest() != row["rgb_sha256"]:
                raise ValueError("Cached image differs from log")
            row.update(
                compare(
                    image,
                    arr(renders / "frozen.png"),
                    arr(folder / "target.png"),
                    arr(folder / "mask.png"),
                )
            )
        save(out / (name + ".json"), rows)
        return summarize(rows)

    if not (out / "warmup.json").exists():
        warm = do(
            dict(stage="render", id=training["training_images"][0], method="frozen")
        )
        save(out / "warmup.json", warm)
    val = phase("validation", protocol["validation_ids"], list(METHODS))
    temporal, tt = choose(val, TEMPORAL, False)
    structural, st = choose(val, STRUCTURAL, True)
    chosen = dict(
        temporal=temporal, temporal_tier=tt, structural=structural, structural_tier=st
    )
    if (out / "selection.json").exists() and json.loads(
        (out / "selection.json").read_text()
    ) != chosen:
        raise ValueError("Validation choice changed after initial selection")
    save(out / "selection.json", chosen)
    print("Validation decision", chosen, flush=True)
    candidates = list(
        dict.fromkeys(
            [
                "frozen",
                temporal,
                structural,
                structural.replace("student", "drop"),
                "student12",
                "drop12",
            ]
        )
    )
    test = phase("test", protocol["test_ids"], candidates)
    save(out / "summary.json", dict(validation=val, test=test))
    if cases is not None:
        rows = (
            json.loads((out / "native.json").read_text())
            if (out / "native.json").exists()
            else []
        )
        for i, e in enumerate(cases):
            methods = ["frozen", temporal, structural]
            if i == 0 and structural != "student12":
                methods.append("student12")
            for method in methods[i % len(methods) :] + methods[: i % len(methods)]:
                if any(
                    row["id"] == e["case"] and row["method"] == method for row in rows
                ):
                    continue
                row = do(dict(stage="native", case=e, method=method))
                rows.append(row)
                save(out / "native.json", rows)
                print(
                    "studio",
                    e["case"],
                    method,
                    row["status"],
                    row["wall_seconds"],
                    flush=True,
                )
        for row in rows:
            if row["status"] != "success":
                continue
            folder = out / "native" / str(row["id"])
            if not (folder / "mask.png").exists():
                continue
            with (
                Image.open(folder / (row["method"] + ".png")) as a,
                Image.open(folder / "frozen.png") as b,
                Image.open(folder / "mask.png") as m,
            ):
                relative = compare(a, b, b, m)
            row.update(
                {
                    k: v
                    for k, v in relative.items()
                    if k not in ("garment_mae", "outside_mae")
                }
            )
        save(out / "native.json", rows)
    return chosen


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--cases", type=Path)
    a = p.parse_args()
    from gpu_service.klein_runtime import KleinRuntime
    from .evaluate_depth import measure

    rt = KleinRuntime(
        Path("model_src/klein-kv/weights"),
        Path("model_src/klein-kv/prompt-cache"),
        Path("model_src/fashn-vton/weights/human-parser"),
    )
    cases = json.loads(a.cases.read_text()) if a.cases else None
    run(a.output, lambda j: measure(rt, a.output, j), cases)
