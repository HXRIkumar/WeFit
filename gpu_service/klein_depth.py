"""Small, time-specific students for the experimental three-pass Klein editor."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from .klein_prompts import MODEL_REVISION


class ResidualStudent(nn.Module):
    def __init__(self, arrays, block):
        super().__init__()
        for name in ("down", "up", "gain", "bias"):
            value = arrays[f"b{block}_{name}"]
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Student coefficients must be finite float32 values.")
            self[name] = mx.array(value)
        rank = self.down.shape[-1]
        if not (
            1 <= rank <= 32
            and self.down.shape == (3, 4096, rank)
            and self.up.shape == (3, rank, 4096)
            and self.gain.shape == self.bias.shape == (3, 4096)
        ):
            raise ValueError("Invalid depth student dimensions")
        self.step = -1

    def __call__(self, hidden_states, kv_cache=None, **kwargs):
        self.step = (
            0 if kv_cache is not None and kv_cache.is_extracting else self.step + 1
        )
        if not 0 <= self.step < 3:
            raise ValueError(
                "The depth student requires three steps and reference caching."
            )
        return _residual(
            hidden_states,
            self.down[self.step],
            self.up[self.step],
            self.gain[self.step],
            self.bias[self.step],
        )


@mx.compile
def _residual(hidden, down, up, gain, bias):
    h = hidden.astype(mx.float32)
    return (h + (h * gain + bias + (h @ down) @ up)).astype(hidden.dtype)


def read_depth_manifest(folder: Path):
    manifest = folder / "wefit-depth.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text())
    removed = data["removed"]
    if (
        data.get("base_revision") != MODEL_REVISION
        or data.get("format") != 1
        or not isinstance(removed, list)
        or not removed
        or len(removed) >= 24
        or any(type(i) is not int or not 0 <= i < 24 for i in removed)
        or sorted(set(removed)) != removed
        or data.get("retained") != [i for i in range(24) if i not in removed]
        or data.get("steps") != 3
        or data["profile"] != f"student{len(removed)}"
        or hashlib.sha256((folder / "students.npz").read_bytes()).hexdigest()
        != data["student_sha256"]
    ):
        raise ValueError("Invalid or incompatible structural checkpoint")
    return data


def restore_positions(kept, arrays, removed):
    if len(kept) + len(removed) != 24:
        raise ValueError("Expected 24 original single-stream positions")
    blocks = iter(kept)
    return [
        ResidualStudent(arrays, i) if i in removed else next(blocks) for i in range(24)
    ]


class Scheduled(nn.Module):
    def __init__(self, base, steps, reuse):
        super().__init__()
        self.base = base
        self.steps = steps
        self.reuse = reuse
        self.delta = None
        self.step = -1

    def __call__(self, hidden_states, kv_cache=None, **kwargs):
        self.step = 0 if kv_cache.is_extracting else self.step + 1
        if self.step == 0:
            self.delta = None
        if self.step in self.steps:
            if self.reuse:
                if self.delta is None or self.delta.shape != hidden_states.shape:
                    raise ValueError("Missing previous-step residual")
                return (hidden_states.astype(mx.float32) + self.delta).astype(
                    hidden_states.dtype
                )
            return hidden_states
        result = self.base(hidden_states=hidden_states, kv_cache=kv_cache, **kwargs)
        if self.reuse and self.step == 1:
            self.delta = result.astype(mx.float32) - hidden_states.astype(mx.float32)
            mx.eval(self.delta)
        return result


def load_schedule(path):
    profile = json.loads(Path(path).read_text())
    method = profile.get("method", "")
    if profile.get("base_revision") != MODEL_REVISION or profile.get("format") != 1:
        raise ValueError("The step schedule belongs to a different renderer revision.")
    if method not in {
        "last1",
        "last2",
        "last4",
        "last12",
        "later8",
        "later16",
        "reuse4",
        "reuse12",
    }:
        raise ValueError("Unknown depth schedule.")
    steps = profile.get("blocks_by_step")
    if not isinstance(steps, list) or len(steps) != 3 or steps[0] != []:
        raise ValueError("All blocks must run on the reference extraction pass.")
    count = int("".join(c for c in method if c.isdigit()))
    expected = [0, count if method.startswith("later") else 0, count]
    for group, size in zip(steps, expected, strict=True):
        if (
            not isinstance(group, list)
            or len(group) != size
            or any(type(i) is not int or not 0 <= i < 24 for i in group)
            or len(set(group)) != size
        ):
            raise ValueError("Invalid block indices for the selected schedule.")
    return profile


@contextmanager
def apply_schedule(model, profile):
    original = model.transformer.single_transformer_blocks
    steps = profile["blocks_by_step"]
    blocks = list(original)
    for index in sorted(set().union(*steps)):
        blocks[index] = Scheduled(
            original[index],
            tuple(s for s, g in enumerate(steps) if index in g),
            profile["method"].startswith("reuse"),
        )
    model.transformer.single_transformer_blocks = blocks
    try:
        yield
    finally:
        model.transformer.single_transformer_blocks = original
