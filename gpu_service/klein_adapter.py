"""Optional low-rank update to the final Klein output projection, final step only."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from .klein_prompts import MODEL_REVISION


@dataclass(frozen=True)
class Adapter:
    down: np.ndarray
    up: np.ndarray

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["revision"]) != MODEL_REVISION
                or str(data["target"]) != "upper.final_step.proj_out"
            ):
                raise ValueError(
                    "The adapter must match this Klein revision and upper-body output head."
                )
            down, up = data["down"], data["up"]
        if (
            down.ndim != 2
            or down.shape[0] != 4096
            or not 1 <= down.shape[1] <= 4
            or up.shape != (down.shape[1], 128)
        ):
            raise ValueError("Expected a 4096-to-128 output adapter of rank 1 to 4.")
        if any(
            matrix.dtype not in (np.float16, np.float32)
            or not np.isfinite(matrix).all()
            for matrix in (down, up)
        ):
            raise ValueError("Invalid adapter weights.")
        return cls(down.astype(np.float32), up.astype(np.float32))


class FinalProjection(nn.Module):
    def __init__(self, base, steps, adapter, capture):
        super().__init__()
        self.base = base
        self.steps = steps
        self.calls = 0
        self.capture = capture
        self.last_hidden = None
        if adapter is not None:
            self.down = mx.array(adapter.down)
            self.up = mx.array(adapter.up)

    def __call__(self, hidden):
        self.calls += 1
        base = self.base(hidden)
        if self.calls == self.steps:
            if self.capture:
                # The native hidden state is bfloat16. Its ordinary values fit losslessly in float16.
                self.last_hidden = np.array(hidden[0].astype(mx.float16), copy=True)
            if "down" in self:
                offset = (hidden.astype(mx.float32) @ self.down) @ self.up
                return (base.astype(mx.float32) + offset).astype(base.dtype)
        return base


@contextmanager
def final_projection(model, steps, adapter=None, capture=False):
    """The serial worker makes exactly one projection call per step at guidance 1."""
    original = model.transformer.proj_out
    wrapper = FinalProjection(original, steps, adapter, capture)
    model.transformer.proj_out = wrapper
    try:
        yield wrapper
        if wrapper.calls != steps:
            raise RuntimeError(
                "Unexpected number of projection calls. This experiment requires guidance 1."
            )
    finally:
        model.transformer.proj_out = original
