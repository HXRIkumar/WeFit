"""Apply research interventions to a serial model for one measurement."""

from contextlib import contextmanager
from mlx import nn

from gpu_service.klein_depth import ResidualStudent, Scheduled
from .protocol import plan


class Bypass(nn.Module):
    def __call__(self, hidden_states, **kwargs):
        return hidden_states


@contextmanager
def intervention(model, method, training, weights):
    original = model.transformer.single_transformer_blocks
    schedule = plan(method, training)
    blocks = list(original)
    for b in sorted(set().union(*schedule)):
        if method.startswith("student"):
            blocks[b] = ResidualStudent(weights, b)
        elif method.startswith("drop"):
            blocks[b] = Bypass()
        else:
            blocks[b] = Scheduled(
                original[b],
                tuple(i for i, s in enumerate(schedule) if b in s),
                method.startswith("reuse"),
            )
    model.transformer.single_transformer_blocks = blocks
    try:
        yield schedule
    finally:
        model.transformer.single_transformer_blocks = original
