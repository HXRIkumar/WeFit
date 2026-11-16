"""Exact, short-lived reuse of deterministic reference encodings on one MLX worker."""

from collections import OrderedDict
import hashlib
import time

import mlx.core as mx
import numpy as np
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class ReferenceLatents:
    def __init__(self, max_bytes=8 * 1024**2, ttl_seconds=600, clock=time.monotonic):
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.enabled = True

    def discard_oldest(self):
        _, (_, value) = self.entries.popitem(last=False)
        self.bytes -= value.nbytes

    def encode(self, image, encoder):
        if not self.enabled:
            return encoder(image)
        now = self.clock()
        while (
            self.entries
            and now - next(iter(self.entries.values()))[0] >= self.ttl_seconds
        ):
            self.discard_oldest()
        raw = mx.contiguous(image).view(mx.uint8)
        key = (image.shape, image.dtype, hashlib.sha256(np.asarray(raw)).digest())
        if key in self.entries:
            _, value = self.entries[key]
            self.entries[key] = (now, value)
            self.entries.move_to_end(key)
            self.hits += 1
            return value
        self.misses += 1
        value = encoder(image)
        if value.nbytes <= self.max_bytes:
            mx.eval(value)
            while self.entries and self.bytes + value.nbytes > self.max_bytes:
                self.discard_oldest()
            self.entries[key] = (self.clock(), value)
            self.bytes += value.nbytes
        return value


class ReferenceCachingVAE(Flux2VAE):
    """The pinned encoder returns its posterior mean and does not sample noise."""

    def __init__(self):
        super().__init__()
        self.reference_latents = ReferenceLatents()

    def encode(self, image):
        return self.reference_latents.encode(image, super().encode)
