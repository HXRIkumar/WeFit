"""Materialize reference K/V slices instead of retaining a lazy layer graph.

The reference values and attention calculation are unchanged. Contiguous,
evaluated slices keep each cached layer from owning the full predecessor graph
and the larger text/target/reference allocation that its view came from.
"""
import mlx.core as mx
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache


class BoundedReferenceCache(Flux2KVCache):
    def store_reference(self, stream, layer_idx, key, value):
        if not self.is_extracting:
            return
        ref_key = mx.contiguous(key[:, :, -self.num_ref_tokens:, :])
        ref_value = mx.contiguous(value[:, :, -self.num_ref_tokens:, :])
        mx.eval(ref_key, ref_value)
        if mx.get_active_memory() > 9 * 1024**3:
            raise MemoryError("The renderer exceeded its local memory budget; this image was not generated.")
        self._slots(stream)[self._layer_idx(layer_idx)] = (ref_key, ref_value)

