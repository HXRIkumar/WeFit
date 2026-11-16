"""The bounded cache must retain exactly the same reference values."""
import pytest


def test_materialized_reference_cache_matches_upstream_values_and_attention():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    from gpu_service.klein_cache import BoundedReferenceCache
    from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache
    key = mx.random.normal((1, 2, 12, 8)).astype(mx.bfloat16)
    value = mx.random.normal((1, 2, 12, 8)).astype(mx.bfloat16)
    query = mx.random.normal((1, 2, 12, 8)).astype(mx.bfloat16)
    native, bounded = Flux2KVCache(1, 1), BoundedReferenceCache(1, 1)
    for cache in (native, bounded):
        cache.configure(mode="extract", num_ref_tokens=4)
        cache.store_reference("double", 0, key, value)
    nk, nv = native._load("double", 0)
    bk, bv = bounded._load("double", 0)
    assert mx.array_equal(nk, bk).item() and mx.array_equal(nv, bv).item()
    kwargs = dict(query=query, key=key, value=value, batch_size=1, num_heads=2, head_dim=8)
    assert mx.array_equal(native.compute_extract_attention(**kwargs), bounded.compute_extract_attention(**kwargs)).item()
    for cache in (native, bounded):
        cache.configure(mode="cached", num_ref_tokens=4)
    nk, nv = native.append_reference("double", 0, key[:, :, :8], value[:, :, :8])
    bk, bv = bounded.append_reference("double", 0, key[:, :, :8], value[:, :, :8])
    assert mx.array_equal(nk, bk).item() and mx.array_equal(nv, bv).item()
