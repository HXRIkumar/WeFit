"""Cached block compilation must not freeze a previous reference image."""
import pytest


def test_compiled_blocks_match_eager_and_use_the_new_reference():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    from gpu_service.klein_compile import CompiledDoubleBlock, CompiledSingleBlock, _InputReference
    from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock
    from mflux.models.flux2.model.flux2_transformer.single_transformer_block import Flux2SingleTransformerBlock

    mx.random.seed(7)
    hidden = mx.random.normal((1, 4, 64))
    encoded = mx.random.normal((1, 2, 64))
    modulation = tuple(mx.random.normal((1, 1, 64)) * .1 for _ in range(3))
    rotary = (mx.ones((6, 16)), mx.zeros((6, 16)))
    double = Flux2TransformerBlock(64, 2, 32)
    single = Flux2SingleTransformerBlock(64, 2, 32)
    compiled_double, compiled_single = CompiledDoubleBlock(double), CompiledSingleBlock(single)
    outputs = []
    for _ in range(2):
        key, value = mx.random.normal((1, 2, 3, 32)), mx.random.normal((1, 2, 3, 32))
        cache = _InputReference(key, value)
        args = dict(hidden_states=hidden, encoder_hidden_states=encoded,
            temb_mod_params_img=(modulation, modulation), temb_mod_params_txt=(modulation, modulation),
            image_rotary_emb=rotary, kv_cache=cache, kv_cache_layer_idx=0)
        expected, actual = double(**args), compiled_double(**args)
        for a, b in zip(expected, actual):
            assert mx.allclose(a, b, atol=2e-5, rtol=2e-5).item()
        combined = mx.concatenate([encoded, hidden], axis=1)
        args = dict(hidden_states=combined, temb_mod_params=modulation,
                    image_rotary_emb=rotary, kv_cache=cache, kv_cache_layer_idx=0)
        expected, actual = single(**args), compiled_single(**args)
        assert mx.allclose(expected, actual, atol=2e-5, rtol=2e-5).item()
        outputs.append(actual)
    assert not mx.allclose(outputs[0], outputs[1]).item()


def test_quantized_blocks_accept_new_photo_and_reference_sizes():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mflux")
    from mlx import nn
    from mlx.utils import tree_map
    from gpu_service.klein_compile import CompiledDoubleBlock, _InputReference
    from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock
    block = Flux2TransformerBlock(64, 2, 32)
    block.update(tree_map(lambda x: x.astype(mx.bfloat16), block.parameters()))
    nn.quantize(block, bits=4)
    mx.eval(block.parameters())  # match the resident runtime before tracing
    compiled = CompiledDoubleBlock(block)
    for length, ref_length in [(4, 3), (7, 5)]:
        hidden = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
        encoded = mx.random.normal((1, 2, 64)).astype(mx.bfloat16)
        mods = tuple(mx.random.normal((1, 1, 64)).astype(mx.bfloat16)*.1 for _ in range(3))
        rotary = (mx.ones((length+2, 16)), mx.zeros((length+2, 16)))
        cache = _InputReference(mx.random.normal((1, 2, ref_length, 32)).astype(mx.bfloat16),
                                mx.random.normal((1, 2, ref_length, 32)).astype(mx.bfloat16))
        kwargs = dict(hidden_states=hidden, encoder_hidden_states=encoded,
            temb_mod_params_img=(mods, mods), temb_mod_params_txt=(mods, mods),
            image_rotary_emb=rotary, kv_cache=cache, kv_cache_layer_idx=0)
        expected, actual = block(**kwargs), compiled(**kwargs)
        for a, b in zip(expected, actual):
            assert a.shape == b.shape
            assert mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item() <= .03125
