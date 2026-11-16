"""Compile pure cached-mode blocks, with reference arrays as explicit inputs.

Extraction stays eager so its bounded cache can materialize each layer. Cached
passes use the original block calculations. No image-dependent array is closed
over by a compiled function, so a later garment cannot reuse an earlier one.
"""
import mlx.core as mx
from mlx import nn
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache


class _InputReference(Flux2KVCache):
    def __init__(self, key, value):
        super().__init__(1, 1)
        self.configure(mode="cached", num_ref_tokens=key.shape[2])
        self._double[0] = self._single[0] = (key, value)


class CompiledDoubleBlock(nn.Module):
    def __init__(self, block):
        super().__init__(); self.block = block

        def pure(hidden, encoded, img_mod, text_mod, rotary, ref_key, ref_value):
            return block(hidden_states=hidden, encoder_hidden_states=encoded,
                temb_mod_params_img=img_mod, temb_mod_params_txt=text_mod, image_rotary_emb=rotary,
                kv_cache=_InputReference(ref_key, ref_value), kv_cache_layer_idx=0)
        self._pure = mx.compile(pure)

    def __call__(self, hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt,
                 image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None):
        if kv_cache is None or not kv_cache.is_cached:
            return self.block(hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt,
                              image_rotary_emb, kv_cache, kv_cache_layer_idx)
        key, value = kv_cache._load("double", kv_cache_layer_idx)
        return self._pure(hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt,
                          image_rotary_emb, key, value)


class CompiledSingleBlock(nn.Module):
    def __init__(self, block):
        super().__init__(); self.block = block

        def pure(hidden, modulation, rotary, ref_key, ref_value):
            return block(hidden_states=hidden, temb_mod_params=modulation, image_rotary_emb=rotary,
                kv_cache=_InputReference(ref_key, ref_value), kv_cache_layer_idx=0)
        self._pure = mx.compile(pure)

    def __call__(self, hidden_states, temb_mod_params, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None):
        if kv_cache is None or not kv_cache.is_cached:
            return self.block(hidden_states, temb_mod_params, image_rotary_emb, kv_cache, kv_cache_layer_idx)
        key, value = kv_cache._load("single", kv_cache_layer_idx)
        return self._pure(hidden_states, temb_mod_params, image_rotary_emb, key, value)


def compile_cached_blocks(transformer):
    transformer.transformer_blocks = [CompiledDoubleBlock(b) for b in transformer.transformer_blocks]
    transformer.single_transformer_blocks = [CompiledSingleBlock(b) for b in transformer.single_transformer_blocks]
