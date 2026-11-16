"""MFLUX's native KV renderer with exact precomputed fixed instructions.

The default Q4 profile uses the upstream transformer, decoder and sampler.
Optional depth checkpoints substitute learned residual students. The pinned
text encoder prepared both fixed instructions offline.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx import nn
import numpy as np

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinNoTextEncoderWeightDefinition

from .klein_prompts import INSTRUCTIONS, MODEL_REPOSITORY, MODEL_REVISION, instruction_hash
from .klein_cache import BoundedReferenceCache


class CachedInstructionKlein(Flux2KleinEdit):
    def __init__(self, weights_path: Path, cache_path: Path):
        nn.Module.__init__(self)
        # MLX's default guideline is 1.5 times recommended GPU memory, which
        # exceeds this 24 GB machine's physical RAM. Leave room for the OS.
        mx.set_memory_limit(8 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)
        manifest=json.loads((cache_path/'manifest.json').read_text())
        if manifest['model_repository']!=MODEL_REPOSITORY or manifest['model_revision']!=MODEL_REVISION:
            raise ValueError('Instruction cache belongs to a different model revision; rebuild it.')
        if manifest['mflux_version']!=importlib.metadata.version('mflux'):
            raise ValueError('MFLUX version changed; rebuild the instruction cache.')
        self._instructions={}
        for kind,prompt in INSTRUCTIONS.items():
            info=manifest['instructions'][kind];path=cache_path/f'{kind}.npz'
            if info['sha256']!=instruction_hash(prompt) or hashlib.sha256(path.read_bytes()).hexdigest()!=info['file_sha256']:
                raise ValueError('Instruction cache is stale or damaged; rebuild it.')
            with np.load(path,allow_pickle=False) as saved:
                embeds=mx.array(saved['prompt_embeds']).astype(mx.bfloat16)
                ids=mx.array(saved['text_ids'])
            self._instructions[prompt]=(embeds,ids,None,None)
        self.model_config=ModelConfig.flux2_klein_9b_kv()
        self.callbacks=CallbackRegistry();self.tiling_config=None;self.prompt_cache={}
        self.small_text_encoder=False;self.small_decoder=False
        precision_name = os.getenv("KLEIN_COMPUTE_PRECISION", "native")
        if precision_name not in ("native", "float16"):
            raise ValueError("KLEIN_COMPUTE_PRECISION must be native or float16")
        self.compute_precision=ComputePrecision(None if precision_name == "native" else mx.float16)
        if os.getenv("WEFIT_KLEIN_REFERENCE_CACHE") == "1":
            from .klein_reference import ReferenceCachingVAE
            self.vae = ReferenceCachingVAE()
        else:
            self.vae = Flux2VAE()
        self.vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
        from .klein_depth import read_depth_manifest, restore_positions
        depth = read_depth_manifest(weights_path)
        config = dict(self.model_config.transformer_overrides)
        if depth:
            config["num_single_layers"] = 24 - len(depth["removed"])
        self.transformer=Flux2Transformer(**config)
        self.text_encoder=None;self.tokenizers={};self.lora_paths=None;self.lora_scales=None
        if weights_path.is_file() and weights_path.suffix.lower() == ".zip":
            from .klein_archive import load as load_archived_weights
            loaded = load_archived_weights(weights_path)
        else:
            loaded = WeightLoader.load(weight_definition=Flux2KleinNoTextEncoderWeightDefinition,
                                       model_path=str(weights_path.resolve()))
        self.bits=WeightApplier.apply_and_quantize(weights=loaded,quantize_arg=4,
            weight_definition=Flux2KleinNoTextEncoderWeightDefinition,models={'vae':self.vae,'transformer':self.transformer})
        if self.compute_precision.dtype is not None:
            self.transformer.apply_compute_precision(self.compute_precision)
        if os.getenv("KLEIN_COMPILE_BLOCKS", "1") == "1":
            from .klein_compile import compile_cached_blocks
            compile_cached_blocks(self.transformer)
        if depth:
            with np.load(weights_path / "students.npz", allow_pickle=False) as students:
                self.transformer.single_transformer_blocks = restore_positions(
                    self.transformer.single_transformer_blocks, students, depth["removed"]
                )
        # Native row-strip decoding retains full-image normalization. No tiled
        # decoder or small text-encoder approximation is introduced here.
        self.callbacks.register(MemorySaver(self,keep_transformer=True,cache_limit_bytes=None))

    def _new_kv_cache(self):
        return BoundedReferenceCache(num_double_layers=len(self.transformer.transformer_blocks),
                                     num_single_layers=len(self.transformer.single_transformer_blocks))

    def _encode_prompt_pair(self, prompt, negative_prompt, guidance):
        if guidance!=1.0 or prompt not in self._instructions:
            raise ValueError('Only the verified fixed instructions and distilled guidance 1.0 are supported.')
        return self._instructions[prompt]
