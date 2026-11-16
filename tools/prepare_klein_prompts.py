"""Encode two fixed instructions once, then optionally release the 8B encoder.

Run in the optional MFLUX environment. These prompts contain no uploaded image
or garment descriptions. Float32 export preserves every BF16 embedding value.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from gpu_service.klein_prompts import INSTRUCTIONS, MODEL_REPOSITORY, MODEL_REVISION, instruction_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,default=Path('model_src/klein-kv/weights'))
    parser.add_argument('--output',type=Path,default=Path('model_src/klein-kv/prompt-cache'))
    parser.add_argument('--drop-text-encoder',action='store_true')
    args=parser.parse_args()
    import mlx.core as mx
    import numpy as np
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
    from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    class EncoderOnly(Flux2KleinWeightDefinition):
        @staticmethod
        def get_components():
            return [c for c in Flux2KleinWeightDefinition.get_components() if c.name=='text_encoder']
        @staticmethod
        def get_download_patterns():
            return ['text_encoder/*','tokenizer/*']

    started=time.perf_counter();mx.set_cache_limit(512*1024**2)
    config=ModelConfig.flux2_klein_9b_kv()
    weights=WeightLoader.load(weight_definition=EncoderOnly,model_path=str(args.weights.resolve()))
    encoder=Qwen3TextEncoder(**config.text_encoder_overrides)
    bits=WeightApplier.apply_and_quantize(weights=weights,quantize_arg=4,weight_definition=EncoderOnly,
                                        models={'text_encoder':encoder})
    tokenizer=TokenizerLoader.load_all(EncoderOnly.get_tokenizers(),model_path=str(args.weights.resolve()))['qwen3']
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'model_repository':MODEL_REPOSITORY,'model_revision':MODEL_REVISION,
              'mflux_version':importlib.metadata.version('mflux'),'quantization_bits':bits,
              'source_dtype':'bfloat16','stored_dtype':'float32 (exact BF16 values)','instructions':{}}
    for kind,prompt in INSTRUCTIONS.items():
        embeds,ids=Flux2PromptEncoder.encode_prompt(prompt=prompt,tokenizer=tokenizer,text_encoder=encoder)
        mx.eval(embeds,ids)
        data=np.asarray(embeds.astype(mx.float32));positions=np.asarray(ids)
        path=args.output/f'{kind}.npz';tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prompt_embeds=data,text_ids=positions);tmp.replace(path)
        with np.load(path,allow_pickle=False) as saved:
            restored=mx.array(saved['prompt_embeds']).astype(mx.bfloat16)
            assert np.array_equal(data,np.asarray(restored.astype(mx.float32))), 'Instruction embedding changed on export'
            assert np.array_equal(positions,saved['text_ids'])
        manifest['instructions'][kind]={'text':prompt,'sha256':instruction_hash(prompt),
            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(data.shape),'roundtrip_exact':True}
        print(f'{kind}: instruction encoded and exact BF16 roundtrip verified',flush=True)
    manifest['seconds']=round(time.perf_counter()-started,2)
    manifest['peak_mlx_gb']=round(mx.get_peak_memory()/1e9,2)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.drop_text_encoder:
        encoder_path=args.weights/'text_encoder'
        assert encoder_path.name=='text_encoder' and encoder_path.parent.resolve()==args.weights.resolve()
        del encoder,weights;mx.clear_cache()
        shutil.rmtree(encoder_path)
        print('Released the downloadable text-encoder checkpoint after verified cache export.',flush=True)
    print(json.dumps({'cache':str(args.output.resolve()),'seconds':manifest['seconds'],
                      'peak_mlx_gb':manifest['peak_mlx_gb']},indent=2))


if __name__=='__main__':main()
