# Attribution and model terms

A collaborator supplied the original fashion application. The complete inherited studio is maintained separately in a private repository. This public renderer contains the WeFit ML integration, benchmark tools, face-redacted examples and engineering study.

Black Forest Labs trained FLUX.2 Klein 9B KV and its step-distilled image model. The MFLUX community provided the four-bit MLX checkpoint, pinned to `d144dcb1f70540f581ed99930ee61aac21b5c6d6`. Its non-commercial, non-production model terms and required notice are bundled with the generic instruction embeddings. Model checkpoints are downloaded from the original repositories. [FLUX model license](model_src/klein-kv/LICENSE) · [Model notice](model_src/klein-kv/NOTICE) · [Community checkpoint](https://huggingface.co/mflux-community/flux2-klein-9b-kv-mflux-q4)

The renderer imports MFLUX 0.24.0 and MLX 0.32.4. Its compiled block wrappers adapt MFLUX calculations. The corresponding MIT notice is included. [MFLUX license](licenses/MFLUX-LICENSE) · [MFLUX](https://github.com/mflux-community/mflux) · [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)

FASHN supplied the human parser, pinned to `1f80c34dbab321c5730dda5c3fea279fd3e97498`. Its model card lists the NVIDIA SegFormer license. The parser runs separately from the garment generator. [FASHN model card and terms](https://huggingface.co/fashn-ai/fashn-human-parser)

The example source is an irreversibly face-blurred derivative of a supplied image. The navy garment in the quick start is a synthetic fixture. The report uses genuine application outputs with publication-only face redaction. Real shop photos used as garment references and the unredacted portraits are excluded from this distribution.
