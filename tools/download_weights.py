"""Fetch the two pinned model components for the public Apple Silicon renderer."""
from pathlib import Path

from huggingface_hub import snapshot_download

from gpu_service.klein_prompts import MODEL_REPOSITORY, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
PARSER_REVISION = "1f80c34dbab321c5730dda5c3fea279fd3e97498"


def main():
    snapshot_download(
        MODEL_REPOSITORY, revision=MODEL_REVISION, allow_patterns=["transformer/*", "vae/*"],
        local_dir=ROOT / "model_src/klein-kv/weights", max_workers=2,
    )
    snapshot_download(
        "fashn-ai/fashn-human-parser", revision=PARSER_REVISION,
        allow_patterns=["config.json", "*.safetensors"],
        local_dir=ROOT / "model_src/fashn-vton/weights/human-parser",
    )


if __name__ == "__main__":
    main()
