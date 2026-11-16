"""Local, serialized photographic garment editing with source-pixel preservation."""
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import os
import tempfile
import time
import uuid

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .photo_preserve import preserve_reference_edit
from .result_checks import reject_invented_hands
from .klein_regions import editing_region, garment_reaches_crop_edge


class RenderRejected(ValueError):
    """A completed generation failed the crop or anatomy check."""
    def __init__(self, message, initial_region, retried):
        super().__init__(message)
        self.initial_region = initial_region
        self.retried = retried


def canvas_size(size, long_side=1024):
    """Keep the source composition, snapped to the renderer's 16-pixel grid."""
    scale = long_side / max(size)
    return tuple(max(32, round(side * scale / 16) * 16) for side in size)


def garment_rgb(upload):
    oriented = ImageOps.exif_transpose(upload)
    if "A" not in oriented.getbands() and "transparency" not in oriented.info:
        return oriented.convert("RGB")
    rgba = oriented.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    return white.convert("RGB")


def garment_reference(image, kind, photo_type, parse):
    """Isolate a model-worn garment without transferring the reference person."""
    image = image.convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    if photo_type == "flat-lay":
        return image
    labels = parse(np.asarray(image))
    selected = np.isin(labels, {"upper": [3, 4, 10], "lower": [5, 6, 7]}[kind])
    ys, xs = np.where(selected)
    if len(xs) < 100:
        raise ValueError("Could not find that garment. Use a garment-only image or a clearer reference.")
    matte = Image.fromarray(selected.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(.6))
    clean = Image.composite(image, Image.new("RGB", image.size, "white"), matte)
    pad = max(5, round(max(xs.max()-xs.min(), ys.max()-ys.min()) * .04))
    return clean.crop((max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                       min(image.width, int(xs.max())+pad+1), min(image.height, int(ys.max())+pad+1)))


def neutral_clothing_condition(source, labels, kind):
    """Remove old garment graphics from the reference, retaining body geometry."""
    import cv2
    selected = np.isin(labels, {"upper": [3, 4, 10], "lower": [5, 6, 7]}[kind]).astype(np.uint8)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(selected)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    # Preserve real exposed skin and overlapping hands/bags inside a garment's
    # silhouette; small parser holes such as a face-like printed badge are gone.
    filled[np.isin(labels, [8, 12, 13, 14, 15, 16])] = 0
    matte = Image.fromarray(filled*255).filter(ImageFilter.GaussianBlur(.7))
    return Image.composite(Image.new("RGB", source.size, (140, 140, 140)), source, matte)


class KleinRuntime:
    """Create and use this object on one worker thread; MLX owns its Metal stream."""
    def __init__(self, weights_path: Path, cache_path: Path, parser_path: Path):
        import mlx.core as mx
        import torch
        from fashn_human_parser import FashnHumanParser
        from .klein_model import CachedInstructionKlein

        torch.set_num_threads(4)
        self.weight_profile = "q4"
        manifest = weights_path / "wefit-mixed.json"
        if manifest.is_file():
            from .klein_prompts import MODEL_REVISION
            info = json.loads(manifest.read_text())
            if info.get("base_revision") != MODEL_REVISION or info.get("profile") != "wefit-q3-six-blocks":
                raise ValueError("Unrecognized local mixed-precision checkpoint.")
            self.weight_profile = info["profile"]
        self.model = CachedInstructionKlein(weights_path, cache_path)
        mx.eval(self.model.parameters())
        self.parser = FashnHumanParser(model_id=str(parser_path), device="cpu")
        self.person_cache = OrderedDict()
        self.upper_adapter = None
        self.upper_adapter_sha256 = ""
        adapter_path = os.getenv("WEFIT_KLEIN_UPPER_ADAPTER")
        if adapter_path:
            from .klein_adapter import Adapter
            path = Path(adapter_path)
            self.upper_adapter = Adapter.load(path)
            self.upper_adapter_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def _person(self, person, size):
        source = ImageOps.exif_transpose(person).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        key = (size, hashlib.sha256(source.tobytes()).digest())
        now = time.monotonic()
        for stale in [k for k, (when, _) in self.person_cache.items() if now-when > 300]:
            self.person_cache.pop(stale)
        if key in self.person_cache:
            self.person_cache.move_to_end(key)
            return source, self.person_cache[key][1], True
        labels = self.parser.predict(np.asarray(source))
        if np.count_nonzero(labels) < labels.size * .005:
            raise ValueError("No person found. Use a clear, front-facing photo with your torso visible.")
        self.person_cache[key] = (now, labels)
        while len(self.person_cache) > 4:
            self.person_cache.popitem(last=False)
        return source, labels, False

    def run(self, person, garment, kind, photo_type, quality, region_mode="full", steps=4,
            conditioning="neutral"):
        import mlx.core as mx
        from .klein_prompts import INSTRUCTIONS

        use_adapter = self.upper_adapter is not None and kind == "upper" and steps == 3
        started = time.perf_counter()
        mx.reset_peak_memory()
        # The tested 768 profile retained the reference design more reliably
        # than the much slower 1024 trial. Do not upscale a smaller generation.
        # More steps are not a detail control for this distilled checkpoint.
        size = canvas_size(person.size, long_side=768)
        source, labels, cached = self._person(person, size)
        conditioned = neutral_clothing_condition(source, labels, kind) if conditioning == "neutral" else source
        reference = garment_reference(garment, kind, photo_type, self.parser.predict)
        region = editing_region(labels, kind, region_mode)
        initial_region = "full" if region.full_frame else "clothing"
        person_reference = region.reference(conditioned)
        person_reference.thumbnail((768, 768), Image.Resampling.LANCZOS)
        with tempfile.TemporaryDirectory(prefix="wefit-reference-") as folder:
            ppath, gpath = Path(folder)/"person.png", Path(folder)/"garment.png"
            person_reference.save(ppath); reference.save(gpath)
            prepared = time.perf_counter()
            try:
                fallback = False
                generation_seconds = 0.0
                while True:
                    render_started = time.perf_counter()
                    context = nullcontext()
                    if use_adapter:
                        from .klein_adapter import final_projection
                        context = final_projection(self.model, steps, adapter=self.upper_adapter)
                    with context:
                        result = self.model.generate_image(seed=42, prompt=INSTRUCTIONS[kind],
                            num_inference_steps=steps, width=region.size[0], height=region.size[1], guidance=1.0,
                            image_paths=[ppath, gpath], use_kv_cache=True)
                    generation_seconds += time.perf_counter()-render_started
                    generated = region.composite(source, result.image.convert("RGB"))
                    generated_labels = self.parser.predict(np.asarray(generated))
                    try:
                        if garment_reaches_crop_edge(generated_labels, kind, region):
                            raise ValueError("The garment reaches the crop edge")
                        reject_invented_hands(labels, generated_labels)
                        break
                    except ValueError as error:
                        if region.full_frame:
                            raise RenderRejected(str(error), initial_region, fallback) from error
                        # Never deliver a visibly clipped ROI to save time.
                        # This fallback is logged and included in wall time.
                        fallback = True
                        region = editing_region(labels, kind, "full")
                        person_reference = conditioned.copy()
                        person_reference.thumbnail((768, 768), Image.Resampling.LANCZOS)
                        person_reference.save(ppath)
                        mx.clear_cache()
                if os.getenv("KLEIN_DEBUG_DIR"):
                    directory = Path(os.environ["KLEIN_DEBUG_DIR"])
                    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                    generated.save(directory/f"{uuid.uuid4().hex}-{kind}-{steps}-raw.png")
                output = preserve_reference_edit(source, generated, labels, kind, generated_labels)
                return output, {"preprocess_seconds": prepared-started, "sampling_seconds": generation_seconds,
                    "postprocess_seconds": time.perf_counter()-prepared-generation_seconds,
                    "width": region.size[0], "height": region.size[1],
                    "export_size": size, "region_box": region.box, "initial_region": initial_region, "region_fallback": fallback,
                    "render_region": "full" if region.full_frame else "clothing",
                    "conditioning": conditioning, "weight_profile": self.weight_profile, "adapter_sha256": self.upper_adapter_sha256 if use_adapter else "",
                    "steps": steps, "person_cached": cached, "peak_mlx_gb": mx.get_peak_memory()/1e9}
            finally:
                mx.synchronize()
                mx.clear_cache()

