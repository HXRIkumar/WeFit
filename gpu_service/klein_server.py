"""Resident MLX service. One dedicated thread owns model loading and inference."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from .klein_profiles import steps_for
from .klein_runtime import garment_rgb
from .klein_prompts import MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
KEY = os.getenv("TRYON_HTTP_KEY", "")
log = logging.getLogger("tryon.klein")


@asynccontextmanager
async def lifespan(app):
    from .klein_runtime import KleinRuntime
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wefit-metal")
    app.state.worker = worker
    try:
        app.state.runtime = await asyncio.get_running_loop().run_in_executor(worker, lambda: KleinRuntime(
            Path(os.getenv("KLEIN_WEIGHTS", str(ROOT/"model_src/klein-kv/weights"))),
            ROOT/"model_src/klein-kv/prompt-cache",
            Path(os.getenv("FASHN_WEIGHTS", str(ROOT/"model_src/fashn-vton/weights")))/"human-parser"))
        yield
    finally:
        # Release Metal objects on their owning worker before joining it.
        if hasattr(app.state, "runtime"):
            await asyncio.get_running_loop().run_in_executor(worker, lambda: del_runtime(app))
        worker.shutdown(wait=True)


def del_runtime(app):
    import gc
    import mlx.core as mx
    mx.synchronize()
    del app.state.runtime
    gc.collect(); mx.clear_cache()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    weight_profile = getattr(app.state.runtime, "weight_profile", "q4")
    return {"ready": True, "instance_id": os.getenv("WEFIT_INSTANCE_ID", ""),
            "model": "flux2-klein-9b-kv-" + weight_profile, "device": "mlx-metal", "native_long_side": 768,
            "trained_steps": 4, "supported_steps": [3] if getattr(app.state.runtime, "depth_profile", False) else [3, 4], "fixed_instruction_cache": True, "trained_reference_kv_cache": True,
            "weight_profile": weight_profile, "weight_storage": getattr(app.state.runtime, "weight_storage", "directory"),
            "upper_three_step_policy": (getattr(app.state.runtime, "temporal_policy", None) or {}).get("method"),
            "reference_latent_cache": getattr(app.state.runtime, "reference_cache", None) is not None,
            "upper_adapter_sha256": getattr(app.state.runtime, "upper_adapter_sha256", ""),
            "compute_precision": os.getenv("KLEIN_COMPUTE_PRECISION", "native"),
            "compiled_cached_blocks": os.getenv("KLEIN_COMPILE_BLOCKS", "1") == "1"}


async def read_reference_pair(person, garment):
    try:
        with Image.open(io.BytesIO(await person.read())) as uploaded:
            photo = ImageOps.exif_transpose(uploaded).convert("RGB")
        with Image.open(io.BytesIO(await garment.read())) as uploaded:
            item = garment_rgb(uploaded)
        return photo, item
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise HTTPException(400, "Use valid JPEG, PNG or WebP images") from error


@app.post("/trace")
async def trace(person: UploadFile = File(...), garment: UploadFile = File(...),
                garment_photo_type: str = Form("flat-lay"), authorization: str = Header("")):
    if KEY and authorization != f"Bearer {KEY}":
        raise HTTPException(401, "bad key")
    if garment_photo_type not in ("flat-lay", "model"):
        raise HTTPException(422, "Choose a product photo or a garment worn by a person.")
    photo, product = await read_reference_pair(person, garment)

    def measure():
        from tools.trace_garment import trace_on_runtime
        runtime = app.state.runtime
        if runtime.weight_profile != "q4" or runtime.upper_adapter is not None:
            raise ValueError("This research trace uses the frozen Q4 worker. Start its ordinary preset.")
        return trace_on_runtime(runtime, photo, product, garment_photo_type, ROOT / "docs/study-v2/evidence")

    try:
        return await asyncio.get_running_loop().run_in_executor(app.state.worker, measure)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except MemoryError as error:
        raise HTTPException(503, "The Mac worker reached its memory limit during tracing.") from error


@app.post("/tryon")
async def tryon(person: UploadFile = File(...), garment: UploadFile = File(...),
                cloth_type: str = Form("upper"), quality: str = Form("balanced"),
                render_region: str = Form(os.getenv("KLEIN_REGION_DEFAULT", "auto")),
                steps: int | None = Form(None),
                conditioning: str = Form("neutral"),
                garment_photo_type: str = Form("flat-lay"), authorization: str = Header("")):
    if KEY and authorization != f"Bearer {KEY}":
        raise HTTPException(401, "bad key")
    if cloth_type not in ("upper", "lower"):
        raise HTTPException(422, "This model supports shirts, T-shirts and trousers.")
    if quality not in ("fast", "balanced", "detail") or garment_photo_type not in ("flat-lay", "model"):
        raise HTTPException(422, "Unknown image setting")
    if render_region not in ("full", "auto"):
        raise HTTPException(422, "Unknown rendering region")
    steps = steps_for(cloth_type, quality) if steps is None else steps
    if steps not in (3, 4):
        raise HTTPException(422, "Use the four-step profile, or three steps for an explicit research comparison.")
    if conditioning not in ("original", "neutral"):
        raise HTTPException(422, "Unknown person conditioning")
    p, g = await read_reference_pair(person, garment)
    queued = time.perf_counter()

    def run():
        started = time.perf_counter()
        image, timing = app.state.runtime.run(p, g, cloth_type, garment_photo_type, quality, render_region, steps, conditioning)
        return image, timing, started

    try:
        output, timing, started = await asyncio.get_running_loop().run_in_executor(app.state.worker, run)
    except ValueError as error:
        from .klein_runtime import RenderRejected
        headers = {}
        if isinstance(error, RenderRejected):
            headers["X-TryOn-Initial-Region"] = error.initial_region
            headers["X-TryOn-Region-Fallback"] = str(error.retried).lower()
        raise HTTPException(422, str(error), headers=headers) from error
    except MemoryError as error:
        raise HTTPException(503, "The renderer reached its memory limit. Close other large apps before trying again.") from error
    finished = time.perf_counter()
    log.info("klein category=%s queue=%.2fs inference=%.2fs", cloth_type, started-queued, finished-started)
    buf = io.BytesIO(); output.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={
        "X-TryOn-Model": "flux2-klein-9b-kv-" + timing.get("weight_profile", "q4"), "X-TryOn-Canvas-Height": str(timing["height"]),
        "X-TryOn-Model-Revision": MODEL_REVISION,
        "X-TryOn-Steps": str(timing.get("steps", steps)), "X-TryOn-Guidance": "1.0",
        "X-TryOn-Region": timing.get("render_region", "full"),
        "X-TryOn-Initial-Region": timing.get("initial_region", timing.get("render_region", "full")),
        "X-TryOn-Region-Fallback": str(timing.get("region_fallback", False)).lower(),
        "X-TryOn-Render-Pixels": str(timing.get("width", output.width)*timing["height"]),
        "X-TryOn-Peak-MLX-GB": f"{timing.get('peak_mlx_gb', 0):.2f}",
        "X-TryOn-Compute-Precision": os.getenv("KLEIN_COMPUTE_PRECISION", "native"),
        "X-TryOn-Conditioning": timing.get("conditioning", "original"),
        "X-TryOn-Weight-Profile": timing.get("weight_profile", "q4"),
        "X-TryOn-Reference-Hits": str(timing.get("reference_hits", 0)),
        "X-TryOn-Reference-Misses": str(timing.get("reference_misses", 0)),
        "X-TryOn-Adapter": timing.get("adapter_sha256") or "none",
        "X-TryOn-Queue-Seconds": f"{started-queued:.2f}", "X-TryOn-Inference-Seconds": f"{finished-started:.2f}",
        "X-TryOn-Preprocess-Seconds": f"{timing['preprocess_seconds']:.2f}",
        "X-TryOn-Sampling-Seconds": f"{timing['sampling_seconds']:.2f}",
        "X-TryOn-Postprocess-Seconds": f"{timing['postprocess_seconds']:.2f}",
    })
