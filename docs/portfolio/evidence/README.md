# Reading the evidence

The CSV preserves the request order. Each pair contains one full-frame request and one automatically selected clothing crop. The six responses report the same pinned revision, person and garment hashes, three steps, seed 42 in the service, neutral conditioning, and 0.00 s queue wait. Use the service time column for the published comparison. It includes parsing, generation and composition. A local client timer can also include file handling and image transfer. Cold starts, power and temperature were outside this experiment.

The JSON records the original ZIP fingerprint, a path-by-path comparison with the pre-portfolio tested release, actual saved-job timings for the three supplied images, and the separate 1024-pixel trial. Before publication, each clipboard output was converted to RGB and matched byte-for-byte against its stored app result. Original uploads, database records and unredacted exports stay on the test computer. Figure checksums refer to the distributable, flattened images. Camera validation used a simulated camera feed.

## Replay the handoff

For the archive replay, use the collaborator’s original ZIP with SHA-256 `614ef5e1442a3b77ef336e1f86df1b18a710d08949cd45e8322d2943faf8909d`. Extract it to a separate folder. Use Python 3.11 in an environment with its base requirements, NumPy, Pillow, OpenCV and pydantic-settings. The recorded replay had OpenCV available and no Python MediaPipe package installed. Leave the example environment file uncopied. From the extracted `tryon` folder run this example with local input paths.

```python
from pathlib import Path
from time import perf_counter
from app.engine import get_engine, OverlayEngine

engine = get_engine()
assert isinstance(engine, OverlayEngine)
person, garment = Path("person.jpg"), Path("shirt.jpg")
for index in range(3):
    start = perf_counter()
    output = engine.run(str(person), str(garment), "shirt", hq=False,
                        adjust={"scale": 1.0, "dx": 0.0, "dy": 0.0})
    print(f"{perf_counter() - start:.3f} seconds")
    if index == 0:
        output.save("replayed-overlay.png")
```

This reconstructs the original CPU configuration. Reproducing the collaborator’s historical AI setup would require its backend selection, reference inputs, hardware and timing record. To review the working code for the current method, start with the [public renderer](https://github.com/HXRIkumar/WeFit/tree/main/gpu_service).
