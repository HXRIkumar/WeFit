# Reproduce the depth experiments

The recorded runs used a 24 GB M4 MacBook Air, Python 3.11, MLX 0.32.4, MFLUX 0.24.0 and SciPy 1.17.1. The editor used native arithmetic, the reference cache, compiled cached blocks, a fixed seed of 42 and three denoising steps. For comparable timings, run one renderer on the GPU. The public record and offline lab can be inspected on any desktop operating system.

## Reuse the fitted students

Follow the renderer setup in the repository README. Activate its virtual environment. The released students file contains just the newly learned coefficients. To build the two-block checkpoint, give the builder its training record and coefficients in one folder.

```sh
mkdir -p artifacts/v3/learned
cp docs/study-v3/evidence/training.json docs/study-v3/students/students.npz artifacts/v3/learned/
python -m tools.depth_study.checkpoint model_src/klein-kv/weights artifacts/v3/learned artifacts/v3/weights-student2 2
```

Use 12 in place of 2 and choose a different destination to reproduce the aggressive checkpoint. The builder rewrites the safetensors index, omits removed dense tensors and preserves the original block positions in a manifest. Shared model files use hard links when supported. Reported sizes count each file at its full length. To start a research worker, choose one of the following examples. Each command runs in its own terminal session.

```sh
# Physically smaller two-block student, three-step upper-body rendering
KLEIN_WEIGHTS="$PWD/artifacts/v3/weights-student2" python -m uvicorn gpu_service.klein_server:app --port 9000

# Full Q4, with two late dense evaluations skipped for standard tops
WEFIT_KLEIN_DEPTH_POLICY="$PWD/docs/study-v3/evidence/selected-policy.json" python -m uvicorn gpu_service.klein_server:app --port 9000
```

The ordinary worker uses the Q4 directory when these variables are unset. In the complete studio, apply the same environment setting to `python run_local.py --model klein`. The depth student was trained for three-step upper-body images. The late schedule also applies only to that profile. The full Q4 worker supports the original detailed and trouser settings.

## Train from the photographs

Install the scientific dependency in the renderer environment. The complete 256 × 320 experiment uses existing Fashionpedia images and masks with known targets. It samples the original V2 training images and discards the ten images used for V2 validation or testing. The three validation and five test photos in V3 are new to both studies’ model selection and evaluation. The internal split is by image. [Fashionpedia dataset and downloads](https://github.com/cvdfoundation/fashionpedia)

```sh
python -m pip install scipy==1.17.1
mkdir -p artifacts/fashion
curl -fL https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json -o artifacts/fashion/annotations.json
curl -fL https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip -o artifacts/fashion/images.zip
python -m tools.depth_study.cohort artifacts/fashion/annotations.json artifacts/fashion/images.zip docs/study-v2/evidence/manifest.json artifacts/v3/pairs
python -m tools.depth_study.capture artifacts/v3/pairs docs/study-v2/evidence/activity.json
VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m tools.depth_study.train artifacts/v3/pairs artifacts/v3/learned --rank 8
python -m tools.depth_study.run artifacts/v3
```

The recorder writes a private sample of 64 tokens for each of 12 blocks at each of three steps. Each token contains 4,096 incoming features and 4,096 update values. Roughly 1.06 GB of private training arrays accumulate across 14 photos. These intermediate records can describe the input photos. The release includes aggregate numbers, the fitted coefficients and masked publication figures.

The selection pass measures 15 methods on each of the three validation photographs. The method order rotates by photo. The validation thresholds are in the protocol. The resulting selection is written before testing. The 12-block stress test and its identity control are always tested. Pass `--cases my-cases.json` to add your own studio images. Each entry uses `case`, `person`, `garment` and `photo_type` with `flat-lay` or `model`. Supply full paths to your own photos. Their output remains in your local study directory.

## Check an independent model load

The next command starts a fresh interpreter, loads a real checkpoint, warms it on one training photo and times the five held-out photos. It compares every output channel with the corresponding saved intervention. For the release’s selected schedule, copy the study policy into the experiment folder first. Native proofs optionally accept `--case private-case.json` containing one case object.

```sh
cp docs/study-v3/evidence/selected-policy.json artifacts/v3/
python -m tools.depth_study.prove artifacts/v3/weights-student2 artifacts/v3 student2
python -m tools.depth_study.prove model_src/klein-kv/weights artifacts/v3 frozen
python -m tools.depth_study.prove model_src/klein-kv/weights artifacts/v3 last2
```

The full studio now sets OpenCV’s RNG seed once per composite. Earlier native images collected during the study allowed GrabCut’s color clusters to depend on which request ran before them. The repeat command reads the historical physical comparisons and rerenders each discrepant in-memory intervention with the corrected compositor.

```sh
python -m tools.depth_study.recheck_matte model_src/klein-kv/weights artifacts/v3 --case private-case.json
```

The secondary quantization experiment keeps every block and stores two projections per single-stream block at three bits. It was specified after initial validation results and before testing. All 24 single-stream blocks were included. No rank or precision choice was fitted on those five test photos.

```sh
python -m tools.adapter_study.mixed_checkpoint model_src/klein-kv/weights artifacts/v3/weights-q3-24 docs/study-v2/evidence/activity.json --blocks 24
python -m tools.depth_study.quantization artifacts/v3/weights-q3-24 artifacts/v3
```

To repeat the VAE projection experiment, copy `docs/study-v3/evidence/encoder-protocol.json` into `artifacts/v3`, then rerun the quantization command with `--cases my-cases.json`. It finishes the ordinary Q3 comparison first. It then writes the original and shortened VAE projection layers under `mean-projection/` and compares full images against those same Q3 renders. The trained teacher and checkpoint files stay in place.

## Use an exact copy with less disk space

The archive command writes a conventional ZIP and verifies every original file against its source. The measured ten-file archive includes the pinned tokenizer and README. A minimal public download contains the required transformer and VAE files and can produce a smaller archive. To recreate the ten-file bundle, also fetch `README.md` and `tokenizer/*` from the same pinned Hugging Face revision into the source directory. The V3 loader streams the four required safetensors shards directly from the ZIP, verifies the pinned digests and loads the original numerical weights. A permanent extracted model directory is unnecessary for a machine using this mode. The fixed instruction cache and human parser remain separate installation files.

```sh
python -m tools.adapter_study.archive model_src/klein-kv/weights artifacts/klein-q4.zip --measurements artifacts/archive-proof.json
KLEIN_WEIGHTS="$PWD/artifacts/klein-q4.zip" python -m uvicorn gpu_service.klein_server:app --port 9000
# Run separately once frozen renders exist in artifacts/v3.
python -m tools.depth_study.prove artifacts/klein-q4.zip artifacts/v3 archive
```

## Reuse the same photograph with a different product

The optional worker setting `WEFIT_KLEIN_REFERENCE_CACHE=1` memoizes the VAE encodings of identically prepared person or garment images. The cache keys include all input values, the shape and the dtype. It is limited to 8 MiB in process memory. Unused entries expire on access after ten minutes. A new crop, a different photo or changed lighting generally produces a different key. The editor still runs each denoising block for every new outfit. To reproduce the timed study, put the priming case first, followed by cases sharing either its person photograph or garment.

```sh
WEFIT_KLEIN_REFERENCE_CACHE=1 python -m uvicorn gpu_service.klein_server:app --port 9000
# In a separate session with exclusive use of the model device
python -m tools.depth_study.reference_reuse model_src/klein-kv/weights my-cases.json artifacts/reuse.json
```

## Try the Mac-hosted studio from a phone

Start the complete studio on the Mac with `python run_local.py --model klein`. For an Android USB demo, connect the phone to desktop Chrome using device debugging. Under `chrome://inspect/#devices`, enable port forwarding and map phone port `8000` to `127.0.0.1:8000` on the Mac. Open `http://localhost:8000/studio` in Android Chrome. The Mac performs inference. [Chrome’s USB port-forwarding instructions](https://developer.chrome.com/docs/devtools/remote-debugging/local-server)

For an iPhone or Android already joined to the same private Tailscale network as the Mac, run `tailscale serve 8000` on the Mac. Open the HTTPS address it prints, followed by `/studio`, on the phone. The model worker stays on the Mac’s loopback port. These are device test instructions. The recorded mobile-layout run used desktop Chromium with a 390 × 844 touch viewport. [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) · [Recorded browser test](evidence/archive-browser.json)

## Read and verify the published results

```sh
python -m tools.verify_depth
python -m tools.depth_study.visualize docs/study-v3 artifacts/depth-lab.html
```

The checker recomputes block means and selection, checks train/test image separation, recalculates per-photo medians and verifies exact pixel comparisons from fresh checkpoint loads. The HTML lab reads the same numeric records. A model download is unnecessary to run the checker. [Photo-level measurements](evidence/photo-metrics.csv) · [Per-block measurements](evidence/block-trace.csv)
