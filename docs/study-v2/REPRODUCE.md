# Reproduce the WeFit minor-project study

The inference and training experiments use Python 3.11, the pinned MLX Klein checkpoint, and an Apple Silicon Mac with 24 GB of unified memory. The complete studio includes a separate Windows FASHN renderer. Windows users can inspect the recorded study and use the offline review form. Running the Klein training and timing experiments requires the Mac renderer.

After installing the Mac renderer, open a terminal in its repository. Run `source .venv/bin/activate` so the `python` commands below use the environment containing MLX, MFLUX and PyTorch. For a classroom demo on either operating system, open `inside-klein.html` directly from the downloaded research kit.

## Keep the baseline

The public renderer and V1 report are archived at tag `v1.1.0`. The private studio baseline is tagged `studio-v1.1`. Run ordinary three-step upper-body inference with the adapter environment variable unset. The study retains the model’s original safetensors files. The hashes in the training record identify the exact baseline.

## Measure startup, warm requests and retries

Install the public renderer following its repository README. From its root, start a new worker from one terminal. This command writes the time at which the worker begins answering HTTP requests. Leave it running while the second terminal measures images.

```sh
python -m tools.time_startup --output artifacts/latency/cold.json
```

In the full studio use `--studio` to include its web server readiness time. Use a free model port and run a single resident Klein worker on this 24 GB machine. The download of the model happens during setup. The startup clock measures an application process starting with the weights available on disk. It does not include downloading several gigabytes or clearing the operating system’s file cache.

Create `cases.json` in the repository root with relative or absolute paths. This example uses the public face-blurred person and synthetic garment. Use different cases when evaluating a different workload.

```json
[{"case":"sample-top","person":"examples/person.png","garment":"examples/garment.png","photo_type":"flat-lay"}]
```

```sh
python -m tools.profile_latency cases.json --rounds 20 --output artifacts/latency
```

With four cases, use five rounds to obtain 20 warm requests. The tool records the first response separately and decodes each returned PNG. A successful warm wall time includes HTTP transfer of the complete output. The server timer covers parsing, sampling and composition. A failed request stays in the attempt count. If the crop triggers a larger render, both attempts are charged to that photo. The fallback rate uses requests that actually started with a crop.

## Train the output-layer adapter

Download the official Fashionpedia validation image archive and annotation JSON into the following paths. Keep photographs and intermediate features local. The split selection uses license ID 0, one eligible upper-body instance, foreground area, a deterministic ordering, duplicate file and URL checks, and a near-image check. The study’s train, selection and test splits are internal splits of that official validation edition. Human identity and the original model’s training-data overlap are unknown. [Fashionpedia downloads and annotation terms](https://github.com/cvdfoundation/fashionpedia)

```sh
mkdir -p artifacts/fashion
curl -fL https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json -o artifacts/fashion/annotations.json
curl -fL https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip -o artifacts/fashion/images.zip
python -m tools.adapter_study.prepare artifacts/fashion/annotations.json artifacts/fashion/images.zip artifacts/pairs
python -m tools.adapter_study.inventory model_src/klein-kv/weights artifacts/study/inventory.json
python -m tools.adapter_study.extract artifacts/pairs
python -m tools.adapter_study.zero_weights model_src/klein-kv/weights artifacts/pairs artifacts/study/zero-weights.json
python -m tools.adapter_study.importance artifacts/pairs artifacts/study
python -m tools.adapter_study.fit artifacts/pairs artifacts/study
python -m tools.adapter_study.evaluate artifacts/pairs artifacts/study
python -m tools.adapter_study.ablate artifacts/pairs artifacts/study
python -m tools.adapter_study.output_maps artifacts/pairs artifacts/study
python -m tools.adapter_study.visualize artifacts/study artifacts/inside-klein.html
```

Stop the inference worker before extraction and held-out evaluation to free its model memory. The ablation command also builds and evaluates a physically separate Q3/Q4 checkpoint under `artifacts/mixed-weights`. Allow roughly 5 GB of logical file space for this candidate. Unchanged files can share disk blocks on the same filesystem. Use `--skip-mixed` to run just the block and channel experiments. The commands load the original checkpoint from your local copy. The extraction reuses previously completed per-image files for the same manifest and model revision. The fitter runs on the CPU and changes the two adapter matrices. It picks an epoch on five validation photos. The compressed adapter is selected on those same validation photos. The final evaluation uses five different photos and actually decodes the output for each. The held-out comparison reruns one case with an all-zero adapter and checks its pixels against the frozen result.

For the same-garment task, the person photograph supplies the known target and the annotated product cutout. The model receives a working copy with that garment greyed out and the cutout on white. Matching that target measures reconstruction. A change of outfit on a new person is a different task.

## Replay the same photograph with different output adapters

During the Q4/mixed comparison, both methods attach the same probes. After the Q4 stopwatch stops, the study writes a private `last-step.safetensors` file next to each test photo. It contains the final layer's input, its original prediction, the image representation before the final denoising step and two noise-schedule values in native precision. These tensors can recreate the photograph. Keep them with the private inputs.

Once that file exists, the following command loads the image decoder alone, applies the supplied adapter, and produces a PNG. Replace the sample ID with an ID in your manifest. The changing part must be confined to the last output projection. A different person, product, seed or earlier transformer change needs another first render.

```sh
python -m tools.adapter_study.replay artifacts/pairs/11446/last-step.safetensors artifacts/replayed-shirt.png --adapter artifacts/study/upper-compact.npz --strength 1
```

The study compares replay pixels with full renders of the frozen, trained and compact settings. It also tries five strengths on validation photographs, chooses the lowest mean decoded garment error and evaluates that one setting on the five test photographs. Replays use the same native decoder as the full model. [Replay measurements](evidence/replay-runs.csv)

## Try the fitted weights in the studio

Run the ordinary setup first. From the repository root, set the optional path when starting the worker. The adapter applies to standard three-step tops. Detailed tops and trousers retain their usual configuration. The response header `X-TryOn-Adapter` contains the applied file’s SHA-256 or `none`. The studio performs its normal crop and hand checks for both configurations.

```sh
WEFIT_KLEIN_UPPER_ADAPTER="$PWD/docs/study-v2/adapters/upper-compact.npz" python -m uvicorn gpu_service.klein_server:app --host 127.0.0.1 --port 9000
```

For the full private studio, use the same environment variable with `python run_local.py --model klein`. To compare both methods on your own camera photographs:

```sh
python -m tools.adapter_study.studio_pairs cases.json docs/study-v2/adapters artifacts/native
```

This direct comparison records successes, rejections, latency, fallback use and how much the two images differ. It has no photograph of the person actually wearing the product, so these differences cannot be scored as fitting accuracy.

## Run an independent blind review

Create a manifest of prepared, anonymized photographs. Include a source, an isolated product and both actual outputs. Paths are relative to the manifest. Remove faces from photographs before supplying them.

```json
{"study":"standard-tops","methods":["frozen","compact"],"cases":[{"id":"shirt-one","source":"person.webp","product":"shirt.webp","frozen":"first.webp","compact":"second.webp"}]}
```

```sh
python -m tools.blind_review build review-cases.json artifacts/reviewer-01
python -m tools.blind_review build review-cases.json artifacts/reviewer-02
python -m tools.blind_review summarize --keys artifacts/reviewer-01/key.json artifacts/reviewer-02/key.json --responses returned/review-01.json returned/review-02.json --output artifacts/ratings.json
```

Send each person only their individual `review.html`. They can open it offline on Mac or Windows. The file embeds the anonymized references and both outputs, strips image metadata and randomizes the order. Reviewers rate neckline, shoulders, sleeves, fabric, anatomy and scene, with a “not visible” option. They download a JSON response at the end. The private key belongs with the organizer. The aggregator reports the number of scores behind each mean and the fraction of common image pairs on which two reviewers chose the same preference, including a tie. With fewer than two completed reviewers, agreement is undefined. Browser automation checks the form separately from human scores.

## Trace your own photo

The full studio links to **Under the hood** at `/research`. It opens a recorded inspector and needs no GPU to explore. To measure a personal photo through the app, generate or reopen a saved upper-body look on the ordinary local Mac worker. Click **Trace this look** next to **Download look**. The same worker runs a separate full-frame, three-step baseline measurement on those two saved inputs. The inspector opens with **Your local trace** selected. Choose **Download this trace** to keep the numbers. Tracing has its own wait time, and the ordinary photo generation path remains uninstrumented.

For the stand-alone command-line workflow, stop the normal worker before loading a second 9B instance, then run:

```sh
python -m tools.trace_garment person.jpg shirt.jpg --photo-type flat-lay --output artifacts/my-trace.json
```

To keep an offline copy that opens with your measured shirt already selected, run the following in the repository. The resulting HTML contains your numeric trace. The public inspector contains only the study examples.

```sh
python -m tools.adapter_study.visualize docs/study-v2/evidence artifacts/my-inspector.html --trace artifacts/my-trace.json
```

Open that JSON using **Open a local trace** in the inspector. The file contains block measurements, a coarse clothing outline, and five channels of the final feature grid. It contains no photographic pixels. Each value comes from a real, instrumented three-step full-frame run. Synchronizing probes changes the running time, so use the separate uninstrumented profiler for latency measurements. Restart the normal worker after tracing.

To regenerate the static figure of a single input-to-output connection, install Matplotlib (the recorded figure used 3.11.2). Choose any photo or channel contained in the published numeric feature file.

```sh
python -m pip install matplotlib==3.11.2
python -m tools.adapter_study.plot docs/study-v2/evidence/feature-maps.json artifacts/connection.png --photo 10471 --channel 89 --token 152 --connection 17
```

## Re-create the smaller backbone on its own

The project identifies six low-change blocks using three training photos. For two large linear projections per block it reads the real Q4 tensor, dequantizes it in small row chunks and stores an affine Q3 tensor with new scales and offsets. It writes a separate destination and preserves the original checkpoint. The pinned MFLUX loader infers each layer’s actual precision from the packed shape. A verification run checks both projections in all 24 single-stream blocks.

```sh
python -m tools.adapter_study.mixed_checkpoint model_src/klein-kv/weights artifacts/mixed-weights docs/study-v2/evidence/activity.json
KLEIN_WEIGHTS="$PWD/artifacts/mixed-weights" WEFIT_KLEIN_UPPER_ADAPTER="$PWD/docs/study-v2/adapters/upper-compact.npz" python -m uvicorn gpu_service.klein_server:app --host 127.0.0.1 --port 9000
```

Use a new output directory if the candidate already exists. The service reports its weight profile in `/health` and `X-TryOn-Weight-Profile`. Use the scope and errors in the V2 report when deciding whether to run the candidate. [MLX quantization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)

## Archive the full checkpoint without changing its weights

With Python 3.11 or newer, the following command packs the original model files, reads every file back from the archive, and compares SHA-256 hashes with the originals. The ZIP contains the model, tokenizer and indexes. The downloader’s hidden cache is excluded. The resulting ZIP stays on your computer. Installations use the extracted directory for inference.

```sh
python -m tools.adapter_study.archive model_src/klein-kv/weights artifacts/klein-q4.zip --measurements artifacts/archive.json
python -m zipfile -e artifacts/klein-q4.zip artifacts/restored-q4
```
