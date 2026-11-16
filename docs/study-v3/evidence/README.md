# What each measurement establishes

| File | What was collected |
| --- | --- |
| `manifest.json`, `protocol.json` | Image hashes, internal splits, planned interventions, image size, seed and selection thresholds. |
| `calibration.json`, `block-trace.csv` | All 1,344 original block visits, with incoming-to-outgoing relative RMS. The CSV uses joint RMS for dual-stream blocks. |
| `training.json` | Per-step teacher prediction error before and after fitting, rank, learned parameter count and trace-derived block order. Training loss is measured on training tokens. |
| `refit.json` | An independent execution of the training command on the recorded activations, with coefficient counts, maximum difference and checksums for both fitted files. |
| `validation.json`, `selection.json`, `selected-policy.json` | All 45 candidate renders, the decision made before test evaluation, and the runnable selected step schedule. |
| `test.json`, `summary.json`, `photo-metrics.csv` | Matched fixed-seed garment reconstructions. The CSV also includes the secondary Q3 experiment. The raw rows contain RGB digests, wall time, peak MLX allocation, target error and reference-relative similarity. |
| `native.json` | Full studio outputs, rejection and retry status for private camera/product pairs. A saved mask labels the frozen garment for reference-relative comparison. |
| `checkpoints.json` | Physical model sizes, original indices, logical parameter counts, exact shard hashes and SHA-256 of every published student tensor. |
| `deployment-files.json` | Actual clothing-parser and fixed-instruction files required alongside the Q4 archive, with lengths, digests and total stored model assets. |
| `physical.json` | Fresh process for each physical model or policy, model load and warmup, five test renders and one camera trial. Channel counts compare each result to its corresponding research intervention. |
| `matte-diagnostic.json`, `native-reload-audit.json` | Isolation of OpenCV's final edge-compositing randomness, followed by new full camera runs and pixel comparisons for every previously discrepant method. |
| `quant-protocol.json`, `quantization.json` | Predeclared secondary arm covering all single-stream blocks, eight labelled photos and two full-size camera cases. |
| `encoder-protocol.json`, `unused-variance.json` | A separate code-proven unused output of the VAE. Saved 64- and 32-row projection files, actual measured encoder activations and full renders with the same Q3 transformer. |
| `archive-stream.json`, `archive-browser.json` | Tensor-by-tensor verification of the streamed contents, followed by a real mobile-width browser upload, generation and downloaded JPEG on the Mac-hosted archive worker. |
| `reference-reuse.json` | One priming request and three paired full-resolution requests using the original encoder or a bounded cache. Each pair changes the person or product from the priming request. The record contains hit counts, wall time and exact RGB comparisons. |
| `counts.json`, `SHA256.json` | Trial totals and integrity hashes of the numerical evidence. |

Times are local wall seconds. MLX peak memory counts allocations tracked by the Metal runtime. It excludes Python, the CPU parser and the operating system. One megabyte is 1,000,000 bytes. Each photograph contributes equally to an aggregate image score. Garment SSIM uses a 7 × 7 local window independently per RGB channel at annotated garment centers. The strict pixel measure requires all three channels to lie within eight levels of the frozen render. The lossless archive figure includes ancillary files that were packed with the model.
