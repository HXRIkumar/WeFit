# Read the raw measurements

All model runs took place locally on an Apple M4 MacBook Air with 24 GB of unified memory. Each file describes a different experiment.

| Evidence | What to read |
|---|---|
| `cold.json`, `requests.jsonl`, `summary.json` | One fresh process start, the first real photo and 20 subsequent serial requests. The downloaded model was present on disk. Use the warm rows for median and nearest-rank p95. |
| `archive.json` | Bytes on disk before and after lossless compression of the entire original model folder. Checksums verify that every archived file decompresses to the original bytes. |
| `manifest.json` | IDs, image hashes, source links, licences and internal train/validation/test assignments for the 24 Fashionpedia photographs. |
| `inventory.json`, `zero-weights.json` | Logical weight counts from tensor headers and an exact round trip of a bitmap encoding that omits numerical zeros from the final layer. Includes byte-exact DEFLATE archive sizes for both layouts. |
| `block-activity.csv`, `double-block-activity.csv`, `head-channels.csv` | Observed changes at each transformer block and the final-layer channel ranking. Block numbers start at zero. The channel measurements use training garment tokens. |
| `activity.json`, `feature-maps.json` | Selection of low-change blocks, selected spatial activation maps and outgoing weight examples used by the offline inspector. |
| `training.csv`, `training.json` | Epoch-by-epoch error, selection on the validation split, optimizer settings, and hashes of the frozen base and learned file. |
| `feature-extraction.csv` | Per-photo wall times printed by the original frozen feature extractor. These cover native rendering, target encoding and saving each feature file. Loading the model was outside these per-photo timers. |
| `compression.json`, `held-out.csv`, `evaluation.json` | Validation-selected rank and precision, followed by true rendered comparisons on the five held-out garments. Each image contributes equally to the summary. |
| `ablation.csv`, `ablation.json` | Fixed-seed, rotated-order comparisons of five inference methods on the same five test photos. |
| `output-maps.json` | Annotated garment pixel counts and sums of absolute RGB errors in each 16 × 16 patch, for all 25 held-out renders. These reproduce every garment score in the ablation table. The inspector compares any two methods without storing source pixels. |
| `mixed-runs.csv`, `mixed-checkpoint.json` | Measured physical file size and Q3/Q4 precision checks, with paired renders against the regular Q4 backbone using the same compact adapter. |
| `replay-runs.csv`, `replay.json` | Validation-only selection of adapter strength on decoded pixels. Fifteen held-out comparisons check whether a decoder-only replay reproduced the full frozen, trained and compact model outputs. The first render is a prerequisite. |
| `native-pairs.json`, `native-differences.json` | Full studio runtime output on private source and product photographs. Pixel differences measure change, and the actual target outfit was not photographed on the source person. |
| `studio-trace-check.json` | A browser registered in a separate local test database, uploaded the real blue product and person photo, generated a saved look, and traced it through the Mac worker. The exported values were compared with a separate command-line trace. |
| `blind-status.json` | Number of prepared independent review packets, actual respondents, valid ratings and agreement. An empty result means a human review is pending. |
| `SHA256.json` | SHA-256 hashes of all the numerical evidence files in this directory. |

Run `python -m tools.verify_study` in the repository root to recompute the reported values from the raw rows.
