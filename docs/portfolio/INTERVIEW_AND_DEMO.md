# Explaining WeFit in an interview

## A thirty second introduction

“I started with a collaborator’s virtual fitting app. I built and evaluated the Apple Silicon rendering path. It takes a person photo and a garment photo, finds the clothing, and uses a pretrained 9-billion-parameter image editor to render the change. I cached repeated work, limited the area the model redraws and added checks for bad edits. With the same shirt, seed and three rendering steps, the clothing region reduced median model-service time from 100.90 to 83.17 seconds across three pairs on one M4 Air.”

## What do the terms mean?

| Term | Explanation you can say aloud |
|---|---|
| Parameter | A number learned during training. “9B” means about nine billion such numbers in the image transformer. Layer count is a different quantity. |
| Inference | Running a trained model on a new example. We changed its runtime and what it sees during each request. |
| Fine-tuning | Using examples, a loss and an optimizer to change learned weights or a trainable adapter. The WeFit checkpoint stayed frozen. |
| Conditioning | The photos and instructions that guide the edit. We neutralize old shirt pixels in a working copy to stop their logo or colour from surviving. |
| Segmentation | Giving each pixel a body or clothing label. It helps separate a shirt from a shop model and locate the editing region. |
| Latent and VAE | The image becomes a smaller grid of numbers. The variational autoencoder converts between those numbers and pixels. |
| Transformer and attention | The image editor learns relationships between parts of the inputs. Attention lets a generated patch use relevant reference information, such as the product collar. |
| Embedding | A numeric representation. The text encoder turns each fixed instruction into numbers once. The runtime loads exactly those values. |
| K/V cache | Keys identify relevant reference features and values carry their information. The upstream model computes them on the first step. We store the needed slices for later steps in the same request. |
| Q4 quantization | The community checkpoint stores eligible weights with four-bit precision to reduce storage and memory traffic. We adopted that checkpoint. |
| Distillation | Upstream training taught the released model to generate in four steps. Our three-step top preset is a runtime choice. |
| Region of interest | The picture area containing the garment and nearby body pixels. Rendering fewer pixels costs less, and a crop or hand failure can trigger a full-frame retry. |

## Which work was yours?

The original collaborator supplied the web application, account and job infrastructure, geometric renderer and optional model adapters. Black Forest Labs trained Klein, the MFLUX community supplied the MLX Q4 conversion, and FASHN supplied the human parser. The WeFit contribution is the local integration, exact text caching, memory-bounded reference reuse, compiled execution, clothing conditioning and region rendering, source preservation, failure handling and measured comparisons.

If an interviewer asks what was switched off, explain the text encoder. We computed its two fixed outputs in advance. The image transformer and its decoder still run. Arbitrary free-text instructions would require another embedding. A Mac also restarted during an early unrestricted experiment. Its log recorded 94 seconds without a watchdog check-in and low swap. Subsequent bounded-cache runs completed. The log leaves the initiating cause uncertain.

## Show a reviewer the project

1. Open the [public project and setup instructions](https://github.com/HXRIkumar/WeFit). Show the archive overlay and the current blue shirt. Trace the collar position and the original dark fabric above the overlay.
2. Open the engineering report and its architecture figure. Explain the source copy, the product reference and the restored background. A generated forearm can look plausible even though the source sleeve hid its actual shape.
3. Open the [six timing rows](evidence/region-runs.csv). The same inputs, seed, model and steps were used in each pair. The clothing crop rendered 25% fewer pixels. The observed reduction in median service time was 17.6%. Each pair used a single subject.
4. For a live demonstration, use the public renderer with the included face-blurred input and synthetic garment fixture, or sign into the complete local studio and use your own photographs. In the studio, capture with the timer, add the garment, generate and move the comparison divider.
5. Share the public repository URL with a recruiter. The full studio is in the owner's [private repository](https://github.com/HXRIkumar/wefit-studio). The owner can invite a reviewer under **Settings → Collaborators**. The reviewer can then clone it and follow its Mac or Windows instructions.

## A résumé bullet

Built and benchmarked a local virtual try-on pipeline around a frozen 9B image editor using MLX, semantic segmentation and reference caching. Clothing-region rendering reduced median inference time by 17.6% in three matched pairs on an Apple M4 Air.

## A research extension

Choose a testable question about collar alignment, garment fidelity or latency. Use consenting subjects and licensed garments. Keep each person's photos in one split and hold out garments to test generalization. Photograph the same people actually wearing target garments when evaluating physical correspondence. Fix inputs, seed, precision and step count. Report cold startup, warm median and p95 over repeated runs, memory, retries and failures.

For visual review, randomize the order, hide the method, and ask independent reviewers to rate collars, cuffs, logos, anatomy and background preservation against the source and target. A V2 training experiment can fit low-rank adapter weights (LoRA) on the training split, then compare the adapter with the frozen baseline on held-out inputs. V1 used the existing weights throughout. The V1 study contains one subject and supplies no population accuracy or physical-sizing score.

## Questions at the demo table

**Which laptop ran the results?** An Apple M4 MacBook Air with 24 GB of unified memory ran the Klein renderer. The full studio also has a Windows setup for an NVIDIA GPU using the FASHN renderer. The owner’s Windows laptop still needs a real generation check. Its speed and output can differ.

**How do I explain the two large numbers on page one?** An early Mac model run took 251.87 seconds to draw a large image. A later end-to-end blue-shirt request took 56.36 seconds. We changed image size, step count and editing area between those observations. The original ZIP’s AI generation time is unknown. For a controlled experiment, show the six paired runs on page four.

**When does the stopwatch start?** For a cold start, start it when you launch the model program and stop it when the program says it is ready. Time the first photo separately. For a warm run, send the next photo while the model remains loaded. Sort the times. The median describes the middle request. With 20 warm runs, the nearest-rank p95 is the 19th fastest. It helps answer how long a slower request takes.

**What is a fallback?** If a crop clips fabric or changes hand positions, the program can redraw the whole picture. Count those retries and divide by the number of requests that used an automatic crop. Write the numerator and denominator so reviewers can see the sample size.

**What did we train in V1?** The image model already knew how to edit pictures. We programmed what inputs it received, how much of the picture it drew, and how results were checked. LoRA would add two small trainable matrices to a chosen model layer. A training loss measures mistakes on labelled examples. An optimizer changes the new matrices to reduce those mistakes while the original weights stay fixed.

Follow [the studio request](diagrams/studio.svg) from the browser to a saved job. Then follow [the Mac image renderer](diagrams/klein-renderer.svg) from the uploaded photos to the final pixels.
