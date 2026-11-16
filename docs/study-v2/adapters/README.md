# WeFit upper-body output adapters

These files are an academic derivative of Black Forest Labs’ FLUX.2 Klein 9B KV. Use the pinned MFLUX Q4 checkpoint at revision `d144dcb1f70540f581ed99930ee61aac21b5c6d6`. The loader checks the base revision, target, dimensions and finite weights. The model license and required notice accompany both files. [Model terms](LICENSE) · [Attribution](NOTICE)

## What they change

On the final step of a standard upper-body render, the native projection maps a 4,096-value patch representation to 128 predicted values. The adapter adds a learned correction using two matrices. The original projection and all earlier transformer operations stay frozen. The architecture uses the trained correction only on that last step.

- `upper-final-r4.npz` stores the trained down and up matrices in 32-bit floating point.
- `upper-compact.npz` stores the validation-selected approximation in 16-bit floating point. It is converted to 32-bit arithmetic for the small correction when loaded. The runtime keeps the main model in its existing Q4 format.

## Intended use and evidence

The training data contains 14 Fashionpedia images with one annotated upper garment. Five separate images selected the checkpoint and compression setting. Five additional images were used for evaluation. Each example reconstructs the known garment from a greyed source photo and its isolated cutout, at 256 × 320. Data provenance and image licensing are recorded per image. Raw Fashionpedia photography is downloaded separately. [Split manifest](../evidence/manifest.json) · [Fashionpedia](https://github.com/cvdfoundation/fashionpedia)

The numerical objective measures reconstruction in the VAE’s image representation. RGB errors measure similarity to the known garment on the held-out photographs. Changes to a camera photo and a different shop garment require separate review. The published native study covers four such cases on the Mac. Anatomical plausibility, strong pose changes and real-world sizing remain open questions. [Evaluation](../README.md) · [Reproduction commands](../REPRODUCE.md)

The five held-out photos cover three top/T-shirt/sweatshirt labels and two shirt/blouse labels. Validation contains five top/T-shirt/sweatshirt labels. Performance on jackets, trousers, other people and unseen camera conditions needs its own evaluation before using the adapter for those cases.
