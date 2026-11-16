"""Restore source regions without reconstructing the new editor's room pixels.

The photographic editor already synthesizes newly exposed background. Filling
those gaps from a semantic silhouette can spread old dark clothing into them.
Keep the generated gap and refine the source head matte against image colors.
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter

HEAD = [1, 2, 9, 11]
TARGETS = {"upper": [3, 4, 10], "lower": [5, 6, 7]}


def refine_matte(source, selected):
    selected = np.asarray(selected, np.uint8)
    if selected.sum() < 25:
        return selected
    radius = max(3, round(source.width * .012))
    kernel = np.ones((radius*2+1, radius*2+1), np.uint8)
    expanded = cv2.dilate(selected, kernel)
    core = cv2.erode(selected, kernel)
    if not core.any():
        return selected
    ys, xs = np.where(expanded)
    box = (max(0, xs.min()-radius), max(0, ys.min()-radius),
           min(source.width, xs.max()+radius+1), min(source.height, ys.max()+radius+1))
    x0, y0, x1, y1 = map(int, box)
    mask = np.full(selected.shape, cv2.GC_BGD, np.uint8)
    mask[expanded > 0] = cv2.GC_PR_BGD
    mask[selected > 0] = cv2.GC_PR_FGD
    mask[core > 0] = cv2.GC_FGD
    roi = mask[y0:y1, x0:x1].copy()
    try:
        cv2.grabCut(np.asarray(source)[y0:y1, x0:x1].copy(), roi, None,
                    np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return selected
    result = np.zeros_like(selected)
    result[y0:y1, x0:x1] = np.isin(roi, [cv2.GC_FGD, cv2.GC_PR_FGD])
    return result


def head_matte(source, labels):
    return refine_matte(source, np.isin(labels, HEAD))


def match_new_background(source, generated, old_foreground, new_foreground):
    """Match only room pixels; the generated garment values remain unchanged."""
    original = np.asarray(source, np.float32)
    pixels = np.asarray(generated, np.float32)
    known = (~old_foreground & ~new_foreground).astype(np.float32)
    sigma = max(6, source.width*.025)
    weight = cv2.GaussianBlur(known, (0, 0), sigma)
    delta = cv2.GaussianBlur((original-pixels)*known[..., None], (0, 0), sigma)
    correction = np.divide(delta, np.maximum(weight[..., None], 1e-4))
    correction = np.clip(correction, -50, 50)
    pixels[~new_foreground] += correction[~new_foreground]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def preserve_reference_edit(source, generated, old_labels, kind, new_labels):
    """Keep identity and unchanged clothes; retain the new natural silhouette."""
    if kind not in TARGETS:
        raise ValueError("Unknown garment region")
    # GrabCut initializes color clusters with OpenCV's thread-local RNG.
    # Start each complete composite at its fresh-worker state so a previous
    # look cannot change this photo's hair, sleeve or background edges.
    cv2.setRNGSeed(0)
    source = source.convert("RGB").resize(generated.size, Image.Resampling.LANCZOS)
    old = np.asarray(Image.fromarray(np.asarray(old_labels, np.uint8)).resize(generated.size, Image.Resampling.NEAREST))
    new = np.asarray(Image.fromarray(np.asarray(new_labels, np.uint8)).resize(generated.size, Image.Resampling.NEAREST))
    head = head_matte(source, old) > 0
    old_person = refine_matte(source, old != 0) > 0
    # Do not take background samples from a possibly missed fringe of the old
    # dark garment. New room pixels in this band are matched from farther out.
    safe_radius = max(8, round(source.width*.02))
    safe_kernel = np.ones((safe_radius*2+1, safe_radius*2+1), np.uint8)
    old_person |= cv2.dilate((old != 0).astype(np.uint8), safe_kernel) > 0
    new_person = refine_matte(generated, new != 0) > 0
    generated = match_new_background(source, generated, old_person, new_person)
    # A parser's face label can spill onto the old neckline/background. The
    # replacement collar takes precedence over those uncertain boundary pixels.
    keep = head & ~np.isin(new, TARGETS[kind])
    other = [5, 6, 7, 14, 15] if kind == "upper" else [3, 4, 10, 12, 16]
    keep |= np.isin(old, other) & ~np.isin(new, TARGETS[kind])
    # Restore hands/bags/feet only where both parses agree. Restoring every old
    # hand label can paste old clothing into a new sleeve or exposed room.
    keep |= (old == new) & np.isin(old, [8, 13, 15, 17])
    radius = max(4, round(source.width * .015))
    kernel = np.ones((radius*2+1, radius*2+1), np.uint8)
    keep |= ~old_person & ~new_person
    # Remove an edited hair fringe where the source had empty room, without
    # cutting into the new collar. The source head itself uses its refined matte.
    near_head = cv2.dilate(head.astype(np.uint8), kernel) > 0
    keep |= near_head & np.isin(new, HEAD) & (old == 0)
    alpha = Image.fromarray(keep.astype(np.uint8)*255).filter(ImageFilter.GaussianBlur(.65))
    return Image.composite(source, generated.convert("RGB"), alpha)
