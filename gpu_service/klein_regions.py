"""Spend denoising work on clothing while retaining the full source photograph."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EditingRegion:
    box: tuple[int, int, int, int]
    size: tuple[int, int]
    full_size: tuple[int, int]

    @property
    def full_frame(self):
        return self.box == (0, 0, *self.full_size)

    def reference(self, source):
        return source.crop(self.box).resize(self.size, Image.Resampling.LANCZOS)

    def composite(self, source, generated):
        frame = source.copy()
        width, height = self.box[2]-self.box[0], self.box[3]-self.box[1]
        frame.paste(generated.resize((width, height), Image.Resampling.LANCZOS), self.box[:2])
        return frame


def editing_region(labels, kind, mode="full"):
    height, width = labels.shape
    full = EditingRegion((0, 0, width, height), (width, height), (width, height))
    if mode == "full":
        return full
    if mode != "auto":
        raise ValueError("Unknown rendering region")
    chosen = [3, 4, 10, 12, 13] if kind == "upper" else [5, 6, 7, 14, 15]
    ys, xs = np.where(np.isin(labels, chosen))
    if len(xs) < labels.size*.02:
        return full
    bw, bh = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    mx, my = max(24, round(bw*.18)), max(24, round(height*.04))
    x0 = max(0, math.floor((xs.min()-mx)/16)*16)
    y0 = max(0, math.floor((ys.min()-my)/16)*16)
    x1 = min(width, math.ceil((xs.max()+mx+1)/16)*16)
    y1 = min(height, math.ceil((ys.max()+my*1.5+1)/16)*16)
    cropped = (x1-x0, y1-y0)
    if cropped[0]*cropped[1] >= width*height*.9:
        return full
    # Maintain every existing garment pixel. Small full-body clothing regions
    # get a 512-pixel render before being reduced into the original photograph.
    scale = max(1, 512/max(cropped))
    size = tuple(max(32, round(side*scale/16)*16) for side in cropped)
    if size[0]*size[1] >= width*height*.95:
        return full
    return EditingRegion((x0, y0, x1, y1), size, (width, height))


def garment_reaches_crop_edge(labels, kind, region):
    """A changed silhouette touching an internal crop edge needs a full render."""
    if region.full_frame:
        return False
    target = np.isin(labels, [3, 4, 10] if kind == "upper" else [5, 6, 7])
    x0, y0, x1, y1 = region.box
    width, height = region.full_size
    edges = []
    if x0 > 0: edges.append(target[y0:y1, x0:x0+2])
    if x1 < width: edges.append(target[y0:y1, x1-2:x1])
    if y0 > 0: edges.append(target[y0:y0+2, x0:x1])
    if y1 < height: edges.append(target[y1-2:y1, x0:x1])
    return any(edge.size and edge.mean() > .15 for edge in edges)
