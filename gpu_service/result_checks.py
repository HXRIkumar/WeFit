"""Reject a severe pose failure instead of saving it as a successful try-on.

Semantic parsing is approximate. This conservative check catches sizable new
hands far from the source positions; passing does not certify a natural fit.
"""
import cv2
import numpy as np


def reject_invented_hands(source_labels, generated_labels):
    source = np.asarray(source_labels, dtype=np.uint8)
    generated = np.asarray(generated_labels, dtype=np.uint8)
    if source.shape != generated.shape:
        source = cv2.resize(source, (generated.shape[1], generated.shape[0]), interpolation=cv2.INTER_NEAREST)
    margin = max(4, round(generated.shape[1] * .035))
    nearby = cv2.dilate((source == 13).astype(np.uint8),
                        np.ones((margin * 2 + 1, margin * 2 + 1), dtype=np.uint8)) > 0
    unexpected = (generated == 13) & ~nearby
    if unexpected.sum() > max(48, generated.size * .004):
        raise ValueError("The result changed your hand position, so it was not saved. Try another photo with your arms at your sides and both hands visible.")
