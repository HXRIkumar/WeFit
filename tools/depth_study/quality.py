"""Image-space comparisons at annotated garment centers, also reporting error to the original."""

import numpy as np
from scipy.ndimage import uniform_filter


def compare(image, reference, target, mask):
    mask = np.asarray(mask) > 127
    a, b, t = [np.asarray(v, dtype=np.float64) for v in (image, reference, target)]
    if (
        a.shape != b.shape
        or a.shape != t.shape
        or mask.shape != a.shape[:2]
        or not mask.any()
        or mask.all()
    ):
        raise ValueError("Unaligned inputs or empty annotation")

    def local(v):
        return uniform_filter(v, size=(7, 7, 1), mode="reflect")

    ma, mb = local(a), local(b)
    va = np.maximum(0.0, local(a * a) - ma * ma)
    vb = np.maximum(0.0, local(b * b) - mb * mb)
    cov = local(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (
        (2 * ma * mb + c1)
        * (2 * cov + c2)
        / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    )
    delta = np.abs(a - b)
    error = np.abs(a - t)
    return dict(
        garment_mae=float(error[mask].mean()),
        outside_mae=float(error[~mask].mean()),
        garment_ssim=float(ssim[mask].mean()),
        garment_within8=float((delta[mask].max(1) <= 8).mean()),
        garment_exact=float((delta[mask].max(1) == 0).mean()),
        garment_rgb_change=float(delta[mask].mean()),
        outside_rgb_change=float(delta[~mask].mean()),
    )


def summarize(rows):
    data = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        subset = [r for r in rows if r["method"] == method]
        secs = [r["wall_seconds"] for r in subset]
        data[method] = dict(
            n=len(subset),
            median_seconds=float(np.median(secs)),
            mean_ssim=float(np.mean([r["garment_ssim"] for r in subset])),
            min_ssim=min(r["garment_ssim"] for r in subset),
            max_peak_mlx_bytes=max(r["peak_mlx_bytes"] for r in subset),
        )
        for metric in (
            "garment_mae",
            "outside_mae",
            "garment_within8",
            "garment_exact",
            "garment_rgb_change",
            "outside_rgb_change",
        ):
            data[method][metric] = float(np.mean([r[metric] for r in subset]))
    return data
