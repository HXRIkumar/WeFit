"""Prepare a reproducible image-disjoint same-garment reconstruction pilot."""

import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

CANVAS = (256, 320)
SEED = "wefit-output-lora-01"


def garment_crop(photo, mask):
    """Use a clothing-centred, aspect-preserving field of view for both inputs."""
    width, height = photo.size
    x0, y0, x1, y1 = mask.getbbox()
    ratio = CANVAS[0] / CANVAS[1]
    field_h = min(
        height, width / ratio, max((x1 - x0) * 1.65 / ratio, (y1 - y0) * 1.85)
    )
    field_w = field_h * ratio
    left = min(width - field_w, max(0, (x0 + x1 - field_w) / 2))
    top = min(height - field_h, max(0, (y0 + y1 - field_h) / 2))
    box = tuple(round(x) for x in (left, top, left + field_w, top + field_h))
    return (
        photo.crop(box).resize(CANVAS, Image.Resampling.LANCZOS),
        mask.crop(box).resize(CANVAS, Image.Resampling.NEAREST),
    )


def prepare(annotations, archive, out, counts):
    if any(n < 1 for n in counts):
        raise ValueError("Each split requires at least one image.")
    corpus = json.loads(annotations.read_text())
    images = {i["id"]: i for i in corpus["images"] if i["license"] == 0}
    categories = {c["id"]: c["name"] for c in corpus["categories"]}
    tops = defaultdict(list)
    for annotation in corpus["annotations"]:
        if annotation["image_id"] in images and annotation["category_id"] in range(6):
            tops[annotation["image_id"]].append(annotation)
    candidates = []
    for image_id, items in tops.items():
        image = images[image_id]
        if len(items) != 1:
            continue
        item = items[0]
        if item["iscrowd"] or not isinstance(item["segmentation"], list):
            continue
        fraction = item["area"] / (image["width"] * image["height"])
        if 0.05 <= fraction <= 0.70:
            candidates.append((image, item))
    candidates.sort(
        key=lambda pair: hashlib.sha256((SEED + pair[0]["file_name"]).encode()).digest()
    )
    rows, seen_urls, seen_hashes, thumbnails = [], set(), set(), []
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for image, item in candidates:
            raw = z.read("test/" + image["file_name"])
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_hashes or image["original_url"] in seen_urls:
                continue
            photo = Image.open(io.BytesIO(raw)).convert("RGB")
            if photo.size != (image["width"], image["height"]):
                raise ValueError("Annotation dimensions do not match the photograph")
            thumb = ImageOps.grayscale(photo).resize((24, 24))
            if any(
                ImageStat.Stat(ImageChops.difference(thumb, old)).mean[0] < 8
                for old in thumbnails
            ):
                continue
            mask = Image.new("L", photo.size)
            draw = ImageDraw.Draw(mask)
            for ring in item["segmentation"]:
                if len(ring) >= 6:
                    draw.polygon(ring, fill=255)
            if not mask.getbbox():
                continue
            target, mask = garment_crop(photo, mask)
            product = Image.composite(target, Image.new("RGB", CANVAS, "white"), mask)
            product = ImageOps.expand(
                product.crop(mask.getbbox()), border=8, fill="white"
            )
            product.thumbnail((256, 256), Image.Resampling.LANCZOS)
            neutral = Image.composite(
                Image.new("RGB", CANVAS, (140, 140, 140)), target, mask
            )
            folder = out / str(image["id"])
            folder.mkdir(exist_ok=True)
            for name, im in [
                ("target", target),
                ("mask", mask),
                ("person", neutral),
                ("item", product),
            ]:
                im.save(folder / (name + ".png"))
            split = (
                "train"
                if len(rows) < counts[0]
                else "val"
                if len(rows) < sum(counts[:2])
                else "test"
            )
            rows.append(
                {
                    "id": image["id"],
                    "split": split,
                    "annotation_id": item["id"],
                    "category": categories[item["category_id"]],
                    "file_name": image["file_name"],
                    "sha256": digest,
                    "original_url": image["original_url"],
                    "license": "https://creativecommons.org/licenses/by/2.0/",
                }
            )
            seen_hashes.add(digest)
            seen_urls.add(image["original_url"])
            thumbnails.append(thumb)
            if len(rows) == sum(counts):
                break
    if len(rows) != sum(counts):
        raise ValueError(
            f"Only {len(rows)} eligible distinct photos for {sum(counts)} requested"
        )
    manifest = {
        "seed": SEED,
        "source": "Fashionpedia validation annotations / val_test2020.zip",
        "canvas": CANVAS,
        "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "split_unit": "image",
        "records": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--counts",
        type=int,
        nargs=3,
        default=[14, 5, 5],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    args = parser.parse_args()
    manifest = prepare(args.annotations, args.archive, args.out, args.counts)
    print(len(manifest["records"]), "licensed reconstruction pairs", args.counts)


if __name__ == "__main__":
    main()
