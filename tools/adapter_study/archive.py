"""Pack a downloaded inference checkpoint and verify the restored bytes of each model file."""

import argparse
import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path


def pack(weights, destination):
    files = sorted(
        path
        for path in weights.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(weights).parts)
    )
    if not files:
        raise ValueError("Choose a downloaded checkpoint directory.")
    if destination.exists() or destination.resolve().is_relative_to(weights.resolve()):
        raise ValueError("Choose a new archive path outside the checkpoint directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".zip", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        started = time.perf_counter()
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for file in files:
                archive.write(file, file.relative_to(weights).as_posix())
        pack_seconds = time.perf_counter() - started
        started = time.perf_counter()
        records = {}
        with zipfile.ZipFile(temporary) as archive:
            for file in files:
                name = file.relative_to(weights).as_posix()
                with file.open("rb") as source, archive.open(name) as restored:
                    original_hash = hashlib.file_digest(source, "sha256").hexdigest()
                    restored_hash = hashlib.file_digest(restored, "sha256").hexdigest()
                if restored_hash != original_hash:
                    raise ValueError(f"Restored bytes failed the checksum for {name}.")
                records[name] = {"bytes": file.stat().st_size, "sha256": restored_hash}
        verify_seconds = time.perf_counter() - started
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    original = sum(file["bytes"] for file in records.values())
    return {
        "method": "ZIP64 with DEFLATE level 1. Restore the model before inference.",
        "source_bytes": original,
        "archive_bytes": destination.stat().st_size,
        "saved_bytes": original - destination.stat().st_size,
        "pack_wall_s": round(pack_seconds, 3),
        "verify_wall_s": round(verify_seconds, 3),
        "all_files_hash_match": True,
        "files": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--measurements", type=Path)
    args = parser.parse_args()
    result = pack(args.weights, args.archive)
    text = json.dumps(result, indent=2) + "\n"
    if args.measurements:
        args.measurements.parent.mkdir(parents=True, exist_ok=True)
        args.measurements.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
