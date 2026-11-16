"""Build offline reviewer packets and decode returned ratings locally."""

import argparse
import base64
import io
import itertools
import json
import secrets
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CRITERIA = ("neckline", "shoulders", "sleeves", "fabric", "anatomy", "scene")


def photo_data(path):
    with Image.open(path) as image:
        clean = image.convert("RGB")
        clean.thumbnail((720, 840), Image.Resampling.LANCZOS)
        data = io.BytesIO()
        clean.save(data, "WEBP", quality=94, method=4)
    return "data:image/webp;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def build(manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"].copy()
    methods = manifest["methods"]
    if (
        len(methods) != 2
        or len(set(methods)) != 2
        or not cases
        or len({c["id"] for c in cases}) != len(cases)
    ):
        raise ValueError(
            "Supply two distinct method names and at least one uniquely identified image pair."
        )
    random = secrets.SystemRandom()
    random.shuffle(cases)
    reverse = [bool(i % 2) for i in range(len(cases))]
    if random.choice((True, False)):
        reverse = [not flag for flag in reverse]
    random.shuffle(reverse)
    packet = secrets.token_hex(10)
    key = {
        "packet": packet,
        "study": manifest["study"],
        "methods": methods,
        "cases": {},
    }
    public = {"packet": packet, "cases": []}
    for case, flip in zip(cases, reverse):
        token = secrets.token_hex(5)
        order = methods[::-1] if flip else methods
        key["cases"][token] = {"id": case["id"], "A": order[0], "B": order[1]}
        images = {
            "source": case["source"],
            "product": case["product"],
            "A": case[order[0]],
            "B": case[order[1]],
        }
        public["cases"].append(
            {
                "id": token,
                "images": {
                    k: photo_data(manifest_path.parent / v) for k, v in images.items()
                },
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).with_name("review_form.html").read_text()
    (out / "review.html").write_text(
        template.replace("__PACKET_DATA__", json.dumps(public))
    )
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(
        f"Send {out / 'review.html'} to one reviewer. Keep {out / 'key.json'} with the study owner."
    )


def aggregate(key_paths, response_paths):
    keys = {}
    for path in key_paths:
        k = json.loads(path.read_text())
        if k["packet"] in keys:
            raise ValueError("Repeated packet key.")
        keys[k["packet"]] = k
    if not keys:
        raise ValueError("At least one private packet key is required.")
    study = next(iter(keys.values()))["study"]
    methods = next(iter(keys.values()))["methods"]
    ids = {r["id"] for r in next(iter(keys.values()))["cases"].values()}
    if any(
        k["study"] != study
        or k["methods"] != methods
        or {r["id"] for r in k["cases"].values()} != ids
        for k in keys.values()
    ):
        raise ValueError("Keys belong to different studies or case sets.")
    by_method = {m: {c: [] for c in CRITERIA} for m in methods}
    preferences = Counter()
    by_case = defaultdict(list)
    used = set()
    for path in response_paths:
        response = json.loads(path.read_text())
        pid = response.get("packet")
        if pid in used or pid not in keys:
            raise ValueError("Unknown or duplicate reviewer packet.")
        key = keys[pid]
        if set(response["ratings"]) != set(key["cases"]):
            raise ValueError("Incomplete or unexpected case set.")
        for token, ratings in response["ratings"].items():
            case = key["cases"][token]
            for side in ["A", "B"]:
                values = ratings[side]
                if set(values) != set(CRITERIA):
                    raise ValueError("Missing scoring dimension.")
                for field, value in values.items():
                    if value is None:
                        continue
                    if type(value) is not int or not 1 <= value <= 5:
                        raise ValueError("Scores must be 1–5 or null.")
                    by_method[case[side]][field].append(value)
            choice = ratings["preference"]
            if choice not in ["A", "B", "tie"]:
                raise ValueError("Choose A, B or tie.")
            winner = "tie" if choice == "tie" else case[choice]
            by_case[case["id"]].append(winner)
            preferences[winner] += 1
        used.add(pid)
    votes = [
        a == b
        for votes in by_case.values()
        for a, b in itertools.combinations(votes, 2)
    ]
    return {
        "study": study,
        "packets_prepared": len(keys),
        "completed_reviewer_packets": len(used),
        "rated_pairs": sum(preferences.values()),
        "preferences": dict(preferences),
        "ratings": {
            m: {
                c: {"n": len(v), "mean": round(sum(v) / len(v), 3) if v else None}
                for c, v in score.items()
            }
            for m, score in by_method.items()
        },
        "preference_agreement": {
            "agreeing_pairs": sum(votes),
            "comparable_pairs": len(votes),
            "fraction": sum(votes) / len(votes) if votes else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("manifest", type=Path)
    make.add_argument("output", type=Path)
    read = commands.add_parser("summarize")
    read.add_argument("--keys", nargs="+", type=Path, required=True)
    read.add_argument("--responses", nargs="*", type=Path, default=[])
    read.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.manifest, args.output)
    else:
        report = json.dumps(aggregate(args.keys, args.responses), indent=2) + "\n"
        if args.output:
            args.output.write_text(report)
        else:
            print(report, end="")


if __name__ == "__main__":
    main()
