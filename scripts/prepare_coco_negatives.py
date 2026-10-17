"""Select textured COCO 2017 validation images with no annotated person."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/coco/annotations/instances_val2017.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/coco_no_person"))
    parser.add_argument("--train-count", type=int, default=1200)
    parser.add_argument("--val-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1021)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download only the candidate COCO images needed for this subset.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.5,
        help="Random candidate pool relative to requested rows before texture ranking.",
    )
    parser.add_argument("--download-workers", type=int, default=12)
    return parser.parse_args()


def texture_score(path: Path) -> float:
    with Image.open(path) as image:
        grayscale = np.asarray(
            image.convert("L").resize((160, 160), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    gradient = np.abs(np.diff(grayscale, axis=0)).mean() + np.abs(np.diff(grayscale, axis=1)).mean()
    return float(gradient + 0.15 * grayscale.std())


def ensure_image(item: dict, path: Path, download_missing: bool) -> None:
    if path.exists():
        return
    if not download_missing:
        raise SystemExit(f"Missing COCO image: {path}; pass --download-missing")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(item["coco_url"], timeout=60) as response:  # noqa: S310 - URL is COCO metadata
        path.write_bytes(response.read())


def score_item(args: argparse.Namespace, item: dict) -> dict:
    path = args.images / item["file_name"]
    ensure_image(item, path, args.download_missing)
    return {
        "id": f"coco-{item['id']}",
        "image": str(path.resolve()),
        "negative": True,
        "source": "coco-val2017-no-annotated-person",
        "hardness": round(texture_score(path), 4),
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.annotations.read_text(encoding="utf-8"))
    person_ids = {category["id"] for category in metadata["categories"] if category["name"] == "person"}
    if not person_ids:
        raise SystemExit("The annotations do not define a person category")
    images_with_people = {
        annotation["image_id"]
        for annotation in metadata["annotations"]
        if annotation["category_id"] in person_ids
    }
    no_person = [item for item in metadata["images"] if item["id"] not in images_with_people]
    needed = args.train_count + args.val_count
    if len(no_person) < needed:
        raise SystemExit(f"Only {len(no_person)} no-person candidates; requested {needed}")
    rng = random.Random(args.seed)
    rng.shuffle(no_person)
    pool_size = min(len(no_person), max(needed, round(needed * args.candidate_multiplier)))
    candidates = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        for index, row in enumerate(
            executor.map(lambda item: score_item(args, item), no_person[:pool_size]), 1
        ):
            candidates.append(row)
            if index % 100 == 0:
                print(f"scored {index}/{pool_size} candidates", flush=True)
    candidates.sort(key=lambda row: row["hardness"], reverse=True)
    selected = candidates[:needed]
    rng.shuffle(selected)
    splits = {
        "training": selected[: args.train_count],
        "validation": selected[args.train_count :],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        destination = args.output / f"{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{split}: wrote {len(rows)} rows to {destination}")
    summary = {
        "source_candidates_without_annotated_people": len(no_person),
        "downloaded_and_scored_pool": len(candidates),
        "selection": "highest grayscale gradient + 0.15 * standard deviation",
        "seed": args.seed,
        "splits": {key: len(value) for key, value in splits.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
