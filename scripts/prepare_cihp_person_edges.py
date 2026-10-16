"""Build a full-body-biased CIHP subset with person-only TEED edge targets.

The source images are not copied. JSONL manifests point to the original CIHP
files while compact 320x320 grayscale targets are written under the output
directory. Targets preserve TEED edges inside every annotated human and add an
explicit silhouette boundary; all other edges become zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter


SIZE = 320
BGR_MEAN = (103.939, 116.779, 123.68)
HEAD = {1, 2, 13}  # hat, hair, face
TORSO = {5, 6, 7, 10, 12}  # upper clothes, dress, coat, torso skin, skirt
LEGS = {16, 17}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / "Downloads/instance-level_human_parsing",
    )
    parser.add_argument("--model", type=Path, default=Path("public/teed-320.onnx"))
    parser.add_argument("--output", type=Path, default=Path("data/cihp_person_edges"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("Training", "Validation"),
        default=("Training", "Validation"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum selected images per split, useful for a smoke test.",
    )
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--cache-targets",
        action="store_true",
        help="Materialize target PNGs. Without this flag they are generated lazily during training.",
    )
    return parser.parse_args()


def qualifying_instances(human_ids: np.ndarray, categories: np.ndarray) -> list[dict]:
    """Return well-framed, full-body person instances for image selection."""
    height, width = human_ids.shape
    matches: list[dict] = []
    for identity in np.unique(human_ids):
        if identity == 0:
            continue
        mask = human_ids == identity
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        labels = set(np.unique(categories[mask]).tolist())
        relative_height = (y1 - y0 + 1) / height
        area_fraction = float(mask.mean())
        if not (
            labels & HEAD
            and labels & TORSO
            and LEGS <= labels
            and relative_height >= 0.50
            and area_fraction >= 0.025
            and y0 > 1
            and y1 < height - 2
        ):
            continue
        matches.append(
            {
                "id": int(identity),
                "bbox": [x0, y0, x1 + 1, y1 + 1],
                "height_fraction": round(relative_height, 5),
                "area_fraction": round(area_fraction, 5),
                "categories": sorted(int(value) for value in labels if value),
            }
        )
    return matches


def teacher_input(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(
        image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    tensor = np.empty((1, 3, SIZE, SIZE), dtype=np.float32)
    tensor[0, 0] = rgb[:, :, 2] - BGR_MEAN[0]
    tensor[0, 1] = rgb[:, :, 1] - BGR_MEAN[1]
    tensor[0, 2] = rgb[:, :, 0] - BGR_MEAN[2]
    return tensor


def soft_person_mask(human_ids: np.ndarray) -> np.ndarray:
    binary = Image.fromarray(np.where(human_ids > 0, 255, 0).astype(np.uint8))
    resized = binary.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255


def silhouette_boundary(human_ids: np.ndarray) -> np.ndarray:
    binary = Image.fromarray(np.where(human_ids > 0, 255, 0).astype(np.uint8))
    resized = binary.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    dilated = resized.filter(ImageFilter.MaxFilter(5))
    eroded = resized.filter(ImageFilter.MinFilter(5))
    boundary = np.asarray(dilated, dtype=np.float32) - np.asarray(eroded, dtype=np.float32)
    softened = Image.fromarray(np.clip(boundary, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.8)
    )
    return np.asarray(softened, dtype=np.float32) / 255


def make_target(
    session: ort.InferenceSession,
    input_name: str,
    image: Image.Image,
    human_ids: np.ndarray,
) -> np.ndarray:
    edge = session.run(None, {input_name: teacher_input(image)})[0][0, 0]
    edge_min, edge_max = float(edge.min()), float(edge.max())
    edge = (edge - edge_min) / max(edge_max - edge_min, 1e-6)
    person = soft_person_mask(human_ids)
    boundary = silhouette_boundary(human_ids)
    target = np.maximum(edge * person, boundary)
    return np.clip(target * 255, 0, 255).astype(np.uint8)


def make_preview(rows: list[dict], destination: Path) -> None:
    if not rows:
        return
    panel = 144
    label_height = 18
    columns = 2
    sample_width = panel * 3
    sheet = Image.new(
        "RGB",
        (sample_width * columns, (panel + label_height) * ((len(rows) + 1) // 2)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * sample_width
        y = (index // columns) * (panel + label_height)
        image = Image.open(row["image"]).convert("RGB").resize((panel, panel), Image.Resampling.BILINEAR)
        human_ids = np.asarray(Image.open(row["human_ids"]))
        mask = Image.fromarray(np.where(human_ids > 0, 255, 0).astype(np.uint8)).resize(
            (panel, panel), Image.Resampling.NEAREST
        ).convert("RGB")
        target = Image.open(row["target"]).convert("RGB").resize((panel, panel), Image.Resampling.BILINEAR)
        sheet.paste(image, (x, y + label_height))
        sheet.paste(mask, (x + panel, y + label_height))
        sheet.paste(target, (x + panel * 2, y + label_height))
        draw.text((x + 4, y + 3), f'{row["id"]}: image / people / target', fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def process_split(args: argparse.Namespace, split: str, session: ort.InferenceSession | None) -> int:
    source = args.dataset / split
    split_name = split.lower()
    target_dir = args.output / "targets" / split_name
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"{split_name}.jsonl"
    preview_rows: list[dict] = []
    selected = 0
    files = sorted((source / "Human_ids").glob("*.png"))
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for scanned, human_path in enumerate(files, 1):
            human_ids = np.asarray(Image.open(human_path))
            category_path = source / "Category_ids" / human_path.name
            categories = np.asarray(Image.open(category_path))
            instances = qualifying_instances(human_ids, categories)
            if not instances:
                continue
            image_path = source / "Images" / f"{human_path.stem}.jpg"
            target_path = target_dir / f"{human_path.stem}.png"
            if session is not None and not target_path.exists():
                with Image.open(image_path) as image:
                    target = make_target(session, session.get_inputs()[0].name, image, human_ids)
                Image.fromarray(target, mode="L").save(target_path, optimize=True)
            row = {
                "id": human_path.stem,
                "image": str(image_path.resolve()),
                "human_ids": str(human_path.resolve()),
                "category_ids": str(category_path.resolve()),
                "target": str(target_path.resolve()),
                "qualifying_instances": instances,
            }
            manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
            selected += 1
            if session is not None and len(preview_rows) < args.preview_count:
                preview_rows.append(row)
            if selected % 250 == 0:
                print(f"{split}: selected {selected} after scanning {scanned}/{len(files)}", flush=True)
            if args.limit is not None and selected >= args.limit:
                break
    if session is not None:
        make_preview(preview_rows, args.output / f"{split_name}_preview.jpg")
    print(f"{split}: wrote {selected} rows to {manifest_path}")
    return selected


def main() -> None:
    args = parse_args()
    for required in ("Training", "Validation"):
        if required in args.splits and not (args.dataset / required).is_dir():
            raise SystemExit(f"Missing dataset split: {args.dataset / required}")
    args.output.mkdir(parents=True, exist_ok=True)
    session = None
    if args.cache_targets:
        session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    summary = {split: process_split(args, split, session) for split in args.splits}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
