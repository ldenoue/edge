"""Compare person-tuned TEED checkpoints on people and hard negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_teed_person_edges import PersonEdgeDataset, choose_device, load_teed, make_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/cihp_person_edges/validation.jsonl"))
    parser.add_argument(
        "--negative-manifest",
        type=Path,
        default=Path("data/cihp_person_edges/validation_negatives.jsonl"),
    )
    parser.add_argument("--teacher", type=Path, default=Path("vendor/TEED/checkpoints/BIPED/5/5_model.pth"))
    parser.add_argument("--source", type=Path, default=Path("vendor/TEED"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.18)
    return parser.parse_args()


def normalized(prediction: torch.Tensor) -> torch.Tensor:
    minimum = prediction.amin(dim=(2, 3), keepdim=True)
    maximum = prediction.amax(dim=(2, 3), keepdim=True)
    return (prediction - minimum) / (maximum - minimum).clamp_min(1e-6)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device()
    TED = load_teed(args.source)
    teacher = TED().to(device).eval()
    teacher.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True), strict=True)
    models = {}
    for checkpoint in args.checkpoints:
        model = TED().to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        models[checkpoint.name + "@" + checkpoint.parent.name] = model

    positive = PersonEdgeDataset(args.manifest, training=False)
    negative = PersonEdgeDataset(args.manifest, training=False, limit=0, negative_manifest=args.negative_manifest)
    loaders = {
        "positive": DataLoader(positive, batch_size=args.batch_size, num_workers=args.workers),
        "negative": DataLoader(negative, batch_size=args.batch_size, num_workers=args.workers),
    }
    totals = {
        name: {
            "positive_background_sum": 0.0,
            "positive_background_count": 0.0,
            "positive_edge_sum": 0.0,
            "positive_edge_count": 0.0,
            "normalized_background_pixels": 0.0,
            "normalized_background_count": 0.0,
            "negative_sum": 0.0,
            "negative_count": 0.0,
            "normalized_negative_pixels": 0.0,
        }
        for name in models
    }

    for kind, loader in loaders.items():
        for images, person in loader:
            images, person = images.to(device), person.to(device)
            target = make_targets(teacher, images, person) if kind == "positive" else None
            for name, model in models.items():
                prediction = torch.sigmoid(model(images)[-1])
                visible = normalized(prediction) > args.threshold
                metrics = totals[name]
                if kind == "positive":
                    outside = (1 - person).clamp(0, 1)
                    edges = (target > 0.20).float()
                    metrics["positive_background_sum"] += float((prediction * outside).sum().cpu())
                    metrics["positive_background_count"] += float(outside.sum().cpu())
                    metrics["positive_edge_sum"] += float((prediction * edges).sum().cpu())
                    metrics["positive_edge_count"] += float(edges.sum().cpu())
                    metrics["normalized_background_pixels"] += float((visible * outside.bool()).sum().cpu())
                    metrics["normalized_background_count"] += float(outside.sum().cpu())
                else:
                    metrics["negative_sum"] += float(prediction.sum().cpu())
                    metrics["negative_count"] += prediction.numel()
                    metrics["normalized_negative_pixels"] += float(visible.sum().cpu())

    report = {}
    for name, metrics in totals.items():
        report[name] = {
            "positive_background_response": metrics["positive_background_sum"]
            / metrics["positive_background_count"],
            "positive_edge_response": metrics["positive_edge_sum"] / metrics["positive_edge_count"],
            "visible_background_fraction": metrics["normalized_background_pixels"]
            / metrics["normalized_background_count"],
            "negative_response": metrics["negative_sum"] / metrics["negative_count"],
            "visible_negative_fraction": metrics["normalized_negative_pixels"]
            / metrics["negative_count"],
        }
    print(json.dumps({"device": str(device), "threshold": args.threshold, "models": report}, indent=2))


if __name__ == "__main__":
    main()
