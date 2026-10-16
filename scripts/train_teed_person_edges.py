"""Fine-tune TEED to retain human edges while suppressing background edges."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset


SIZE = 320
BGR_MEAN = (103.939, 116.779, 123.68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("data/cihp_person_edges/training.jsonl"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/cihp_person_edges/validation.jsonl"))
    parser.add_argument("--source", type=Path, default=Path("vendor/TEED"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vendor/TEED/checkpoints/BIPED/5/5_model.pth"),
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/teed_person"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--max-steps", type=int, help="Stop each epoch early; useful for smoke tests.")
    parser.add_argument("--seed", type=int, default=1021)
    return parser.parse_args()


def load_teed(source: Path):
    sys.path.insert(0, str(source.resolve()))
    image_utils = ModuleType("utils.img_processing")
    image_utils.count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    sys.modules["utils.img_processing"] = image_utils
    from ted import TED  # type: ignore[import-not-found]  # noqa: PLC0415

    return TED


class PersonEdgeDataset(Dataset):
    def __init__(self, manifest: Path, training: bool, limit: int | None = None):
        with manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row["image"]).convert("RGB")
        human_ids = Image.open(row["human_ids"]).convert("L")
        if self.training:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                human_ids = human_ids.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        rgb = np.asarray(image.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
        source = np.empty((3, SIZE, SIZE), dtype=np.float32)
        source[0] = rgb[:, :, 2] - BGR_MEAN[0]
        source[1] = rgb[:, :, 1] - BGR_MEAN[1]
        source[2] = rgb[:, :, 0] - BGR_MEAN[2]
        mask = np.asarray(
            human_ids.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32
        )
        mask = np.clip(mask, 0, 1)[None]
        return torch.from_numpy(source), torch.from_numpy(mask)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def make_targets(teacher, images: torch.Tensor, person: torch.Tensor) -> torch.Tensor:
    edge = torch.sigmoid(teacher(images)[-1])
    edge_min = edge.amin(dim=(2, 3), keepdim=True)
    edge_max = edge.amax(dim=(2, 3), keepdim=True)
    edge = (edge - edge_min) / (edge_max - edge_min).clamp_min(1e-6)
    binary = (person > 0.5).float()
    dilated = F.max_pool2d(binary, 5, stride=1, padding=2)
    eroded = -F.max_pool2d(-binary, 5, stride=1, padding=2)
    boundary = F.avg_pool2d((dilated - eroded).clamp(0, 1), 3, stride=1, padding=1)
    return torch.maximum(edge * person, boundary)


def supervised_loss(logit: torch.Tensor, target: torch.Tensor, person: torch.Tensor) -> torch.Tensor:
    positive = (target > 0.10).float()
    positive_fraction = positive.mean(dim=(1, 2, 3), keepdim=True)
    weights = positive * (1 - positive_fraction) + (1 - positive) * positive_fraction * 1.1
    edge_loss = F.binary_cross_entropy_with_logits(logit, target, weight=weights)
    outside = (1 - person).clamp(0, 1)
    background_loss = (torch.sigmoid(logit) * outside).sum() / outside.sum().clamp_min(1)
    return edge_loss + 0.5 * background_loss


def run_epoch(
    student,
    teacher,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    student.train(training)
    totals = {"loss": 0.0, "background": 0.0, "edge_response": 0.0}
    steps = 0
    for step, (images, person) in enumerate(loader, 1):
        images = images.to(device)
        person = person.to(device)
        target = make_targets(teacher, images, person)
        with torch.set_grad_enabled(training):
            outputs = student(images)
            losses = [supervised_loss(output, target, person) for output in outputs]
            loss = sum(losses[:-1]) * 0.25 + losses[-1]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
        with torch.no_grad():
            prediction = torch.sigmoid(outputs[-1])
            outside = (1 - person).clamp(0, 1)
            edge_pixels = (target > 0.20).float()
            totals["loss"] += float(loss.detach().cpu())
            totals["background"] += float(
                ((prediction * outside).sum() / outside.sum().clamp_min(1)).detach().cpu()
            )
            totals["edge_response"] += float(
                ((prediction * edge_pixels).sum() / edge_pixels.sum().clamp_min(1)).detach().cpu()
            )
        steps += 1
        if training and step % 50 == 0:
            print(f"step {step}/{len(loader)} loss={totals['loss'] / steps:.4f}", flush=True)
        if max_steps is not None and step >= max_steps:
            break
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    TED = load_teed(args.source)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    teacher = TED().to(device).eval()
    student = TED().to(device)
    teacher.load_state_dict(state, strict=True)
    student.load_state_dict(state, strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    train_set = PersonEdgeDataset(args.train_manifest, training=True, limit=args.train_limit)
    val_set = PersonEdgeDataset(args.val_manifest, training=False, limit=args.val_limit)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    print(
        f"device={device} train={len(train_set)} val={len(val_set)} "
        f"parameters={sum(p.numel() for p in student.parameters()):,}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(student, teacher, train_loader, device, optimizer, args.max_steps)
        val_metrics = run_epoch(student, teacher, val_loader, device, None, args.max_steps)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        checkpoint = args.output / f"epoch-{epoch}.pth"
        torch.save(student.state_dict(), checkpoint)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(student.state_dict(), args.output / "best.pth")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
