"""Convert an official PiDiNet checkpoint into a vanilla-convolution ONNX graph."""

from pathlib import Path
from types import SimpleNamespace
import argparse
import sys

import torch
import torch.nn as nn


class FusedEdgeMap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--variant", choices=("standard", "small", "tiny"), default="small")
    args = parser.parse_args()

    if args.size % 8:
        parser.error("--size must be divisible by 8")

    sys.path.insert(0, str(args.source.resolve()))
    from models.convert_pidinet import convert_pidinet  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.pidinet import (  # type: ignore[import-not-found]  # noqa: PLC0415
        pidinet_converted,
        pidinet_small_converted,
        pidinet_tiny_converted,
    )

    model_args = SimpleNamespace(config="carv4", dil=True, sa=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    converted_state = convert_pidinet(state, model_args.config)
    constructors = {
        "standard": pidinet_converted,
        "small": pidinet_small_converted,
        "tiny": pidinet_tiny_converted,
    }
    model = constructors[args.variant](model_args)
    model.load_state_dict(converted_state, strict=True)
    wrapped = FusedEdgeMap(model.eval())
    sample = torch.zeros(1, 3, args.size, args.size, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        sample,
        args.output,
        input_names=["image"],
        output_names=["mask"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
