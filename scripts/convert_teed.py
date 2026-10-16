"""Convert the official TEED checkpoint to a browser-friendly ONNX graph.

The upstream architecture imports its image utility module only to expose a
parameter-count helper. Stubbing that unused helper avoids pulling OpenCV and
the rest of TEED's training dependencies into this small conversion tool.
"""

from pathlib import Path
from types import ModuleType
import argparse
import sys

import torch
import torch.nn as nn


class FusedEdgeMap(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a checkout of https://github.com/xavysp/TEED",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.source.resolve()))
    image_utils = ModuleType("utils.img_processing")
    image_utils.count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    sys.modules["utils.img_processing"] = image_utils
    from ted import TED  # type: ignore[import-not-found]  # noqa: PLC0415

    model = TED()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    wrapped = FusedEdgeMap(model.eval())
    if args.size % 8:
        parser.error("--size must be divisible by 8")
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
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
