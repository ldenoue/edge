"""Convert the Hugging Face PyTorch checkpoint to a browser-friendly ONNX graph."""

from pathlib import Path
import argparse

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class WithSigmoid(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    wrapped = WithSigmoid(model.eval())
    sample = torch.zeros(1, 3, 256, 256, dtype=torch.float32)

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
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
