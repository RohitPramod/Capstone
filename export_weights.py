"""Export a trained first-layer RGB filter and its bias as floating-point JSON."""

import argparse
import json
from pathlib import Path

import torch

from cnn_model import EdgeDetectionCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-index", type=int, default=0,
                        help="zero-based Conv1 output channel (default: 0)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    model = EdgeDetectionCNN()
    if not 0 <= args.filter_index < model.conv1.out_channels:
        parser.error(f"--filter-index must be in [0, {model.conv1.out_channels - 1}]")
    checkpoint = project_dir / "models" / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    # Use output channel 0 by default as a fixed, reproducible baseline.
    # PyTorch stores Conv2d weights as [output channel, input channel, row, column].
    index = args.filter_index
    weights = model.conv1.weight[index].detach()
    bias = model.conv1.bias
    if bias is None:
        raise ValueError("Expected Conv1 to have one bias per output channel")
    channels = ["R", "G", "B"]
    export = {
        "checkpoint": checkpoint.relative_to(project_dir).as_posix(),
        "layer": "conv1",
        "filter_index": index,
        "channel_order": channels,
        "weight_layout": "channel, row, column",
        "weights": weights.tolist(),
        "bias": bias[index].item(),
    }
    output = project_dir / "outputs" / f"conv1_filter_{index}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")

    print(f"Conv1 filter {index} (zero-based)")
    for channel, kernel in zip(channels, export["weights"]):
        print(f"{channel} kernel")
        for row in kernel:
            print("  " + " ".join(f"{value:+.8f}" for value in row))
    print(f"Bias: {export['bias']:+.8f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
