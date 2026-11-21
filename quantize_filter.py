"""Generate a fixed-point Conv1 test vector and compare it with PyTorch."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def integer_convolution(patch: np.ndarray, weights: np.ndarray, bias: int) -> np.ndarray:
    """Compute stride-1, zero-padded cross-correlation using INT32 arithmetic."""
    # This absolute bound covers every partial sum, including the bias.
    bound = abs(bias) + 255 * int(np.abs(weights.astype(np.int32)).sum())
    if bound > np.iinfo(np.int32).max:
        raise OverflowError("Bias and products could overflow an INT32 accumulator")
    padded = np.pad(patch.astype(np.int32), ((0, 0), (1, 1), (1, 1)))
    kernel = weights.astype(np.int32)
    height, width = patch.shape[1:]
    result = np.empty((height, width), dtype=np.int32)
    for row in range(height):
        for col in range(width):
            window = padded[:, row : row + 3, col : col + 3]
            result[row, col] = (window * kernel).sum(dtype=np.int32) + np.int32(bias)
    return result


def pytorch_convolution(patch, weights, bias):
    """Compute an independent reference using PyTorch's Conv2d convention."""
    return F.conv2d(
        torch.tensor(patch, dtype=torch.float64)[None],
        torch.tensor(weights, dtype=torch.float64)[None],
        torch.tensor([bias], dtype=torch.float64),
        padding=1,
    )[0, 0].numpy()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    filter_path = project_dir / "outputs" / "conv1_filter_0.json"
    exported = json.loads(filter_path.read_text(encoding="utf-8"))
    weights = np.asarray(exported["weights"], dtype=np.float64)
    bias = exported["bias"]
    if weights.shape != (3, 3, 3) or exported["channel_order"] != ["R", "G", "B"]:
        raise ValueError("Expected one RGB 3x3 filter in channel-first layout")

    # Use the same center crop from the same held-out image on every run.
    image_path = project_dir / "dataset" / "images" / "test" / "100007.jpg"
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not decode {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    top, left = (height - 8) // 2, (width - 8) // 2
    patch = rgb[top : top + 8, left : left + 8].transpose(2, 0, 1)

    # Pixel bytes represent normalized inputs exactly under scale 1/255.
    # Symmetric weights use [-127, 127] and round ties to even.
    input_scale = 1 / 255
    weight_scale = float(np.max(np.abs(weights))) / 127 or 1.0
    accumulator_scale = input_scale * weight_scale
    weights_q = np.clip(np.rint(weights / weight_scale), -127, 127).astype(np.int8)
    bias_q = int(np.rint(bias / accumulator_scale))
    accumulators = integer_convolution(patch, weights_q, bias_q)

    # Float64 represents the checked INT32 sums exactly, allowing an exact check.
    reference = pytorch_convolution(patch, weights_q, bias_q)
    np.testing.assert_array_equal(accumulators, reference)
    float_output = pytorch_convolution(patch * input_scale, weights, bias)
    dequantized = accumulators * accumulator_scale
    error = dequantized - float_output
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }
    vector = {
        "filter_index": exported["filter_index"],
        "source_image": image_path.relative_to(project_dir).as_posix(),
        "crop_top_left": [top, left],
        "input_and_weight_layout": "RGB channel, row, column",
        "output_layout": "row, column",
        "stride": 1,
        "zero_padding": 1,
        "rounding": "nearest, ties to even",
        "overflow": "reject if any INT32 partial sum could overflow",
        "weight_clipping": [-127, 127],
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "accumulator_scale": accumulator_scale,
        "input_uint8": patch.tolist(),
        "weights_int8": weights_q.tolist(),
        "bias_int32": bias_q,
        "expected_output_int32": accumulators.tolist(),
        "float_output": float_output.tolist(),
        "dequantized_output": dequantized.tolist(),
        "error": metrics,
    }
    output = project_dir / "outputs" / "quantized_conv1_filter_0.json"
    output.write_text(json.dumps(vector, indent=2) + "\n", encoding="utf-8")
    print(f"Verified all {accumulators.size} integer outputs against PyTorch.")
    print(f"Crop origin (row, column): ({top}, {left})")
    print(f"Input scale: {input_scale:.10g} | Weight scale: {weight_scale:.10g}")
    print(f"Accumulator scale: {accumulator_scale:.10g} | INT32 bias: {bias_q}")
    print(f"INT32 output range: [{accumulators.min()}, {accumulators.max()}]")
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
