"""Minimal CNN architecture for RGB edge detection.

The model preserves the spatial resolution of the input and returns one raw
logit per pixel, making its output suitable for ``BCEWithLogitsLoss``.
"""

import torch
from torch import nn


class EdgeDetectionCNN(nn.Module):
    """Three-layer CNN that maps an RGB image to a single-channel logit map."""

    def __init__(self) -> None:
        super().__init__()

        # Every convolution keeps height and width unchanged: kernel=3,
        # stride=1, and padding=1.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw per-pixel logits with shape ``[N, 1, H, W]``."""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


if __name__ == "__main__":
    # A single randomly generated RGB image with the required dimensions.
    input_tensor = torch.randn(1, 3, 128, 128)
    model = EdgeDetectionCNN()
    output_tensor = model(input_tensor)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Input dimensions: {list(input_tensor.shape)}")
    print(f"Output dimensions: {list(output_tensor.shape)}")
    print(f"Total trainable parameters: {trainable_parameters}")

    # Verify the proof-of-concept contract explicitly.
    assert list(output_tensor.shape) == [1, 1, 128, 128]
    assert trainable_parameters == 1537
