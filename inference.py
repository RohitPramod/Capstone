"""Evaluate the trained edge detector on the full-resolution BSDS500 test split."""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cnn_model import EdgeDetectionCNN
from dataset_prepare import BSDSDataset


def save_comparison(path, rgb, target, probabilities, prediction):
    """Save aligned input, soft annotation, probability and binary panels."""
    figure, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")
    panels = (
        (target, "Mean human boundary map"),
        (probabilities, "Predicted edge probability"),
        (prediction, "Binary prediction"),
    )
    for axis, (data, title) in zip(axes[1:], panels):
        axis.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


@torch.inference_mode()
def evaluate(project_dir: Path, threshold: float, gt_threshold: float, samples: int):
    """Evaluate every test image and save a deterministic subset of comparisons."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeDetectionCNN().to(device)
    checkpoint = project_dir / "models" / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    dataset = BSDSDataset(project_dir / "dataset", split="test")
    output_dir = project_dir / "outputs" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = set(
        np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int)
    )

    true_positive = predicted_positive = actual_positive = total_pixels = 0
    for index, (image_path, annotation_path) in enumerate(dataset.samples):
        # Load the original image and reuse the training annotation averaging.
        # The dataset's item accessor center-crops, so use the paired paths here.
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target = dataset._load_ground_truth(annotation_path)
        if target.shape != rgb.shape[:2]:
            raise ValueError(f"Image and annotation sizes differ for {image_path.name}")

        # The fully convolutional model accepts either orientation at native size.
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=device, dtype=torch.float32).div_(255)
        probabilities = model(image).sigmoid()[0, 0].cpu().numpy()
        if probabilities.shape != target.shape or not np.isfinite(probabilities).all():
            raise ValueError(f"Invalid probability map for {image_path.name}")
        prediction = probabilities >= threshold
        ground_truth = target >= gt_threshold

        # Pool pixel counts across images for exact-pixel micro-averaged scores.
        true_positive += int(np.count_nonzero(prediction & ground_truth))
        predicted_positive += int(np.count_nonzero(prediction))
        actual_positive += int(np.count_nonzero(ground_truth))
        total_pixels += target.size

        if index in sample_indices:
            path = output_dir / f"{image_path.stem}_comparison.png"
            save_comparison(path, rgb, target, probabilities, prediction)
            print(f"Saved {path.name}")

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    positives = predicted_positive + actual_positive
    f1 = 2 * true_positive / positives if positives else 0.0
    print(f"Device: {device} | Full-resolution images: {len(dataset)} | Pixels: {total_pixels:,}")
    print(f"Prediction threshold: >= {threshold:g} | Annotation fraction: >= {gt_threshold:g}")
    print(f"TP: {true_positive:,} | FP: {predicted_positive - true_positive:,} "
          f"| FN: {actual_positive - true_positive:,}")
    print(f"Pixelwise micro precision: {precision:.6f} | Recall: {recall:.6f} | F1: {f1:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.5, help="edge probability cutoff")
    parser.add_argument("--gt-threshold", type=float, default=0.5, help="annotator agreement cutoff")
    parser.add_argument("--samples", type=int, default=6, help="number of comparisons to save")
    args = parser.parse_args()
    for flag, value in (("--threshold", args.threshold), ("--gt-threshold", args.gt_threshold)):
        if not 0 < value <= 1:
            parser.error(f"{flag} must be in (0, 1]")
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    evaluate(Path(__file__).resolve().parent, args.threshold, args.gt_threshold, args.samples)


if __name__ == "__main__":
    main()
