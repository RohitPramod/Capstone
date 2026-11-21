"""Dataset preparation utilities for the BSDS500 edge-detection dataset."""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset


class BSDSDataset(Dataset):
    """Load BSDS500 images and averaged human-annotated boundary maps."""

    def __init__(
        self,
        root_dir: str | Path = "dataset",
        split: str = "train",
        patch_size: int = 128,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.image_dir = self.root_dir / "images" / split
        self.ground_truth_dir = self.root_dir / "groundTruth" / split
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        if not self.image_paths:
            raise FileNotFoundError(f"No JPG images found in {self.image_dir}")

        # Keep only images with a matching MAT annotation file.
        self.samples = [
            (image_path, self.ground_truth_dir / f"{image_path.stem}.mat")
            for image_path in self.image_paths
            if (self.ground_truth_dir / f"{image_path.stem}.mat").exists()
        ]
        if len(self.samples) != len(self.image_paths):
            raise FileNotFoundError("One or more images have no matching MAT file")

    @staticmethod
    def _get_boundaries(annotation: Any) -> np.ndarray:
        """Extract a Boundaries field from one scipy MATLAB annotation object."""
        if isinstance(annotation, dict):
            boundaries = annotation["Boundaries"]
        else:
            boundaries = getattr(annotation, "Boundaries")
        return np.asarray(boundaries, dtype=np.float32)

    def _load_ground_truth(self, mat_path: Path) -> np.ndarray:
        """Average the boundary map from every human annotation."""
        mat_data = scipy.io.loadmat(
            mat_path, squeeze_me=True, struct_as_record=False
        )
        annotations = np.atleast_1d(mat_data["groundTruth"])
        boundary_maps = [self._get_boundaries(annotation) for annotation in annotations]
        return np.mean(np.stack(boundary_maps, axis=0), axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mat_path = self.samples[index]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        edge_map = self._load_ground_truth(mat_path)
        height, width = image.shape[:2]
        if edge_map.shape != (height, width):
            raise ValueError(
                f"Image and boundary dimensions differ: {image.shape[:2]} vs "
                f"{edge_map.shape} for {image_path.name}"
            )
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(f"Image is smaller than {self.patch_size}x{self.patch_size}")

        if self.split == "train":
            top = np.random.randint(0, height - self.patch_size + 1)
            left = np.random.randint(0, width - self.patch_size + 1)
        else:
            top = (height - self.patch_size) // 2
            left = (width - self.patch_size) // 2

        image_patch = image[top : top + self.patch_size, left : left + self.patch_size]
        edge_patch = edge_map[top : top + self.patch_size, left : left + self.patch_size]

        # Convert HWC to CHW and normalize RGB pixels to [0, 1].
        image_tensor = torch.from_numpy(image_patch.transpose(2, 0, 1).copy()).float()
        image_tensor /= 255.0
        edge_tensor = torch.from_numpy(edge_patch[None, ...].copy()).float()
        return image_tensor, edge_tensor


def create_dataloaders(
    root_dir: str | Path = "dataset", batch_size: int = 8
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    train_dataset = BSDSDataset(root_dir, "train")
    val_dataset = BSDSDataset(root_dir, "val")
    test_dataset = BSDSDataset(root_dir, "test")
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


if __name__ == "__main__":
    dataset_root = Path(__file__).parent / "dataset"
    train_loader, val_loader, test_loader = create_dataloaders(dataset_root)
    print(f"Dataset sizes: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    image_batch, edge_batch = next(iter(train_loader))
    print(f"Image tensor shape: {list(image_batch.shape)}")
    print(f"Edge tensor shape: {list(edge_batch.shape)}")
    print(f"Image value range: [{image_batch.min().item():.4f}, {image_batch.max().item():.4f}]")
    print(f"Edge value range: [{edge_batch.min().item():.4f}, {edge_batch.max().item():.4f}]")

    # Save one RGB training patch beside its soft ground-truth edge map.
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    sample_image, sample_edges = train_loader.dataset[0]
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(sample_image.permute(1, 2, 0).numpy())
    axes[0].set_title("RGB training patch")
    axes[1].imshow(sample_edges.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Soft ground-truth edges")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "dataset_sample.png", dpi=150)
    plt.close(figure)
    print(f"Saved sample visualization to {output_dir / 'dataset_sample.png'}")
