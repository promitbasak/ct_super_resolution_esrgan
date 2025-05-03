import ast
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from utils import convert_to_hu, crop_or_pad, resample_volume, extract_patches


class CustomTransform:
    def __call__(self, image):
        # Normalize to [0, 1]
        image = (image + 1000.0) / 2000.0  # HU values are in [-1000, 1000] range
        T.ToTensor(),  # Convert image to PyTorch tensor
        T.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor values to [-1, 1]
        return image


class CTRateDataset(Dataset):
    def __init__(
        self,
        root_dir,
        metadata_csv,
        transform=None,
        depth=3,
        size_low=128,
        size_high=512,
        drop=0,
        target_spacing=(0.75, 0.75, 1.5),  # x,y,z
        target_shape=(512, 512, 240),
    ):
        """
        Args:
            root_dir (str): Directory where the .nii.gz files are located.
            metadata_csv (str): Path to the CSV file containing metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_spacing (tuple): Desired spacing for resampling (default is (0.75, 0.75, 1.5)).
            target_shape (tuple): Desired shape for padding/cropping (default is (480, 480, 240)).
            depth (int): Number of consecutive slices to include in each patch.
            size (int): Desired size for each patch (default is 128).
            drop (int): Number of slices to skip between patches.
        """
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_csv(metadata_csv)  # Load metadata
        self.transform = transform
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.depth = depth
        self.size_low = size_low
        self.size_high = size_high
        self.drop = drop
        self.samples = list(self.root_dir.rglob("*.nii.gz"))
        self.valid_samples = self.filter_invalid_samples()

    def filter_invalid_samples(self):
        """Filters out samples with NaN values in critical metadata columns."""
        valid_samples = []
        invalid_samples = self.metadata.loc[
            self.metadata["ZSpacing"].isna(), "VolumeName"
        ]
        for sample in self.samples:
            if sample.name not in invalid_samples:
                valid_samples.append(sample)
        return valid_samples

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        nii_path = self.valid_samples[idx]
        volume_name = nii_path.name
        volume_metadata = self.get_metadata_for_volume(volume_name)

        # Load the NIfTI image
        img = nib.load(nii_path)
        volume = img.get_fdata()

        # Use metadata for RescaleSlope, RescaleIntercept, and spacing
        rescale_intercept = volume_metadata["RescaleIntercept"]
        rescale_slope = volume_metadata["RescaleSlope"]
        spacing = ast.literal_eval(volume_metadata["XYSpacing"]) + [
            int(volume_metadata["ZSpacing"])
        ]

        # 1. Convert volume to HU using metadata
        volume = convert_to_hu(volume, rescale_slope, rescale_intercept)

        # 2. Resample volume to target spacing
        volume = resample_volume(
            volume, spacing[::-1], target_spacing=self.target_spacing
        )

        # 3. Crop or pad the volume to the desired shape
        volume = crop_or_pad(volume, target_shape=self.target_shape)

        # 5. Extract patches (slices) from the volume
        hr_patches = extract_patches(volume, self.depth, self.drop, self.size_high)
        lr_patches = extract_patches(volume, self.depth, self.drop, self.size_low)

        # Apply any transformations (if provided)
        if self.transform:
            lr_patches = [self.transform(patch) for patch in lr_patches]
            hr_patches = [self.transform(patch) for patch in hr_patches]

        # Return a tuple of (low-resolution patches, high-resolution target)
        return lr_patches, hr_patches


def create_dataloader(
    data_dir,
    metadata_csv,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    depth=3,
    size_low=128,
    size_high=512,
    drop=0,
    target_spacing=(0.75, 0.75, 1.5),
    target_shape=(512, 512, 240),
):
    # Define the transformation for the data (optional)
    transform = CustomTransform()

    # Initialize the dataset
    dataset = CTRateDataset(
        data_dir,
        metadata_csv,
        transform,
        depth,
        size_low,
        size_high,
        drop,
        target_spacing,
        target_shape,
    )

    # Create DataLoader for batching
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader
