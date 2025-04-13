from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader


class CTDataset(Dataset):
    def __init__(
        self, data_dir, transform=None, slice_depth=5, slice_drop=2, target_size=128
    ):
        """
        Args:
            data_dir (str): Path to the directory where .nii.gz files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            slice_depth (int): Number of slices to be taken as input tensor channels.
            slice_drop (int): Number of slices to drop between tensors.
            target_size (int): The target size for resizing images.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.slice_depth = slice_depth
        self.slice_drop = slice_drop
        self.target_size = target_size
        self.samples = []  # List to store file paths

        # Traverse through dataset directory and gather all .nii.gz file paths
        for split in ["train", "valid"]:  # Assuming we have train/valid directories
            split_path = self.data_dir / split
            for case_folder in split_path.iterdir():
                if case_folder.is_dir():
                    for nii_file in case_folder.glob("*.nii.gz"):
                        self.samples.append(str(nii_file))  # Append path to the list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the CT volume from the .nii.gz file
        nii_file = self.samples[idx]
        volume = self.load_nii(nii_file)

        # Extract patches with the slice depth and drop strategy
        patches = self.extract_patches(volume)

        # Apply any transformations (e.g., normalization, augmentation) if provided
        if self.transform:
            patches = [self.transform(patch) for patch in patches]

        return patches

    def load_nii(self, filepath):
        """Load NIfTI image and normalize it."""
        img = nib.load(filepath).get_fdata()
        img = np.clip(img, -1000, 1000)  # HU windowing
        img = (img + 1000) / 2000  # Normalize to 0–1
        return img

    def extract_patches(self, volume):
        """Extract 3D patches with given slice depth and drop strategy."""
        inputs = []
        total_slices = volume.shape[2]
        i = 0

        while i + self.slice_depth <= total_slices:
            stack = volume[:, :, i : i + self.slice_depth]  # H x W x depth
            stack = np.transpose(stack, (2, 0, 1))  # depth x H x W
            stack_resized = np.zeros(
                (self.slice_depth, self.target_size, self.target_size)
            )

            for j in range(self.slice_depth):
                stack_resized[j] = resize(
                    stack[j], (self.target_size, self.target_size), preserve_range=True
                )

            tensor = torch.tensor(stack_resized, dtype=torch.float32)
            inputs.append(tensor)
            i += self.slice_depth + self.slice_drop

        return inputs


class CustomTransform:
    def __call__(self, tensor):
        # Normalizing the input tensor to [-1, 1] (optional, can be skipped)
        return (tensor - 0.5) * 2  # If needed, normalize to [-1, 1] from [0, 1]


def create_dataloader(data_dir, batch_size=8, shuffle=True, num_workers=4):
    # Define the transformation for the data (optional)
    transform = CustomTransform()

    # Initialize the dataset
    dataset = CTDataset(data_dir=data_dir, transform=transform)

    # Create DataLoader for batching
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader
