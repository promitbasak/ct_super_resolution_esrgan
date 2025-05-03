import ast
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import torch


def get_metadata_for_volume(metadata, volume_name):
    volume_metadata = metadata[metadata["VolumeName"] == volume_name]

    if not volume_metadata.empty:
        return volume_metadata.iloc[0].to_dict()
    else:
        raise ValueError(f"No metadata found for volume: {volume_name}")


def convert_to_hu(image, slope, intercept):
    """Convert image data to Hounsfield Units using slope and intercept."""
    hu_image = image * slope + intercept
    hu_image = np.clip(hu_image, -1000, 1000)  # Clip HU to the valid range
    return hu_image


def resample_volume(volume, current_spacing, target_spacing=(0.75, 0.75, 1.5)):
    """Resample the volume to the desired spacing using zoom."""
    zoom_factors = [curr / tgt for curr, tgt in zip(current_spacing, target_spacing)]
    resampled = zoom(volume, zoom=zoom_factors, order=1)  # Linear interpolation
    return resampled


def crop_or_pad(volume, target_shape=(480, 480, 240)):
    """Center crop or pad the volume to the target shape."""
    output = np.zeros(target_shape, dtype=np.float32)
    min_shape = np.minimum(volume.shape, target_shape)
    start_src = [(s - m) // 2 for s, m in zip(volume.shape, min_shape)]
    start_dst = [(t - m) // 2 for t, m in zip(target_shape, min_shape)]

    output[
        start_dst[0] : start_dst[0] + min_shape[0],
        start_dst[1] : start_dst[1] + min_shape[1],
        start_dst[2] : start_dst[2] + min_shape[2],
    ] = volume[
        start_src[0] : start_src[0] + min_shape[0],
        start_src[1] : start_src[1] + min_shape[1],
        start_src[2] : start_src[2] + min_shape[2],
    ]
    return output


def extract_patches(volume, depth=5, drop=2, size=128):
    slices = volume.shape[2]
    outputs = []
    idx = 0
    while idx + depth <= slices:
        chunk = volume[:, :, idx : idx + depth]  # Shape: H x W x depth
        chunk = np.transpose(chunk, (2, 0, 1))  # Shape: depth x H x W
        chunk = resize(chunk, (depth, size, size), mode="constant")
        outputs.append(torch.tensor(chunk, dtype=torch.float32))
        idx += depth + drop
    return outputs


def preprocess_ct_rate_volume(
    nii_path,
    metadata,
    output_path,
    target_spacing=(0.75, 0.75, 1.5),
    target_shape=(480, 480, 240),
):
    """Preprocess the CT volume by using metadata from the CSV."""

    # Extract the volume name (without extension)
    volume_name = Path(nii_path).name

    # Get metadata for the volume from the CSV
    volume_metadata = get_metadata_for_volume(metadata, volume_name)

    # Load the NIfTI image
    img = nib.load(nii_path)
    volume = img.get_fdata()
    affine = img.affine

    # Use metadata for RescaleSlope and RescaleIntercept
    rescale_intercept = volume_metadata["RescaleIntercept"]
    rescale_slope = volume_metadata["RescaleSlope"]

    # 1. Convert to HU using metadata
    volume = convert_to_hu(volume, rescale_slope, rescale_intercept)

    # 2. Resample volume to target spacing
    spacing = ast.literal_eval(volume_metadata["XYSpacing"]) + [
        int(volume_metadata["ZSpacing"])
    ]
    volume = resample_volume(volume, spacing[::-1], target_spacing=target_spacing)

    # 3. Crop or pad volume to target shape
    volume = crop_or_pad(volume, target_shape=target_shape)

    # 4. Save the preprocessed volume as .nii.gz
    new_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_img, output_path.with_suffix(".nii.gz"))


def batch_preprocess_ct_rate(input_root="dataset", output_root="preprocessed"):
    input_root = Path(input_root)
    output_root = Path(output_root)

    for split in ["train", "valid"]:
        split_dir = input_root / split
        for case_folder in split_dir.iterdir():
            if case_folder.is_dir():
                for nii_file in case_folder.glob("*.nii.gz"):
                    out_file = output_root / split / case_folder.name / nii_file.stem
                    preprocess_ct_rate_volume(nii_file, out_file)
