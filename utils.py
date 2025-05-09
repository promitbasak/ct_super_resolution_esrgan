import ast
from pathlib import Path
import warnings
import pandas as pd
from typing import List

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import imageio
import SimpleITK as sitk


def get_metadata_for_volume(metadata_df: pd.DataFrame, volume_name: str) -> dict:
    """Fetches metadata for a specific volume name from the DataFrame."""
    volume_metadata = metadata_df[metadata_df["VolumeName"] == volume_name]
    if not volume_metadata.empty:
        return volume_metadata.iloc[0].to_dict()
    else:
        raise ValueError(f"No metadata found for volume: {volume_name}")


def get_affine_from_metadata(
    image_orientation_patient: np.ndarray,
    image_position_patient: np.ndarray,
    xy_spacing: List,
    z_spacing: float,
):
    row_vec = np.array(image_orientation_patient[0:3])
    col_vec = np.array(image_orientation_patient[3:6])
    slice_vec = np.cross(row_vec, col_vec)

    di = xy_spacing[0]
    dj = xy_spacing[1]
    dk = z_spacing

    origin = np.array(image_position_patient)

    affine = np.eye(4)
    affine[0:3, 0] = row_vec * di
    affine[0:3, 1] = col_vec * dj
    affine[0:3, 2] = slice_vec * dk
    affine[0:3, 3] = origin
    return affine


def convert_to_hu(
    image: np.ndarray,
    slope: float,
    intercept: float,
    hu_min: int = -1000,
    hu_max: int = 1000,
) -> np.ndarray:
    """
    Convert image data (numpy array) to Hounsfield Units using slope and intercept.
    """
    if slope is None or not np.isfinite(slope) or slope == 0:
        warnings.warn(f"Invalid RescaleSlope received: {slope}. Using 1.0.")
        slope = 1.0
    if intercept is None or not np.isfinite(intercept):
        warnings.warn(f"Invalid RescaleIntercept received: {intercept}. Using 0.0.")
        intercept = 0.0

    hu_image = image.astype(np.float32) * slope + intercept
    hu_image = np.clip(hu_image, hu_min, hu_max)
    return hu_image


def resample_volume(
    volume: np.ndarray,
    original_spacing: tuple,
    target_spacing: tuple,
    original_affine: np.ndarray,
    interpolation_order: int = 3,
    hu_min: int = -1000,
) -> tuple[np.ndarray, np.ndarray]:  # Returns volume AND updated affine
    """Resample using SimpleITK, handling geometry correctly."""

    original_image = sitk.GetImageFromArray(volume)
    original_image.SetSpacing(original_spacing)
    direction_matrix = original_affine[:3, :3] / np.array(original_spacing)
    direction_matrix_flat = direction_matrix.flatten()
    original_image.SetDirection(direction_matrix_flat.tolist())
    origin = original_affine[:3, 3].tolist()
    original_image.SetOrigin(origin)

    original_size = np.array(original_image.GetSize())
    spacing_ratio = np.array(original_spacing) / np.array(target_spacing)
    new_size = np.round(original_size * spacing_ratio).astype(int).tolist()

    if interpolation_order == 0:
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation_order == 1:
        interpolator = sitk.sitkLinear
    elif interpolation_order == 3:
        interpolator = sitk.sitkBSpline
    else:
        warnings.warn(
            f"SimpleITK resampling: Unsupported interpolation order {interpolation_order}. Using Linear."
        )
        interpolator = sitk.sitkLinear

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetInterpolator(interpolator)

    resample_filter.SetOutputOrigin(original_image.GetOrigin())
    resample_filter.SetOutputDirection(original_image.GetDirection())

    # Set default pixel value an minimum HU
    default_pixel_value = hu_min
    resample_filter.SetDefaultPixelValue(default_pixel_value)

    resampled_image = resample_filter.Execute(original_image)
    resampled_volume = sitk.GetArrayFromImage(resampled_image)

    new_spacing = resampled_image.GetSpacing()
    new_origin = resampled_image.GetOrigin()
    new_direction_flat = resampled_image.GetDirection()
    new_direction = np.array(new_direction_flat).reshape(3, 3)

    new_affine = np.eye(4)
    new_affine[:3, :3] = new_direction * np.array(new_spacing)
    new_affine[:3, 3] = new_origin

    return resampled_volume, new_affine


def crop_or_pad(
    volume: np.ndarray,
    target_shape_in: tuple = (512, 512, -1),
    pad_value: float = -1000.0,
) -> np.ndarray:
    """
    Center crop or pad the volume (numpy array) to the target shape.
    Revised implementation using np.pad and slicing for robustness.

    Args:
        volume: Numpy array representing the volume data (X, Y, Z).
        target_shape_in: Desired shape (X, Y, Z). If Z is -1, keeps original Z dim.
                         Use tuple to avoid mutable default issues.
        pad_value: Value to use for padding (should typically be hu_min).

    Returns:
        The cropped or padded volume.
    """
    current_shape = np.array(volume.shape)
    target_shape_list = list(target_shape_in)

    if target_shape_list[2] == -1:
        target_shape_list[2] = current_shape[2]
    target_shape = np.array(target_shape_list)

    # Calculate differences: positive means padding needed, negative means cropping needed
    diff = target_shape - current_shape

    # Calculate padding amounts (before, after) for each dimension
    pad_before = np.maximum(0, diff) // 2
    pad_after = np.maximum(0, diff) - pad_before
    pad_width = tuple(zip(pad_before, pad_after))  # Format for np.pad

    # Calculate cropping amounts (before, after) for each dimension
    crop_before = np.maximum(0, -diff) // 2
    crop_after = np.maximum(0, -diff) - crop_before

    # Create slices for cropping
    crop_slices = tuple(
        slice(cb, cs - ca) for cb, ca, cs in zip(crop_before, crop_after, current_shape)
    )

    # Perform cropping first
    cropped_volume = volume[crop_slices]

    # Then perform padding
    # Ensure pad_value has the same dtype as the volume to avoid warnings/errors
    padded_volume = np.pad(
        cropped_volume,
        pad_width,
        mode="constant",
        constant_values=np.array(pad_value, dtype=volume.dtype),
    )

    # Final check (should match exactly if logic is correct)
    if padded_volume.shape != tuple(target_shape):
        warnings.warn(
            f"Final shape {padded_volume.shape} doesn't match target {tuple(target_shape)}. This might indicate an issue in calculation."
        )
        # Implement a hard crop/pad if necessary, though the above should work
        # Example: Force shape by creating target array and placing data
        final_vol = np.full(tuple(target_shape), pad_value, dtype=volume.dtype)
        min_final_shape = np.minimum(padded_volume.shape, target_shape)
        final_crop_slices = tuple(slice(0, m) for m in min_final_shape)
        final_vol[final_crop_slices] = padded_volume[final_crop_slices]
        return final_vol

    return padded_volume


def update_affine(
    original_affine: np.ndarray, original_spacing: tuple, target_spacing: tuple
) -> np.ndarray:
    """Updates the affine matrix to reflect new voxel spacing after resampling."""
    if not len(original_spacing) == len(target_spacing) == 3:
        raise ValueError("Spacings must be 3D tuples (X, Y, Z)")
    if any(s == 0 for s in original_spacing):
        raise ValueError(
            f"Original spacing contains zero, cannot calculate scale factors: {original_spacing}"
        )

    new_affine = original_affine.copy()
    scale_factors = np.array(target_spacing, dtype=float) / np.array(
        original_spacing, dtype=float
    )
    new_affine[:3, :3] = original_affine[:3, :3] @ np.diag(scale_factors)
    return new_affine


# --- Slice Saving Function (Unchanged) ---
def save_slices(
    volume: np.ndarray,
    output_dir: str,
    base_filename: str,
    slice_axis: int = 2,
    img_format: str = "png",
    hu_range: tuple[int, int] = (-1000, 1000),
):
    """Saves slices of a 3D volume as individual 2D image files."""
    if slice_axis < 0 or slice_axis >= volume.ndim:
        raise ValueError(
            f"Invalid slice_axis {slice_axis} for volume ndim {volume.ndim}"
        )

    num_slices = volume.shape[slice_axis]
    slice_output_path = Path(output_dir)
    slice_output_path.mkdir(parents=True, exist_ok=True)

    hu_min, hu_max = hu_range
    range_diff = float(hu_max - hu_min)
    if range_diff <= 0:
        warnings.warn(
            f"HU range max <= min ({hu_range}). Normalization might produce uniform images."
        )
        range_diff = 1.0

    zfill_count = len(str(num_slices - 1))

    for i in range(num_slices):
        slice_data = np.take(volume, i, axis=slice_axis).astype(np.float32)
        normalized_slice = np.clip((slice_data - hu_min) / range_diff, 0.0, 1.0)

        if img_format.lower() in ["tif", "tiff"]:
            img_slice = (normalized_slice * 65535).astype(np.uint16)
        else:
            img_slice = (normalized_slice * 255).astype(np.uint8)

        slice_filename = (
            f"{base_filename}_slice_{str(i).zfill(zfill_count)}.{img_format}"
        )
        output_filepath = slice_output_path / slice_filename

        try:
            imageio.imwrite(output_filepath, np.ascontiguousarray(img_slice))
        except Exception as e:
            print(f"  Error saving slice {i} to {output_filepath}: {e}")

    print(f"Finished saving slices.")


def preprocess_ct_volume(
    nii_path: Path,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    target_spacing: tuple = (0.75, 0.75, 1.5),
    target_shape: tuple = (480, 480, -1),
    hu_range: tuple = (-1000, 1000),
    resample_order: int = 1,
    save_nifti: bool = True,
    save_slice_images: bool = True,
    slice_axis: int = 2,
    slice_format: str = "png",
):
    """Preprocesses a CT volume using external metadata and saves results."""
    volume_name = nii_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = nii_path.stem

    try:
        # 1. Get Metadata
        try:
            volume_metadata = get_metadata_for_volume(metadata_df, volume_name)
        except ValueError:
            warnings.warn(f"No metadata found for {volume_name}. Skipping.")
            return
        rescale_intercept = volume_metadata.get("RescaleIntercept")
        rescale_slope = volume_metadata.get("RescaleSlope")
        xy_spacing = ast.literal_eval(volume_metadata.get("XYSpacing", "None"))
        z_spacing = volume_metadata.get("ZSpacing")
        image_position_patient = ast.literal_eval(
            volume_metadata.get("ImagePositionPatient", "None")
        )
        image_orientation_patient = ast.literal_eval(
            volume_metadata.get("ImageOrientationPatient", "None")
        )

        if (
            rescale_intercept is None
            or rescale_slope is None
            or xy_spacing is None
            or z_spacing is None
            or image_position_patient is None
            or image_orientation_patient is None
        ):
            raise ValueError(f"Missing critical metadata for {volume_name}")

        if not isinstance(xy_spacing, (list, tuple)) or len(xy_spacing) != 2:
            raise ValueError("XYSpacing format")
        z_spacing = float(z_spacing)
        original_spacing = tuple(xy_spacing) + (z_spacing,)

        # 2. Load NIfTI Data and Affine
        img = nib.load(nii_path)
        volume = img.get_fdata(dtype=np.float32)
        original_affine = get_affine_from_metadata(
            image_orientation_patient, image_position_patient, xy_spacing, z_spacing
        )

        affine_spacing = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))
        if not np.allclose(affine_spacing, original_spacing, atol=0.1):
            warnings.warn(
                f"Calculated affine spacing {affine_spacing} differs significantly from metadata spacing {original_spacing}."
            )

        # 3. Convert to HU
        hu_min, hu_max = hu_range
        volume_hu = convert_to_hu(
            volume, rescale_slope, rescale_intercept, hu_min, hu_max
        )

        # 4. Resample volume
        volume_resampled, updated_affine = resample_volume(
            volume=volume_hu,
            original_spacing=original_spacing,
            target_spacing=target_spacing,
            original_affine=original_affine,
            interpolation_order=resample_order,
        )

        # 5. Calculate the updated affine matrix
        updated_affine = update_affine(
            original_affine, original_spacing, target_spacing
        )

        # 6. Crop or pad volume
        volume_processed = crop_or_pad(
            volume_resampled, target_shape_in=target_shape, pad_value=float(hu_min)
        )

        # 7. Save Results
        if save_nifti:
            nifti_filename = output_dir / f"{base_name}_processed.nii.gz"
            new_img = nib.Nifti1Image(
                volume_processed.astype(np.float32), updated_affine
            )
            nib.save(new_img, nifti_filename)

        if save_slice_images:
            slice_output_subdir = output_dir / f"{base_name}_slices_{slice_format}"
            save_slices(
                volume=volume_processed,
                output_dir=str(slice_output_subdir),
                base_filename=base_name,
                slice_axis=slice_axis,
                img_format=slice_format,
                hu_range=hu_range,
            )

    except Exception as e:
        print(f"!!! FAILED processing {volume_name}: {type(e).__name__} - {e}")
        import traceback

        traceback.print_exc()


def batch_preprocess_ct_rate(
    input_root: str,
    output_root: str,
    metadata_csv: str,
    target_spacing: tuple = (0.75, 0.75, 1.5),
    target_shape: tuple = (512, 512, -1),
    hu_range: tuple = (-1000, 1000),
    resample_order: int = 1,
    save_nifti: bool = False,
    save_slices: bool = True,
    slice_axis: int = 2,
    slice_format: str = "png",
):
    """Runs preprocessing on all NIfTI files found in train/valid subdirs."""
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)

    try:
        metadata_df = pd.read_csv(metadata_csv)
        print(f"Loaded metadata from {metadata_csv}")
        if "VolumeName" not in metadata_df.columns:
            raise ValueError("'VolumeName' column missing")
    except Exception as e:
        print(f"Error loading metadata CSV: {e}")
        return

    for split in ["train", "valid"]:
        split_dir = input_root_path / split
        if not split_dir.is_dir():
            print(f"Warning: Input split directory not found: {split_dir}")
            continue

        for case_folder in split_dir.iterdir():
            if case_folder.is_dir():
                for nii_file in case_folder.glob("*.nii.gz"):
                    out_vol_dir = output_root_path / split / case_folder.name
                    preprocess_ct_volume(
                        nii_path=nii_file,
                        metadata_df=metadata_df,
                        output_dir=out_vol_dir,
                        target_spacing=target_spacing,
                        target_shape=target_shape,
                        hu_range=hu_range,
                        resample_order=resample_order,
                        save_nifti=save_nifti,
                        save_slice_images=save_slices,
                        slice_axis=slice_axis,
                        slice_format=slice_format,
                    )
