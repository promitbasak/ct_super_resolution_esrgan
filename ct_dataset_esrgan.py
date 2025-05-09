import ast
import math
import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import get_root_logger, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import (
    get_affine_from_metadata,
    convert_to_hu,
    resample_volume,
    crop_or_pad,
)


class CTRateDatasetBase(Dataset):
    """
    Base class using XYZ convention and "NumberofSlices" metadata.
    Handles target_shape[2] == -1 for cropping/padding.
    """

    def __init__(
        self,
        root_dir,
        metadata_csv,
        target_spacing=(0.75, 0.75, 1.5),  # Expects (X, Y, Z)
        target_shape=(512, 512, 240),  # Expects (X, Y, Z), Z can be -1
        hu_min=-1000,
        hu_max=1000,
        interpolation_order=1,
    ):
        self.root_dir = Path(root_dir)
        self.target_spacing = tuple(target_spacing)
        self.target_shape = tuple(target_shape)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.interpolation_order = interpolation_order
        self.logger = get_root_logger()

        # --- Metadata Loading and Filtering ---
        try:
            self.metadata_df = pd.read_csv(metadata_csv)
            required_cols = [
                "VolumeName",
                "RescaleSlope",
                "RescaleIntercept",
                "XYSpacing",
                "ZSpacing",
                "ImageOrientationPatient",
                "ImagePositionPatient",
                "NumberofSlices",
            ]
            if not all(col in self.metadata_df.columns for col in required_cols):
                missing = [
                    col for col in required_cols if col not in self.metadata_df.columns
                ]
                raise ValueError(f"Metadata CSV missing required columns: {missing}")
            for col in ["ImageOrientationPatient", "ImagePositionPatient", "XYSpacing"]:
                try:
                    self.metadata_df[col] = self.metadata_df[col].apply(
                        lambda x: ast.literal_eval(str(x)) if pd.notna(x) else None
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error parsing column `{col}` in metadata CSV. Ensure it contains valid list/tuple representations. Error: {e}"
                    )

        except FileNotFoundError:
            self.logger.error(f"Metadata CSV not found at {metadata_csv}")
            raise
        except ValueError as e:
            self.logger.error(f"Error loading or processing metadata: {e}")
            raise

        self.samples = []

        self.metadata_lookup = {}
        all_files = list(self.root_dir.rglob("*.nii.gz"))
        self.logger.info(f"Found {len(all_files)} NIfTI files. Filtering...")
        metadata_indexed = self.metadata_df.set_index("VolumeName")

        num_slices_total = 0
        for nii_path in tqdm(all_files, desc="Filtering Samples"):
            volume_name = nii_path.name
            if volume_name in metadata_indexed.index:
                meta = metadata_indexed.loc[volume_name].to_dict()
                essential_fields = [
                    "RescaleSlope",
                    "RescaleIntercept",
                    "XYSpacing",
                    "ZSpacing",
                    "ImageOrientationPatient",
                    "ImagePositionPatient",
                    "NumberofSlices",
                ]
                if all(pd.notna(meta.get(field)) for field in essential_fields):
                    self.samples.append(nii_path)
                    num_slices_total += meta["NumberofSlices"]
                    self.metadata_lookup[volume_name] = {
                        "RescaleIntercept": float(meta["RescaleIntercept"]),
                        "RescaleSlope": float(meta["RescaleSlope"]),
                        "XYSpacing": tuple(map(float, meta["XYSpacing"])),
                        "ZSpacing": float(meta["ZSpacing"]),
                        "ImageOrientationPatient": tuple(
                            map(float, meta["ImageOrientationPatient"])
                        ),
                        "ImagePositionPatient": tuple(
                            map(float, meta["ImagePositionPatient"])
                        ),
                        "NumberofSlices": int(meta["NumberofSlices"]),  # Store as int
                    }
                else:
                    missing_meta = [f for f in essential_fields if pd.isna(meta.get(f))]
                    self.logger.warning(
                        f"Skipping {volume_name} due to missing essential metadata fields: {missing_meta}."
                    )
            else:
                self.logger.warning(
                    f"Skipping {volume_name} as it's not found in metadata CSV."
                )

        self.num_slices_total = num_slices_total
        if not self.samples:
            raise ValueError(
                "No valid samples found after filtering. Check paths, metadata content, and required columns."
            )
        self.logger.info(f"Kept {len(self.samples)} valid samples after filtering.")

    def __len__(self):
        return len(self.samples)

    def get_volume_slice_count(self, idx):
        """Returns the original number of slices for a given volume index using metadata."""
        if idx < 0 or idx >= len(self.samples):
            self.logger.error(
                f"Invalid index {idx} requested in get_volume_slice_count."
            )
            return None
        volume_name = self.samples[idx].name
        if volume_name in self.metadata_lookup:
            return self.metadata_lookup[volume_name].get("NumberofSlices")
        else:
            self.logger.error(
                f"Metadata not found for volume {volume_name} (index {idx}) in lookup table."
            )
            return None

    def get_processed_volume(self, idx):
        """Loads and preprocesses a single volume using XYZ convention throughout."""
        nii_path = self.samples[idx]
        volume_name = nii_path.name
        volume_metadata = self.metadata_lookup[volume_name]

        try:
            # Load NIfTI (XYZ)
            img_nii = nib.load(nii_path)
            volume_xyz = img_nii.get_fdata(dtype=np.float32)
            # img_nii.uncache()

            # Get Original Spacing (X, Y, Z) and Affine
            original_spacing_xyz = tuple(volume_metadata["XYSpacing"]) + (
                volume_metadata["ZSpacing"],
            )
            original_affine = get_affine_from_metadata(
                image_orientation_patient=volume_metadata["ImageOrientationPatient"],
                image_position_patient=volume_metadata["ImagePositionPatient"],
                xy_spacing=volume_metadata["XYSpacing"],
                z_spacing=volume_metadata["ZSpacing"],
            )

            # Convert volume to HU (XYZ)
            volume_xyz = convert_to_hu(
                volume_xyz,
                slope=volume_metadata["RescaleSlope"],
                intercept=volume_metadata["RescaleIntercept"],
                hu_min=self.hu_min,
                hu_max=self.hu_max,
            )

            volume_xyz, _ = resample_volume(
                volume=volume_xyz,  # Pass XYZ
                original_spacing=original_spacing_xyz,  # Pass XYZ
                target_spacing=self.target_spacing,  # Pass XYZ
                original_affine=original_affine,
                interpolation_order=self.interpolation_order,
                hu_min=self.hu_min,
            )

            # Crop or Pad Volume (XYZ -> XYZ)
            volume_xyz = crop_or_pad(
                volume=volume_xyz,  # Pass XYZ
                target_shape_in=self.target_shape,  # Pass flexible shape (X, Y, Z or -1)
                pad_value=float(self.hu_min),
            )

            return volume_xyz, nii_path.name  # Return XYZ

        except Exception as e:
            self.logger.error(f"Error processing volume {nii_path}: {e}", exc_info=True)
            return None, nii_path.name


# --- Updated RealESRGANCustomCTDataset Wrapper ---
@DATASET_REGISTRY.register()
class RealESRGANCustomCTDataset(Dataset):
    """
    Wrapper Dataset using CTRateDatasetBase
    """

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.scale = opt["scale"]
        self.gt_size = opt["gt_size"]  # Target HR patch size (H, W)
        self.lq_size = self.gt_size // self.scale
        self.depth = opt.get("depth", 1)  # Slices per patch
        self.drop = opt.get("drop", 0)  # Slice drop
        self.logger = get_root_logger()

        self.hu_window_min = opt.get("hu_window_min", -1000)
        self.hu_window_max = opt.get("hu_window_max", 1000)

        target_spacing = opt.get("target_spacing", [0.75, 0.75, 1.5])
        target_shape = opt.get("target_shape", [512, 512, -1])  # Default Z to -1
        interpolation_order = opt.get("interpolation_order", 3)

        self.base_dataset = CTRateDatasetBase(
            root_dir=opt["dataroot_gt"],
            metadata_csv=opt["metadata_csv"],
            target_spacing=target_spacing,
            target_shape=target_shape,
            hu_min=self.hu_window_min,
            hu_max=self.hu_window_max,
            interpolation_order=interpolation_order,
        )

        # --- Pre-calculate total patches using NumberofSlices metadata ---
        self.patch_indices = []  # List of (volume_idx, slice_start_idx)
        self.logger.info(
            "Calculating total number of patches using NumberofSlices metadata..."
        )
        total_patches = 0
        target_z_dim = self.base_dataset.target_shape[2]

        for vol_idx in tqdm(
            range(len(self.base_dataset)), desc="Scanning Volumes for Patch Counts"
        ):
            num_slices_available = 0
            if target_z_dim == -1:
                num_slices_available = self.base_dataset.get_volume_slice_count(vol_idx)
                if num_slices_available is None:
                    self.logger.warning(
                        f"Could not get slice count for volume index {vol_idx} ({self.base_dataset.samples[vol_idx].name}). Skipping patch calculation."
                    )
                    continue
            else:
                num_slices_available = target_z_dim

            # Calculate patches extractable from the available slices
            num_patches_in_vol = (num_slices_available - self.depth) // (
                self.drop + 1
            ) + 1
            for i in range(num_patches_in_vol):
                slice_start = i * (self.drop + 1)
                self.patch_indices.append((vol_idx, slice_start))
            total_patches += num_patches_in_vol

        self.total_patches = total_patches
        if self.total_patches == 0:
            raise ValueError(
                f"No patches could be extracted. Check target_shape, patch depth/drop, and `NumberofSlices` in metadata."
            )
        self.logger.info(f"Total number of extractable patches: {self.total_patches}")
        # --- End Patch Calculation ---

        # --- Degradation parameters (same as previous) ---
        self.blur_kernel_size = opt.get("blur_kernel_size", 0)
        self.kernel_list = opt.get("kernel_list", [])
        self.kernel_prob = opt.get(
            "kernel_prob", [0 for _ in range(len(self.kernel_list))]
        )
        self.blur_sigma = opt["blur_sigma"]
        self.betag_range = opt["betag_range"]
        self.betap_range = opt["betap_range"]
        self.sinc_prob = opt.get("sinc_prob", 0.1)
        self.blur_kernel_size2 = opt.get("blur_kernel_size2")
        self.kernel_list2 = opt.get("kernel_list2")
        self.kernel_prob2 = opt.get("kernel_prob2")
        self.blur_sigma2 = opt.get("blur_sigma2")
        self.betag_range2 = opt.get("betag_range2")
        self.betap_range2 = opt.get("betap_range2")
        self.sinc_prob2 = opt.get("sinc_prob2", 0.1)
        self.noise_range = opt.get("noise_range")
        self.jpeg_range = opt.get("jpeg_range")
        self.final_sinc_prob = opt.get("final_sinc_prob", 0.1)
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    # --- _apply_degradations method remains the same ---
    def _apply_degradations(self, img_gt):
        """Applies the Real-ESRGAN degradation pipeline. Assumes img_gt is HWC uint8 [0, 255]."""
        h_gt, w_gt, _ = img_gt.shape
        img_lq = img_gt.copy()

        # Optional probability for applying degradation stages
        degrad_prob1 = self.opt.get("degrad_prob1", 0.5)
        degrad_prob2 = self.opt.get("degrad_prob2", 0.5)
        noise_prob = self.opt.get("noise_prob", 0)
        jpeg_prob = self.opt.get("jpeg_prob", 0)

        # 1st stage blur+downsample
        if np.random.uniform() < degrad_prob1:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
            if np.random.uniform() < self.sinc_prob:
                kernel_size = (
                    random.choice(self.kernel_range)
                    if self.blur_kernel_size < 7
                    else self.blur_kernel_size
                )
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                kernel = circular_lowpass_kernel(
                    omega_c, kernel_size, pad_to=self.blur_kernel_size
                )
            d = self.blur_kernel_size // 2
            img_lq_padded = cv2.copyMakeBorder(
                img_lq, d, d, d, d, cv2.BORDER_REFLECT_101
            )
            img_lq = cv2.filter2D(img_lq_padded, -1, kernel)[d:-d, d:-d, :]
            img_lq = cv2.resize(
                img_lq, (self.lq_size, self.lq_size), interpolation=cv2.INTER_LINEAR
            )

        # 2nd stage blur
        if self.kernel_list2 and np.random.uniform() < degrad_prob2:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                self.blur_kernel_size2,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )
            if np.random.uniform() < self.sinc_prob2:
                kernel_size2 = (
                    random.choice(self.kernel_range)
                    if self.blur_kernel_size2 < 7
                    else self.blur_kernel_size2
                )
                omega_c2 = np.random.uniform(np.pi / 3, np.pi)
                kernel2 = circular_lowpass_kernel(
                    omega_c2, kernel_size2, pad_to=self.blur_kernel_size2
                )
            d2 = self.blur_kernel_size2 // 2
            img_lq_padded = cv2.copyMakeBorder(
                img_lq, d2, d2, d2, d2, cv2.BORDER_REFLECT_101
            )
            img_lq = cv2.filter2D(img_lq_padded, -1, kernel2)[d2:-d2, d2:-d2, :]

        # Noise
        if self.noise_range is not None and np.random.uniform() < noise_prob:
            noise_level = np.random.uniform(self.noise_range[0], self.noise_range[1])
            img_lq_float = img_lq.astype(np.float32) / 255.0
            noise = np.random.normal(0, noise_level / 255.0, img_lq_float.shape).astype(
                np.float32
            )
            img_lq_float = img_lq_float + noise
            img_lq = np.clip(img_lq_float * 255.0, 0, 255)

        # JPEG
        # if self.jpeg_range is not None and np.random.uniform() < jpeg_prob:
        #     img_lq_uint8 = np.round(img_lq).astype(np.uint8)
        #     jpeg_quality = int(
        #         np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
        #     )
        #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        #     _, encimg = cv2.imencode(".jpg", img_lq_uint8, encode_param)
        #     img_lq = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)

        # Final Sinc
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            pad_to_size = (
                self.blur_kernel_size2 if self.kernel_list2 else self.blur_kernel_size
            )
            sinc_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=pad_to_size
            )
            ds = pad_to_size // 2
            img_lq_padded = cv2.copyMakeBorder(
                img_lq, ds, ds, ds, ds, cv2.BORDER_REFLECT_101
            )
            img_lq = cv2.filter2D(img_lq_padded, -1, sinc_kernel)[ds:-ds, ds:-ds, :]

        img_lq = np.round(img_lq).astype(np.uint8)
        return img_lq

    def __getitem__(self, index):
        volume_idx, slice_start_idx = self.patch_indices[index]
        max_retries = 5
        for retry in range(max_retries):
            try:
                volume_xyz, volume_name = self.base_dataset.get_processed_volume(
                    volume_idx
                )
                if volume_xyz is None:
                    self.logger.warning(
                        f"Attempt {retry+1}: Failed loading vol {volume_idx}. Trying next index."
                    )
                    index = (index + 1) % self.total_patches
                    volume_idx, slice_start_idx = self.patch_indices[index]
                    continue

                # Extract patch (slice along Z from XYZ) -> Shape (X, Y, depth)
                patch_xy_d = volume_xyz[
                    :, :, slice_start_idx : slice_start_idx + self.depth
                ]

                # Transpose to HWC (Y, X, depth) for image processing
                patch_hr_hwd = patch_xy_d.transpose(1, 0, 2)  # Y -> H, X -> W

                h_patch, w_patch, _ = patch_hr_hwd.shape
                if h_patch != self.gt_size or w_patch != self.gt_size:
                    patch_hr_hwd = cv2.resize(
                        patch_hr_hwd,
                        (self.gt_size, self.gt_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    if patch_hr_hwd.ndim == 2:
                        patch_hr_hwd = np.expand_dims(patch_hr_hwd, axis=2)

                img_gt_float = np.clip(
                    patch_hr_hwd, self.hu_window_min, self.hu_window_max
                )
                img_gt_float = (img_gt_float - self.hu_window_min) / (
                    self.hu_window_max - self.hu_min + 1e-6
                )
                img_gt = (img_gt_float * 255.0).round().astype(np.uint8)

                if self.depth == 1 and img_gt.shape[2] == 1:
                    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2BGR)
                elif self.depth == 3:
                    pass
                else:
                    raise ValueError(f"Unsupported depth {self.depth}.")

                # Augmentation (HWC)
                img_gt = augment(
                    img_gt,
                    self.opt.get("use_hflip", False),
                    self.opt.get("use_rot", False),
                )

                # Generate LQ (HWC)
                img_lq = self._apply_degradations(img_gt.copy())

                # Convert to Tensor (CHW)
                img_gt_tensor = img2tensor(img_gt, bgr2rgb=True, float32=True)
                img_lq_tensor = img2tensor(img_lq, bgr2rgb=True, float32=True)

                gt_path_info = f"{volume_name}_vol{volume_idx}_slice{slice_start_idx}"
                return {
                    "lq": img_lq_tensor,
                    "gt": img_gt_tensor,
                    "gt_path": gt_path_info,
                }

            except Exception as e:
                self.logger.error(
                    f"ERROR in __getitem__ index {index} (Vol {volume_idx}, Slice {slice_start_idx}) attempt {retry+1}: {e}",
                    exc_info=True,
                )
                if retry == max_retries - 1:
                    raise e
                index = (index + 1) % self.total_patches
                volume_idx, slice_start_idx = self.patch_indices[index]

    def __len__(self):
        return self.total_patches
