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

from .utils import (
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
                missing = [col for col in required_cols if col not in self.metadata_df.columns]
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
                if np.array((pd.notna(meta.get(field)) for field in essential_fields)).all():
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
                        "ImagePositionPatient": tuple(map(float, meta["ImagePositionPatient"])),
                        "NumberofSlices": int(meta["NumberofSlices"]),  # Store as int
                    }
                else:
                    missing_meta = [f for f in essential_fields if pd.isna(meta.get(f))]
                    self.logger.warning(
                        f"Skipping {volume_name} due to missing essential metadata fields: {missing_meta}."
                    )
            else:
                self.logger.warning(f"Skipping {volume_name} as it's not found in metadata CSV.")

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
            self.logger.error(f"Invalid index {idx} requested in get_volume_slice_count.")
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

            # Affine calculation not required if we do not resample
            # original_spacing_xyz = tuple(volume_metadata["XYSpacing"]) + (
            #     volume_metadata["ZSpacing"],
            # )
            # original_affine = get_affine_from_metadata(
            #     image_orientation_patient=volume_metadata["ImageOrientationPatient"],
            #     image_position_patient=volume_metadata["ImagePositionPatient"],
            #     xy_spacing=volume_metadata["XYSpacing"],
            #     z_spacing=volume_metadata["ZSpacing"],
            # )

            # Convert volume to HU (XYZ)
            volume_xyz = convert_to_hu(
                volume_xyz,
                slope=volume_metadata["RescaleSlope"],
                intercept=volume_metadata["RescaleIntercept"],
                hu_min=self.hu_min,
                hu_max=self.hu_max,
            )

            # Resampling changes the number of slices, which can be fixed
            # But resampling is not necessary for Super resolution
            # So, dropped this part
            # But it is necessary for volume generation, so commented out
            # volume_xyz, _ = resample_volume(
            #     volume=volume_xyz,  # Pass XYZ
            #     original_spacing=original_spacing_xyz,  # Pass XYZ
            #     target_spacing=self.target_spacing,  # Pass XYZ
            #     original_affine=original_affine,
            #     interpolation_order=self.interpolation_order,
            #     hu_min=self.hu_min,
            # )

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


#


@DATASET_REGISTRY.register()
class RealESRGANCustomCTDataset(Dataset):
    """
    Custom CT Dataset for Real-ESRGAN.
    Outputs: GT image tensor, kernel1, kernel2, sinc_kernel tensors.
    ALWAYS outputs 3-channel GT images.
    """

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.gt_size = opt["gt_size"]  # Target HR patch size (H, W) for the output GT
        self.depth = opt.get("depth", 1)  # Slices per patch from NIfTI
        self.drop = opt.get("drop", 0)
        self.logger = get_root_logger()

        self.hu_min = opt.get("hu_min", -1000)
        self.hu_max = opt.get("hu_max", 1000)

        target_spacing = opt.get("target_spacing", [0.75, 0.75, 1.5])
        target_shape = opt.get("target_shape", [512, 512, -1])  # (X,Y,Z), Z=-1 for flexibility
        interpolation_order = opt.get("interpolation_order", 1)

        self.base_dataset = CTRateDatasetBase(
            root_dir=opt["dataroot_gt"],
            metadata_csv=opt["metadata_csv"],
            target_spacing=target_spacing,
            target_shape=target_shape,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
            interpolation_order=interpolation_order,
        )

        self.patch_indices = []
        self.logger.info("Calculating total number of patches using NumberofSlices metadata...")
        total_patches = 0
        target_z_dim_config = self.base_dataset.target_shape[2]
        for vol_idx in tqdm(
            range(len(self.base_dataset)), desc="Scanning Volumes for Patch Counts"
        ):
            num_slices_available = 0
            if target_z_dim_config == -1:
                num_slices_available = self.base_dataset.get_volume_slice_count(vol_idx)
                if num_slices_available is None:
                    self.logger.warning(f"Could not get slice count for vol {vol_idx}. Skipping.")
                    continue
            else:
                num_slices_available = target_z_dim_config

            num_patches_in_vol = 0
            if num_slices_available >= self.depth:
                num_patches_in_vol = (num_slices_available - self.depth) // (self.drop + 1) + 1

            if num_patches_in_vol > 0:
                for i in range(num_patches_in_vol):
                    slice_start = i * (self.drop + 1)
                    if slice_start + self.depth <= num_slices_available:
                        self.patch_indices.append((vol_idx, slice_start))
                        total_patches += 1
                    else:
                        break
        self.total_patches = total_patches
        if self.total_patches == 0:
            raise ValueError(f"No patches could be extracted.")
        self.logger.info(f"Total number of extractable patches: {self.total_patches}")

        # --- Kernel Generation Parameters (from dataset opt, like original RealESRGANDataset) ---
        # Blur settings for the first degradation
        self.blur_kernel_size = opt.get("blur_kernel_size", 7)
        self.kernel_list = opt.get(
            "kernel_list",
            [
                "iso",
                "aniso",
                "generalized_iso",
                "generalized_aniso",
                "plateau_iso",
                "plateau_aniso",
            ],
        )
        self.kernel_prob = opt.get(
            "kernel_prob", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        )  # Sum to 1.0
        self.blur_sigma = opt.get("blur_sigma", [0.2, 3.0])
        self.betag_range = opt.get("betag_range", [0.5, 4.0])
        self.betap_range = opt.get("betap_range", [1.0, 2.0])
        self.sinc_prob = opt.get("sinc_prob", 0.1)  # Prob for sinc kernel in first degradation

        # Blur settings for the second degradation
        self.blur_kernel_size2 = opt.get("blur_kernel_size2", 7)
        self.kernel_list2 = opt.get(
            "kernel_list2",
            [
                "iso",
                "aniso",
                "generalized_iso",
                "generalized_aniso",
                "plateau_iso",
                "plateau_aniso",
            ],
        )
        self.kernel_prob2 = opt.get("kernel_prob2", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03])
        self.blur_sigma2 = opt.get("blur_sigma2", [0.2, 1.5])
        self.betag_range2 = opt.get("betag_range2", [0.5, 4.0])
        self.betap_range2 = opt.get("betap_range2", [1.0, 2.0])
        self.sinc_prob2 = opt.get("sinc_prob2", 0.1)  # Prob for sinc kernel in second degradation

        # Final sinc filter
        self.final_sinc_prob = opt.get("final_sinc_prob", 0.8)  # Original higher prob

        # Kernel generation helpers
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1  # Represents no blur, for cases where sinc is not chosen

    def _generate_kernel(
        self,
        kernel_size,
        kernel_list_opt,
        kernel_prob_opt,
        blur_sigma_opt,
        betag_range_opt,
        betap_range_opt,
        sinc_prob_opt,
        fixed_kernel_size=21,
    ):  # fixed_kernel_size for padding
        """Helper to generate a single blur kernel tensor."""
        kernel_size = random.choice(self.kernel_range)  # Randomly select a size from [7, 21]

        if np.random.uniform() < sinc_prob_opt:
            if kernel_size < 13:  # Original logic for sinc omega_c
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                kernel_list_opt,
                kernel_prob_opt,
                kernel_size,  # Use the randomly chosen kernel_size
                blur_sigma_opt,
                blur_sigma_opt,
                [-math.pi, math.pi],  # Default orientation range
                betag_range_opt,
                betap_range_opt,
                noise_range=None,
            )

        # Pad kernel to fixed_kernel_size (e.g., 21)
        pad_size = (fixed_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel)

    def __getitem__(self, index):
        volume_idx, slice_start_idx = self.patch_indices[index]
        max_retries = 5
        for retry in range(max_retries):
            try:
                volume_xyz, volume_name = self.base_dataset.get_processed_volume(volume_idx)
                if volume_xyz is None:
                    self.logger.warning(
                        f"Attempt {retry+1}: Failed loading vol {volume_idx}. Trying next index."
                    )
                    index = (index + 1) % self.total_patches
                    volume_idx, slice_start_idx = self.patch_indices[index]
                    continue

                # Extract patch (X, Y, self.depth)
                patch_xy_d = volume_xyz[:, :, slice_start_idx : slice_start_idx + self.depth]

                # Transpose to HWC (Y, X, self.depth)
                patch_hr_hwd = patch_xy_d.transpose(1, 0, 2)

                # Resize to self.gt_size if necessary
                h_patch, w_patch, d_patch_actual = patch_hr_hwd.shape
                if h_patch != self.gt_size or w_patch != self.gt_size:
                    patch_hr_hwd_resized = cv2.resize(
                        patch_hr_hwd,
                        (self.gt_size, self.gt_size),  # cv2.resize takes (width, height)
                        interpolation=cv2.INTER_CUBIC,
                    )
                    if patch_hr_hwd_resized.ndim == 2 and d_patch_actual == 1:
                        patch_hr_hwd = np.expand_dims(patch_hr_hwd_resized, axis=2)
                    elif (
                        patch_hr_hwd_resized.ndim == 3
                        and patch_hr_hwd_resized.shape[2] == d_patch_actual
                    ):
                        patch_hr_hwd = patch_hr_hwd_resized
                    else:
                        raise RuntimeError(
                            f"Unexpected shape after resizing patch: {patch_hr_hwd_resized.shape}"
                        )

                # Convert HU Patch to Image [0, 255] uint8 (still H, W, self.depth)
                img_gt_float = np.clip(patch_hr_hwd, self.hu_min, self.hu_max)
                img_gt_float = (img_gt_float - self.hu_min) / (self.hu_max - self.hu_min + 1e-6)
                img_gt_uint8_hwd = (img_gt_float * 255.0).round().astype(np.uint8)

                # --- Enforce 3-Channel BGR Output for GT ---
                if self.depth == 1:
                    if img_gt_uint8_hwd.shape[2] != 1:  # Should be (H,W,1)
                        img_gt_uint8_hwd = img_gt_uint8_hwd[:, :, 0:1]  # Safety net
                    img_gt_bgr_hwc = cv2.cvtColor(img_gt_uint8_hwd, cv2.COLOR_GRAY2BGR)
                elif self.depth == 3:
                    img_gt_bgr_hwc = img_gt_uint8_hwd  # Assume it's HWC and BGR-compatible
                else:
                    raise ValueError(
                        f"Dataset depth is {self.depth}. Must be 1 or 3 for 3-channel GT output."
                    )

                # Augmentation (on 3-channel HWC BGR image)
                img_gt_aug_bgr_hwc = augment(
                    img_gt_bgr_hwc,
                    self.opt.get("use_hflip", False),
                    self.opt.get("use_rot", False),
                )

                # --- Convert GT to Tensor (CHW, RGB, [0,1] float32) ---
                # img2tensor expects a list of images if you pass multiple, or a single image
                gt_tensor = img2tensor(img_gt_aug_bgr_hwc, bgr2rgb=True, float32=True)

                # --- Generate Kernels ---
                kernel1 = self._generate_kernel(
                    self.blur_kernel_size,
                    self.kernel_list,
                    self.kernel_prob,
                    self.blur_sigma,
                    self.betag_range,
                    self.betap_range,
                    self.sinc_prob,
                )

                kernel2 = self._generate_kernel(
                    self.blur_kernel_size2,
                    self.kernel_list2,
                    self.kernel_prob2,
                    self.blur_sigma2,
                    self.betag_range2,
                    self.betap_range2,
                    self.sinc_prob2,
                )

                if np.random.uniform() < self.final_sinc_prob:
                    kernel_size = random.choice(self.kernel_range)
                    omega_c = np.random.uniform(np.pi / 3, np.pi)  # from RealESRGANDataset
                    sinc_kernel_np = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                    sinc_kernel = torch.FloatTensor(sinc_kernel_np)
                else:
                    sinc_kernel = self.pulse_tensor  # No final sinc

                gt_path_info = f"{volume_name}_vol{volume_idx}_slice{slice_start_idx}"

                return_dict = {
                    "gt": gt_tensor,
                    "kernel1": kernel1,
                    "kernel2": kernel2,
                    "sinc_kernel": sinc_kernel,
                    "gt_path": gt_path_info,
                }
                return return_dict

            except Exception as e:
                self.logger.error(
                    f"ERROR in __getitem__ for index {index} (Vol {volume_idx}, Slice {slice_start_idx}), attempt {retry+1}: {e}",
                    exc_info=True,
                )
                if retry == max_retries - 1:
                    raise e
                index = (index + 1) % self.total_patches
                volume_idx, slice_start_idx = self.patch_indices[index]

    def __len__(self):
        return self.total_patches


@DATASET_REGISTRY.register()
class RealESRGANCustomCTPairedDataset(Dataset):
    """
    Wrapper Dataset using CTRateDatasetBase.
    ALWAYS outputs 3-channel images.
    Ensures LQ image is ALWAYS self.lq_size.
    """

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.scale = opt["scale"]
        self.gt_size = opt["gt_size"]
        self.lq_size = self.gt_size // self.scale  # CRITICAL for LQ output size
        self.depth = opt.get("depth", 1)
        self.drop = opt.get("drop", 0)
        self.logger = get_root_logger()

        # ... (rest of __init__ is the same as your last version) ...
        self.hu_window_min = opt.get("hu_window_min", -1000)
        self.hu_window_max = opt.get("hu_window_max", 1000)
        target_spacing = opt.get("target_spacing", [0.75, 0.75, 1.5])
        target_shape = opt.get("target_shape", [512, 512, -1])
        interpolation_order = opt.get("interpolation_order", 1)

        self.base_dataset = CTRateDatasetBase(
            root_dir=opt["dataroot_gt"],
            metadata_csv=opt["metadata_csv"],
            target_spacing=target_spacing,
            target_shape=target_shape,
            hu_min=self.hu_window_min,
            hu_max=self.hu_window_max,
            interpolation_order=interpolation_order,
        )

        self.patch_indices = []
        self.logger.info("Calculating total number of patches using NumberofSlices metadata...")
        total_patches = 0
        target_z_dim = self.base_dataset.target_shape[2]
        for vol_idx in tqdm(
            range(len(self.base_dataset)), desc="Scanning Volumes for Patch Counts"
        ):
            num_slices_available = 0
            if target_z_dim == -1:
                num_slices_available = self.base_dataset.get_volume_slice_count(vol_idx)
                if num_slices_available is None:
                    self.logger.warning(f"Could not get slice count for vol {vol_idx}. Skipping.")
                    continue
            else:
                num_slices_available = target_z_dim
            num_patches_in_vol = 0
            if num_slices_available >= self.depth:
                num_patches_in_vol = (num_slices_available - self.depth) // (self.drop + 1) + 1
            if num_patches_in_vol > 0:
                for i in range(num_patches_in_vol):
                    slice_start = i * (self.drop + 1)
                    if slice_start + self.depth <= num_slices_available:
                        self.patch_indices.append((vol_idx, slice_start))
                        total_patches += 1
                    else:
                        break
        self.total_patches = total_patches
        if self.total_patches == 0:
            raise ValueError(f"No patches could be extracted.")
        self.logger.info(f"Total number of extractable patches: {self.total_patches}")

    def _apply_degradations(self, img_gt_hwc_3ch):  # Input is ALWAYS HWC 3-channel uint8 [0,255]
        """
        Applies Real-ESRGAN degradations. Input is HWC 3-channel uint8 [0,255].
        Ensures LQ output is always self.lq_size.
        """
        img_lq_resized = cv2.resize(
            img_gt_hwc_3ch, (self.lq_size, self.lq_size), interpolation=cv2.INTER_LINEAR
        )
        img_lq_output = np.round(img_lq_resized).astype(np.uint8)
        return img_lq_output

    def __getitem__(self, index):
        volume_idx, slice_start_idx = self.patch_indices[index]
        max_retries = 5
        for retry in range(max_retries):
            try:
                volume_xyz, volume_name = self.base_dataset.get_processed_volume(volume_idx)
                if volume_xyz is None:
                    self.logger.warning(
                        f"Attempt {retry+1}: Failed loading vol {volume_idx}. Trying next index."
                    )
                    index = (index + 1) % self.total_patches
                    volume_idx, slice_start_idx = self.patch_indices[index]
                    continue

                patch_xy_d = volume_xyz[:, :, slice_start_idx : slice_start_idx + self.depth]
                patch_hr_hwd = patch_xy_d.transpose(1, 0, 2)

                h_patch, w_patch, d_patch_actual = patch_hr_hwd.shape
                if h_patch != self.gt_size or w_patch != self.gt_size:
                    patch_hr_hwd_resized = cv2.resize(
                        patch_hr_hwd,
                        (self.gt_size, self.gt_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    if patch_hr_hwd_resized.ndim == 2 and d_patch_actual == 1:
                        patch_hr_hwd = np.expand_dims(patch_hr_hwd_resized, axis=2)
                    elif (
                        patch_hr_hwd_resized.ndim == 3
                        and patch_hr_hwd_resized.shape[2] == d_patch_actual
                    ):
                        patch_hr_hwd = patch_hr_hwd_resized
                    else:
                        raise RuntimeError(
                            f"Unexpected shape after resizing patch: {patch_hr_hwd_resized.shape}, expected depth {d_patch_actual}"
                        )

                img_gt_float = np.clip(patch_hr_hwd, self.hu_window_min, self.hu_window_max)
                img_gt_float = (img_gt_float - self.hu_window_min) / (
                    self.hu_window_max - self.hu_min + 1e-6
                )
                img_gt_uint8_hwd = (img_gt_float * 255.0).round().astype(np.uint8)

                if self.depth == 1:
                    if img_gt_uint8_hwd.shape[2] != 1:
                        img_gt_uint8_hwd = img_gt_uint8_hwd[:, :, 0:1]
                    img_gt_final_3ch_hwc = cv2.cvtColor(img_gt_uint8_hwd, cv2.COLOR_GRAY2BGR)
                elif self.depth == 3:
                    img_gt_final_3ch_hwc = img_gt_uint8_hwd
                else:
                    raise ValueError(
                        f"Dataset depth is {self.depth}. To always output 3 channels, input depth must be 1 or 3."
                    )

                img_gt_aug_3ch_hwc = augment(
                    img_gt_final_3ch_hwc,
                    self.opt.get("use_hflip", False),
                    self.opt.get("use_rot", False),
                )
                img_lq_3ch_hwc = self._apply_degradations(img_gt_aug_3ch_hwc.copy())

                gt_tensor = img2tensor(img_gt_aug_3ch_hwc, bgr2rgb=True, float32=True)
                lq_tensor = img2tensor(img_lq_3ch_hwc, bgr2rgb=True, float32=True)

                gt_path_info = f"gt_{volume_name}_vol{volume_idx}_slice{slice_start_idx}"
                lq_path_info = f"lq_{volume_name}_vol{volume_idx}_slice{slice_start_idx}"

                return {
                    "lq": lq_tensor,
                    "gt": gt_tensor,
                    "lq_path": gt_path_info,
                    "gt_path": lq_path_info,
                }
            except Exception as e:
                self.logger.error(
                    f"ERROR in __getitem__ for index {index} (Vol {volume_idx}, Slice {slice_start_idx}), attempt {retry+1}: {e}",
                    exc_info=True,
                )
                if retry == max_retries - 1:
                    raise e
                index = (index + 1) % self.total_patches
                volume_idx, slice_start_idx = self.patch_indices[index]

    def __len__(self):
        return self.total_patches
