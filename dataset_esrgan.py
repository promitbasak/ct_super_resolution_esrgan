import ast
import math
import random
from pathlib import Path
import warnings
from typing import List

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import convert_to_hu, resample_volume, crop_or_pad


class CTRateDatasetBase(Dataset):
    """
    Base class to load and preprocess NIfTI volumes.
    Modified to facilitate patch extraction by a wrapper.
    """

    def __init__(
        self,
        root_dir,
        metadata_csv,
        target_spacing=(0.75, 0.75, 1.5),
        target_shape=(512, 512, 240),
    ):
        self.root_dir = Path(root_dir)
        self.metadata_df = pd.read_csv(metadata_csv)
        self.target_spacing = target_spacing
        self.target_shape = target_shape

        # Pre-filter samples and store metadata lookup
        self.samples: List[Path]
        self.samples = []
        self.num_slices = 0
        self.metadata_lookup = {}
        all_files = list(self.root_dir.rglob("*.nii.gz"))
        logger = get_root_logger()
        logger.info(
            f"Found {len(all_files)} NIfTI files. Filtering and reading metadata..."
        )

        # Create a quick lookup for metadata
        self.metadata_df.set_index("VolumeName", inplace=True)

        for nii_path in tqdm(all_files, desc="Filtering Samples"):
            volume_name = nii_path.name
            if volume_name in self.metadata_df.index:
                meta = self.metadata_df.loc[volume_name]
                # Check essential fields
                if (
                    pd.notna(meta.get("NumberofSlices"))
                    and pd.notna(meta.get("ZSpacing"))
                    and pd.notna(meta.get("RescaleIntercept"))
                    and pd.notna(meta.get("RescaleSlope"))
                    and pd.notna(meta.get("XYSpacing"))
                ):
                    # Store necessary metadata fields only
                    self.metadata_lookup[volume_name] = {
                        "NumberofSlices": meta["NumberofSlices"],
                        "RescaleIntercept": meta["RescaleIntercept"],
                        "RescaleSlope": meta["RescaleSlope"],
                        "XYSpacing": ast.literal_eval(meta["XYSpacing"]),
                        "ZSpacing": float(meta["ZSpacing"]),
                    }
                    self.samples.append(nii_path)
                    self.num_slices += (
                        int(meta["NumberofSlices"])
                        if self.target_shape[2] == -1
                        else self.target_shape[2]
                    )

                else:
                    logger.warning(
                        f"Skipping {volume_name} due to missing metadata (ZSpacing, Intercept, Slope, or XYSpacing)."
                    )
            else:
                logger.warning(
                    f"Skipping {volume_name} as it's not found in metadata CSV."
                )

        if not self.samples:
            raise ValueError(
                "No valid samples found after filtering. Check paths and metadata."
            )
        logger.info(f"Kept {len(self.samples)} valid samples after filtering.")

    def __len__(self):
        return len(self.samples)

    def get_processed_volume(self, idx):
        """Loads and preprocesses a single volume."""
        nii_path: Path
        nii_path = self.samples[idx]
        volume_name = nii_path.name
        volume_metadata = self.metadata_lookup[volume_name]

        try:
            img = nib.load(nii_path)
            volume = img.get_fdata(dtype=np.float32)
            # img_header = img.header
            # img.uncache()

            rescale_intercept = volume_metadata["RescaleIntercept"]
            rescale_slope = volume_metadata["RescaleSlope"]
            spacing = tuple(map(float, volume_metadata["XYSpacing"])) + (
                volume_metadata["ZSpacing"],
            )

            # 1. Convert volume to HU
            volume = convert_to_hu(volume, rescale_slope, rescale_intercept)

            # 2. Resample volume
            current_spacing_zyx = (spacing[2], spacing[1], spacing[0])
            volume = resample_volume(
                volume, current_spacing_zyx, target_spacing=self.target_spacing
            )

            # 3. Crop or pad
            target_shape = [
                volume.shape[i] if shape == -1 else shape
                for i, shape in enumerate(self.target_shape)
            ]
            volume = crop_or_pad(volume, target_shape=target_shape)

            return volume, nii_path.name

        except Exception as e:
            logger = get_root_logger()
            logger.error(f"Error processing volume {nii_path}: {e}")
            return None, nii_path.name


# --- RealESRGAN Wrapper Dataset ---
@DATASET_REGISTRY.register()
class RealESRGANCustomCTDataset(Dataset):
    """
    Wrapper Dataset for Real-ESRGAN fine-tuning using CTRateDatasetBase.
    Extracts single HR patches, applies HU windowing/scaling,
    and generates LQ images on the fly using Real-ESRGAN degradations.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt["scale"]
        self.gt_size = opt["gt_size"]  # The target HR patch size for training
        self.lq_size = self.gt_size // self.scale
        self.depth = opt.get("depth", 1)  # Number of slices per patch (default 1)
        self.drop = opt.get("drop", 0)  # Slice drop (usually 0 if depth=1)

        self.hu_window_min = opt.get("hu_window_min", -1000)
        self.hu_window_max = opt.get("hu_window_max", 1000)

        # Initialize the base dataset loader
        self.base_dataset = CTRateDatasetBase(
            root_dir=opt["dataroot_gt"],  # Use dataroot_gt for convention
            metadata_csv=opt["metadata_csv"],
            target_spacing=opt["target_spacing"],
            target_shape=opt["target_shape"],
        )

        # Pre-calculate total patches and index mapping
        self.patch_indices = []  # List of (volume_idx, slice_start_idx)
        logger = get_root_logger()
        logger.info("Calculating total number of patches...")
        total_patches = 0
        for vol_idx in tqdm(
            range(len(self.base_dataset)), desc="Scanning Volumes for Patches"
        ):
            # Get volume shape *after* preprocessing (use target_shape)
            num_slices = self.base_dataset.num_slices

            # Calculate number of patches in this volume
            num_patches_in_vol = (
                0
                if num_slices < self.depth
                else (num_slices - self.depth) // (self.drop + 1) + 1
            )

            if num_patches_in_vol > 0:
                for i in range(num_patches_in_vol):
                    slice_start = i * (self.drop + 1)
                    self.patch_indices.append((vol_idx, slice_start))
                total_patches += num_patches_in_vol
            else:
                logger.warning(
                    f"Volume {vol_idx} ({self.base_dataset.samples[vol_idx].name}) has {num_slices} slices, fewer than depth {self.depth}. Skipping."
                )

        self.total_patches = total_patches
        if self.total_patches == 0:
            raise ValueError(
                f"No patches could be extracted with depth={self.depth}, drop={self.drop}. Check dataset and parameters."
            )

        logger.info(f"Total number of extractable patches: {self.total_patches}")

        # blur settings for the first degradation
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.kernel_list = opt["kernel_list"]
        self.kernel_prob = opt["kernel_prob"]  # a list for each kernel probability
        self.blur_sigma = opt["blur_sigma"]
        self.betag_range = opt[
            "betag_range"
        ]  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt["betap_range"]  # betap used in plateau blur kernels
        self.sinc_prob = opt["sinc_prob"]  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt["blur_kernel_size2"]
        self.kernel_list2 = opt["kernel_list2"]
        self.kernel_prob2 = opt["kernel_prob2"]
        self.blur_sigma2 = opt["blur_sigma2"]
        self.betag_range2 = opt["betag_range2"]
        self.betap_range2 = opt["betap_range2"]
        self.sinc_prob2 = opt["sinc_prob2"]

        self.noise_range = opt["noise_range"]
        self.jpeg_range = opt["jpeg_range"]

        # a final sinc filter
        self.final_sinc_prob = opt["final_sinc_prob"]

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def _apply_degradations(self, img_gt):
        """Applies the Real-ESRGAN degradation pipeline."""
        h_gt, w_gt, _ = img_gt.shape  # Assumes HWC

        # --- Apply 1st degradation stage ---
        # 1. Blur
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
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        # 2. Downsample (Resize)
        # Determine resize dimensions carefully based on gt_size and scale
        # If img_gt patch IS self.gt_size, then lq should be self.lq_size
        # But the blur happens first. Let's assume we resize *after* blur.
        # Original Real-ESRGAN might resize the *kernel* instead of the image sometimes. Check BasicSR code if needed.
        # Simpler: Resize blurred image to target LQ size.
        img_lq = cv2.resize(
            img_lq, (self.lq_size, self.lq_size), interpolation=cv2.INTER_LINEAR
        )  # Use self.lq_size

        # --- Apply 2nd degradation stage (Optional) ---
        if self.kernel_list2:
            # 3. Blur (2nd)
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.opt["kernel_prob2"],
                self.blur_kernel_size2,
                self.opt["blur_sigma2"],
                self.opt["blur_sigma2"],
                [-math.pi, math.pi],
                self.opt["betag_range2"],
                self.opt["betap_range2"],
                noise_range=None,
            )
            if np.random.uniform() < self.opt.get("sinc_prob2", 0.1):
                kernel_size2 = (
                    random.choice(self.kernel_range)
                    if self.blur_kernel_size2 < 7
                    else self.blur_kernel_size2
                )
                omega_c2 = np.random.uniform(np.pi / 3, np.pi)
                kernel2 = circular_lowpass_kernel(
                    omega_c2, kernel_size2, pad_to=self.blur_kernel_size2
                )
            img_lq = cv2.filter2D(img_lq, -1, kernel2)

        # 4. Add noise (Gaussian)
        if self.noise_range is not None:
            noise_level = np.random.uniform(self.noise_range[0], self.noise_range[1])
            noise = np.random.normal(0, noise_level / 255.0, img_lq.shape).astype(
                np.float32
            )
            # Add noise in float domain, then clip
            img_lq = (
                img_lq / 255.0 + noise
            ) * 255.0  # Add noise after scaling img to [0,1]
            img_lq = np.clip(img_lq, 0, 255)

        # # 5. Add JPEG compression
        # if self.jpeg_range is not None:
        #     # Convert to uint8 before JPEG
        #     img_lq_uint8 = np.round(img_lq).astype(np.uint8)
        #     jpeg_quality = int(
        #         np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
        #     )
        #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        #     _, encimg = cv2.imencode(".jpg", img_lq_uint8, encode_param)
        #     img_lq = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)  # Read back channels
        #     # Convert back to float if subsequent steps need it, otherwise keep uint8 is fine before tensor conversion

        # # 6. Final Sinc filter (optional) - applied after noise/jpeg
        # if np.random.uniform() < self.final_sinc_prob:
        #     kernel_size = random.choice(self.kernel_range)
        #     omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     sinc_kernel = circular_lowpass_kernel(
        #         omega_c,
        #         kernel_size,
        #         pad_to=self.blur_kernel_size2 or self.blur_kernel_size,
        #     )  # Pad to a valid kernel size
        #     img_lq = cv2.filter2D(img_lq, -1, sinc_kernel)

        # Round and cast final LQ to uint8
        img_lq = np.round(img_lq).astype(np.uint8)

        return img_lq

    def __getitem__(self, index):
        # Map global index to volume and patch index
        volume_idx, slice_start_idx = self.patch_indices[index]

        # Get the fully preprocessed volume (Z, Y, X)
        volume, volume_name = self.base_dataset.get_processed_volume(volume_idx)
        if volume is None:
            # Handle error: return dummy data or skip? For simplicity, let's retry another index.
            warnings.warn(
                f"Failed to load volume {volume_idx}, returning sample 0 instead."
            )
            return self.__getitem__(
                0
            )  # Recursive call - potential issue if 0 fails. Better error handling needed in production.

        # Extract the HR patch (slices)
        # Assuming volume is ZYX: [slice_start_idx : slice_start_idx + self.depth, :, :]
        # We need YXZ or HWD usually for image processing. Let's assume output should be (H, W, D)
        slices: np.ndarray
        slices = volume[
            slice_start_idx : slice_start_idx + self.depth, :, :
        ]  # Shape: (depth, H, W) e.g. (1, 512, 512)
        # Transpose to HWD if needed, e.g., (H, W, depth)
        patch_hr = slices.transpose(1, 2, 0)  # Shape: (H, W, depth) e.g. (512, 512, 1)

        # Crop/Resize HR patch to self.gt_size if needed
        # The base dataset already crops/pads volume to target_shape (e.g., 512x512 slices).
        # If self.gt_size is different, we need to crop/resize *here*.
        h_vol, w_vol, _ = patch_hr.shape
        if h_vol != self.gt_size or w_vol != self.gt_size:
            # Random crop if volume slice is larger
            if h_vol > self.gt_size and w_vol > self.gt_size:
                top = random.randint(0, h_vol - self.gt_size)
                left = random.randint(0, w_vol - self.gt_size)
                patch_hr = patch_hr[
                    top : top + self.gt_size, left : left + self.gt_size, :
                ]
            else:
                # Resize if smaller or non-square (adjust logic as needed)
                patch_hr = cv2.resize(
                    patch_hr,
                    (self.gt_size, self.gt_size),
                    interpolation=cv2.INTER_CUBIC,
                )  # Use high-quality resize
                if (
                    patch_hr.ndim == 2
                ):  # Handle resize output for single channel depth=1
                    patch_hr = np.expand_dims(patch_hr, axis=2)

        # --- Convert HU to Image Format (e.g., [0, 255] uint8) ---
        # Apply windowing and scaling
        patch_hr = np.clip(patch_hr, self.hu_window_min, self.hu_window_max)
        # Scale to 0-1, then 0-255
        patch_hr = (patch_hr - self.hu_window_min) / (
            self.hu_window_max - self.hu_window_min + 1e-6
        )
        img_gt = (patch_hr * 255.0).astype(np.uint8)

        # --- Handle Channels (Depth) ---
        # Assuming depth=1 for now, and replicating for 3-channel model
        if self.depth == 1 and img_gt.shape[2] == 1:
            img_gt = cv2.cvtColor(
                img_gt, cv2.COLOR_GRAY2BGR
            )  # Shape: (gt_size, gt_size, 3)
        elif self.depth == 3:
            # Already (gt_size, gt_size, 3), assuming correct channel order
            pass
        else:
            # Need to handle other depths or modify model input channels
            raise ValueError(
                f"Unsupported depth {self.depth}. Model likely expects 1 or 3 channels."
            )

        # --- Augmentation (on HR) ---
        img_gt = augment(img_gt, self.opt["use_hflip"], self.opt["use_rot"])

        # --- Generate LQ image using degradations ---
        # Degradations expect HWC, uint8 [0, 255]
        img_lq = self._apply_degradations(img_gt.copy())  # Pass a copy

        # --- Convert to Tensor ---
        # bgr2rgb=True because cvtColor makes BGR. float32=True for [0,1] range.
        img_gt_tensor = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq_tensor = img2tensor(img_lq, bgr2rgb=True, float32=True)

        gt_path_info = f"{volume_name}_vol{volume_idx}_slice{slice_start_idx}"
        return {"lq": img_lq_tensor, "gt": img_gt_tensor, "gt_path": gt_path_info}

    def __len__(self):
        return self.total_patches
