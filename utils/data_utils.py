"""
Data processing utilities for Sentinel-2 image loading, preprocessing, augmentation, and dataset creation.
Specifically supports 6-band data (blue, green, red, NIR, SWIR1, SWIR2).
"""
import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import rasterio
import logging
import random
from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger(__name__)

def load_sentinel2_image(image_path: str, bands: List[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Load Sentinel-2 image with optional band selection.
    
    Args:
        image_path: Path to the image file
        bands: List of band indices to load; loads all bands if None
        
    Returns:
        tuple: (loaded image data with shape [bands, height, width], image metadata profile)
    """
    try:
        with rasterio.open(image_path) as src:
            height, width = src.shape
            count = src.count
            
            if bands is None:
                bands = list(range(1, count + 1))
            else:
                valid_bands = [b for b in bands if 1 <= b <= count]
                if not valid_bands:
                    raise ValueError(f"No valid band indices. File contains {count} bands.")
                bands = valid_bands
            
            try:
                image = np.zeros((len(bands), height, width), dtype=np.float32)
                
                for i, band_idx in enumerate(bands):
                    try:
                        band_data = src.read(band_idx)
                        image[i] = band_data.astype(np.float32)
                        del band_data
                    except MemoryError:
                        logger.warning(f"Insufficient memory. Reading band {band_idx} row by row.")
                        for row in range(height):
                            try:
                                row_data = src.read(band_idx, window=((row, row+1), (0, width)))
                                image[i, row, :] = row_data.astype(np.float32)
                                del row_data
                            except MemoryError:
                                logger.warning(f"Row reading failed. Reading band {band_idx} in chunks.")
                                chunk_size = max(1, height // 10)
                                for chunk_start in range(0, height, chunk_size):
                                    chunk_end = min(chunk_start + chunk_size, height)
                                    try:
                                        chunk_data = src.read(band_idx, window=((chunk_start, chunk_end), (0, width)))
                                        image[i, chunk_start:chunk_end, :] = chunk_data.astype(np.float32)
                                        del chunk_data
                                    except MemoryError as e:
                                        raise IOError(f"Failed to allocate memory for band {band_idx} even with chunked reading: {str(e)}")
                    
                    if i % 2 == 0:
                        import gc
                        gc.collect()
                
                profile = src.profile
                return image, profile
                
            except MemoryError:
                logger.warning(f"Failed to allocate memory for full image. Using chunked processing.")
                
                band_chunks = []
                chunk_height = max(1, height // 4)
                
                for band_idx in bands:
                    band_data = np.zeros((height, width), dtype=np.float32)
                    
                    for chunk_start in range(0, height, chunk_height):
                        chunk_end = min(chunk_start + chunk_height, height)
                        
                        try:
                            chunk_data = src.read(band_idx, window=((chunk_start, chunk_end), (0, width)))
                            band_data[chunk_start:chunk_end, :] = chunk_data.astype(np.float32)
                            del chunk_data
                        except MemoryError as e:
                            raise IOError(f"Failed to allocate memory for band {band_idx} chunk [{chunk_start}:{chunk_end}]: {str(e)}")
                        
                        import gc
                        gc.collect()
                    
                    band_chunks.append(band_data)
                
                try:
                    image = np.stack(band_chunks, axis=0)
                    del band_chunks
                    profile = src.profile
                    return image, profile
                except MemoryError as e:
                    raise IOError(f"Failed to stack bands into full image: {str(e)}")
                    
    except Exception as e:
        raise IOError(f"Failed to load image file: {image_path}. Error: {str(e)}")

def load_mask(mask_path: str) -> np.ndarray:
    """
    Load water segmentation mask.
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        Mask data with shape [1, height, width], values 0 or 1
    """
    try:
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
        except:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        
        return mask
    except Exception as e:
        raise IOError(f"Failed to load mask file: {mask_path}. Error: {str(e)}")

def normalize_image(image: np.ndarray, method: str = 'minmax', params: Dict = None) -> np.ndarray:
    """
    Normalize image data.
    
    Args:
        image: Input image array
        method: Normalization method: 'minmax', 'zscore', 'sentinel'
        params: Normalization parameters (min, max, mean, std)
        
    Returns:
        Normalized image array
    """
    normalized_image = image
    
    if method == 'minmax':
        for i in range(image.shape[0]):
            band_min = params.get('min', {}).get(i, image[i].min()) if params else image[i].min()
            band_max = params.get('max', {}).get(i, image[i].max()) if params else image[i].max()
            if band_max - band_min > 0:
                normalized_image[i] = (image[i] - band_min) / (band_max - band_min)
            else:
                normalized_image[i] = 0
    
    elif method == 'zscore':
        for i in range(image.shape[0]):
            band_mean = params.get('mean', {}).get(i, image[i].mean()) if params else image[i].mean()
            band_std = params.get('std', {}).get(i, image[i].std()) if params else image[i].std()
            if band_std > 0:
                normalized_image[i] = (image[i] - band_mean) / band_std
            else:
                normalized_image[i] = image[i] - band_mean
    
    elif method == 'sentinel':
        sentinel_means = np.array([1379.82, 1287.89, 1210.28, 1055.91, 2199.17, 1595.94])
        sentinel_stds = np.array([156.83, 235.39, 298.42, 462.18, 537.47, 503.59])
        
        num_bands = min(image.shape[0], len(sentinel_means))
        
        for i in range(num_bands):
            mean = params.get('mean', {}).get(i, sentinel_means[i]) if params else sentinel_means[i]
            std = params.get('std', {}).get(i, sentinel_stds[i]) if params else sentinel_stds[i]
            if std > 0:
                normalized_image[i] = (image[i] - mean) / std
            else:
                normalized_image[i] = image[i] - mean
            
        normalized_image = np.clip(normalized_image, -3, 3)
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized_image

def augment_image_and_mask(image: np.ndarray, mask: np.ndarray, 
                          rotation_range: float = 15, 
                          width_shift_range: float = 0.1, 
                          height_shift_range: float = 0.1, 
                          scale_range: Tuple[float, float] = (0.9, 1.1), 
                          horizontal_flip: bool = True, 
                          vertical_flip: bool = True, 
                          brightness_range: Tuple[float, float] = (0.9, 1.1), 
                          contrast_range: Tuple[float, float] = (0.9, 1.1),
                          saturation_range: Optional[Tuple[float, float]] = None,
                          hue_range: Optional[Tuple[float, float]] = None,
                          noise_std: float = 0.0,
                          elastic_alpha: float = 1000,
                          elastic_sigma: float = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply spatial and photometric augmentation to image and mask.
    
    Args:
        image: Input image
        mask: Input mask
        rotation_range: Rotation angle range
        width_shift_range: Horizontal shift range
        height_shift_range: Vertical shift range
        scale_range: Scaling range
        horizontal_flip: Enable horizontal flip
        vertical_flip: Enable vertical flip
        brightness_range: Brightness adjustment range
        contrast_range: Contrast adjustment range
        saturation_range: Saturation adjustment (RGB only)
        hue_range: Hue adjustment (RGB only)
        noise_std: Gaussian noise standard deviation
        elastic_alpha: Elastic deformation alpha
        elastic_sigma: Elastic deformation sigma
        
    Returns:
        Augmented image and mask
    """
    _, height, width = image.shape
    transform_matrix = np.eye(3)
    
    if rotation_range > 0:
        angle = random.uniform(-rotation_range, rotation_range)
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        transform_matrix[:2, :] = rotation_matrix
    
    if width_shift_range > 0 or height_shift_range > 0:
        tx = random.uniform(-width_shift_range, width_shift_range) * width
        ty = random.uniform(-height_shift_range, height_shift_range) * height
        transform_matrix[0, 2] += tx
        transform_matrix[1, 2] += ty
    
    if scale_range is not None and len(scale_range) == 2:
        zoom = random.uniform(scale_range[0], scale_range[1])
        transform_matrix[0, 0] *= zoom
        transform_matrix[1, 1] *= zoom
        
        transform_matrix[0, 2] = (1 - zoom) * width / 2 + transform_matrix[0, 2]
        transform_matrix[1, 2] = (1 - zoom) * height / 2 + transform_matrix[1, 2]
    
    augmented_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        augmented_image[i] = cv2.warpAffine(
            image[i], transform_matrix[:2, :], (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    
    augmented_mask = cv2.warpAffine(
        mask[0], transform_matrix[:2, :], (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
    )
    augmented_mask = np.expand_dims(augmented_mask, axis=0)
    
    if horizontal_flip and random.random() > 0.5:
        augmented_image = np.flip(augmented_image, axis=2).copy()
        augmented_mask = np.flip(augmented_mask, axis=2).copy()
    
    if vertical_flip and random.random() > 0.5:
        augmented_image = np.flip(augmented_image, axis=1).copy()
        augmented_mask = np.flip(augmented_mask, axis=1).copy()
    
    if brightness_range is not None and random.random() > 0.5:
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        augmented_image = augmented_image * brightness_factor
    
    if contrast_range is not None and random.random() > 0.5:
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
        for i in range(augmented_image.shape[0]):
            mean = augmented_image[i].mean()
            augmented_image[i] = (augmented_image[i] - mean) * contrast_factor + mean
    
    if (saturation_range is not None and 
        random.random() > 0.5 and 
        augmented_image.shape[0] >= 3 and
        saturation_range[0] != saturation_range[1]):
        print("Warning: Saturation adjustment may not be suitable for Sentinel-2 multispectral data.")
        saturation_factor = random.uniform(saturation_range[0], saturation_range[1])
        rgb_image = np.transpose(augmented_image[:3], (1, 2, 0))
        rgb_norm = np.clip(rgb_image / 255.0, 0, 1) if rgb_image.max() > 1 else rgb_image
        hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 1)
        rgb_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if rgb_image.max() > 1:
            rgb_adjusted *= 255
        augmented_image[:3] = np.transpose(rgb_adjusted, (2, 0, 1))
    
    if (hue_range is not None and 
        random.random() > 0.5 and 
        augmented_image.shape[0] >= 3 and
        hue_range[0] != hue_range[1]):
        print("Warning: Hue adjustment may not be suitable for Sentinel-2 multispectral data.")
        hue_shift = random.uniform(hue_range[0], hue_range[1])
        rgb_image = np.transpose(augmented_image[:3], (1, 2, 0))
        rgb_norm = np.clip(rgb_image / 255.0, 0, 1) if rgb_image.max() > 1 else rgb_image
        hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2HSV)
        hue_shift_scaled = hue_shift * 90
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_scaled) % 180
        rgb_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if rgb_image.max() > 1:
            rgb_adjusted *= 255
        augmented_image[:3] = np.transpose(rgb_adjusted, (2, 0, 1))
    
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, augmented_image.shape).astype(np.float32)
        augmented_image += noise
    
    if elastic_alpha > 0 and elastic_sigma > 0:
        shape = augmented_image.shape[1:]
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), elastic_sigma) * elastic_alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), elastic_sigma) * elastic_alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        for i in range(augmented_image.shape[0]):
            augmented_image[i] = cv2.remap(augmented_image[i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        augmented_mask[0] = cv2.remap(augmented_mask[0], map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    return augmented_image, augmented_mask

def create_sentinel2_transform(normalize_method: str = 'sentinel'):
    """
    Create transformation pipeline for Sentinel-2 data.
    
    Args:
        normalize_method: Normalization method
        
    Returns:
        Transform function
    """
    def transform(image):
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
            if image_tensor.dim() == 3 and image_tensor.shape[2] <= image_tensor.shape[0]:
                image_tensor = image_tensor.permute(2, 0, 1)
        else:
            image_tensor = image.float()
        
        if normalize_method == 'sentinel':
            pass
        
        return image_tensor
    
    return transform

class Sentinel2WaterDataset(Dataset):
    """
    Dataset class for Sentinel-2 water segmentation.
    Loads image-mask pairs from specified directories.
    """
    def __init__(self, data_dir: str, split: str = 'train', bands: List[int] = None, 
                 augment: bool = False, normalize_method: str = 'sentinel', 
                 splits_dir: str = None, images_dir: str = None, masks_dir: str = None,
                 augmentation_config: Dict = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            bands: Band indices to use
            augment: Enable augmentation
            normalize_method: Normalization method
            splits_dir: Directory with split text files
            images_dir: Custom image directory
            masks_dir: Custom mask directory
            augmentation_config: Augmentation parameters
        """
        self.data_dir = data_dir
        self.split = split.lower()
        self.bands = bands
        self.augment = augment
        self.normalize_method = normalize_method
        self.splits_dir = splits_dir
        
        if augmentation_config is None:
            self.augmentation_config = {
                'rotation_range': 15,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'scale_range': [0.9, 1.1],
                'horizontal_flip': True,
                'vertical_flip': True,
                'brightness_range': [0.9, 1.1],
                'contrast_range': [0.9, 1.1],
                'saturation_range': [0.9, 1.1],
                'hue_range': [-0.1, 0.1],
                'noise_std': 0.0,
                'elastic_alpha': 1000,
                'elastic_sigma': 8
            }
        else:
            self.augmentation_config = augmentation_config
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}. Use train/val/test.")
        
        if images_dir is not None:
            self.image_dir = images_dir
        else:
            self.image_dir = os.path.join(data_dir, self.split, 'images')
            
        if masks_dir is not None:
            self.mask_dir = masks_dir
        else:
            self.mask_dir = os.path.join(data_dir, self.split, 'masks')
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        
        if self.splits_dir is not None:
            split_file = os.path.join(self.splits_dir, f"{self.split}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            
            with open(split_file, 'r') as f:
                file_names = [line.strip() for line in f.readlines() if line.strip()]
            
            self.image_files = []
            for name in file_names:
                for ext in ['.tif', '.tiff', '.png']:
                    img_file = name + ext
                    if os.path.exists(os.path.join(self.image_dir, img_file)):
                        self.image_files.append(img_file)
                        break
                else:
                    if '_msk_' in name:
                        img_name = name.replace('_msk_', '_img_')
                        for ext in ['.tif', '.tiff', '.png']:
                            img_file = img_name + ext
                            if os.path.exists(os.path.join(self.image_dir, img_file)):
                                self.image_files.append(img_file)
                                break
                        else:
                            raise FileNotFoundError(f"Image not found for: {name} or {img_name}")
                    else:
                        raise FileNotFoundError(f"Image not found for: {name}")
        else:
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                      if f.endswith(('.tif', '.tiff', '.png'))])
        
        self.mask_files = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = None
            
            for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(potential_mask):
                    mask_file = base_name + ext
                    break
            
            if mask_file is None and '_img_' in base_name:
                mask_base_name = base_name.replace('_img_', '_msk_')
                for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                    potential_mask = os.path.join(self.mask_dir, mask_base_name + ext)
                    if os.path.exists(potential_mask):
                        mask_file = mask_base_name + ext
                        break
            
            if mask_file is None and self.splits_dir is not None:
                img_base_name = os.path.splitext(img_file)[0]
                if '_img_' in img_base_name:
                    original_mask_name = img_base_name.replace('_img_', '_msk_')
                    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                        potential_mask = os.path.join(self.mask_dir, original_mask_name + ext)
                        if os.path.exists(potential_mask):
                            mask_file = original_mask_name + ext
                            break
            
            if mask_file is None:
                raise FileNotFoundError(f"No matching mask found for image: {img_file}")
            
            self.mask_files.append(mask_file)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image, _ = load_sentinel2_image(image_path, self.bands)
        mask = load_mask(mask_path)
        
        if self.augment and self.split == 'train':
            image, mask = augment_image_and_mask(
                image, mask, 
                rotation_range=self.augmentation_config.get('rotation_range', 15),
                width_shift_range=self.augmentation_config.get('width_shift_range', 0.1),
                height_shift_range=self.augmentation_config.get('height_shift_range', 0.1),
                scale_range=self.augmentation_config.get('scale_range', [0.9, 1.1]),
                horizontal_flip=self.augmentation_config.get('horizontal_flip', True),
                vertical_flip=self.augmentation_config.get('vertical_flip', True),
                brightness_range=self.augmentation_config.get('brightness_range', [0.9, 1.1]),
                contrast_range=self.augmentation_config.get('contrast_range', [0.9, 1.1]),
                saturation_range=self.augmentation_config.get('saturation_range', [0.9, 1.1]),
                hue_range=self.augmentation_config.get('hue_range', [-0.1, 0.1]),
                noise_std=self.augmentation_config.get('noise_std', 0.0),
                elastic_alpha=self.augmentation_config.get('elastic_alpha', 1000),
                elastic_sigma=self.augmentation_config.get('elastic_sigma', 8)
            )
        
        image = normalize_image(image, method=self.normalize_method)
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()
        
        del image
        del mask
        
        if self.split == 'train' or idx % 10 == 0:
            import gc
            gc.collect()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': self.image_files[idx]
        }

class SingleImageDataset(Dataset):
    """
    Dataset for single image inference.
    Uses identical preprocessing as Sentinel2WaterDataset.
    """
    def __init__(self, image_path: str, mask_path: str, bands: List[int] = None, 
                 normalize_method: str = 'sentinel'):
        self.image_path = image_path
        self.mask_path = mask_path
        self.normalize_method = normalize_method
        self.image_filename = os.path.basename(image_path)
        
        if bands is None:
            self.bands = [1, 2, 3, 4, 5, 6]
        else:
            self.bands = bands
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        valid_methods = ['minmax', 'zscore', 'sentinel']
        if normalize_method not in valid_methods:
            raise ValueError(f"Invalid normalization: {normalize_method}. Use {valid_methods}.")
    
    def __len__(self) -> int:
        return 1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx != 0:
            raise IndexError("Single-image dataset supports only index 0.")
        
        try:
            image, _ = load_sentinel2_image(self.image_path, self.bands)
            mask = load_mask(self.mask_path)
            
            image = normalize_image(image, method=self.normalize_method)
            
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask).float()
            
            if image_tensor.dim() == 3 and image_tensor.shape[2] <= image_tensor.shape[0]:
                image_tensor = image_tensor.permute(2, 0, 1)
            
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            del image
            del mask
            
            return {
                'image': image_tensor,
                'mask': mask_tensor,
                'filename': self.image_filename
            }
        except Exception as e:
            logger.error(f"Error loading single image: {e}")
            raise

class Sentinel2WaterDatasetWithAdvancedAug(Sentinel2WaterDataset):
    """
    Dataset with advanced batch augmentation (MixUp, CutMix).
    """
    def __init__(self, data_dir: str, split: str = 'train', bands: List[int] = None, 
                 augment: bool = False, normalize_method: str = 'sentinel', 
                 splits_dir: str = None, images_dir: str = None, masks_dir: str = None,
                 augmentation_config: Dict = None,
                 mixup_alpha=1.0, cutmix_alpha=1.0, mixup_prob=0.5, cutmix_prob=0.5):
        super().__init__(data_dir, split, bands, augment, normalize_method, splits_dir, images_dir, masks_dir, augmentation_config)
        
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        
        try:
            from utils.augmentation_utils import apply_random_augmentation
            self.apply_random_augmentation = apply_random_augmentation
            self.has_advanced_aug = True
        except ImportError:
            self.has_advanced_aug = False
            print("Warning: Advanced augmentation utilities not found. Using basic augmentation.")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return super().__getitem__(idx)
    
    def apply_batch_augmentation(self, images, masks):
        if not (self.augment and self.split == 'train' and self.has_advanced_aug):
            return images, masks, {'type': 'none', 'lambda': 1.0}
        
        augmented_images, augmented_masks, aug_type, lam = self.apply_random_augmentation(
            images, masks, 
            mixup_prob=self.mixup_prob, 
            cutmix_prob=self.cutmix_prob,
            mixup_alpha=self.mixup_alpha, 
            cutmix_alpha=self.cutmix_alpha
        )
        
        return augmented_images, augmented_masks, {'type': aug_type, 'lambda': lam}

def collate_fn(batch):
    """
    Memory-efficient collate function for DataLoader.
    """
    images = list(item['image'] for item in batch)
    masks = list(item['mask'] for item in batch)
    filenames = list(item['filename'] for item in batch)
    
    stacked_images = torch.stack(images)
    stacked_masks = torch.stack(masks)
    
    del images
    del masks
    import gc
    gc.collect()
    
    return {
        'image': stacked_images,
        'mask': stacked_masks,
        'filename': filenames
    }

def create_data_loader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True, 
                      num_workers: int = 4, pin_memory: bool = True, persistent_workers: bool = False,
                      memory_optimized: bool = False, prefetch_factor: int = None) -> DataLoader:
    """
    Create optimized DataLoader.
    """
    if memory_optimized:
        batch_size = min(batch_size, 2)
        num_workers = min(num_workers, 1)
        pin_memory = False
        persistent_workers = False
    
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = int(prefetch_factor)

    return DataLoader(**loader_kwargs)

def calculate_dataset_statistics(dataset: Dataset, num_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    Compute dataset mean, std, min, max from random samples.
    """
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    means = []
    stds = []
    mins = []
    maxs = []
    
    for idx in sample_indices:
        sample = dataset[idx]
        image = sample['image'].numpy()
        
        band_means = np.mean(image, axis=(1, 2))
        band_stds = np.std(image, axis=(1, 2))
        band_mins = np.min(image, axis=(1, 2))
        band_maxs = np.max(image, axis=(1, 2))
        
        means.append(band_means)
        stds.append(band_stds)
        mins.append(band_mins)
        maxs.append(band_maxs)
    
    results = {
        'mean': np.mean(means, axis=0),
        'std': np.mean(stds, axis=0),
        'min': np.min(mins, axis=0),
        'max': np.max(maxs, axis=0)
    }
    
    return results

def save_predictions(predictions: torch.Tensor, filenames: List[str], output_dir: str, threshold: float = 0.5, 
                    save_probabilities: bool = True, is_probabilities: bool = False, 
                    postprocessing_config: dict = None, reference_dir: Optional[str] = None) -> None:
    """
    Save model predictions to GeoTIFF/PNG files with optional post-processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    reference_dir = os.path.abspath(reference_dir) if reference_dir else None

    def _load_reference_profile(filename: str) -> Optional[dict]:
        if not reference_dir:
            return None
        ref_path = os.path.join(reference_dir, filename)
        if not os.path.exists(ref_path):
            return None
        try:
            with rasterio.open(ref_path) as src:
                return src.profile.copy()
        except Exception:
            return None

    from utils.postprocessing_utils import apply_postprocessing_pipeline
    
    if save_probabilities:
        prob_dir = os.path.join(output_dir, 'probabilities')
        os.makedirs(prob_dir, exist_ok=True)
        
        for i, pred in enumerate(predictions):
            if pred.shape[0] == 1:
                if is_probabilities:
                    prob = pred.cpu().numpy()
                else:
                    prob = torch.sigmoid(pred).cpu().numpy()
            else:
                if is_probabilities:
                    prob = pred.cpu().numpy()
                else:
                    prob = torch.softmax(pred, dim=0).cpu().numpy()
            
            filename = filenames[i]
            output_path = os.path.join(prob_dir, f"{filename}_prob.tif")
            
            try:
                save_image = (prob[0] * 65535).astype(np.uint16)
                profile = _load_reference_profile(filename)
                if profile:
                    profile.update(count=1, dtype=rasterio.uint16)
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(save_image, 1)
                else:
                    cv2.imwrite(output_path, save_image)
            except Exception as e:
                print(f"Failed to save probability map: {output_path}. Error: {str(e)}")
    
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    for i, pred in enumerate(predictions):
        if pred.shape[0] == 1:
            if is_probabilities:
                binary_pred = (pred > threshold).float().cpu().numpy()
            else:
                binary_pred = (torch.sigmoid(pred) > threshold).float().cpu().numpy()
        else:
            if is_probabilities:
                binary_pred = torch.argmax(pred, dim=0, keepdim=True).float().cpu().numpy()
            else:
                binary_pred = torch.argmax(torch.softmax(pred, dim=0), dim=0, keepdim=True).float().cpu().numpy()
        
        if postprocessing_config and pred.shape[0] == 1:
            try:
                if is_probabilities:
                    prob_tensor = pred
                else:
                    prob_tensor = torch.sigmoid(pred)
                
                binary_pred_tensor = (prob_tensor > threshold).float()
                
                processed_pred = apply_postprocessing_pipeline(
                    prob_tensor, 
                    binary_pred_tensor,
                    gaussian_sigma=postprocessing_config.get('gaussian_sigma', 1.0),
                    median_kernel_size=postprocessing_config.get('median_kernel_size', 3),
                    morph_close_kernel_size=postprocessing_config.get('morph_close_kernel_size', 3),
                    morph_open_kernel_size=postprocessing_config.get('morph_open_kernel_size', 0),
                    min_object_size=postprocessing_config.get('min_object_size', 50),
                    hole_area_threshold=postprocessing_config.get('hole_area_threshold', 30),
                    adaptive_threshold=postprocessing_config.get('adaptive_threshold', False)
                )
                binary_pred = processed_pred.astype(np.float32, copy=False)
                if binary_pred.ndim == 4:
                    binary_pred = binary_pred[0,0]
                elif binary_pred.ndim == 3:
                    binary_pred = binary_pred[0]
            except Exception as e:
                print(f"Post-processing failed: {str(e)}")
                if is_probabilities:
                    binary_pred = (pred > threshold).float().cpu().numpy()
                else:
                    binary_pred = (torch.sigmoid(pred) > threshold).float().cpu().numpy()
        
        filename = filenames[i]
        output_path = os.path.join(pred_dir, filename)
        
        try:
            if binary_pred.ndim == 3:
                save_image = (binary_pred[0] * 255).astype(np.uint8)
            else:
                save_image = (binary_pred * 255).astype(np.uint8)
            profile = _load_reference_profile(filename)
            if profile:
                profile.update(count=1, dtype=rasterio.uint8)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(save_image, 1)
            else:
                cv2.imwrite(output_path, save_image)
        except Exception as e:
            print(f"Failed to save prediction: {output_path}. Error: {str(e)}")
