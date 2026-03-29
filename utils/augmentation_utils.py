import torch
import numpy as np
import random

def mixup_data(x, y, alpha=1.0):
    """
    Perform MixUp data augmentation.
    
    Args:
        x: Input image tensor (batch_size, C, H, W)
        y: Target mask tensor (batch_size, 1, H, W) or (batch_size, H, W)
        alpha: MixUp interpolation parameter
        
    Returns:
        mixed_x: Mixed image tensor
        mixed_y: Mixed mask tensor
        lam: Lambda mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y, lam

def cutmix_data(x, y, alpha=1.0):
    """
    Perform CutMix data augmentation.
    
    Args:
        x: Input image tensor (batch_size, C, H, W)
        y: Target mask tensor (batch_size, 1, H, W) or (batch_size, H, W)
        alpha: CutMix interpolation parameter
        
    Returns:
        mixed_x: Mixed image tensor
        mixed_y: Mixed mask tensor
        lam: Lambda mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    _, _, H, W = x.shape
    
    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Bounding box boundaries
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_y = y.clone()
    
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_y[:, :, bbx1:bbx2, bby1:bby2] = y[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to match actual region size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    
    return mixed_x, mixed_y, lam

def apply_random_augmentation(x, y, mixup_prob=0.5, cutmix_prob=0.5, mixup_alpha=1.0, cutmix_alpha=1.0):
    """
    Randomly apply either MixUp or CutMix augmentation.
    
    Args:
        x: Input image tensor
        y: Target mask tensor
        mixup_prob: Probability to apply MixUp
        cutmix_prob: Probability to apply CutMix
        mixup_alpha: Alpha parameter for MixUp
        cutmix_alpha: Alpha parameter for CutMix
        
    Returns:
        augmented_x: Augmented image tensor
        augmented_y: Augmented mask tensor
        aug_type: Type of augmentation applied ('none', 'mixup', 'cutmix')
        lam: Mixing coefficient (1.0 if no augmentation)
    """
    r = random.random()
    
    if r < mixup_prob:
        augmented_x, augmented_y, lam = mixup_data(x, y, alpha=mixup_alpha)
        return augmented_x, augmented_y, 'mixup', lam
    elif r < mixup_prob + cutmix_prob:
        augmented_x, augmented_y, lam = cutmix_data(x, y, alpha=cutmix_alpha)
        return augmented_x, augmented_y, 'cutmix', lam
    else:
        return x, y, 'none', 1.0
