"""
CRF (Conditional Random Field) post-processing utilities for refining segmentation results.
"""
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple

class DenseCRF:
    """
    Dense Conditional Random Field implementation for segmentation post-processing.
    This is a simplified CRF implementation accelerated with PyTorch.
    """
    def __init__(self, iter_max: int = 10, 
                 pos_w: float = 3.0, pos_xy_std: float = 1.0,
                 bi_w: float = 4.0, bi_xy_std: float = 67.0, bi_rgb_std: float = 3.0):
        """
        Initialize CRF parameters
        
        Args:
            iter_max: Number of iterations
            pos_w: Position kernel weight
            pos_xy_std: Spatial standard deviation for position kernel
            bi_w: Bilateral kernel weight
            bi_xy_std: Spatial standard deviation for bilateral kernel
            bi_rgb_std: Color standard deviation for bilateral kernel
        """
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
    
    def _create_affinity_kernel(self, image: torch.Tensor, 
                                xy_std: float, rgb_std: Optional[float] = None) -> torch.Tensor:
        """
        Create pairwise affinity kernel
        
        Args:
            image: Input image [C, H, W]
            xy_std: Spatial standard deviation
            rgb_std: Color standard deviation (None for position-only kernel)
            
        Returns:
            Affinity kernel [H*W, H*W]
        """
        device = image.device
        H, W = image.shape[1], image.shape[2]
        
        # Create position grid
        xx, yy = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=0)
        
        # Compute position differences
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        pos_dist = torch.sum(pos_diff ** 2, dim=0)
        
        # Position affinity
        pos_affinity = torch.exp(-pos_dist / (2 * xy_std ** 2))
        
        if rgb_std is not None:
            # Compute color differences
            pixels = image.view(image.shape[0], -1)
            rgb_diff = pixels.unsqueeze(2) - pixels.unsqueeze(1)
            rgb_dist = torch.sum(rgb_diff ** 2, dim=0)
            
            # Bilateral affinity
            bi_affinity = torch.exp(-rgb_dist / (2 * rgb_std ** 2))
            affinity = pos_affinity * bi_affinity
        else:
            affinity = pos_affinity
            
        return affinity
    
    def __call__(self, image: torch.Tensor, unary: torch.Tensor) -> torch.Tensor:
        """
        Apply CRF post-processing
        
        Args:
            image: Input image [C, H, W]
            unary: Unary potential [H, W] or [1, H, W]
            
        Returns:
            Refined probability map [H, W]
        """
        device = image.device
        
        # Ensure unary shape is correct
        if unary.dim() == 3:
            unary = unary.squeeze(0)
        
        # Convert unary to probabilities
        if unary.min() < 0:
            prob = torch.sigmoid(unary)
        else:
            prob = unary
        
        # Create Q matrix [2, H*W]
        Q = torch.stack([1 - prob, prob], dim=0)
        
        # Create kernels
        pos_kernel = self._create_affinity_kernel(image, self.pos_xy_std)
        bi_kernel = self._create_affinity_kernel(image, self.bi_xy_std, self.bi_rgb_std)
        
        # Iterative inference
        for _ in range(self.iter_max):
            pos_message = self.pos_w * torch.matmul(pos_kernel, Q)
            bi_message = self.bi_w * torch.matmul(bi_kernel, Q)
            Q = torch.exp(unary.flatten() + pos_message + bi_message)
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-10)
        
        return Q[1].view_as(unary)


def apply_crf_postprocessing(image: torch.Tensor, prediction: torch.Tensor, 
                           iterations: int = 10) -> torch.Tensor:
    """
    Apply CRF post-processing to segmentation predictions
    
    Args:
        image: Original image [C, H, W]
        prediction: Model prediction [1, H, W] or [H, W]
        iterations: Number of CRF iterations
        
        Returns:
            Refined prediction [H, W]
    """
    device = image.device
    
    # Convert logits to probabilities
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    if prediction.dim() == 3:
        prediction = prediction.squeeze(0)
    
    # Expand single-channel image to 3 channels for bilateral kernel
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    
    crf = DenseCRF(iter_max=iterations)
    refined_prediction = crf(image, prediction)
    
    return refined_prediction


def batch_apply_crf(images: torch.Tensor, predictions: torch.Tensor, 
                   iterations: int = 10) -> torch.Tensor:
    """
    Apply CRF post-processing to a batch of images
    
    Args:
        images: Batch of images [B, C, H, W]
        predictions: Batch of predictions [B, 1, H, W] or [B, H, W]
        iterations: Number of CRF iterations
        
    Returns:
        Refined batch predictions [B, H, W]
    """
    batch_size = images.shape[0]
    device = images.device
    
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)
    
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    
    refined_predictions = torch.zeros_like(predictions)
    
    for i in range(batch_size):
        refined_predictions[i] = apply_crf_postprocessing(
            images[i], predictions[i], iterations
        )
    
    return refined_predictions
