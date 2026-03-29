#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing utilities for segmentation predictions.
Includes smoothing, morphological operations, small object removal, and hole filling.
"""

import numpy as np
import cv2
from skimage import morphology
from skimage.filters import threshold_otsu


def apply_postprocessing_pipeline(
    probability_prediction,
    binary_prediction,
    median_kernel_size=0,
    gaussian_sigma=0.0,
    morph_close_kernel_size=0,
    morph_open_kernel_size=0,
    min_object_size=0,
    hole_area_threshold=0,
    adaptive_threshold=False,
    threshold=0.5
):
    """
    Apply a full post-processing pipeline to segmentation predictions.

    Args:
        probability_prediction: Probability map [H, W] or [1, 1, H, W]
        binary_prediction: Binary prediction [H, W] or [1, 1, H, W]
        median_kernel_size: Median blur kernel size (0 to disable)
        gaussian_sigma: Gaussian blur sigma (0 to disable)
        morph_close_kernel_size: Morphological closing kernel size (0 to disable)
        morph_open_kernel_size: Morphological opening kernel size (0 to disable)
        min_object_size: Minimum object area in pixels (0 to disable)
        hole_area_threshold: Max hole area to fill in pixels (0 to disable)
        adaptive_threshold: Use Otsu's thresholding
        threshold: Fixed binarization threshold

    Returns:
        Processed binary prediction in shape [1, 1, H, W]
    """
    # Convert torch tensors to numpy
    if hasattr(probability_prediction, 'detach') and hasattr(probability_prediction, 'cpu'):
        probability_prediction = probability_prediction.detach().cpu().numpy()
    if hasattr(binary_prediction, 'detach') and hasattr(binary_prediction, 'cpu'):
        binary_prediction = binary_prediction.detach().cpu().numpy()

    # Ensure numpy arrays
    if not isinstance(probability_prediction, np.ndarray):
        probability_prediction = np.array(probability_prediction)
    if not isinstance(binary_prediction, np.ndarray):
        binary_prediction = np.array(binary_prediction)

    # Squeeze to [H, W]
    prob = np.squeeze(probability_prediction)
    binary = np.squeeze(binary_prediction)

    # Adaptive thresholding (Otsu)
    if adaptive_threshold:
        threshold = threshold_otsu(prob)
        binary = (prob > threshold).astype(np.float32)

    # Gaussian smoothing on probability map
    if gaussian_sigma > 0:
        prob = cv2.GaussianBlur(prob, (0, 0), gaussian_sigma)

    # Median filtering on binary mask
    if median_kernel_size > 0:
        binary = cv2.medianBlur(
            (binary * 255).astype(np.uint8),
            median_kernel_size
        ).astype(np.float32) / 255.0

    # Morphological closing (fill small holes)
    if morph_close_kernel_size > 0:
        kernel = morphology.disk(morph_close_kernel_size)
        binary = morphology.binary_closing(binary, kernel).astype(np.float32)

    # Morphological opening (remove small noise)
    if morph_open_kernel_size > 0:
        kernel = morphology.disk(morph_open_kernel_size)
        binary = morphology.binary_opening(binary, kernel).astype(np.float32)

    # Remove small connected components
    if min_object_size > 0:
        labeled = morphology.label(binary > 0)
        keep_mask = np.zeros_like(binary, dtype=bool)
        for region in morphology.regionprops(labeled):
            if region.area >= min_object_size:
                keep_mask |= (labeled == region.label)
        binary = keep_mask.astype(np.float32)

    # Fill small holes
    if hole_area_threshold > 0:
        inv = 1.0 - binary
        labeled = morphology.label(inv > 0)
        hole_mask = np.zeros_like(inv, dtype=bool)
        for region in morphology.regionprops(labeled):
            if region.area < hole_area_threshold:
                hole_mask |= (labeled == region.label)
        binary[hole_mask] = 1.0

    # Restore shape to [1, 1, H, W]
    processed = np.expand_dims(np.expand_dims(binary, axis=0), axis=0)
    return processed
