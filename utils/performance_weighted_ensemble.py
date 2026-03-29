"""
Performance-based weighted ensemble strategy.
Dynamically adjusts weights based on model performance on the validation set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import pandas as pd
from pathlib import Path


class PerformanceWeightedEnsemble(nn.Module):
    """
    Performance-weighted ensemble for semantic segmentation.
    Dynamically computes weights from validation-set performance metrics.
    """
    def __init__(self, models: List[nn.Module],
                 performance_metrics: Dict[str, float],
                 metric_name: str = 'iou',
                 temperature: float = 2.0):
        """
        Initialize performance-weighted ensemble.
        
        Args:
            models: List of models to ensemble
            performance_metrics: Dict {model_name: metric_value}
            metric_name: Metric to use for weighting (iou, dice, f1_score)
            temperature: Temperature to sharpen weight distribution
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.performance_metrics = performance_metrics
        self.metric_name = metric_name
        self.temperature = temperature

        self._compute_weights()

    def _compute_weights(self) -> None:
        """
        Compute normalized weights from performance scores using softmax.
        """
        scores = list(self.performance_metrics.values())
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        scaled = score_tensor / self.temperature
        weights = F.softmax(scaled, dim=0)
        self.register_buffer('weights', weights)
        print(f"Ensemble weights from {self.metric_name}: {weights.tolist()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted average in probability space.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Fused logits from weighted ensemble
        """
        probs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(x)
                prob = torch.sigmoid(logits)
                probs.append(prob)

        weighted = [p * w for p, w in zip(probs, self.weights)]
        fused = torch.sum(torch.stack(weighted), dim=0)
        fused = torch.clamp(fused, 1e-6, 1.0 - 1e-6)
        return torch.logit(fused)

    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """
        Update performance scores and recompute weights.
        """
        self.performance_metrics = new_metrics
        self._compute_weights()

    def get_weights(self) -> torch.Tensor:
        """Return current ensemble weights."""
        return self.weights


def load_performance_metrics(csv_paths: List[str],
                             model_names: List[str],
                             metric_name: str = 'iou') -> Dict[str, float]:
    """
    Load model performance scores from CSV evaluation results.
    
    Args:
        csv_paths: Paths to model evaluation CSVs
        model_names: Names corresponding to each model
        metric_name: Target metric to load
    
    Returns:
        Dict {model_name: metric_value}
    """
    metrics = {}
    for path, name in zip(csv_paths, model_names):
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'Metric' in df.columns and 'Value' in df.columns:
                row = df[df['Metric'] == metric_name]
                val = row['Value'].iloc[0] if not row.empty else 0.5
                metrics[name] = val
            elif metric_name in df.columns:
                metrics[name] = df[metric_name].iloc[-1]
            else:
                print(f"Warning: Metric {metric_name} not found in {path}")
                metrics[name] = 0.5
        else:
            print(f"Warning: File {path} not found")
            metrics[name] = 0.5
    return metrics


def create_performance_weighted_ensemble(
    model_paths: List[str],
    model_classes: List,
    model_names: List[str],
    csv_paths: List[str],
    metric_name: str = 'iou',
    temperature: float = 2.0,
    device: str = 'cuda'
):
    """
    Build a ready-to-use performance-weighted ensemble from saved checkpoints.
    
    Args:
        model_paths: Paths to model checkpoint files
        model_classes: Model class constructors
        model_names: Names for each model
        csv_paths: Paths to evaluation CSVs
        metric_name: Weighting metric
        temperature: Softmax temperature
        device: Target device
    
    Returns:
        PerformanceWeightedEnsemble on device
    """
    performance = load_performance_metrics(csv_paths, model_names, metric_name)
    models = []

    for path, cls in zip(model_paths, model_classes):
        model = cls()
        ckpt = torch.load(path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            else:
                sd = ckpt
        else:
            sd = ckpt

        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        models.append(model)

    ensemble = PerformanceWeightedEnsemble(
        models=models,
        performance_metrics=performance,
        metric_name=metric_name,
        temperature=temperature
    )
    return ensemble.to(device)
