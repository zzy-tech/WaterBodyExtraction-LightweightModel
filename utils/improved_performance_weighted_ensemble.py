import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def _normalize_model_name(name: str) -> str:
    if not name:
        return ""
    normalized = name.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized == "aer_u_net":
        normalized = "aer_unet"
    elif normalized == "ultra_lightweight_deeplabv3_":
        normalized = "ultra_lightweight_deeplabv3_plus"
    return normalized

class ImprovedPerformanceWeightedEnsemble(nn.Module):
    """
    Improved performance-weighted ensemble
    Solves prediction cancellation issues and supports multi-metric fusion
    """
    def __init__(self, models: List[nn.Module], 
                 performance_metrics: Dict[str, Dict[str, float]],
                 metric_weights: Dict[str, float] = None,
                 metric_name: str = 'iou',
                 temperature: float = 1.0,
                 power: float = 2.0,
                 ensemble_method: str = 'gated_ensemble',
                 diff_threshold: float = 0.20,
                 conf_threshold: float = 0.22,
                 binary_threshold: float = 0.5,
                 model_names: List[str] = None):
        """
        Initialize improved performance-weighted ensemble
        
        Args:
            models: List of models to ensemble
            performance_metrics: Dict of model performance {model_name: {iou, dice, f1, ...}}
            metric_weights: Weights for each metric
            metric_name: Primary metric for logging
            temperature: Softmax temperature
            power: Power to amplify score differences
            ensemble_method: 'logits_weighted', 'prob_weighted', 'gated_ensemble'
            diff_threshold: Threshold for prediction disagreement
            conf_threshold: Threshold for model confidence
            binary_threshold: Threshold for binary segmentation
            model_names: List of model names for gated selection
        """
        super(ImprovedPerformanceWeightedEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.model_names = [_normalize_model_name(name) for name in model_names] if model_names else []
        self.aer_index = None
        self.ultra_index = None
        if self.model_names:
            for idx, name in enumerate(self.model_names):
                if name == 'aer_unet' and self.aer_index is None:
                    self.aer_index = idx
                if name == 'ultra_lightweight_deeplabv3_plus' and self.ultra_index is None:
                    self.ultra_index = idx
            if self.aer_index is None or self.ultra_index is None:
                logger.warning("gated_ensemble: could not find both 'aer_unet' and 'ultra_lightweight_deeplabv3_plus' in model_names; falling back to first two models.")
        self.performance_metrics = performance_metrics
        
        if metric_weights is None:
            all_metrics = set()
            for model_metrics in performance_metrics.values():
                all_metrics.update(model_metrics.keys())
            num_metrics = len(all_metrics)
            self.metric_weights = {metric: 1.0 / num_metrics for metric in all_metrics}
        else:
            self.metric_weights = metric_weights
        
        self.metric_name = metric_name
        self.temperature = temperature
        self.power = power
        self.ensemble_method = ensemble_method
        self.diff_threshold = diff_threshold
        self.conf_threshold = conf_threshold
        self.binary_threshold = binary_threshold
        
        initial_weights = torch.ones(len(models)) / len(models)
        self.register_buffer('weights', initial_weights)
        self._compute_weights()
    
    def _compute_weights(self) -> None:
        metric_weight_sum = sum(self.metric_weights.values())
        if metric_weight_sum > 0:
            normalized_metric_weights = {k: v / metric_weight_sum for k, v in self.metric_weights.items()}
            logger.info(f"Normalized metric weights: {normalized_metric_weights}")
        else:
            num_metrics = len(self.metric_weights)
            normalized_metric_weights = {k: 1.0 / num_metrics for k in self.metric_weights.keys()}
            logger.warning(f"Metric weight sum is zero; using uniform weights: {normalized_metric_weights}")
        
        composite_scores = []
        for model_name in self.performance_metrics.keys():
            model_metrics = self.performance_metrics[model_name]
            total_weight = sum(weight for metric, weight in self.metric_weights.items() if metric in model_metrics)
            
            if total_weight > 0:
                norm_w = {k: v/total_weight for k, v in self.metric_weights.items() if k in model_metrics}
                score = sum(model_metrics.get(m, 0) * w for m, w in norm_w.items())
                composite_scores.append(score)
            else:
                composite_scores.append(0.5)
                logger.warning(f"Model {model_name} has no valid metrics; using default score 0.5")
        
        score_tensor = torch.tensor(composite_scores, dtype=torch.float32)
        min_s, max_s = torch.min(score_tensor), torch.max(score_tensor)
        if max_s > min_s:
            norm = (score_tensor - min_s) / (max_s - min_s)
        else:
            norm = torch.ones_like(score_tensor) / len(score_tensor)
        
        amplified = torch.pow(norm, self.power)
        scaled = amplified / self.temperature
        weights = F.softmax(scaled, dim=0)
        self.weights.copy_(weights)
        self.performance_weights = weights.tolist()
        
        logger.info(f"Multi-metric fusion weights: {self.metric_weights}")
        logger.info(f"Composite scores: {composite_scores}")
        logger.info(f"Final model weights: {weights.tolist()}")
        logger.info(f"Temperature: {self.temperature}, Power: {self.power}")
        
        for i, (name, mets) in enumerate(self.performance_metrics.items()):
            logger.info(f"{name}: metrics={mets}, weight={weights[i].item():.4f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs.append(model(x))
        
        if self.ensemble_method == 'logits_weighted':
            weighted = [o * w for o, w in zip(outputs, self.weights)]
            return torch.sum(torch.stack(weighted), dim=0)
        
        elif self.ensemble_method == 'prob_weighted':
            probs = [torch.sigmoid(o) for o in outputs]
            weighted = [p * w for p, w in zip(probs, self.weights)]
            fused = torch.sum(torch.stack(weighted), dim=0)
            fused = torch.clamp(fused, 1e-6, 1 - 1e-6)
            return torch.logit(fused)
        
        elif self.ensemble_method == 'gated_ensemble':
            if len(outputs) < 2:
                return outputs[0]
            probs = [torch.sigmoid(o) for o in outputs]
            
            aer_idx = self.aer_index if self.aer_index is not None else 0
            ultra_idx = self.ultra_index if self.ultra_index is not None else 1
            if aer_idx == ultra_idx:
                aer_idx, ultra_idx = 0, 1
            
            p_aer = probs[aer_idx]
            p_ultra = probs[ultra_idx]
            diff = torch.abs(p_aer - p_ultra)
            
            conf_aer = torch.abs(p_aer - self.binary_threshold)
            conf_ultra = torch.abs(p_ultra - self.binary_threshold)
            avg_conf = (conf_aer + conf_ultra) / 2
            
            uncertain = avg_conf < self.conf_threshold
            consistent = (diff < self.diff_threshold) & (~uncertain)
            inconsistent = (diff >= self.diff_threshold) & (~uncertain)
            prefer_aer = conf_aer >= conf_ultra
            
            fused = torch.where(
                uncertain,
                (p_aer * conf_aer + p_ultra * conf_ultra) / (conf_aer + conf_ultra + 1e-8),
                torch.where(
                    consistent | inconsistent,
                    torch.where(prefer_aer, p_aer, p_ultra),
                    (p_aer * conf_aer + p_ultra * conf_ultra) / (conf_aer + conf_ultra + 1e-8)
                )
            )
            fused = torch.clamp(fused, 1e-6, 1 - 1e-6)
            return torch.logit(fused)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def update_metrics(self, new_metrics: Dict[str, Dict[str, float]]) -> None:
        self.performance_metrics = new_metrics
        self._compute_weights()
    
    def get_weights(self) -> torch.Tensor:
        return self.weights

def create_improved_performance_weighted_ensemble(
        model_paths: List[str],
        model_classes: List,
        model_names: List[str],
        csv_paths: List[str],
        metric_names: List[str] = None,
        metric_weights: Dict[str, float] = None,
        metric_name: str = 'iou',
        temperature: float = 1.0,
        power: float = 2.0,
        ensemble_method: str = 'gated_ensemble',
        device: str = 'cuda',
        n_channels: int = 6,
        n_classes: int = 1,
        model_configs: List[Dict] = None,
        diff_threshold: float = 0.20,
        conf_threshold: float = 0.22,
        binary_threshold: float = 0.5):
    
    if metric_names is None:
        metric_names = ['iou', 'dice', 'f1']
    
    performance_metrics = load_performance_metrics(csv_paths, model_names, metric_names)
    models = []
    
    for i, (path, cls) in enumerate(zip(model_paths, model_classes)):
        cfg = model_configs[i] if (model_configs and i < len(model_configs)) else {}
        norm_name = _normalize_model_name(model_names[i])
        
        if norm_name == 'aer_unet':
            model = cls(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                base_features=cfg.get('base_features', 32),
                dropout_rate=cfg.get('dropout_rate', 0.3)
            )
        elif norm_name in ['lightweight_unet', 'lightweight_deeplabv3_plus']:
            model = cls(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                base_features=cfg.get('base_features', 64),
                dropout_rate=cfg.get('dropout_rate', 0.3),
                output_stride=cfg.get('output_stride', 16),
                pretrained_backbone=cfg.get('pretrained_backbone', True)
            )
        elif norm_name == 'deeplabv3_plus':
            model = cls(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                output_stride=cfg.get('output_stride', 16),
                pretrained_backbone=cfg.get('pretrained_backbone', True)
            )
        elif norm_name == 'ultra_lightweight_deeplabv3_plus':
            model = cls(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes),
                pretrained_backbone=cfg.get('pretrained_backbone', True),
                aspp_out=cfg.get('aspp_out', 64),
                dec_ch=cfg.get('dec_ch', 64),
                low_ch_out=cfg.get('low_ch_out', 32),
                use_cbam=cfg.get('use_cbam', False),
                cbam_reduction_ratio=cfg.get('cbam_reduction_ratio', 16),
                output_stride=cfg.get('output_stride', 16)
            )
        else:
            model = cls(
                n_channels=cfg.get('n_channels', n_channels),
                n_classes=cfg.get('n_classes', n_classes)
            )
        
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
        
        if norm_name == 'ultra_lightweight_deeplabv3_plus':
            sd = {k.replace('low_reduce', 'low_proj'): v for k, v in sd.items()}
        
        missing, unexp = model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded {model_names[i]} | missing: {len(missing)}, unexpected: {len(unexp)}")
        
        model.to(device)
        model.eval()
        models.append(model)
    
    ensemble = ImprovedPerformanceWeightedEnsemble(
        models=models,
        performance_metrics=performance_metrics,
        metric_weights=metric_weights,
        metric_name=metric_name,
        temperature=temperature,
        power=power,
        ensemble_method=ensemble_method,
        diff_threshold=diff_threshold,
        conf_threshold=conf_threshold,
        binary_threshold=binary_threshold,
        model_names=model_names
    )
    return ensemble.to(device)

def load_performance_metrics(csv_paths: List[str], 
                           model_names: List[str],
                           metric_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    if metric_names is None:
        metric_names = ['iou', 'dice', 'f1']
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    default_val = 0.5
    metrics = {}
    
    for path, name in zip(csv_paths, model_names):
        if not os.path.exists(path):
            metrics[name] = {m: default_val for m in metric_names}
            continue
        
        try:
            df = pd.read_csv(path)
            model_mets = {}
            
            if 'Metric' in df.columns and 'Value' in df.columns:
                for m in metric_names:
                    row = df[df['Metric'] == m]
                    if not row.empty:
                        try:
                            val = float(row['Value'].iloc[0])
                            if 0 <= val <= 1:
                                model_mets[m] = val
                            else:
                                model_mets[m] = default_val
                        except:
                            model_mets[m] = default_val
                    else:
                        model_mets[m] = default_val
            elif all(m in df.columns for m in metric_names):
                for m in metric_names:
                    try:
                        val = df[m].iloc[-1]
                        if isinstance(val, str) and '%' in val:
                            val = float(val.replace('%', '')) / 100
                        val = float(val)
                        model_mets[m] = val if 0 <= val <= 1 else default_val
                    except:
                        model_mets[m] = default_val
            else:
                for m in metric_names:
                    model_mets[m] = default_val
            
            metrics[name] = model_mets
            logger.info(f"Loaded metrics for {name}: {model_mets}")
        
        except Exception as e:
            logger.warning(f"Failed to load {path}: {str(e)}")
            metrics[name] = {m: default_val for m in metric_names}
    
    return metrics
