"""
Utility functions for model ensembling, supporting multiple strategies
such as averaging, voting, weighted averaging, and adaptive fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import os

class ModelEnsemble(nn.Module):
    """
    Model ensemble class that combines predictions from multiple models
    using various ensemble strategies.
    """
    def __init__(self, models: List[nn.Module], strategy: str = 'mean', weights: Optional[List[float]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of models to ensemble
            strategy: Ensemble strategy - 'mean', 'weighted_mean', 'vote', 'logits_mean', 'performance_weighted'
            weights: Weights for weighted averaging
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy.lower()
        
        # Validate strategy
        valid_strategies = ['mean', 'weighted_mean', 'vote', 'logits_mean', 'performance_weighted']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Options: {', '.join(valid_strategies)}")
        
        # Initialize weights
        if self.strategy == 'weighted_mean':
            if weights is None:
                self.weights = nn.Parameter(torch.ones(len(models)) / len(models), requires_grad=False)
            else:
                if len(weights) != len(models):
                    raise ValueError("Number of weights must match number of models")
                self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
        else:
            self.weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ensemble fusion.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            Fused prediction
        """
        # Get predictions from all models
        outputs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs.append(model(x))
        
        # Apply ensemble strategy
        if self.strategy == 'mean':
            return self._mean_ensemble(outputs)
        elif self.strategy == 'weighted_mean':
            return self._weighted_mean_ensemble(outputs)
        elif self.strategy == 'vote':
            return self._vote_ensemble(outputs)
        elif self.strategy == 'logits_mean':
            return self._logits_mean_ensemble(outputs)
        elif self.strategy == 'performance_weighted':
            return self._performance_weighted_ensemble(outputs)
        
        raise ValueError(f"Unknown ensemble strategy: {self.strategy}")
    
    def _mean_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Simple mean averaging in probability space"""
        probs = [torch.sigmoid(output) for output in outputs]
        mean_prob = torch.mean(torch.stack(probs), dim=0)
        return torch.logit(torch.clamp(mean_prob, 1e-6, 1-1e-6))
    
    def _weighted_mean_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Weighted averaging in probability space"""
        probs = [torch.sigmoid(output) for output in outputs]
        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_outputs = [prob * weight for prob, weight in zip(probs, normalized_weights)]
        weighted_prob = torch.sum(torch.stack(weighted_outputs), dim=0)
        return torch.logit(torch.clamp(weighted_prob, 1e-6, 1-1e-6))
    
    def _vote_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Hard voting ensemble"""
        predictions = []
        for output in outputs:
            if output.shape[1] == 1:
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True).float()
            predictions.append(pred)
        
        vote_result = torch.mean(torch.stack(predictions), dim=0) >= 0.5
        return vote_result.float()
    
    def _logits_mean_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Direct averaging of logits"""
        return torch.mean(torch.stack(outputs), dim=0)
    
    def _performance_weighted_ensemble(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Weighted ensemble using validation performance"""
        if self.weights is None:
            weights = torch.ones(len(outputs)) / len(outputs)
        else:
            weights = self.weights
        
        probs = [torch.sigmoid(output) for output in outputs]
        normalized_weights = F.softmax(weights, dim=0)
        weighted_outputs = [prob * weight for prob, weight in zip(probs, normalized_weights)]
        weighted_prob = torch.sum(torch.stack(weighted_outputs), dim=0)
        return torch.logit(torch.clamp(weighted_prob, 1e-6, 1-1e-6))
    
    def set_strategy(self, strategy: str, weights: Optional[List[float]] = None) -> None:
        """Update ensemble strategy and weights"""
        self.strategy = strategy.lower()
        
        if self.strategy == 'weighted_mean' and weights is not None:
            if len(weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            self.weights = nn.Parameter(torch.tensor(weights), requires_grad=False)
    
    def get_model_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return individual predictions without fusion"""
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                predictions.append(model(x))
        return predictions

class WeightedModelEnsemble(ModelEnsemble):
    """Ensemble with learnable model weights"""
    def __init__(self, models: List[nn.Module], n_classes: int = 1):
        super(WeightedModelEnsemble, self).__init__(models, strategy='weighted_mean')
        self.n_classes = n_classes
        
        self.learnable_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        if n_classes > 1:
            self.class_weights = nn.Parameter(torch.ones(len(models), n_classes) / len(models))
        else:
            self.class_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        normalized_weights = F.softmax(self.learnable_weights, dim=0)
        
        if self.class_weights is not None and self.n_classes > 1:
            weighted_outputs = []
            for i, output in enumerate(outputs):
                class_norm_weights = F.softmax(self.class_weights[i], dim=0)
                weighted_output = output * class_norm_weights.view(1, -1, 1, 1)
                weighted_outputs.append(weighted_output * normalized_weights[i])
            return torch.sum(torch.stack(weighted_outputs), dim=0)
        else:
            weighted_outputs = [output * weight for output, weight in zip(outputs, normalized_weights)]
            return torch.sum(torch.stack(weighted_outputs), dim=0)

class MultiHeadEnsemble(nn.Module):
    """Feature-level fusion ensemble with learnable fusion layers"""
    def __init__(self, models: List[nn.Module], input_channels: int, n_classes: int = 1):
        super(MultiHeadEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        
        # Infer output channels
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, 64, 64)
            output_sizes = [model(dummy).size(1) for model in models]
        
        total_channels = sum(output_sizes)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(total_channels // 2),
            nn.ReLU(),
            nn.Conv2d(total_channels // 2, n_classes, kernel_size=1)
        )
        
        for m in self.fusion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        
        # Resize to match spatial dimensions
        target_size = outputs[0].shape[2:]
        outputs = [F.interpolate(o, size=target_size, mode='bilinear', align_corners=False) for o in outputs]
        
        combined = torch.cat(outputs, dim=1)
        return self.fusion(combined)

def load_ensemble_models(model_paths: List[str], model_classes: List[nn.Module], 
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[nn.Module]:
    """Load multiple trained models for ensembling"""
    models = []
    
    if len(model_paths) != len(model_classes):
        raise ValueError("Number of paths must match number of model classes")
    
    for path, model_cls in zip(model_paths, model_classes):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        model = model_cls()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        models.append(model)
    
    return models

class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner"""
    def __init__(self, models: List[nn.Module], n_classes: int = 1, 
                 fusion_layers: int = 2, hidden_units: int = 64, 
                 dropout_rate: float = 0.2, use_batch_norm: bool = True):
        super(StackingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        
        with torch.no_grad():
            dummy = torch.randn(1, 6, 64, 64)
            output_sizes = [model(dummy).size(1) for model in models]
        
        total_channels = sum(output_sizes)
        
        layers = []
        in_dim = total_channels
        for _ in range(fusion_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_units
        
        layers.append(nn.Linear(in_dim, n_classes))
        self.fusion_network = nn.Sequential(*layers)
        
        for m in self.fusion_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        
        target_size = outputs[0].shape[2:]
        outputs = [F.interpolate(o, size=target_size, mode='bilinear', align_corners=False) for o in outputs]
        
        pooled = []
        for o in outputs:
            if o.shape[1] == 1:
                p = F.adaptive_avg_pool2d(torch.sigmoid(o), 1)
            else:
                p = F.adaptive_avg_pool2d(torch.softmax(o, dim=1), 1)
            pooled.append(p.view(p.size(0), -1))
        
        combined = torch.cat(pooled, dim=1)
        result = self.fusion_network(combined)
        
        if self.n_classes == 1:
            return result.view(-1, 1, 1, 1)
        else:
            b, h, w = x.size(0), target_size[0], target_size[1]
            return result.view(b, self.n_classes, 1, 1).expand(b, self.n_classes, h, w)

def create_water_segmentation_ensemble(aer_unet=None, lightweight_unet=None, deeplabv3_plus=None, strategy='mean'):
    """Create ensemble for water segmentation using available models"""
    models = []
    if aer_unet is not None:
        models.append(aer_unet)
    if lightweight_unet is not None:
        models.append(lightweight_unet)
    if deeplabv3_plus is not None:
        models.append(deeplabv3_plus)
    
    if len(models) == 0:
        raise ValueError("At least one model must be provided")
    
    return ModelEnsemble(models=models, strategy=strategy)

def compute_ensemble_metrics(predictions: List[torch.Tensor], targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute evaluation metrics for ensemble predictions"""
    from .metrics import compute_iou, compute_dice, compute_precision_recall, compute_accuracy
    
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    
    if ensemble_pred.shape[1] == 1:
        binary = (torch.sigmoid(ensemble_pred) > threshold).float()
    else:
        binary = torch.argmax(torch.softmax(ensemble_pred, dim=1), dim=1, keepdim=True).float()
    
    precision, recall = compute_precision_recall(binary, targets)
    
    return {
        'iou': compute_iou(binary, targets).item(),
        'dice': compute_dice(binary, targets).item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': compute_accuracy(binary, targets).item()
    }

class AdvancedAdaptiveWeightedEnsemble(nn.Module):
    """Pixel-adaptive weighted ensemble with attention and input-aware weight generation"""
    def __init__(self, models: List[nn.Module], n_classes: int = 1, 
                 input_channels: int = 6, hidden_units: int = 128, 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        self.num_models = len(models)
        self.use_attention = use_attention
        
        self.input_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        self.weight_generator = nn.Sequential(
            nn.Conv2d(hidden_units + self.num_models * n_classes, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units, hidden_units//2, 3, padding=1),
            nn.BatchNorm2d(hidden_units//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units//2, self.num_models, 1),
            nn.Sigmoid()
        )
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(self.num_models, self.num_models, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.attention = None
        
        for m in self.input_feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_feature_extractor(x)
        outputs = [model(x) for model in self.models]
        
        target_size = outputs[0].shape[2:]
        outputs = [F.interpolate(o, target_size, mode='bilinear', align_corners=False) for o in outputs]
        
        concat_out = torch.cat(outputs, dim=1)
        combined = torch.cat([feat, concat_out], dim=1)
        weights = self.weight_generator(combined)
        
        if self.attention is not None:
            weights = self.attention(weights)
        else:
            weights = F.softmax(weights, dim=1)
        
        weighted = [out * weights[:, i:i+1] for i, out in enumerate(outputs)]
        return torch.sum(torch.stack(weighted), dim=0)

class AdaptiveWeightedEnsemble(nn.Module):
    """Input-adaptive weighted ensemble with global weight generation"""
    def __init__(self, models: List[nn.Module], n_classes: int = 1, 
                 input_channels: int = 6, hidden_units: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_classes = n_classes
        self.num_models = len(models)
        
        self.weight_generator = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units, hidden_units//2, 3, padding=1),
            nn.BatchNorm2d(hidden_units//2),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_units//2, self.num_models, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        for m in self.weight_generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.weight_generator(x)
        weights = F.softmax(weights, dim=1)
        
        outputs = [model(x) for model in self.models]
        target_size = outputs[0].shape[2:]
        outputs = [F.interpolate(o, target_size, mode='bilinear', align_corners=False) for o in outputs]
        
        weighted = [out * weights[:, i:i+1] for i, out in enumerate(outputs)]
        return torch.sum(torch.stack(weighted), dim=0)
