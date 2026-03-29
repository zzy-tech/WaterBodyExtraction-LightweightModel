# models/__init__.py
"""
Model module initialization file, providing a unified model creation interface.
"""

# Import all model creation functions
from models.aer_unet import get_aer_unet_model
try:
    from models.lightweight_unet import get_lightweight_unet_model
    _HAS_LIGHTWEIGHT_UNET = True
except ModuleNotFoundError:
    get_lightweight_unet_model = None
    _HAS_LIGHTWEIGHT_UNET = False
from models.unet_model import get_unet_model
from models.deeplabv3_plus import get_deeplabv3_plus_model

# Model Type Enumeration
class ModelType:
    AER_UNET = "aer_unet"
    LIGHTWEIGHT_UNET = "lightweight_unet"
    UNET = "unet"
    DEEPLABV3_PLUS = "deeplabv3_plus"

# Model Creation Factory
def create_model(model_type: str, **kwargs):
    """
    Create a model instance based on the specified model type.
    
    Args:
        model_type: Type of model to create. Options:
            - "aer_unet": AER U-Net model
            - "lightweight_unet": Lightweight U-Net model
            - "deeplabv3_plus": DeepLabV3+ model
        **kwargs: Model-specific parameters
    
    Returns:
        Created model instance
    
    Raises:
        ValueError: If the model type is not supported
    """
    if model_type == ModelType.AER_UNET:
        return get_aer_unet_model(**kwargs)
    elif model_type == ModelType.LIGHTWEIGHT_UNET:
        if not _HAS_LIGHTWEIGHT_UNET:
            raise ValueError("lightweight_unet module is missing; cannot create this model.")
        return get_lightweight_unet_model(**kwargs)
    elif model_type == ModelType.UNET:
        return get_unet_model(**kwargs)
    elif model_type == ModelType.DEEPLABV3_PLUS:
        return get_deeplabv3_plus_model(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Get all supported model types
def get_supported_model_types():
    """Get a list of all supported model types."""
    return [
        ModelType.AER_UNET,
        *( [ModelType.LIGHTWEIGHT_UNET] if _HAS_LIGHTWEIGHT_UNET else [] ),
        ModelType.UNET,
        ModelType.DEEPLABV3_PLUS
    ]

# Get model information
def get_model_info(model_type: str):
    """
    Get detailed information about a model.
    
    Args:
        model_type: Model type
    
    Returns:
        Dictionary containing model information
    
    Raises:
        ValueError: If the model type is not supported
    """
    model_info = {
        ModelType.AER_UNET: {
            "name": "AER U-Net",
            "description": "Attention-Enhanced Multi-Scale Residual U-Net",
            "architecture": "CNN",
            "complexity": "Medium",
            "parameters": "~5–10M",
            "suitable_for": "High-precision water segmentation, requires moderate computing resources"
        },
        ModelType.LIGHTWEIGHT_UNET: {
            "name": "Lightweight U-Net",
            "description": "Lightweight U-Net using depthwise separable convolutions",
            "architecture": "CNN",
            "complexity": "Low",
            "parameters": "~1–2M",
            "suitable_for": "Fast water segmentation, ideal for resource-limited environments"
        },
        ModelType.UNET: {
            "name": "U-Net",
            "description": "Standard U-Net encoder-decoder with skip connections",
            "architecture": "CNN",
            "complexity": "Medium",
            "parameters": "~30M",
            "suitable_for": "General segmentation baseline for comparison"
        },
        ModelType.DEEPLABV3_PLUS: {
            "name": "DeepLabV3+",
            "description": "Semantic segmentation model with atrous convolutions and ASPP module",
            "architecture": "CNN with Atrous Convolutions",
            "complexity": "High",
            "parameters": "~40–50M",
            "suitable_for": "High-precision semantic segmentation, requires strong computing resources"
        }
    }
    
    if model_type in model_info:
        return model_info[model_type]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Print information for all models
def print_all_models_info():
    """Print detailed information for all supported models."""
    print("Supported Model Types:")
    print("=" * 80)
    
    for model_type in get_supported_model_types():
        info = get_model_info(model_type)
        print(f"Model Type: {model_type}")
        print(f"Name: {info['name']}")
        print(f"Description: {info['description']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Complexity: {info['complexity']}")
        print(f"Parameters: {info['parameters']}")
        print(f"Suitable For: {info['suitable_for']}")
        print("-" * 80)

if __name__ == "__main__":
    print_all_models_info()
