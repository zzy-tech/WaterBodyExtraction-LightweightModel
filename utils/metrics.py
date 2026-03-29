"""
Evaluation metrics for water segmentation models.
Includes IoU, Dice, Precision, Recall, Accuracy, F1, confusion matrix, and threshold analysis.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_iou_no_threshold(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) without binarization.
    Used for threshold analysis; predictions are already binary.
    
    Args:
        prediction: Binary prediction map, shape [batch, 1, height, width]
        target: Ground truth label
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Mean IoU score
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    intersection = (prediction * target).sum(dtype=torch.float32)
    union = prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def compute_dice_no_threshold(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice coefficient (F1) without binarization.
    Used for threshold analysis; predictions are already binary.
    
    Args:
        prediction: Binary prediction map
        target: Ground truth label
        smooth: Smoothing factor
        
    Returns:
        Mean Dice score
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    intersection = (prediction * target).sum(dtype=torch.float32)
    dice = (2. * intersection + smooth) / (prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) + smooth)
    return dice


def compute_iou(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) with thresholding.
    
    Args:
        prediction: Model output (logits or probabilities), shape [batch, 1, H, W]
        target: Ground truth
        smooth: Smoothing factor
        threshold: Binarization threshold
        
    Returns:
        Mean IoU
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()
    
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    intersection = (prediction * target).sum(dtype=torch.float32)
    union = prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def compute_dice(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute Dice coefficient with thresholding.
    
    Args:
        prediction: Model output
        target: Ground truth
        smooth: Smoothing factor
        threshold: Binarization threshold
        
    Returns:
        Mean Dice score
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()
    
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    intersection = (prediction * target).sum(dtype=torch.float32)
    dice = (2. * intersection + smooth) / (prediction.sum(dtype=torch.float32) + target.sum(dtype=torch.float32) + smooth)
    return dice


def compute_precision_recall(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Precision and Recall.
    
    Returns:
        (precision, recall)
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()
    
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    tp = (prediction * target).sum(dtype=torch.float32)
    fp = (prediction * (1 - target)).sum(dtype=torch.float32)
    fn = ((1 - prediction) * target).sum(dtype=torch.float32)
    
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    precision = torch.clamp(precision, 0.0, 1.0)
    recall = torch.clamp(recall, 0.0, 1.0)
    
    return precision, recall


def compute_accuracy(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute pixel-wise accuracy.
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    prediction = (prediction > threshold).float()
    target = (target > 0.5).float()
    
    prediction = prediction.view(-1)
    target = target.view(-1)
    
    correct = (prediction == target).sum(dtype=torch.float32)
    accuracy = correct / prediction.numel()
    return float(torch.clamp(accuracy, 0.0, 1.0))


def compute_metrics_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute IoU, Dice, Precision, Recall, F1, Accuracy from confusion matrix counts.
    """
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = (2.0 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    false_discovery_rate = fp / (fp + tp + eps)
    false_negative_rate = fn / (fn + tp + eps)

    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'accuracy': float(accuracy),
        'false_discovery_rate': float(false_discovery_rate),
        'false_negative_rate': float(false_negative_rate)
    }


def compute_classification_report_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-6) -> Dict[str, Dict[str, float]]:
    """
    Build classification report similar to sklearn from confusion counts.
    """
    precision_1 = tp / (tp + fp + eps)
    recall_1 = tp / (tp + fn + eps)
    f1_1 = (2.0 * precision_1 * recall_1) / (precision_1 + recall_1 + eps)
    support_1 = tp + fn

    tp_0 = tn
    fp_0 = fn
    fn_0 = fp
    precision_0 = tp_0 / (tp_0 + fp_0 + eps)
    recall_0 = tp_0 / (tp_0 + fn_0 + eps)
    f1_0 = (2.0 * precision_0 * recall_0) / (precision_0 + recall_0 + eps)
    support_0 = tp_0 + fn_0

    total_support = support_0 + support_1 + eps

    macro_precision = (precision_0 + precision_1) / 2.0
    macro_recall = (recall_0 + recall_1) / 2.0
    macro_f1 = (f1_0 + f1_1) / 2.0

    weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / total_support
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / total_support
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total_support

    return {
        '0': {
            'precision': float(precision_0),
            'recall': float(recall_0),
            'f1-score': float(f1_0),
            'support': float(support_0)
        },
        '1': {
            'precision': float(precision_1),
            'recall': float(recall_1),
            'f1-score': float(f1_1),
            'support': float(support_1)
        },
        'macro avg': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1-score': float(macro_f1),
            'support': float(support_0 + support_1)
        },
        'weighted avg': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1-score': float(weighted_f1),
            'support': float(support_0 + support_1)
        }
    }


def compute_global_binary_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    threshold: float = 0.5, 
    eps: float = 1e-6
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute aggregated binary segmentation metrics over the entire dataset.
    
    Returns:
        metrics: dict of IoU, Dice, Precision, Recall, F1, Accuracy
        counts: dict of TP, FP, FN, TN
    """
    predictions = predictions.float()
    targets = targets.float()

    if predictions.shape != targets.shape:
        predictions = torch.nn.functional.interpolate(
            predictions, size=targets.shape[2:], mode='bilinear', align_corners=False
        )

    if predictions.min() < 0 or predictions.max() > 1:
        predictions = torch.sigmoid(predictions)
    predictions = predictions.clamp(0.0, 1.0)
    targets = (targets > 0.5).float()

    binary_preds = (predictions > threshold).float()

    tp = (binary_preds * targets).sum(dtype=torch.float64).item()
    fp = (binary_preds * (1.0 - targets)).sum(dtype=torch.float64).item()
    fn = ((1.0 - binary_preds) * targets).sum(dtype=torch.float64).item()
    tn = ((1.0 - binary_preds) * (1.0 - targets)).sum(dtype=torch.float64).item()

    metrics = compute_metrics_from_counts(tp, fp, fn, tn, eps=eps)
    counts = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    return metrics, counts


def compute_f1_score(precision: torch.Tensor, recall: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute F1 score from precision and recall."""
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1.item()


def compute_confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Compute 2x2 confusion matrix."""
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    if prediction.min() >= 0 and prediction.max() <= 1 and not torch.all((prediction == 0) | (prediction == 1)):
        if prediction.min() < 0 or prediction.max() > 1:
            prediction = torch.sigmoid(prediction)
        prediction = (prediction > threshold).float()
    
    target = (target > 0.5).float()
    
    pred_np = prediction.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    
    tp = np.sum((pred_np == 1) & (target_np == 1))
    fp = np.sum((pred_np == 1) & (target_np == 0))
    fn = np.sum((pred_np == 0) & (target_np == 1))
    tn = np.sum((pred_np == 0) & (target_np == 0))
    
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics_from_prob(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute all metrics from probability maps (no sigmoid applied).
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    target = (target > 0.5).float()
    binary_prediction = (prediction > threshold).float()
    
    binary_prediction_flat = binary_prediction.view(-1)
    target_flat = target.view(-1)
    
    smooth = 1e-6
    intersection = (binary_prediction_flat * target_flat).sum(dtype=torch.float32)
    union = binary_prediction_flat.sum(dtype=torch.float32) + target_flat.sum(dtype=torch.float32) - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (binary_prediction_flat.sum(dtype=torch.float32) + target_flat.sum(dtype=torch.float32) + smooth)
    
    tp = (binary_prediction_flat * target_flat).sum(dtype=torch.float32)
    fp = (binary_prediction_flat * (1 - target_flat)).sum(dtype=torch.float32)
    fn = ((1 - binary_prediction_flat) * target_flat).sum(dtype=torch.float32)
    tn = ((1 - binary_prediction_flat) * (1 - target_flat)).sum(dtype=torch.float32)
    
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    correct = (binary_prediction_flat == target_flat).sum(dtype=torch.float32)
    accuracy = correct / binary_prediction_flat.numel()
    
    false_discovery_rate = fp / (fp + tp + eps)
    false_negative_rate = fn / (fn + tp + eps)
    
    return {
        'iou': float(torch.clamp(iou, 0.0, 1.0)),
        'dice': float(torch.clamp(dice, 0.0, 1.0)),
        'precision': float(torch.clamp(precision, 0.0, 1.0)),
        'recall': float(torch.clamp(recall, 0.0, 1.0)),
        'f1_score': float(torch.clamp(f1, 0.0, 1.0)),
        'accuracy': float(torch.clamp(accuracy, 0.0, 1.0)),
        'false_discovery_rate': float(torch.clamp(false_discovery_rate, 0.0, 1.0)),
        'false_negative_rate': float(torch.clamp(false_negative_rate, 0.0, 1.0))
    }


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute all standard segmentation metrics and return as a dictionary.
    """
    device = prediction.device
    target = target.to(device)
    
    prediction = prediction.float()
    target = target.float()
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)
    
    pred_bin = (prediction > threshold).float()
    mask_bin = (target > 0.5).float()
    
    tp = ((pred_bin == 1) & (mask_bin == 1)).sum()
    fp = ((pred_bin == 1) & (mask_bin == 0)).sum()
    fn = ((pred_bin == 0) & (mask_bin == 1)).sum()
    tn = ((pred_bin == 0) & (mask_bin == 0)).sum()
    
    iou = compute_iou(prediction, target, threshold=threshold)
    dice = compute_dice(prediction, target, threshold=threshold)
    precision, recall = compute_precision_recall(prediction, target, threshold)
    f1 = compute_f1_score(precision, recall)
    accuracy = compute_accuracy(prediction, target, threshold)
    
    eps = 1e-7
    false_discovery_rate = fp / (fp + tp + eps)
    false_negative_rate = fn / (fn + tp + eps)
    confusion_mat = compute_confusion_matrix(prediction, target, threshold)
    
    return {
        'iou': iou.item() if isinstance(iou, torch.Tensor) else iou,
        'dice': dice.item() if isinstance(dice, torch.Tensor) else dice,
        'precision': precision.item() if isinstance(precision, torch.Tensor) else precision,
        'recall': recall.item() if isinstance(recall, torch.Tensor) else recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'false_discovery_rate': false_discovery_rate.item() if isinstance(false_discovery_rate, torch.Tensor) else false_discovery_rate,
        'false_negative_rate': false_negative_rate.item() if isinstance(false_negative_rate, torch.Tensor) else false_negative_rate,
        'confusion_matrix': confusion_mat.tolist()
    }


def compute_classification_report(
    prediction: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5, 
    batch_size: int = 1000
) -> Dict[str, float]:
    """
    Compute per-class classification report with batched processing to save memory.
    """
    device = prediction.device
    target = target.to(device)
    
    if prediction.shape != target.shape:
        prediction = torch.nn.functional.interpolate(
            prediction, size=target.shape[2:], mode='bilinear', align_corners=False
        )

    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)

    total_samples = prediction.shape[0]
    batch_size = max(1, min(batch_size, total_samples))

    tp0, fp0, fn0 = 0.0, 0.0, 0.0
    tp1, fp1, fn1 = 0.0, 0.0, 0.0
    sup0, sup1 = 0.0, 0.0

    for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        p = prediction[i:end]
        t = target[i:end]

        bp = (p > threshold).float()
        tb = (t > 0.5).float()
        ip = 1 - bp
        it = 1 - tb

        t1 = (bp * tb).sum().item()
        f1 = (bp * it).sum().item()
        n1 = (ip * tb).sum().item()

        t0 = (ip * it).sum().item()
        f0 = n1
        n0 = f1

        tp0 += t0
        fp0 += f0
        fn0 += n0
        tp1 += t1
        fp1 += f1
        fn1 += n1

        sup0 += it.sum().item()
        sup1 += tb.sum().item()

    eps = 1e-6
    p0 = tp0 / (tp0 + fp0 + eps)
    r0 = tp0 / (tp0 + fn0 + eps)
    f0 = 2 * p0 * r0 / (p0 + r0 + eps)

    p1 = tp1 / (tp1 + fp1 + eps)
    r1 = tp1 / (tp1 + fn1 + eps)
    f1 = 2 * p1 * r1 / (p1 + r1 + eps)

    macro_p = (p0 + p1) / 2
    macro_r = (r0 + r1) / 2
    macro_f = (f0 + f1) / 2

    total = sup0 + sup1 + eps
    w_p = (p0 * sup0 + p1 * sup1) / total
    w_r = (r0 * sup0 + r1 * sup1) / total
    w_f = (f0 * sup0 + f1 * sup1) / total

    return {
        '0': {'precision': p0, 'recall': r0, 'f1-score': f0, 'support': sup0},
        '1': {'precision': p1, 'recall': r1, 'f1-score': f1, 'support': sup1},
        'macro avg': {'precision': macro_p, 'recall': macro_r, 'f1-score': macro_f, 'support': sup0+sup1},
        'weighted avg': {'precision': w_p, 'recall': w_r, 'f1-score': w_f, 'support': sup0+sup1}
    }


def calculate_threshold_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    thresholds: Optional[List[float]] = None, 
    batch_size: int = 1000
) -> Dict[str, List[float]]:
    """
    Evaluate metrics across thresholds to select the optimal one.
    Uses batched inference to reduce memory usage.
    """
    device = predictions.device
    targets = targets.to(device)
    
    predictions = predictions.float()
    targets = targets.float()
    
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9).tolist()
    
    results = {
        'thresholds': thresholds,
        'iou': [], 'dice': [], 'precision': [], 'recall': [], 'f1_score': [], 'accuracy': []
    }
    
    if predictions.min() < 0 or predictions.max() > 1:
        prob = torch.sigmoid(predictions)
    else:
        prob = predictions
    prob = prob.clamp(0.0, 1.0)
    
    N = prob.shape[0]
    
    for th in thresholds:
        iou_s, dice_s, prec_s, rec_s, f1_s, acc_s = 0.0,0.0,0.0,0.0,0.0,0.0
        cnt = 0
        
        for i in range(0, N, batch_size):
            j = min(i+batch_size, N)
            p = prob[i:j]
            t = targets[i:j]
            b = (p > th).float()
            
            try:
                iou = compute_iou_no_threshold(b, t)
                dice = compute_dice_no_threshold(b, t)
                pr, re = compute_precision_recall(b, t, 0.5)
                f1 = compute_f1_score(pr, re)
                acc = compute_accuracy(b, t, 0.5)
                
                iou_s += iou.item()
                dice_s += dice.item()
                prec_s += pr.item()
                rec_s += re.item()
                f1_s += f1
                acc_s += acc
                cnt +=1
            except Exception as e:
                continue
        
        if cnt == 0:
            results['iou'].append(0.0)
            results['dice'].append(0.0)
            results['precision'].append(0.0)
            results['recall'].append(0.0)
            results['f1_score'].append(0.0)
            results['accuracy'].append(0.0)
        else:
            results['iou'].append(iou_s/cnt)
            results['dice'].append(dice_s/cnt)
            results['precision'].append(prec_s/cnt)
            results['recall'].append(rec_s/cnt)
            results['f1_score'].append(f1_s/cnt)
            results['accuracy'].append(acc_s/cnt)
    
    best_idx = np.argmax(results['f1_score'])
    results['best_threshold'] = thresholds[best_idx]
    results['best_f1_score'] = results['f1_score'][best_idx]
    
    return results
