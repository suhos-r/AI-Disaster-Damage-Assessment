import numpy as np
import cv2
import torch
from sklearn.metrics import roc_curve, auc

def calculate_iou(preds, targets, num_classes=5):
    '''Calculates Intersection over Union per class.'''
    ious = []
    preds = preds.argmax(dim=1).view(-1)
    targets = targets.view(-1)
    
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.nanmean(ious)

def calculate_dice(preds, targets, num_classes=5):
    '''Calculates Dice Coefficient per class.'''
    dices = []
    preds_cls = preds.argmax(dim=1).view(-1)
    targets_flat = targets.view(-1)
    
    for cls in range(num_classes):
        pred_inds = preds_cls == cls
        target_inds = targets_flat == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        total = pred_inds.long().sum().item() + target_inds.long().sum().item()
        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append((2. * intersection) / max(total, 1))
    return np.nanmean(dices)

def calculate_metrics(preds, targets, num_classes=5):
    '''Calculates Accuracy, AUC'''
    preds_cls = preds.argmax(dim=1).view(-1)
    targets_flat = targets.view(-1)
    
    correct = (preds_cls == targets_flat).sum().item()
    total = targets_flat.size(0)
    accuracy = correct / total
    
    # Calculate AUC for each class vs all
    aucs = []
    # Needs probabilities for ROC
    probs = torch.softmax(preds, dim=1) # (N, C, H, W)
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, num_classes).cpu().numpy()
    targets_np = targets_flat.cpu().numpy()
    
    # Subsample for faster ROC if too large (using a fixed grid to avoid memory issues)
    if len(targets_np) > 10000:
        idx = np.linspace(0, len(targets_np)-1, 10000, dtype=int)
        probs_flat = probs_flat[idx]
        targets_np = targets_np[idx]
        
    for cls in range(num_classes):
        y_true = (targets_np == cls).astype(int)
        y_scores = probs_flat[:, cls]
        # Only compute if we have both positive and negative examples
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.0
    
    return {'accuracy': accuracy, 'auc': mean_auc}

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlays a damage classification mask onto an image.
    Classes:
    0: Background (Transparent)
    1: No Damage (Green)
    2: Minor Damage (Yellow)
    3: Major Damage (Orange)
    4: Destroyed (Red)
    """
    colors = {
        0: [0, 0, 0],       # Background
        1: [0, 255, 0],     # No Damage
        2: [255, 255, 0],   # Minor
        3: [255, 165, 0],   # Major
        4: [255, 0, 0]      # Destroyed
    }
    
    color_mask = np.zeros_like(image)
    for cls, color in colors.items():
        color_mask[mask == cls] = color
        
    overlay_idx = mask > 0
    overlay_img = image.copy()
    
    blended = image.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha
    overlay_img[overlay_idx] = blended[overlay_idx].astype(np.uint8)
    
    return overlay_img
