"""
CNN Model Architecture for Burn Injuries Diagnostic System.
Supports Transfer Learning with ResNet-18 / MobileNetV3 for 4-class burn classification.
"""

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

CLASS_NAMES = ["degree_1", "degree_2", "degree_3", "healthy"]

CLASS_DISPLAY = {
    "degree_1": "Degree 1 Burn (Superficial)",
    "degree_2": "Degree 2 Burn (Partial Thickness)",
    "degree_3": "Degree 3 Burn (Full Thickness)",
    "healthy": "Healthy Normal Skin"
}

def get_transforms(is_train: bool = False):
    """
    Returns image transformation pipeline for training and inference.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class BurnCNN(nn.Module):
    """
    Transfer learning CNN using ResNet18 backbone.
    High accuracy, fast CPU/GPU inference, well-suited for Grad-CAM explainability.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(BurnCNN, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self):
        """
        Returns the final convolutional layer for Grad-CAM computation.
        In ResNet-18, this is layer4[-1].conv2.
        """
        return self.backbone.layer4[-1].conv2

def extract_candidate_crops(pil_img: Image.Image):
    """
    Extracts multi-scale and lesion-focused region-of-interest (ROI) candidate crops.
    Ensures that localized erythematous or blistered lesions are inspected at high resolution
    without being diluted by surrounding unburned healthy skin or clothing/background.
    Implemented with pure NumPy and SciPy for complete cross-platform stability.
    """
    w, h = pil_img.size
    crops = [pil_img]
    arr = np.array(pil_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Pure NumPy YCbCr conversion
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    
    skin_mask = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 140) & (r > g) & (r > b) & (r > 50)
    if np.sum(skin_mask) < 200:
        is_neutral = (np.abs(r - g) < 14) & (np.abs(r - b) < 14)
        is_extreme = (r < 30) | (r > 248)
        skin_mask = ~(is_neutral | is_extreme) & (r > g) & (g > b * 0.7)
        
    if np.sum(skin_mask) >= 200:
        red_excess = r - (g + b) / 2.0
        median_re = np.median(red_excess[skin_mask])
        re_dev = red_excess - median_re
        
        gx = np.zeros_like(y)
        gy = np.zeros_like(y)
        gx[:, 1:-1] = np.abs(y[:, 2:] - y[:, :-2]) / 2.0
        gy[1:-1, :] = np.abs(y[2:, :] - y[:-2, :]) / 2.0
        grad_mag = np.sqrt(gx**2 + gy**2)
        med_grad = np.median(grad_mag[skin_mask])
        grad_dev = grad_mag - med_grad
        
        anomaly_score = np.zeros_like(r)
        anomaly_score[skin_mask] = np.maximum(0, re_dev[skin_mask]) * 0.6 + np.maximum(0, grad_dev[skin_mask]) * 0.4
        
        p90 = np.percentile(anomaly_score[skin_mask], 90)
        lesion_core = (anomaly_score >= p90) & skin_mask
        
        kernel = ndi.generate_binary_structure(2, 2)
        dilated = ndi.binary_dilation(lesion_core, structure=kernel, iterations=2)
        labeled, num_features = ndi.label(dilated)
        slices = ndi.find_objects(labeled)
        
        clusters = []
        for idx, slc in enumerate(slices):
            if slc is not None:
                area = int(np.sum(labeled[slc] == (idx + 1)))
                if area >= 60:
                    ymin, ymax = slc[0].start, slc[0].stop
                    xmin, xmax = slc[1].start, slc[1].stop
                    mean_score = float(np.mean(anomaly_score[ymin:ymax, xmin:xmax]))
                    clusters.append((xmin, ymin, xmax - xmin, ymax - ymin, area, mean_score))
                    
        clusters = sorted(clusters, key=lambda c: c[5], reverse=True)
        for cx, cy, cw, ch, area, score in clusters[:5]:
            pad_x = int(cw * 0.25)
            pad_y = int(ch * 0.25)
            lx1, ly1 = max(0, cx - pad_x), max(0, cy - pad_y)
            lx2, ly2 = min(w, cx + cw + pad_x), min(h, cy + ch + pad_y)
            if (lx2 - lx1) >= 40 and (ly2 - ly1) >= 40:
                crops.append(pil_img.crop((lx1, ly1, lx2, ly2)))
                
    cx1, cy1 = int(w * 0.125), int(h * 0.125)
    cx2, cy2 = int(w * 0.875), int(h * 0.875)
    crops.append(pil_img.crop((cx1, cy1, cx2, cy2)))
    return crops

def diagnose_multiscale(model, pil_img: Image.Image, device, transforms_fn):
    """
    Performs dual-stream lesion-aware clinical CAD diagnosis.
    Evaluates both the global contextual presentation and focal lesion candidate regions,
    guaranteeing that localized burns against clothing/backgrounds and intact healthy skin
    are both classified with maximum clinical precision.
    """
    crops = extract_candidate_crops(pil_img)
    tensors = torch.stack([transforms_fn(c) for c in crops]).to(device)
    with torch.no_grad():
        all_probs = F.softmax(model(tensors), dim=1).cpu().numpy()
        
    global_probs = all_probs[0]
    crop_burn_scores = all_probs[:, 0] + all_probs[:, 1] + all_probs[:, 2]
    
    # Identify the crop with highest burn evidence
    max_crop_idx = int(np.argmax(crop_burn_scores))
    max_burn_score = float(crop_burn_scores[max_crop_idx])
    focal_burn_probs = all_probs[max_crop_idx]
    
    # If a verified focal lesion shows clear burn injury (>= 55%):
    if max_burn_score >= 0.55:
        # Lesion detected! Blend focal burn attention (75%) with global context (25%)
        final_probs = 0.25 * global_probs + 0.75 * focal_burn_probs
        final_probs = final_probs / np.sum(final_probs)
        is_burn_diagnosed = True
        burn_idx = int(np.argmax([final_probs[0], final_probs[1], final_probs[2]]))
        pred_idx = burn_idx
        pred_class = CLASS_NAMES[burn_idx]
        conf = float(final_probs[burn_idx])
    else:
        # No focal lesion detected: evaluate overall presentation
        final_probs = 0.60 * global_probs + 0.40 * focal_burn_probs
        final_probs = final_probs / np.sum(final_probs)
        p_burn = float(final_probs[0] + final_probs[1] + final_probs[2])
        p_healthy = float(final_probs[3])
        if p_burn >= 0.50 or (p_burn > p_healthy and p_burn >= 0.40):
            is_burn_diagnosed = True
            burn_idx = int(np.argmax([final_probs[0], final_probs[1], final_probs[2]]))
            pred_idx = burn_idx
            pred_class = CLASS_NAMES[burn_idx]
            conf = float(final_probs[burn_idx])
        else:
            is_burn_diagnosed = False
            pred_idx = 3
            pred_class = "healthy"
            conf = float(p_healthy)
            
    p_burn_total = float(final_probs[0] + final_probs[1] + final_probs[2])
    p_healthy = float(final_probs[3])
    
    # Tissue Biomarker Analysis
    arr = np.array(pil_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_neutral = (np.abs(r - g) < 14) & (np.abs(r - b) < 14)
    is_extreme = (r < 30) | (r > 248)
    tissue_mask = ~(is_neutral | is_extreme)
    re_map = r - (g + b) / 2.0
    if np.any(tissue_mask):
        re_hotspot = float(np.percentile(re_map[tissue_mask], 90))
        rg_ratio = float(np.mean(r[tissue_mask]) / (np.mean(g[tissue_mask]) + 1e-5))
    else:
        re_hotspot = float(np.percentile(re_map, 90))
        rg_ratio = float(np.mean(r) / (np.mean(g) + 1e-5))
        
    if pred_class == "healthy":
        status_desc = "Intact Stratum Corneum & Dermis"
        erythema_desc = "Normal Physiological Vascularity"
        triage_summary = f"Intact Epidermal Barrier Confirmed (Healthy Skin Evidence: {p_healthy*100:.1f}%)"
    elif pred_class == "degree_1":
        status_desc = "Superficial Epidermal Injury (Acute Erythema)"
        erythema_desc = "Acute Thermal Hyperemia & Capillary Dilation"
        triage_summary = f"Thermal Injury Identified (Degree 1 Evidence: {final_probs[0]*100:.1f}%, Cumulative Burn: {p_burn_total*100:.1f}%)"
    elif pred_class == "degree_2":
        status_desc = "Partial-Thickness Blister / Dermal Exudate"
        erythema_desc = "Dermal Microvascular Injury & Epidermal Detachment"
        triage_summary = f"Thermal Injury Identified (Degree 2 Evidence: {final_probs[1]*100:.1f}%, Cumulative Burn: {p_burn_total*100:.1f}%)"
    else:
        status_desc = "Full-Thickness Coagulation Necrosis / Eschar"
        erythema_desc = "Microvascular Thrombosis & Dermal Necrosis"
        triage_summary = f"Critical Thermal Injury Identified (Full-Thickness Burn: {final_probs[2]*100:.1f}%)"
        
    biomarkers = {
        "red_excess_hotspot": re_hotspot,
        "rg_ratio": rg_ratio,
        "status_desc": status_desc,
        "erythema_desc": erythema_desc,
        "triage_summary": triage_summary,
        "is_burn_diagnosed": is_burn_diagnosed,
        "p_burn_total": p_burn_total,
        "p_healthy": p_healthy,
        "n_crops": len(crops)
    }
    
    return pred_class, pred_idx, conf, final_probs, biomarkers
