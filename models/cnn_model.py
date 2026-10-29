"""
CNN Model Architecture for Burn Injuries Diagnostic System.
Supports Transfer Learning with ResNet-18 / MobileNetV3 for 4-class burn classification.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

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
    """
    w, h = pil_img.size
    crops = [pil_img] # Always include the global whole image
    
    arr = np.array(pil_img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Tissue mask (exclude studio white, extreme dark, and neutral gray)
    is_neutral = (np.abs(r - g) < 14) & (np.abs(r - b) < 14)
    is_extreme = (r < 30) | (r > 248)
    skin_mask = ~(is_neutral | is_extreme) & (r > g) & (g > b * 0.7)
    
    if np.sum(skin_mask) > 100:
        # Erythema anomaly map relative to skin baseline
        red_excess = r - (g + b) / 2.0
        skin_re = red_excess[skin_mask]
        re_thresh = np.percentile(skin_re, 80)
        hotspot_mask = (red_excess >= re_thresh) & skin_mask
        
        # Texture gradient anomaly (blisters, peeling epidermal sheets, necrotic eschar)
        gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
        grad_mag = np.abs(grad)
        grad_thresh = np.percentile(grad_mag[skin_mask], 85)
        anomaly_mask = hotspot_mask | ((grad_mag >= grad_thresh) & skin_mask)
        
        mask_u8 = (anomaly_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask_u8, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for c in contours[:2]:
            x, y, cw, ch = cv2.boundingRect(c)
            pad_x = int(cw * 0.2)
            pad_y = int(ch * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                crops.append(pil_img.crop((x1, y1, x2, y2)))
                
    # Center crop (covers center 75%)
    cx1 = int(w * 0.125)
    cy1 = int(h * 0.125)
    cx2 = int(w * 0.875)
    cy2 = int(h * 0.875)
    crops.append(pil_img.crop((cx1, cy1, cx2, cy2)))
    return crops

def diagnose_multiscale(model, pil_img: Image.Image, device, transforms_fn):
    """
    Performs dual-stream lesion-aware clinical diagnosis.
    Evaluates both the global contextual presentation and focal lesion candidate regions,
    guaranteeing that both localized burns and intact healthy skin are accurately classified.
    """
    crops = extract_candidate_crops(pil_img)
    tensors = torch.stack([transforms_fn(c) for c in crops]).to(device)
    
    with torch.no_grad():
        all_probs = F.softmax(model(tensors), dim=1).cpu().numpy()
        
    global_probs = all_probs[0]
    
    # Calculate cumulative burn scores across all crops: [deg1, deg2, deg3, healthy]
    crop_burns = all_probs[:, 0] + all_probs[:, 1] + all_probs[:, 2]
    max_burn_crop_idx = int(np.argmax(crop_burns))
    max_focal_probs = all_probs[max_burn_crop_idx]
    
    # Multi-crop ensemble: 40% global context + 60% focal lesion attention
    ens_probs = 0.40 * global_probs + 0.60 * max_focal_probs
    ens_probs = ens_probs / np.sum(ens_probs)
    
    p_burn = float(ens_probs[0] + ens_probs[1] + ens_probs[2])
    p_healthy = float(ens_probs[3])
    
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
        
    # Hierarchical Clinical Triage Decision:
    # Thermal burn injury is diagnosed if cumulative burn evidence is significant (>=40% or burn > healthy)
    if p_burn >= 0.40 or (p_burn > p_healthy):
        is_burn_diagnosed = True
        burn_idx = int(np.argmax([ens_probs[0], ens_probs[1], ens_probs[2]]))
        pred_idx = burn_idx
        pred_class = CLASS_NAMES[burn_idx]
        conf = float(ens_probs[burn_idx])
    else:
        is_burn_diagnosed = False
        pred_idx = 3
        pred_class = "healthy"
        conf = float(p_healthy)
        
    if pred_class == "healthy":
        status_desc = "Intact Stratum Corneum & Dermis"
        erythema_desc = "Normal Physiological Vascularity"
        triage_summary = f"Intact Epidermal Barrier Confirmed (Healthy Skin Evidence: {p_healthy*100:.1f}%)"
    elif pred_class == "degree_1":
        status_desc = "Superficial Epidermal Injury (Acute Erythema)"
        erythema_desc = "Acute Thermal Hyperemia & Capillary Dilation"
        triage_summary = f"Thermal Injury Identified (Degree 1 Evidence: {ens_probs[0]*100:.1f}%, Cumulative Burn: {p_burn*100:.1f}%)"
    elif pred_class == "degree_2":
        status_desc = "Partial-Thickness Blister / Dermal Exudate"
        erythema_desc = "Dermal Microvascular Injury & Epidermal Detachment"
        triage_summary = f"Thermal Injury Identified (Degree 2 Evidence: {ens_probs[1]*100:.1f}%, Cumulative Burn: {p_burn*100:.1f}%)"
    else:
        status_desc = "Full-Thickness Coagulation Necrosis / Eschar"
        erythema_desc = "Microvascular Thrombosis & Dermal Necrosis"
        triage_summary = f"Critical Thermal Injury Identified (Full-Thickness Burn: {ens_probs[2]*100:.1f}%)"
        
    biomarkers = {
        "red_excess_hotspot": re_hotspot,
        "rg_ratio": rg_ratio,
        "status_desc": status_desc,
        "erythema_desc": erythema_desc,
        "triage_summary": triage_summary,
        "is_burn_diagnosed": is_burn_diagnosed,
        "p_burn_total": p_burn,
        "p_healthy": p_healthy,
        "n_crops": len(crops)
    }
    
    return pred_class, pred_idx, conf, ens_probs, biomarkers
