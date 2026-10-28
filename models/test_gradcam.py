"""
Grad-CAM Verification and Test Image Predictor.
Runs inference on sample test images and generates Grad-CAM heatmap overlays.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.cnn_model import BurnCNN, get_transforms, CLASS_NAMES, CLASS_DISPLAY
from models.gradcam import GradCAM

WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "weights", "best_burn_model.pt")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_test_images")

def run_gradcam_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Grad-CAM tests on {device}...")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return

    # Load Model
    model = BurnCNN(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Initialize GradCAM
    target_layer = model.get_target_layer()
    gradcam = GradCAM(model, target_layer)
    transforms_pipeline = get_transforms(is_train=False)

    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(("_sample.jpg", "_sample.png")) and "gradcam" not in f]
    print(f"Found {len(sample_files)} sample test images.")

    for sf in sample_files:
        img_path = os.path.join(SAMPLES_DIR, sf)
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transforms_pipeline(pil_img).unsqueeze(0).to(device)

        with torch.set_grad_enabled(True):
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item() * 100
            pred_class = CLASS_NAMES[pred_idx]

            # Generate Heatmap
            heatmap = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)
            blended_img, colored_cam = gradcam.overlay_heatmap(pil_img, heatmap, alpha=0.5)

            # Save overlay
            base_name = sf.rsplit(".", 1)[0]
            out_overlay_path = os.path.join(SAMPLES_DIR, f"{base_name}_gradcam_overlay.jpg")
            out_heatmap_path = os.path.join(SAMPLES_DIR, f"{base_name}_heatmap.jpg")
            blended_img.save(out_overlay_path, quality=95)
            colored_cam.save(out_heatmap_path, quality=95)

            print(f"Sample: {sf:<25} -> Pred: {CLASS_DISPLAY[pred_class]} ({confidence:.2f}%)")
            print(f"  Overlay saved: {out_overlay_path}")

    print("\nAll Grad-CAM test visualizations generated successfully!")

if __name__ == "__main__":
    run_gradcam_tests()
