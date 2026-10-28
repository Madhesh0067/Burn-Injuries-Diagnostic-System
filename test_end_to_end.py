"""
End-to-End Verification Test Script for Burn Injuries Diagnostic System.
Validates Database Auth, Ayurvedic Engine, CNN Model, Grad-CAM, PDF Generator, and Streamlit integrity.
"""

import os
import sys
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from web_app.auth import register_user, authenticate_user, log_scan, get_scans_for_user
from web_app.recommendation_engine import AyurvedicRecommendationEngine
from web_app.report_generator import generate_diagnostic_pdf
from models.cnn_model import BurnCNN, get_transforms, CLASS_NAMES
from models.gradcam import GradCAM

def run_all_tests():
    print("=" * 60)
    print("BURN INJURIES DIAGNOSTIC SYSTEM - END-TO-END VERIFICATION")
    print("=" * 60)

    # 1. Test Authentication & Database
    print("\n[1/5] Testing Authentication & SQLite Database...")
    test_email = "clinician_test@burnhospital.org"
    test_pwd = "SecureMedicalPassword123"
    test_name = "Dr. Ananya Sharma"

    reg_res = register_user(test_email, test_pwd, test_name)
    print(f"  Registration: {reg_res['message']}")
    
    auth_user = authenticate_user(test_email, test_pwd)
    assert auth_user is not None, "Authentication failed for valid credentials!"
    print(f"  Auth Success: Logged in as {auth_user['full_name']} ({auth_user['email']})")

    # 2. Test Ayurvedic Recommendation Engine
    print("\n[2/5] Testing Ayurvedic Recommendation Engine...")
    engine = AyurvedicRecommendationEngine()
    for c in CLASS_NAMES:
        rec = engine.get_recommendations(c, 0.88)
        assert rec["title"], f"Missing title for {c}"
        assert len(rec["immediate_first_aid"]) > 0, f"Missing first aid for {c}"
        assert len(rec["safety_precautions"]) > 0, f"Missing precautions for {c}"
        print(f"  Class [{c}]: Title='{rec['title']}' | Emergency={rec['is_emergency']} | Formulations={len(rec['ayurvedic_topical_therapies'])}")

    # 3. Test Model & Grad-CAM Inference
    print("\n[3/5] Testing PyTorch CNN Model & Grad-CAM Hook...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(BASE_DIR, "models", "weights", "best_burn_model.pt")
    assert os.path.exists(weights_path), f"Weights file not found at {weights_path}"

    model = BurnCNN(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model, model.get_target_layer())
    transforms = get_transforms(is_train=False)

    sample_img_path = os.path.join(BASE_DIR, "sample_test_images", "degree_3_test_sample.jpg")
    pil_img = Image.open(sample_img_path).convert("RGB")
    tensor_input = transforms(pil_img).unsqueeze(0).to(device)

    with torch.set_grad_enabled(True):
        out = model(tensor_input)
        pred_idx = torch.argmax(out, dim=1).item()
        cam = gradcam.generate_heatmap(tensor_input, target_class=pred_idx)
        blended, colored_cam = gradcam.overlay_heatmap(pil_img, cam, alpha=0.5)

    assert cam.shape == (7, 7) or len(cam.shape) == 2, "Heatmap dimensions unexpected!"
    assert blended.size == pil_img.size, "Overlay size must match original!"
    print(f"  Inference & Grad-CAM: Predicted class index={pred_idx} ({CLASS_NAMES[pred_idx]}), heatmap range=[{cam.min():.2f}, {cam.max():.2f}]")

    # 4. Test PDF Clinical Report Generation
    print("\n[4/5] Testing PDF Report Generation...")
    pdf_dir = os.path.join(BASE_DIR, "data", "reports")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_out = os.path.join(pdf_dir, "test_clinical_report.pdf")
    temp_orig = os.path.join(pdf_dir, "test_orig.jpg")
    temp_cam = os.path.join(pdf_dir, "test_cam.jpg")
    pil_img.save(temp_orig)
    blended.save(temp_cam)

    rec_result = engine.get_recommendations(CLASS_NAMES[pred_idx], 0.92)
    generated_pdf = generate_diagnostic_pdf(
        output_pdf_path=pdf_out,
        patient_identifier=auth_user["full_name"],
        prediction_data=rec_result,
        original_image_path=temp_orig,
        gradcam_image_path=temp_cam
    )
    assert os.path.exists(generated_pdf) and os.path.getsize(generated_pdf) > 1000, "PDF generation failed!"
    print(f"  PDF Report: Created successfully ({os.path.getsize(generated_pdf)} bytes) at {generated_pdf}")

    # 5. Test Scan Logging in Database
    print("\n[5/5] Testing Scan Logging to SQLite...")
    log_scan(
        user_email=auth_user["email"],
        image_name="degree_3_test_sample.jpg",
        predicted_class=rec_result["title"],
        confidence=rec_result["confidence"],
        gradcam_path=temp_cam,
        summary=rec_result.get("severity_level", "")
    )
    user_scans = get_scans_for_user(auth_user["email"])
    assert len(user_scans) > 0, "Scan was not retrieved from database!"
    print(f"  Logged Scans: Retrieved {len(user_scans)} scans for {auth_user['email']}")

    print("\n" + "=" * 60)
    print("ALL 5 SUBSYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
