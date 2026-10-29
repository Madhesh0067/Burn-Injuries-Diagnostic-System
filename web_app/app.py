"""
Burn Injuries Diagnostic System - Clinical AI & Ayurvedic Management Platform.
Main Streamlit Application.
"""

import os
import sys
import json
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st

# Configure parent path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.cnn_model import BurnCNN, get_transforms, CLASS_NAMES, CLASS_DISPLAY
from models.gradcam import GradCAM
from web_app.recommendation_engine import AyurvedicRecommendationEngine
from web_app.auth import register_user, authenticate_user, log_scan, get_scans_for_user
from web_app.report_generator import generate_diagnostic_pdf

# Set Page Config
st.set_page_config(
    page_title="Burn Injuries Diagnostic System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .badge-deg1 {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-deg2 {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-deg3 {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #DC2626;
    }
    .badge-healthy {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .card-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_email" not in st.session_state:
    st.session_state.user_email = "guest@clinical.local"
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest Clinician"
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = "sample.jpg"

@st.cache_resource
def load_diagnostic_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(BASE_DIR, "models", "weights", "best_burn_model.pt")
    
    model = BurnCNN(num_classes=len(CLASS_NAMES), pretrained=False)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    target_layer = model.get_target_layer()
    gradcam = GradCAM(model, target_layer)
    transforms = get_transforms(is_train=False)
    rec_engine = AyurvedicRecommendationEngine()

    return model, gradcam, transforms, rec_engine, device

model, gradcam, transforms, rec_engine, device = load_diagnostic_resources()

# Sidebar Navigation & Auth
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
    st.title("Clinical Panel")
    
    # Auth Status
    if st.session_state.authenticated:
        st.success(f"Logged in: **{st.session_state.user_name}**")
        st.caption(f"Email: {st.session_state.user_email}")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.user_email = "guest@clinical.local"
            st.session_state.user_name = "Guest Clinician"
            st.rerun()
    else:
        st.info("Operating in **Guest Mode**")
        with st.expander("🔐 Clinician Sign In / Sign Up"):
            tab_login, tab_signup = st.tabs(["Sign In", "Sign Up"])
            
            with tab_login:
                login_email = st.text_input("Email ID", key="login_email")
                login_pwd = st.text_input("Password", type="password", key="login_pwd")
                if st.button("Log In", use_container_width=True):
                    res = authenticate_user(login_email, login_pwd)
                    if res:
                        st.session_state.authenticated = True
                        st.session_state.user_email = res["email"]
                        st.session_state.user_name = res["full_name"]
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
            
            with tab_signup:
                reg_name = st.text_input("Full Name", key="reg_name")
                reg_email = st.text_input("Email ID", key="reg_email")
                reg_pwd = st.text_input("Password (min 6 chars)", type="password", key="reg_pwd")
                if st.button("Create Account", use_container_width=True):
                    res = register_user(reg_email, reg_pwd, reg_name)
                    if res["success"]:
                        st.success(res["message"])
                    else:
                        st.error(res["message"])

    st.markdown("---")
    menu = st.radio(
        "Navigation",
        [
            "🔍 Diagnostic Studio",
            "🌿 Ayurvedic Protocols",
            "📊 Model Analytics",
            "📁 Scan History & Reports",
            "ℹ️ Clinical Knowledge Base"
        ]
    )

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🩺 Burn Injuries Diagnostic System</h1>
    <p>AI-Assisted Thermal Injury Triage, Grad-CAM Visual Explainability & Classical Ayurvedic Management Protocols</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SECTION 1: DIAGNOSTIC STUDIO -----------------
if menu == "🔍 Diagnostic Studio":
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("1. Ingest Skin Image")
        upload_choice = st.radio("Choose Input Mode:", ["Upload File", "Select Test Preset"], horizontal=True)

        selected_pil = None
        img_name = "burn_image.jpg"

        if upload_choice == "Upload File":
            uploaded_file = st.file_uploader("Upload dermatological skin photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_pil = Image.open(uploaded_file).convert("RGB")
                img_name = uploaded_file.name
        else:
            samples_dir = os.path.join(BASE_DIR, "sample_test_images")
            presets = {
                "Degree 1 Sample (Superficial Erythema)": "degree_1_test_sample.jpg",
                "Degree 2 Sample (Partial Thickness Blister)": "degree_2_test_sample.jpg",
                "Degree 3 Sample (Full Thickness Eschar)": "degree_3_test_sample.jpg",
                "Healthy Skin Sample (Normal Barrier)": "healthy_test_sample.jpg"
            }
            preset_choice = st.selectbox("Select verified test sample:", list(presets.keys()))
            sample_filename = presets[preset_choice]
            sample_file_path = os.path.join(samples_dir, sample_filename)
            if os.path.exists(sample_file_path):
                selected_pil = Image.open(sample_file_path).convert("RGB")
                img_name = sample_filename

        if selected_pil is not None:
            st.image(selected_pil, caption="Selected Skin Area for Assessment", use_container_width=True)
            alpha_val = st.slider("Grad-CAM Heatmap Opacity (Alpha Blend):", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        else:
            st.info("Please upload an image or choose a preset to begin diagnostic analysis.")

    with col_right:
        st.subheader("2. AI Diagnosis & Grad-CAM Analysis")
        if selected_pil is not None:
            # Run Inference
            input_tensor = transforms(selected_pil).unsqueeze(0).to(device)
            
            with torch.set_grad_enabled(True):
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]

                # ----------------- HIERARCHICAL CLINICAL TRIAGE -----------------
                # CLASS_NAMES = ["degree_1", "degree_2", "degree_3", "healthy"]
                # Medical decision protocol: Thermal Injury Detection vs. Intact Integument
                p_burn_total = float(probs[0] + probs[1] + probs[2])
                p_healthy = float(probs[3])

                if p_burn_total >= p_healthy:
                    # Definite thermal injury detected: Select predominant burn severity grade
                    burn_indices = [0, 1, 2]
                    pred_idx = int(burn_indices[int(np.argmax([probs[0], probs[1], probs[2]]))])
                    pred_class = CLASS_NAMES[pred_idx]
                    confidence = round(max(float(probs[pred_idx]), p_burn_total) * 100, 2)
                    is_burn_diagnosed = True
                else:
                    # Intact skin confirmed
                    pred_idx = 3
                    pred_class = "healthy"
                    confidence = round(p_healthy * 100, 2)
                    is_burn_diagnosed = False

                # Tissue Hotspot Biomarker Analysis (excludes neutral studio walls & gray sheets)
                arr_rgb = np.array(selected_pil, dtype=np.float32)
                r_chan, g_chan, b_chan = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]
                is_neutral_bg = (np.abs(r_chan - g_chan) < 14) & (np.abs(r_chan - b_chan) < 14)
                is_extreme = (r_chan < 30) | (r_chan > 245)
                tissue_mask = ~(is_neutral_bg | is_extreme)
                red_excess_map = r_chan - (g_chan + b_chan) / 2.0

                if np.any(tissue_mask):
                    red_excess_hotspot = float(np.percentile(red_excess_map[tissue_mask], 90))
                    rg_ratio = float(np.mean(r_chan[tissue_mask]) / (np.mean(g_chan[tissue_mask]) + 1e-5))
                else:
                    red_excess_hotspot = float(np.percentile(red_excess_map, 90))
                    rg_ratio = float(np.mean(r_chan) / (np.mean(g_chan) + 1e-5))

                # Grad-CAM heatmap
                heatmap = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)
                blended_pil, colored_cam = gradcam.overlay_heatmap(selected_pil, heatmap, alpha=alpha_val)

            # Retrieve Ayurvedic Recommendations
            rec_data = rec_engine.get_recommendations(pred_class, confidence)

            # Display Result Badge
            badge_class = {
                "degree_1": "badge-deg1",
                "degree_2": "badge-deg2",
                "degree_3": "badge-deg3",
                "healthy": "badge-healthy"
            }.get(pred_class, "badge-deg1")

            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <span class="{badge_class}" style="font-size: 1.25rem;">
                    {rec_data['title']}
                </span>
                <span style="font-size: 1.1rem; font-weight: 600; margin-left: 10px; color: #374151;">
                    Confidence: {rec_data['confidence']}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            if rec_data.get("is_emergency"):
                st.error("🚨 **CRITICAL MEDICAL EMERGENCY**: Full-thickness tissue destruction identified. Mandatory immediate emergency burn-unit hospital referral.")
            
            # Probability Bars
            st.write("**Prediction Probability Distribution:**")
            for c_idx, c_name in enumerate(CLASS_NAMES):
                pct = float(probs[c_idx]) * 100
                st.write(f"• **{CLASS_DISPLAY[c_name]}**: {pct:.1f}%")
                st.progress(min(pct / 100.0, 1.0))

            # Tissue Biomarker Summary
            status_desc = "Intact Stratum Corneum & Dermis" if pred_class == "healthy" else ("Superficial Epidermal Injury (Erythema)" if pred_class == "degree_1" else ("Partial-Thickness Blister / Dermal Exudate" if pred_class == "degree_2" else "Full-Thickness Coagulation Necrosis / Eschar"))
            erythema_desc = "Normal Baseline" if red_excess_hotspot < 30.0 else ("Moderate Erythema" if red_excess_hotspot < 50.0 else "Acute Inflammatory Hyperemia / Dermal Injury")
            triage_summary = f"Thermal Injury Identified (Cumulative Burn Evidence: {p_burn_total*100:.1f}% vs. Healthy: {p_healthy*100:.1f}%)" if is_burn_diagnosed else f"Intact Epidermal Barrier Confirmed (Healthy Skin Evidence: {p_healthy*100:.1f}%)"

            st.markdown(f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px; margin-bottom: 12px;">
                <p style="margin: 0; font-size: 0.85rem; color: #475569;">
                    <b>🔬 Clinical Triage & Biomarker Assessment:</b><br/>
                    • <b>Triage Assessment:</b> {triage_summary}<br/>
                    • <b>Lesion Hotspot Erythema:</b> {red_excess_hotspot:.1f} ({erythema_desc})<br/>
                    • <b>Tissue Chromophore Ratio (R/G):</b> {rg_ratio:.2f}<br/>
                    • <b>Integumentary Barrier Status:</b> {status_desc}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Visual Explainability (Grad-CAM)
            st.markdown("---")
            st.write("### 🔍 Grad-CAM Visual Interpretability")
            st.caption("Visual activation highlights anatomical regions (erythema, blistering, or tissue eschar) driving the CNN classification decision.")
            
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(colored_cam, caption="Class Activation Heatmap", use_container_width=True)
            with g_col2:
                st.image(blended_pil, caption=f"Superimposed Diagnostic Overlay (α={alpha_val})", use_container_width=True)

            # Action Buttons: Save & PDF
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("💾 Save Scan to History", use_container_width=True):
                    # Save temporary images for logging
                    temp_dir = os.path.join(BASE_DIR, "data", "scans_cache")
                    os.makedirs(temp_dir, exist_ok=True)
                    cam_save_path = os.path.join(temp_dir, f"cam_{img_name}")
                    blended_pil.save(cam_save_path)
                    
                    log_scan(
                        user_email=st.session_state.user_email,
                        image_name=img_name,
                        predicted_class=rec_data["title"],
                        confidence=rec_data["confidence"],
                        gradcam_path=cam_save_path,
                        summary=rec_data.get("severity_level", "")
                    )
                    st.success("Diagnostic record successfully logged in patient scan history!")

            with btn_col2:
                # Generate PDF report
                pdf_output_dir = os.path.join(BASE_DIR, "data", "reports")
                os.makedirs(pdf_output_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_output_dir, f"report_{img_name.rsplit('.',1)[0]}.pdf")
                
                # Save temp images for PDF inclusion
                temp_orig = os.path.join(pdf_output_dir, "temp_orig.jpg")
                temp_cam = os.path.join(pdf_output_dir, "temp_cam.jpg")
                selected_pil.save(temp_orig)
                blended_pil.save(temp_cam)
                
                generate_diagnostic_pdf(
                    output_pdf_path=pdf_path,
                    patient_identifier=st.session_state.user_name,
                    prediction_data=rec_data,
                    original_image_path=temp_orig,
                    gradcam_image_path=temp_cam
                )

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="📄 Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name=f"Burn_Diagnostic_Report_{img_name.rsplit('.',1)[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    # Comprehensive Ayurvedic Panel below Studio
    if selected_pil is not None:
        st.markdown("---")
        st.subheader("3. Prescribed Ayurvedic Regimen & Clinical Safety Protocol")
        
        tab_firstaid, tab_herbs, tab_diet, tab_precautions = st.tabs([
            "🚨 Immediate First Aid",
            "🌿 Classical Topical Therapies",
            "🥗 Dietary Guidelines (Ahara)",
            "⚠️ Safety Alerts & Red Flags"
        ])

        with tab_firstaid:
            st.write(f"**Dosha Involvement:** `{rec_data.get('dosha_involvement', 'N/A')}`")
            st.write(f"**Clinical Manifestation:** {rec_data.get('clinical_presentation', 'N/A')}")
            st.markdown("#### Primary Action Steps:")
            for step in rec_data.get("immediate_first_aid", []):
                st.markdown(f"- {step}")

        with tab_herbs:
            therapies = rec_data.get("ayurvedic_topical_therapies", [])
            if therapies:
                for t in therapies:
                    with st.container():
                        st.markdown(f"#### 🧴 {t.get('remedy', '')}")
                        st.write(f"**Application Technique:** {t.get('method', '')}")
                        st.info(f"**Therapeutic Rationale (Pharmacology):** {t.get('benefit', '')}")
            else:
                st.write("No topical application permitted in acute full-thickness tissue destruction.")

        with tab_diet:
            st.markdown("#### Nutritional Support (*Pitta Shamaka* & Tissue Rebuilding):")
            for d in rec_data.get("dietary_guidelines", []):
                st.markdown(f"- {d}")

        with tab_precautions:
            if rec_data.get("is_emergency"):
                st.error("### ⚠️ CRITICAL EMERGENCY HOSPITAL REFERRAL")
            for p in rec_data.get("safety_precautions", []):
                st.warning(f"⚠️ {p}")

# ----------------- SECTION 2: AYURVEDIC PROTOCOLS -----------------
elif menu == "🌿 Ayurvedic Protocols":
    st.subheader("Classical Ayurvedic Burn Wound Care (*Dagdha Vrana Chikitsa*)")
    st.write("In classical Ayurvedic texts (*Sushruta Samhita, Sutrasthana Chapter 12*), burns (*Dagdha*) are classified based on the depth of tissue damage and dosha vitiation.")
    
    deg_tabs = st.tabs([
        "Degree 1 (Prathama Dagdha)",
        "Degree 2 (Dvitiya Dagdha)",
        "Degree 3 (Tritiya Dagdha)",
        "Healthy Skin Maintenance"
    ])
    
    for idx, (tab_obj, key) in enumerate(zip(deg_tabs, ["degree_1", "degree_2", "degree_3", "healthy"])):
        with tab_obj:
            info = rec_engine.knowledge_base.get(key, {})
            st.markdown(f"### {info.get('title', '')}")
            st.write(f"**Severity Category:** {info.get('severity_level', '')}")
            st.write(f"**Classical Pathophysiology:** {info.get('dosha_involvement', '')}")
            st.write(info.get("clinical_presentation", ""))
            
            st.markdown("#### Classical Herbal Formulations:")
            for t in info.get("ayurvedic_topical_therapies", []):
                st.markdown(f"- **{t.get('remedy', '')}**: {t.get('method', '')} (*{t.get('benefit', '')}*)")

            st.markdown("#### Clinical Precautions:")
            for p in info.get("safety_precautions", []):
                st.markdown(f"⚠️ {p}")

# ----------------- SECTION 3: MODEL ANALYTICS -----------------
elif menu == "📊 Model Analytics":
    st.subheader("Model Training & Validation Performance Analytics")
    
    weights_dir = os.path.join(BASE_DIR, "models", "weights")
    history_file = os.path.join(weights_dir, "train_history.json")
    metrics_file = os.path.join(weights_dir, "evaluation_metrics.json")
    learning_curve_img = os.path.join(weights_dir, "learning_curves.png")
    cm_img = os.path.join(weights_dir, "confusion_matrix.png")

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            eval_data = json.load(f)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Test Accuracy", f"{eval_data.get('test_accuracy', 0)}%")
        m_col2.metric("Total Diagnostic Classes", "4")
        m_col3.metric("Architecture", "ResNet-18 (CNN)")
        m_col4.metric("Interpretability", "Grad-CAM")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Learning Curves (Loss & Accuracy)")
            if os.path.exists(learning_curve_img):
                st.image(learning_curve_img, use_container_width=True)
            else:
                st.info("Learning curves plot generating...")

        with c2:
            st.write("### Multi-Class Confusion Matrix")
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)
            else:
                st.info("Confusion matrix generating...")

        st.markdown("### Per-Class Performance Breakdown")
        cr = eval_data.get("classification_report", {})
        if cr:
            import pandas as pd
            table_data = []
            for c in CLASS_NAMES:
                if c in cr:
                    table_data.append({
                        "Class": CLASS_DISPLAY[c],
                        "Precision (%)": round(cr[c]["precision"] * 100, 2),
                        "Recall (%)": round(cr[c]["recall"] * 100, 2),
                        "F1-Score (%)": round(cr[c]["f1-score"] * 100, 2),
                        "Test Support": cr[c]["support"]
                    })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    else:
        st.info("Training analytics will be displayed here once model training completes.")

# ----------------- SECTION 4: SCAN HISTORY -----------------
elif menu == "📁 Scan History & Reports":
    st.subheader(f"Patient Diagnostic Scan History ({st.session_state.user_email})")
    scans = get_scans_for_user(st.session_state.user_email)

    if scans:
        for s in scans:
            with st.container():
                st.markdown(f"#### 📅 Scan: {s['created_at']} - {s['predicted_class']}")
                sc_col1, sc_col2 = st.columns([1, 3])
                with sc_col1:
                    if s.get("gradcam_image_path") and os.path.exists(s["gradcam_image_path"]):
                        st.image(s["gradcam_image_path"], width=160)
                with sc_col2:
                    st.write(f"• **Image:** {s['image_name']}")
                    st.write(f"• **Confidence:** {s['confidence']}%")
                    st.write(f"• **Severity Notes:** {s['summary']}")
                st.markdown("---")
    else:
        st.info("No saved scans yet for this account. Run a diagnosis in the Studio and click 'Save Scan to History'.")

# ----------------- SECTION 5: CLINICAL KNOWLEDGE -----------------
elif menu == "ℹ️ Clinical Knowledge Base":
    st.subheader("Clinical Research & Pharmacological References")
    st.markdown("""
    ### 1. Convolutional Neural Network & Grad-CAM Methodology
    - **Dataset**: Trained on peer-reviewed medical burn imagery from *Baid et al. (2020)*, verified and augmented across 4 distinct thermal severity stages.
    - **Grad-CAM Layer**: Hooks into the terminal residual block (`layer4[-1].conv2`) to project visual gradients back onto spatial feature maps, delineating inflamed vs. necrosis boundaries.

    ### 2. Classical Ayurvedic References
    - **Sushruta Samhita, Sutrasthana (Ch. 12)**: *Dagdha Vrana Lakshana & Chikitsa*
    - **Charaka Samhita, Chikitsasthana (Ch. 25)**: *Dwi-Vraniyo Adhyaya*
    - **Herbal Principles**:
      - *Shatadhauta Ghrita*: Washer 100 times in cold water in copper vessels, reducing particle size for epidermal penetrance and profound cooling (*Sheeta Virya*).
      - *Jatyadi Ghrita*: Infused with *Jati*, *Nimba*, *Haridra*, *Patola*, and *Karanja* for antimicrobial protection and rapid collagen re-epithelialization (*Vrana Ropana*).

    ### 3. Medical Emergency Disclaimers
    > ⚠️ **Clinical Alert**: This software is an algorithmic clinical decision-support and educational triage tool. Third-degree burns and burns exceeding 10% TBSA require immediate emergency trauma management.
    """)
