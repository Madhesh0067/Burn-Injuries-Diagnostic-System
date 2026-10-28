# Burn Injuries Diagnostic System

An algorithmic clinical decision-support and educational triage platform combining **Deep Convolutional Neural Networks (CNN)**, **Grad-CAM visual explainability**, and **Classical Ayurvedic Burn Wound Care (*Dagdha Vrana Chikitsa*)**.

---

## 🌟 Key Features

1. **4-Class Dermatological Burn Severity Classification**:
   - **Degree 1 (Superficial / *Prathama Dagdha*)**: Epidermal erythema, pain, no blisters.
   - **Degree 2 (Partial Thickness / *Dvitiya Dagdha*)**: Fluid blisters, dermis involvement, weeping.
   - **Degree 3 (Full Thickness / *Tritiya Dagdha*)**: Charred, leathery eschar — **Immediate Emergency Hospital Referral**.
   - **Healthy Skin (*Prakriti Twak*)**: Normal intact skin barrier.

2. **Grad-CAM Visual Interpretability**:
   - Computes activation gradients from the final convolutional layer of the ResNet-18 backbone.
   - Generates heatmaps highlighting the exact tissue areas (blister borders, thermal erythema, or necrotic eschar) driving the diagnosis.
   - Includes interactive opacity slider for fine-grained clinical visual comparison.

3. **Evidence-Based Ayurvedic Recommendation Engine**:
   - Classical formulations (*Shatadhauta Ghrita*, *Kumari Swarasa*, *Jatyadi Ghrita*, *Triphala Kashaya*).
   - Dosha imbalance identification (*Pitta-Rakta*, *Pitta-Vata*).
   - Dietary (*Ahara*) and behavioral (*Vihara*) guidelines.
   - Safety red flags (e.g. strict contraindication of home remedies for full-thickness Degree 3 burns).

4. **Structured Clinical Web Dashboard (Streamlit)**:
   - **Diagnostic Studio**: Image upload & one-click test presets.
   - **Ayurvedic Care Hub**: Structured treatment tabs.
   - **Model Analytics**: Loss curves, accuracy history, and confusion matrix.
   - **Patient History & PDF Export**: Downloadable clinical summary reports with embedded Grad-CAM visuals.
   - **Flexible Authentication**: Optional email login/signup with SQLite backend, plus instant **Guest Mode**.

---

## 🚀 Quick Start Guide

### Option 1: Double-Click Launcher
Simply double-click `run_dashboard.bat` inside the folder on your Desktop:
```text
C:\Users\madhe\OneDrive\Desktop\Burn Injuries Diagnostic System\run_dashboard.bat
```

### Option 2: Command Line
Open terminal in the project directory and run:
```bash
venv\Scripts\activate
streamlit run web_app\app.py
```
Then open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
Burn Injuries Diagnostic System/
├── data/
│   ├── raw/                      # Downloaded dataset archives
│   ├── processed/                # Preprocessed train, val, test splits (640 balanced images)
│   ├── reports/                  # Generated diagnostic PDF reports
│   └── ayurvedic_knowledge.json  # Comprehensive Ayurvedic clinical knowledge base
├── models/
│   ├── cnn_model.py              # ResNet-18 CNN architecture
│   ├── gradcam.py                # Grad-CAM heatmap generation & image superimposition
│   ├── train.py                  # Training pipeline with data augmentation & metrics logging
│   ├── test_gradcam.py           # Grad-CAM verification on sample images
│   └── weights/                  # Best model checkpoint (.pt) and evaluation plots
├── sample_test_images/           # Instant test samples for each burn degree and healthy skin
├── web_app/
│   ├── app.py                    # Streamlit clinical dashboard
│   ├── auth.py                   # SQLite email auth and scan history logging
│   ├── recommendation_engine.py  # Ayurvedic recommendation engine
│   └── report_generator.py       # ReportLab PDF clinical report generator
├── test_end_to_end.py            # Comprehensive verification test suite
├── run_dashboard.bat             # 1-Click Desktop launcher
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🔬 Scientific & Clinical References
1. **Skin Burn Dataset**: Baid et al. (2020), *"Detection of Different Degrees of Skin Burn using YOLOv3"*.
2. **ISIC Dermatological Archive**: International Skin Imaging Collaboration (benign human skin).
3. **Grad-CAM**: Selvaraju et al. (2017), *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization"*.
4. **Classical Ayurvedic Texts**: *Sushruta Samhita (Sutrasthana Ch. 12: Dagdha Vrana)* & *Charaka Samhita (Chikitsasthana Ch. 25)*.

---

> ⚠️ **Medical Disclaimer**: This application is an automated decision-support and educational triage system. Third-degree burns and large-surface thermal injuries require immediate hospital burn ICU admission.
