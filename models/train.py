"""
Model Training and Evaluation Pipeline for Burn Injuries Diagnostic System.
Trains ResNet-18 Transfer Learning model on 4 burn classes, logs metrics,
and evaluates performance with confusion matrix and classification report.
"""

import os
import sys
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.cnn_model import BurnCNN, get_transforms, CLASS_NAMES

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def train_model(epochs: int = 6, batch_size: int = 16, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Datasets and DataLoaders
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(is_train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(is_train=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(is_train=False))

    print(f"Dataset classes detected: {train_dataset.classes}")
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = BurnCNN(num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epochs": list(range(1, epochs + 1))
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = running_corrects / total_samples

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_running_corrects / val_total

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc * 100, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc * 100, 2))

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc*100:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f"\nTraining completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    # Save Best Model Weights
    best_weights_path = os.path.join(WEIGHTS_DIR, "best_burn_model.pt")
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, best_weights_path)
    print(f"Best model weights saved to {best_weights_path}")

    # Save History JSON
    history_path = os.path.join(WEIGHTS_DIR, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot & Save Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["epochs"], history["train_acc"], label="Train Accuracy", marker="o", color="#2563EB")
    ax1.plot(history["epochs"], history["val_acc"], label="Val Accuracy", marker="s", color="#10B981")
    ax1.set_title("Model Accuracy Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history["epochs"], history["train_loss"], label="Train Loss", marker="o", color="#DC2626")
    ax2.plot(history["epochs"], history["val_loss"], label="Val Loss", marker="s", color="#F59E0B")
    ax2.set_title("Model Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(WEIGHTS_DIR, "learning_curves.png"), dpi=200)
    plt.close()

    # Evaluation on Test Set
    print("\n--- Evaluating Best Model on Held-out Test Set ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print("\nClassification Report:\n", report_text)

    # Save metrics JSON
    metrics_path = os.path.join(WEIGHTS_DIR, "evaluation_metrics.json")
    metrics_data = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_accuracy": round(report["accuracy"] * 100, 2),
        "class_names": CLASS_NAMES
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="left")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Burn Severity Confusion Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(WEIGHTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    print("All training artifacts & evaluation charts successfully saved in models/weights/")

if __name__ == "__main__":
    train_model(epochs=6, batch_size=16, lr=1e-3)
