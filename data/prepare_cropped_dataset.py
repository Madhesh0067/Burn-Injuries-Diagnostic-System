"""
High-Precision Lesion-Focused Dataset Builder.
Extracts cropped burn lesions for Degree 1, Degree 2, Degree 3 from Baid et al. 2020,
and extracts verified authentic human skin patches (tested via chromophore & YCrCb analysis)
from the same clinical photos for Healthy Skin.
Eliminates background bias and drape/linen contamination so normal human skin is recognized accurately.
"""

import os
import io
import zipfile
import random
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_test_images")
ARCHIVE_ZIP = os.path.join(RAW_DIR, "archive.zip")

CLASSES = ["degree_1", "degree_2", "degree_3", "healthy"]
SPLITS = ["train", "val", "test"]

def overlaps(b1, b2):
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

def is_genuine_healthy_skin(crop_np: np.ndarray) -> bool:
    """
    Validates that a patch is authentic human skin (Fitzpatrick I-VI)
    and NOT hospital linens, blue/green drapes, dark shadows, or flash glare.
    """
    r = crop_np[:, :, 0].astype(np.float32)
    g = crop_np[:, :, 1].astype(np.float32)
    b = crop_np[:, :, 2].astype(np.float32)

    # 1. Luminance filter (reject deep shadows and flash glare)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mean_lum = np.mean(lum)
    if mean_lum < 40 or mean_lum > 235:
        return False

    # 2. Human skin chromophore dominance: R > G and R > B
    skin_rgb = (r > g) & (g >= b) & ((r - g) >= 8) & ((r - b) >= 12)
    rgb_pct = np.mean(skin_rgb)

    # 3. YCrCb space skin clustering (Kovac model: Cr in 130-180, Cb in 75-135)
    ycbcr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2YCrCb)
    cr = ycbcr[:, :, 1]
    cb = ycbcr[:, :, 2]
    skin_ycbcr = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)
    ycbcr_pct = np.mean(skin_ycbcr)

    # 4. Check for normal erythema (burns have red_excess > 35, healthy skin has red_excess <= 30)
    red_excess = np.mean(r - (g + b) / 2.0)
    is_normal_erythema = (red_excess <= 30.0)

    # 5. Reject completely uniform flat synthetic blocks (linens/plastics)
    std_lum = np.std(lum)
    has_texture = 4.0 <= std_lum <= 50.0

    return (rgb_pct >= 0.65 or ycbcr_pct >= 0.65) and is_normal_erythema and has_texture

def extract_all_crops(target_per_class: int = 240):
    print("Extracting lesion crops and verified healthy skin from archive.zip...", flush=True)

    burn_crops = {
        "degree_1": [],
        "degree_2": [],
        "degree_3": []
    }
    healthy_crops = []

    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zf:
        img_names = [n for n in zf.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.seed(42)
        random.shuffle(img_names)

        for img_name in img_names:
            base = img_name.rsplit(".", 1)[0]
            txt_name = f"{base}.txt"
            if txt_name not in zf.namelist():
                continue

            txt_data = zf.read(txt_name).decode("utf-8", errors="ignore").strip()
            if not txt_data:
                continue

            try:
                raw_bytes = zf.read(img_name)
                im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            except Exception:
                continue

            w, h = im.size
            if w < 50 or h < 50:
                continue

            burn_boxes = []

            # 1. Parse burn bounding boxes and crop lesions
            for idx, line in enumerate(txt_data.splitlines()):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in [0, 1, 2]:
                    continue

                cx, cy, bw, bh = map(float, parts[1:5])
                # Pixel box with 10% context margin
                margin_x = bw * 0.10
                margin_y = bh * 0.10
                bx1 = max(0, int((cx - bw/2 - margin_x) * w))
                by1 = max(0, int((cy - bh/2 - margin_y) * h))
                bx2 = min(w, int((cx + bw/2 + margin_x) * w))
                by2 = min(h, int((cy + bh/2 + margin_y) * h))

                box_w = bx2 - bx1
                box_h = by2 - by1
                if box_w >= 28 and box_h >= 28:
                    burn_boxes.append((bx1, by1, bx2, by2))
                    cls_key = {0: "degree_1", 1: "degree_2", 2: "degree_3"}[cls_id]
                    if len(burn_crops[cls_key]) < target_per_class * 2:
                        crop_im = im.crop((bx1, by1, bx2, by2)).resize((224, 224), Image.Resampling.BICUBIC)
                        buf = io.BytesIO()
                        crop_im.save(buf, format="JPEG", quality=92)
                        burn_crops[cls_key].append((f"{base}_lesion_{idx}.jpg", buf.getvalue(), crop_im))

            # 2. Extract genuine human skin patches from unburned regions of the same patient photo
            if len(healthy_crops) < target_per_class * 2 and burn_boxes:
                buffered_burns = []
                for (b1, b2, b3, b4) in burn_boxes:
                    pad = 25
                    buffered_burns.append((max(0, b1 - pad), max(0, b2 - pad), min(w, b3 + pad), min(h, b4 + pad)))

                im_np = np.array(im)
                patches_from_img = 0

                for patch_sz in [120, 90, 70, 50]:
                    if w > patch_sz and h > patch_sz:
                        for _ in range(12):
                            rx = random.randint(0, w - patch_sz)
                            ry = random.randint(0, h - patch_sz)
                            cand = (rx, ry, rx + patch_sz, ry + patch_sz)
                            if not any(overlaps(cand, b) for b in buffered_burns):
                                patch_arr = im_np[ry:ry+patch_sz, rx:rx+patch_sz]
                                if is_genuine_healthy_skin(patch_arr):
                                    patch_im = Image.fromarray(patch_arr).resize((224, 224), Image.Resampling.BICUBIC)
                                    buf = io.BytesIO()
                                    patch_im.save(buf, format="JPEG", quality=92)
                                    healthy_crops.append((f"{base}_healthy_{patches_from_img}.jpg", buf.getvalue(), patch_im))
                                    buffered_burns.append(cand)
                                    patches_from_img += 1
                                    if patches_from_img >= 2:
                                        break
                        if patches_from_img >= 2:
                            break

    print("Crops extracted:")
    for k in ["degree_1", "degree_2", "degree_3"]:
        print(f"  {k}: {len(burn_crops[k])} available", flush=True)
    print(f"  healthy: {len(healthy_crops)} verified human skin patches available", flush=True)

    # Balance each class to target_per_class
    balanced = {}
    for k in ["degree_1", "degree_2", "degree_3"]:
        items = burn_crops[k]
        random.shuffle(items)
        balanced[k] = items[:target_per_class]

    random.shuffle(healthy_crops)
    balanced["healthy"] = healthy_crops[:target_per_class]

    return balanced

def save_and_split(balanced_data):
    # Initialize and clean directories
    for split in SPLITS:
        for cls in CLASSES:
            d = os.path.join(PROCESSED_DIR, split, cls)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    os.remove(p)
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    random.seed(42)
    split_counts = defaultdict(lambda: defaultdict(int))

    for cls in CLASSES:
        items = balanced_data[cls]
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        splits_dict = {
            "train": items[:n_train],
            "val": items[n_train:n_train+n_val],
            "test": items[n_train+n_val:]
        }

        for split_name, split_items in splits_dict.items():
            for fname, byte_data, pil_img in split_items:
                target_path = os.path.join(PROCESSED_DIR, split_name, cls, fname)
                with open(target_path, "wb") as f:
                    f.write(byte_data)
                split_counts[split_name][cls] += 1

        # Also save high-quality verified sample test image for this class
        if items:
            best_sample = items[0]
            sample_dest = os.path.join(SAMPLES_DIR, f"{cls}_test_sample.jpg")
            with open(sample_dest, "wb") as f:
                f.write(best_sample[1])
            print(f"Saved representative sample: {sample_dest}")

    print("\nDataset split summary:")
    for split in SPLITS:
        total = sum(split_counts[split].values())
        print(f"  {split.upper()}: {total} images -> {dict(split_counts[split])}")

if __name__ == "__main__":
    balanced = extract_all_crops(target_per_class=240)
    save_and_split(balanced)
    print("Cropped dataset preparation complete!")
