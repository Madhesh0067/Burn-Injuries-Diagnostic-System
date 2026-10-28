"""
Automated GitHub Repository Creator & Project Uploader.
Uploads the Burn Injuries Diagnostic System to GitHub via GitHub Git Data API.
No local git installation required.
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.error

# Ensure stdout handles utf-8 safely
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES_TO_UPLOAD = [
    "app.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "run_dashboard.bat",
    "data/ayurvedic_knowledge.json",
    "models/cnn_model.py",
    "models/gradcam.py",
    "models/train.py",
    "models/test_gradcam.py",
    "models/weights/best_burn_model.pt",
    "models/weights/confusion_matrix.png",
    "models/weights/evaluation_metrics.json",
    "models/weights/learning_curves.png",
    "models/weights/train_history.json",
    "sample_test_images/degree_1_test_sample.jpg",
    "sample_test_images/degree_2_test_sample.jpg",
    "sample_test_images/degree_3_test_sample.jpg",
    "sample_test_images/healthy_test_sample.jpg",
    "sample_test_images/degree_1_test_sample_gradcam_overlay.jpg",
    "sample_test_images/degree_2_test_sample_gradcam_overlay.jpg",
    "sample_test_images/degree_3_test_sample_gradcam_overlay.jpg",
    "sample_test_images/healthy_test_sample_gradcam_overlay.jpg",
    "web_app/app.py",
    "web_app/auth.py",
    "web_app/recommendation_engine.py",
    "web_app/report_generator.py"
]

def make_request(url: str, method: str = "GET", data: dict = None, token: str = None) -> dict:
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", "Burn-Diagnosis-Uploader")
    req.add_header("Accept", "application/vnd.github.v3+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode("utf-8")
    else:
        body = None

    try:
        with urllib.request.urlopen(req, data=body, timeout=60) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        try:
            err_json = json.loads(err_msg)
            return {"error": True, "status": e.code, "message": err_json.get("message", err_msg)}
        except Exception:
            return {"error": True, "status": e.code, "message": err_msg}

def upload_project(token: str, repo_name: str = "Burn-Injuries-Diagnostic-System", is_private: bool = False):
    print("=" * 65)
    print("BURN INJURIES DIAGNOSTIC SYSTEM - GITHUB UPLOADER")
    print("=" * 65)

    # 1. Verify User
    print("\n[1/5] Verifying GitHub Authentication...")
    user_info = make_request("https://api.github.com/user", token=token)
    if "login" not in user_info:
        print(f"❌ Authentication failed: {user_info.get('message', 'Invalid Token')}")
        return False
    
    username = user_info["login"]
    print(f"  ✓ Authenticated as: {username} ({user_info.get('name', '')})")

    # 2. Create or Find Repository
    print(f"\n[2/5] Creating GitHub Repository '{repo_name}'...")
    create_payload = {
        "name": repo_name,
        "description": "Clinical AI Diagnostic & Ayurvedic Management System for Burn Injuries using Deep CNN and Grad-CAM",
        "private": is_private,
        "auto_init": True
    }
    repo_res = make_request("https://api.github.com/user/repos", method="POST", data=create_payload, token=token)
    
    if repo_res.get("error"):
        if "already exists" in str(repo_res.get("message", "")):
            print(f"  ℹ️ Repository '{repo_name}' already exists. Will update it.")
        else:
            print(f"❌ Failed to create repo: {repo_res.get('message')}")
            return False
    else:
        print(f"  ✓ Repository created: https://github.com/{username}/{repo_name}")

    repo_url = f"https://api.github.com/repos/{username}/{repo_name}"

    # 3. Upload Blobs
    print(f"\n[3/5] Uploading project files ({len(FILES_TO_UPLOAD)} files)...")
    tree_items = []
    
    for rel_path in FILES_TO_UPLOAD:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"  ⚠️ Skipping missing file: {rel_path}")
            continue

        size = os.path.getsize(full_path)
        print(f"  • Uploading {rel_path:<45} ({size / 1024:.1f} KB)...", end="", flush=True)

        with open(full_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")

        blob_res = make_request(
            f"{repo_url}/git/blobs",
            method="POST",
            data={"content": content_b64, "encoding": "base64"},
            token=token
        )

        if blob_res.get("error") or "sha" not in blob_res:
            print(f" ❌ Error: {blob_res.get('message')}")
            return False

        tree_items.append({
            "path": rel_path.replace("\\", "/"),
            "mode": "100644",
            "type": "blob",
            "sha": blob_res["sha"]
        })
        print(" ✓ Done")

    # 4. Create Tree & Commit
    print("\n[4/5] Committing changes to branch 'main'...")
    # Get current commit SHA
    ref_res = make_request(f"{repo_url}/git/ref/heads/main", token=token)
    parent_sha = None
    base_tree = None
    if not ref_res.get("error") and "object" in ref_res:
        parent_sha = ref_res["object"]["sha"]
        commit_res = make_request(f"{repo_url}/git/commits/{parent_sha}", token=token)
        if "tree" in commit_res:
            base_tree = commit_res["tree"]["sha"]

    tree_payload = {"tree": tree_items}
    if base_tree:
        tree_payload["base_tree"] = base_tree

    new_tree = make_request(f"{repo_url}/git/trees", method="POST", data=tree_payload, token=token)
    if new_tree.get("error") or "sha" not in new_tree:
        print(f"❌ Failed to create git tree: {new_tree.get('message')}")
        return False

    commit_payload = {
        "message": "Clinical AI: Burn Injuries Diagnostic System with CNN, Grad-CAM and Ayurvedic Recommendations",
        "tree": new_tree["sha"]
    }
    if parent_sha:
        commit_payload["parents"] = [parent_sha]

    new_commit = make_request(f"{repo_url}/git/commits", method="POST", data=commit_payload, token=token)
    if new_commit.get("error") or "sha" not in new_commit:
        print(f"❌ Failed to create commit: {new_commit.get('message')}")
        return False

    # Update or Create Ref
    if parent_sha:
        update_ref = make_request(
            f"{repo_url}/git/refs/heads/main",
            method="PATCH",
            data={"sha": new_commit["sha"], "force": True},
            token=token
        )
    else:
        update_ref = make_request(
            f"{repo_url}/git/refs",
            method="POST",
            data={"ref": "refs/heads/main", "sha": new_commit["sha"]},
            token=token
        )

    # 5. Success & Deployment Guide
    print("\n" + "=" * 65)
    print("🎉 PROJECT SUCCESSFULLY UPLOADED TO GITHUB!")
    print("=" * 65)
    print(f"\n📁 GitHub Repository URL: https://github.com/{username}/{repo_name}")
    print(f"\n🚀 Direct Streamlit Community Cloud Deployment Link:")
    deploy_url = f"https://share.streamlit.io/deploy?repository={username}/{repo_name}&branch=main&mainModule=app.py"
    print(f"👉 {deploy_url}\n")
    return True

if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN")
    if len(sys.argv) > 1:
        token = sys.argv[1]
    
    if not token:
        print("Please provide your GitHub Personal Access Token.")
        print("Usage: python upload_to_github.py <YOUR_GITHUB_TOKEN>")
        sys.exit(1)
        
    upload_project(token)
