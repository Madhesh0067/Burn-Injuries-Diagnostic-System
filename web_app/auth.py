"""
Authentication and Scan History module using SQLite.
Supports Email sign-up, sign-in, guest access, and patient scan logging.
"""

import sqlite3
import hashlib
import os
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "burn_diagnostics.db"
)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            image_name TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            gradcam_image_path TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def register_user(email: str, password: str, full_name: str) -> Dict[str, Any]:
    email = email.strip().lower()
    if not email or "@" not in email:
        return {"success": False, "message": "Invalid email address format."}
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters long."}
    
    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (email, full_name, salt, password_hash) VALUES (?, ?, ?, ?)",
            (email, full_name.strip(), salt, pwd_hash)
        )
        conn.commit()
        conn.close()
        return {"success": True, "message": "Account created successfully! You can now log in."}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "An account with this email address already exists."}
    except Exception as e:
        return {"success": False, "message": f"Database error: {str(e)}"}

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None
    
    computed_hash = _hash_password(password, user["salt"])
    if computed_hash == user["password_hash"]:
        return {
            "email": user["email"],
            "full_name": user["full_name"],
            "created_at": user["created_at"]
        }
    return None

def log_scan(user_email: str, image_name: str, predicted_class: str, confidence: float, gradcam_path: str = "", summary: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scans (user_email, image_name, predicted_class, confidence, gradcam_image_path, summary) VALUES (?, ?, ?, ?, ?, ?)",
        (user_email, image_name, predicted_class, confidence, gradcam_path, summary)
    )
    conn.commit()
    conn.close()

def get_scans_for_user(user_email: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM scans WHERE user_email = ? ORDER BY created_at DESC LIMIT 50",
        (user_email,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize DB tables on import
init_db()
