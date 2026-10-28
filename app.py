"""
Root Entrypoint for Streamlit Community Cloud Deployment.
Redirects execution to the main application in web_app/app.py.
"""

import os
import sys

# Ensure root directory is present in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Run the app
from web_app import app
