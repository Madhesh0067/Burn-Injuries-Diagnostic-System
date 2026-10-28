@echo off
title Burn Injuries Diagnostic System
cd /d "%~dp0"
echo ==============================================================
echo       Starting Burn Injuries Diagnostic System Dashboard...
echo ==============================================================
echo.
call venv\Scripts\activate.bat
streamlit run web_app\app.py --server.port 8501
pause
