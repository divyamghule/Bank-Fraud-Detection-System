@echo off
REM Bank Fraud Detection System - Quick Start Script (Windows)

echo.
echo ========================================
echo 🏦 Bank Fraud Detection System
echo ========================================
echo.

REM Check if venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -q -r requirements.txt

REM Generate dataset
echo Generating dataset...
python src/predict_sample.py

REM Run Streamlit
echo.
echo ========================================
echo 🚀 Starting Web UI...
echo ========================================
echo.
echo Opening: http://localhost:8501
echo Press Ctrl+C to stop
echo.

streamlit run src/app.py
