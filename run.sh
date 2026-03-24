#!/bin/bash
# Bank Fraud Detection System - Quick Start Script (macOS/Linux)

echo ""
echo "========================================"
echo "🏦 Bank Fraud Detection System"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Generate dataset
echo "Generating dataset..."
python src/predict_sample.py

# Run Streamlit
echo ""
echo "========================================"
echo "🚀 Starting Web UI..."
echo "========================================"
echo ""
echo "Opening: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/app.py
