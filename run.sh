#!/bin/bash
# Bank Fraud Detection System - Quick Start Script (macOS/Linux)

set -e

echo ""
echo "========================================"
echo "🏦 Bank Fraud Detection System"
echo "========================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_CMD=""

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_CMD="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_CMD="python3.11"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11 or 3.12 was not found."
    echo "Install Python 3.12 and try again."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Check if venv exists
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate venv
echo "Activating virtual environment..."
. "$VENV_DIR/bin/activate"

echo "Upgrading packaging tools..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing dependencies..."
python -m pip install -r requirements.txt

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

python -m streamlit run src/app.py
