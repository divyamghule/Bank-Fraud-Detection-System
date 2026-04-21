@echo off
REM Bank Fraud Detection System - Quick Start Script (Windows)

setlocal
set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%.venv"
set "PYTHON_CMD="

echo.
echo ========================================
echo 🏦 Bank Fraud Detection System
echo ========================================
echo.

REM Prefer Python 3.12 or 3.11 for compatibility with project dependencies
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.12"
    if not defined PYTHON_CMD (
        py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>nul && set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo ERROR: Python 3.11 or 3.12 was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/ and try again.
    exit /b 1
)

echo Using Python: %PYTHON_CMD%

REM Create virtual environment if needed
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
)

REM Activate venv
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade packaging tools first to improve wheel installation
echo Upgrading packaging tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip/setuptools/wheel.
    exit /b 1
)

REM Install requirements
echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    echo Try Python 3.12 or reinstall the virtual environment.
    exit /b 1
)

REM Generate dataset
echo Generating dataset...
python src/predict_sample.py
if errorlevel 1 (
    echo ERROR: Dataset setup failed.
    exit /b 1
)

REM Run Streamlit
echo.
echo ========================================
echo 🚀 Starting Web UI...
echo ========================================
echo.
echo Opening: http://localhost:8501
echo Press Ctrl+C to stop
echo.

python -m streamlit run src/app.py

endlocal
