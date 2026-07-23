@echo off
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install it from https://www.python.org/downloads/ and try again.
    echo IMPORTANT: during install, check the box "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Setting up the app for the first time, this may take a minute...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

streamlit run app.py
