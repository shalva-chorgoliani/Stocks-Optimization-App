#!/bin/bash
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install it from https://www.python.org/downloads/ and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Setting up the app for the first time (this may take a minute)..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

streamlit run app.py
