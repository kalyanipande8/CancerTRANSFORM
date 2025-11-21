#!/usr/bin/env bash
set -euo pipefail

# Creates a virtual environment in .venv and installs requirements
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment created at .venv. Activate with: source .venv/bin/activate"