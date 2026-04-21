#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "==========================================="
echo " PodLab | Thumbnail Extractor - Installer"
echo "==========================================="

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
echo "Python: $(python3 --version)"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading starter models..."
python tools/download_models.py || echo "WARNING: model download failed; heuristic mode still works."

echo "Done. Start with: source venv/bin/activate && python src/app.py"
