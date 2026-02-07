#!/bin/bash
# Launch Research Intelligence Platform UI
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

streamlit run streamlit_app.py --server.port 8503
