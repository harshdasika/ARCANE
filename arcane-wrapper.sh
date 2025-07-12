#!/bin/bash
set -e  # Exit on any error
echo "Starting arcane-wrapper.sh" >&2
cd "$(dirname "$0")"
echo "Changed to directory: $(pwd)" >&2
echo "Activating virtual environment..." >&2
source venv/bin/activate
echo "Virtual environment activated. Python path: $(which python)" >&2
echo "Running ARCANE..." >&2
exec arcane 