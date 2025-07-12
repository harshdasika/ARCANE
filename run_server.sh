#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python src/arcane_mcp/server.py 