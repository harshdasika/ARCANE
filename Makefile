.PHONY: install dev test clean run setup

# Installation
install:
	pip install -e .

install-uv:
	uv pip install -e .

# Development
dev:
	pip install -e ".[dev]"

# Quick setup
setup: install
	mkdir -p data
	@echo "✅ Setup complete!"
	@echo "Add configuration to Claude Desktop and restart it"
	@echo "Then try: arcane"

# Running
run:
	arcane

run-dev:
	python -m arcane_mcp.server

# Testing
test:
	python -c "from arcane_mcp.core.identifier_resolver import IdentifierResolver; print('✅ Import test passed')"

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf data/academic_papers.db
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Help
help:
	@echo "ARCANE MCP Server"
	@echo "Available commands:"
	@echo "  make setup    - Install and set up the project"
	@echo "  make run      - Start the MCP server"
	@echo "  make test     - Run basic tests"
	@echo "  make clean    - Clean build artifacts"
