"""Configuration management"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from environment and files"""
    
    config = {
        "semantic_scholar_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        "opencitations_access_token": os.getenv("OPENCITATIONS_ACCESS_TOKEN"),
        "database_path": os.getenv("ARCANE_DB_PATH", "academic_papers.db"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "arxiv_rate_limit": float(os.getenv("ARXIV_RATE_LIMIT", "1.0")),
        "semantic_scholar_rate_limit": float(os.getenv("S2_RATE_LIMIT", "1.0")),
        "opencitations_rate_limit": float(os.getenv("OC_RATE_LIMIT", "2.0"))
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config