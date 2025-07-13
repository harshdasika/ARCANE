"""
Configuration Manager for ARCANE MCP
Handles loading and managing configuration from files and environment variables
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management with environment variable overrides"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files (defaults to package config dir)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self._config_cache = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files from the config directory"""
        try:
            # Load main configuration
            self._config_cache['main'] = self._load_config_file('main_config.yaml')
            
            # Load API configuration
            self._config_cache['api'] = self._load_config_file('api_config.yaml')
            
            # Load similarity configuration
            self._config_cache['similarity'] = self._load_config_file('similarity_config.yaml')
            
            # Load domain configuration
            self._config_cache['domain'] = self._load_config_file('domain_config.yaml')
            
            # Load validation configuration
            self._config_cache['validation'] = self._load_config_file('validation_config.yaml')
            
            # Load temporal configuration
            self._config_cache['temporal'] = self._load_config_file('temporal_config.yaml')
            
            logger.info("All configuration files loaded successfully")
            
        except Exception as e:
            logger.warning(f"Error loading some configuration files: {e}")
            self._create_default_configs()
    
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load a single configuration file with environment variable overrides"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif filename.endswith('.json'):
                    config = json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {filename}")
                    return {}
            
            # Apply environment variable overrides
            config = self._apply_env_overrides(config, filename)
            return config or {}
            
        except Exception as e:
            logger.error(f"Error loading config file {filename}: {e}")
            return {}
    
    def _apply_env_overrides(self, config: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        if not config:
            return {}
        
        # Define environment variable mappings
        env_mappings = {
            'api_config.yaml': {
                'semantic_scholar.base_url': 'S2_API_URL',
                'semantic_scholar.rate_limit': 'S2_RATE_LIMIT',
                'opencitations.base_url': 'OC_API_URL', 
                'opencitations.rate_limit': 'OC_RATE_LIMIT',
                'arxiv.base_url': 'ARXIV_API_URL',
                'arxiv.rate_limit': 'ARXIV_RATE_LIMIT'
            },
            'main_config.yaml': {
                'database.path': 'ARCANE_DB_PATH',
                'semantic_model.name': 'SEMANTIC_MODEL_NAME',
                'cache.ttl_hours': 'CACHE_TTL_HOURS'
            }
        }
        
        if filename in env_mappings:
            for config_path, env_var in env_mappings[filename].items():
                env_value = os.getenv(env_var)
                if env_value is not None:
                    self._set_nested_value(config, config_path, env_value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: str):
        """Set a nested configuration value using dot notation"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Try to convert value to appropriate type
        final_value = self._convert_env_value(value)
        current[keys[-1]] = final_value
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _create_default_configs(self):
        """Create default configurations if files don't exist"""
        self._config_cache = {
            'main': self._get_default_main_config(),
            'api': self._get_default_api_config(),
            'similarity': self._get_default_similarity_config(),
            'domains': self._get_default_domain_config(),
            'validation': self._get_default_validation_config(),
            'temporal': self._get_default_temporal_config()
        }
        logger.info("Using default configurations")
    
    def get(self, config_section: str, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            config_section: Configuration section (main, api, similarity, etc.)
            key_path: Dot-separated path to configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if config_section not in self._config_cache:
            logger.warning(f"Configuration section '{config_section}' not found")
            return default
        
        config = self._config_cache[config_section]
        keys = key_path.split('.')
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_section(self, config_section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config_cache.get(config_section, {})
    
    def reload(self):
        """Reload all configuration files"""
        self._config_cache.clear()
        self._load_all_configs()
        logger.info("Configuration reloaded")
    
    # Default configuration methods
    def _get_default_main_config(self) -> Dict[str, Any]:
        """Default main configuration"""
        return {
            'database': {
                'path': os.getenv('ARCANE_DB_PATH', 'academic_papers.db'),
                'cache_ttl_hours': int(os.getenv('CACHE_TTL_HOURS', '24'))
            },
            'semantic_model': {
                'name': os.getenv('SEMANTIC_MODEL_NAME', 'all-MiniLM-L6-v2'),
                'fallback_models': ['all-MiniLM-L12-v2', 'sentence-transformers/all-mpnet-base-v2']
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))
            }
        }
    
    def _get_default_api_config(self) -> Dict[str, Any]:
        """Default API configuration"""
        return {
            'semantic_scholar': {
                'base_url': os.getenv('S2_API_URL', 'https://api.semanticscholar.org/graph/v1/'),
                'rate_limit': float(os.getenv('S2_RATE_LIMIT', '1.0')),
                'enhanced_rate_limit': 10.0
            },
            'opencitations': {
                'base_url': os.getenv('OC_API_URL', 'https://opencitations.net/index/api/v2/'),
                'rate_limit': float(os.getenv('OC_RATE_LIMIT', '2.0'))
            },
            'arxiv': {
                'base_url': os.getenv('ARXIV_API_URL', 'http://export.arxiv.org/api/'),
                'rate_limit': float(os.getenv('ARXIV_RATE_LIMIT', '1.0'))
            }
        }
    
    def _get_default_similarity_config(self) -> Dict[str, Any]:
        """Default similarity configuration"""
        return {
            'weights': {
                'mathematical_structures': 0.25,
                'problem_types': 0.20,
                'semantic_similarity': 0.25,
                'mathematical_patterns': 0.15,
                'methodological_keywords': 0.10,
                'structural_complexity': 0.05
            },
            'thresholds': {
                'base_threshold': 0.3,
                'adaptive_adjustments': {
                    'complex_pattern_reduction': 0.1,
                    'moderate_pattern_reduction': 0.05,
                    'uncertainty_iterative_boost': 0.05,
                    'rich_abstraction_boost': 0.03
                },
                'bounds': {
                    'min_threshold': 0.15,
                    'max_threshold': 0.45
                }
            },
            'confidence': {
                'base_confidence': 0.3,
                'text_length_boost': 0.2,
                'abstract_boost': 0.2,
                'technical_terms_boost': 0.2,
                'structure_boost': 0.1
            }
        }
    
    def _get_default_domain_config(self) -> Dict[str, Any]:
        """Default domain configuration - this will be expanded in the next step"""
        return {
            'search_strategies': {},  # Will be populated from existing hard-coded data
            'concept_mappings': {},   # Will be populated from existing hard-coded data
            'math_priorities': {}     # Will be populated from existing hard-coded data
        }
    
    def _get_default_validation_config(self) -> Dict[str, Any]:
        """Default validation configuration"""
        return {
            'known_cases': [
                {
                    'name': 'conformal_prediction_biomolecular',
                    'description': 'Conformal prediction applied to biomolecular design',
                    'conditions': {
                        'problem_types': ['uncertainty_quantification', 'prediction'],
                        'iterative_process': True
                    },
                    'indicators': {
                        'conformal': ['conformal', 'uncertainty', 'feedback', 'covariate shift'],
                        'biomolecular': ['biomolecular', 'molecular', 'drug', 'protein', 'chemical']
                    },
                    'validation_weight': 0.3,
                    'required_matches': {
                        'conformal': 2,
                        'biomolecular': 1
                    }
                }
            ]
        }
    
    def _get_default_temporal_config(self) -> Dict[str, Any]:
        """Default temporal configuration"""
        return {
            'filters': {
                'outdated_penalty_year': 2000,
                'recent_boost_year': 2020,
                'strict_outdated_year': 2015
            },
            'citation_thresholds': {
                'high_impact': 50,
                'medium_impact': 20,
                'boost_threshold': 50
            },
            'confidence_weights': {
                'coverage_optimal_papers': 50,
                'diversity_optimal_domains': 3
            }
        }

# Global configuration manager instance
config_manager = ConfigManager()