import logging
import os
import yaml
from typing import List, Dict, Any
from app.faults.models import FaultConfig

logger = logging.getLogger(__name__)

class FaultConfigLoader:
    """
    Utility to parse and load FaultConfig listings from YAML configuration files,
    Environment Variables, or custom Python dictionary formats.
    """
    
    @staticmethod
    def load_from_dict(data: List[Dict[str, Any]]) -> List[FaultConfig]:
        """Convert a list of dictionaries into Pydantic models."""
        configs = []
        for item in data:
            configs.append(FaultConfig(**item))
        return configs

    @staticmethod
    def load_from_yaml(file_path: str) -> List[FaultConfig]:
        """Loads configuration from a YAML file."""
        if not os.path.exists(file_path):
            return []
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            
        if not content or "faults" not in content:
            return []
            
        return FaultConfigLoader.load_from_dict(content["faults"])

    @staticmethod
    def load_from_env() -> List[FaultConfig]:
        """
        Loads configuration from an environment variable 'FAULTS_CONFIG' 
        which holds a JSON representation of the configuration.
        """
        env_content = os.getenv("FAULTS_CONFIG")
        if not env_content:
            return []
            
        try:
            import json
            data = json.loads(env_content)
            if isinstance(data, list):
                return FaultConfigLoader.load_from_dict(data)
            elif isinstance(data, dict) and "faults" in data:
                return FaultConfigLoader.load_from_dict(data["faults"])
        except Exception as e:
            logger.warning("FAULTS_CONFIG could not be parsed: %s", e)
            
        return []
