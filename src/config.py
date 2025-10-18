"""Configuration loader with inheritance and grid expansion."""

import yaml
from pathlib import Path
from typing import Dict, Any, List
from itertools import product
import copy


class ConfigLoader:
    """Load and process VendoMini configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load a YAML config file, handling inheritance.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            Merged configuration dictionary
        """
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if 'inherit' in config:
            parent_path = config_path.parent / config['inherit']
            parent_config = ConfigLoader.load_config(str(parent_path))
            # Merge parent with child (child overrides parent)
            merged = ConfigLoader._deep_merge(parent_config, config)
            # Remove inherit key
            merged.pop('inherit', None)
            return merged
        
        return config
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    @staticmethod
    def _set_nested(d: Dict, path: str, value: Any):
        """Set a value in a nested dict using dot notation."""
        keys = path.split('.')
        current = d
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    @staticmethod
    def expand_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand grid parameters into individual configurations.
        
        Args:
            config: Configuration with 'grid' and 'fixed' sections
            
        Returns:
            List of individual run configurations
        """
        grid = config.get('grid', {})
        fixed = config.get('fixed', {})
        replications = config.get('experiment', {}).get('replications', 1)
        base_seed = config.get('experiment', {}).get('seed', 42)
        
        if not grid:
            # No grid, return single config with replications
            configs = []
            for rep in range(replications):
                run_config = copy.deepcopy(config)
                run_config['experiment']['replication_id'] = rep
                run_config['experiment']['seed'] = base_seed + rep if base_seed else rep
                configs.append(run_config)
            return configs
        
        # Generate all combinations
        grid_keys = list(grid.keys())
        grid_values = [grid[k] if isinstance(grid[k], list) else [grid[k]] 
                       for k in grid_keys]
        
        combinations = list(product(*grid_values))
        
        configs = []
        for combo_idx, combo in enumerate(combinations):
            for rep in range(replications):
                run_config = copy.deepcopy(config)
                
                # Apply fixed parameters
                for path, value in fixed.items():
                    ConfigLoader._set_nested(run_config, path, value)
                
                # Apply grid parameters
                for path, value in zip(grid_keys, combo):
                    ConfigLoader._set_nested(run_config, path, value)
                
                # Set replication info
                run_config['experiment']['combination_id'] = combo_idx
                run_config['experiment']['replication_id'] = rep
                run_config['experiment']['seed'] = base_seed + combo_idx * replications + rep if base_seed else combo_idx * replications + rep
                
                # Create run ID
                run_config['experiment']['run_id'] = f"{config['experiment']['name']}_c{combo_idx}_r{rep}"
                
                configs.append(run_config)
        
        return configs


class Config:
    """Configuration manager for VendoMini experiments."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        self.config = config_dict
        
        # Validate required sections
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration structure."""
        # Remove strict validation - allow flexible config structures
        pass
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config.get('env', {})
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return self.config.get('agent', {})
    
    def get_crash_config(self) -> Dict[str, Any]:
        """Get crash detector configuration."""
        return self.config.get('crash_detector', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.config.get('experiment', {})
    
    def get_grid_config(self) -> Dict[str, Any]:
        """Get grid search configuration."""
        return self.config.get('grid', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
