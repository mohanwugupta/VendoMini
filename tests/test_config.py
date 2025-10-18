"""Test configuration loader."""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import ConfigLoader


def test_load_simple_config():
    """Test loading a simple config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'experiment': {'name': 'test'},
            'simulation': {'max_steps': 100}
        }
        yaml.dump(config, f)
        f.flush()
        
        loaded = ConfigLoader.load_config(f.name)
        
        assert loaded['experiment']['name'] == 'test'
        assert loaded['simulation']['max_steps'] == 100
        
        Path(f.name).unlink()


def test_config_inheritance():
    """Test config file inheritance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create base config
        base_config = {
            'experiment': {'name': 'base'},
            'simulation': {'max_steps': 100, 'budget': 200}
        }
        base_path = tmpdir / 'base.yaml'
        with open(base_path, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create child config
        child_config = {
            'inherit': 'base.yaml',
            'experiment': {'name': 'child'},
            'simulation': {'max_steps': 500}
        }
        child_path = tmpdir / 'child.yaml'
        with open(child_path, 'w') as f:
            yaml.dump(child_config, f)
        
        # Load child
        loaded = ConfigLoader.load_config(str(child_path))
        
        assert loaded['experiment']['name'] == 'child'
        assert loaded['simulation']['max_steps'] == 500
        assert loaded['simulation']['budget'] == 200  # Inherited
        assert 'inherit' not in loaded


def test_grid_expansion():
    """Test grid parameter expansion."""
    config = {
        'experiment': {'name': 'test', 'replications': 2, 'seed': 42},
        'grid': {
            'pe_induction.p_shock': [0.1, 0.2],
            'model.name': ['llama', 'gpt']
        },
        'fixed': {
            'simulation.max_steps': 100
        }
    }
    
    expanded = ConfigLoader.expand_grid(config)
    
    # Should have 2 x 2 x 2 (replications) = 8 configs
    assert len(expanded) == 8
    
    # Check that parameters are set correctly
    for run_config in expanded:
        assert run_config['simulation']['max_steps'] == 100
        assert run_config['pe_induction']['p_shock'] in [0.1, 0.2]
        assert run_config['model']['name'] in ['llama', 'gpt']
        assert 'run_id' in run_config['experiment']
        assert 'seed' in run_config['experiment']


def test_nested_parameter_setting():
    """Test setting nested parameters with dot notation."""
    config = {}
    ConfigLoader._set_nested(config, 'a.b.c', 123)
    
    assert config['a']['b']['c'] == 123


def test_empty_grid():
    """Test config with no grid (should still create replications)."""
    config = {
        'experiment': {'name': 'test', 'replications': 3, 'seed': 0},
        'simulation': {'max_steps': 100}
    }
    
    expanded = ConfigLoader.expand_grid(config)
    
    assert len(expanded) == 3
    for i, run_config in enumerate(expanded):
        assert run_config['experiment']['replication_id'] == i
        assert run_config['experiment']['seed'] == i
