"""Integration tests for the full experiment pipeline."""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import ConfigLoader
from src.experiment_runner import run_single_experiment


@pytest.fixture
def test_config():
    """Create a minimal test configuration."""
    return {
        'experiment': {
            'name': 'integration_test',
            'replications': 1,
            'seed': 42,
            'run_id': 'test_run',
            'combination_id': 0,
            'replication_id': 0
        },
        'simulation': {
            'max_steps': 20,  # Short for testing
            'complexity_level': 0,  # Minimal complexity
            'initial_budget': 200,
            'pressure_level': 'low'
        },
        'pe_induction': {
            'p_shock': 0.0,  # No shocks for deterministic testing
            'pe_mag': 'low',
            'pe_type_mix': 'realistic',
            'observability': 'full'
        },
        'interface': {
            'prediction_mode': 'optional',  # Optional to simplify
            'prediction_format': 'structured',
            'memory_tools': 'none',
            'recovery_tools': 'none'
        },
        'model': {
            'name': 'test-model',
            'context_length': 8000,
            'temperature': 0.0,
            'max_tokens_per_call': 1000
        },
        'measurement': {
            'crash_threshold': 'moderate',
            'pe_windows': [10, 100],
            'success_metric': 'time_to_crash',
            'log_every_n_steps': 1,
            'checkpoint_every_n_steps': 100
        },
        'paths': {
            'logs_dir': 'test_logs',
            'checkpoints_dir': 'test_checkpoints',
            'results_dir': 'test_results'
        }
    }


def test_single_experiment_run(test_config, tmp_path):
    """Test running a single experiment end-to-end."""
    # Update paths to use temp directory
    test_config['paths']['logs_dir'] = str(tmp_path / 'logs')
    test_config['paths']['results_dir'] = str(tmp_path / 'results')
    
    # Run experiment
    summary = run_single_experiment(test_config)
    
    # Check summary
    assert 'run_id' in summary
    assert summary['run_id'] == 'test_run'
    assert 'total_steps' in summary
    assert summary['total_steps'] > 0
    assert summary['total_steps'] <= 20  # Max steps
    assert 'crashed' in summary
    assert 'time_to_crash' in summary
    assert 'final_budget' in summary
    
    # Check that log files were created
    log_dir = Path(test_config['paths']['logs_dir']) / 'test_run'
    assert log_dir.exists()
    assert (log_dir / 'steps.jsonl').exists()
    assert (log_dir / 'summary.json').exists()


def test_grid_expansion_integration():
    """Test that grid expansion works correctly."""
    config = {
        'experiment': {
            'name': 'grid_test',
            'replications': 2,
            'seed': 0
        },
        'grid': {
            'pe_induction.p_shock': [0.0, 0.1],
            'simulation.complexity_level': [0, 1]
        },
        'fixed': {
            'simulation.max_steps': 10
        },
        'simulation': {
            'initial_budget': 200,
            'pressure_level': 'low'
        },
        'pe_induction': {
            'pe_mag': 'low',
            'pe_type_mix': 'realistic',
            'observability': 'full'
        },
        'interface': {
            'prediction_mode': 'optional',
            'prediction_format': 'structured',
            'memory_tools': 'none',
            'recovery_tools': 'none'
        },
        'model': {
            'name': 'test',
            'context_length': 8000,
            'temperature': 0.0
        },
        'measurement': {
            'crash_threshold': 'moderate',
            'pe_windows': [10]
        },
        'paths': {
            'logs_dir': 'test_logs',
            'results_dir': 'test_results'
        }
    }
    
    # Expand grid
    expanded = ConfigLoader.expand_grid(config)
    
    # Should have 2 x 2 x 2 = 8 configs
    assert len(expanded) == 8
    
    # Check that each has correct structure
    for run_config in expanded:
        assert 'experiment' in run_config
        assert 'run_id' in run_config['experiment']
        assert run_config['simulation']['max_steps'] == 10
        assert run_config['pe_induction']['p_shock'] in [0.0, 0.1]
        assert run_config['simulation']['complexity_level'] in [0, 1]


def test_config_file_loading():
    """Test loading actual config files."""
    # Test base config
    base_path = Path('configs/base.yaml')
    if base_path.exists():
        config = ConfigLoader.load_config(str(base_path))
        
        assert 'experiment' in config
        assert 'simulation' in config
        assert 'pe_induction' in config
        assert 'model' in config


def test_pe_calculation_in_run(test_config, tmp_path):
    """Test that PE calculation works in full run."""
    test_config['paths']['logs_dir'] = str(tmp_path / 'logs')
    test_config['interface']['prediction_mode'] = 'required'
    test_config['simulation']['max_steps'] = 5  # Very short
    
    summary = run_single_experiment(test_config)
    
    # Should have PE stats
    assert 'cumulative_pes' in summary
    assert 'temporal' in summary['cumulative_pes']


def test_crash_detection_in_run(test_config, tmp_path):
    """Test that crash detection works in full run."""
    test_config['paths']['logs_dir'] = str(tmp_path / 'logs')
    test_config['measurement']['crash_threshold'] = 'strict'
    
    summary = run_single_experiment(test_config)
    
    # Should have crash info
    assert 'crashed' in summary
    assert 'crash_type' in summary
    assert 'time_to_crash' in summary
