#!/usr/bin/env python3
"""Test that model loading failures are properly recorded."""

import yaml
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiment_runner import ExperimentRunner

print("="*60)
print("Testing Model Failure Detection")
print("="*60)

# Test 1: Create a config with a non-existent model
print("\nTest 1: Non-existent Model")
print("-"*60)

test_config = {
    'experiment': {
        'name': 'test_model_failure',
        'replications': 1
    },
    'env': {
        'max_steps': 10,
        'complexity_level': 1,
        'initial_budget': 10000,
        'pressure_level': 'medium'
    },
    'agent': {
        'model': {
            'name': 'fake-model/does-not-exist',
            'temperature': 0.3,
            'max_tokens_per_call': 1000
        },
        'interface': {
            'prediction_mode': 'required',
            'prediction_format': 'structured',
            'memory_tools': 'full',
            'recovery_tools': 'none'
        }
    },
    'crash_detector': {
        'threshold': 'moderate',
        'window_size': 20,
        'continue_after_crash': 50
    },
    'pe_induction': {
        'p_shock': 0.0,
        'pe_mag': 'low',
        'pe_type_mix': 'realistic',
        'observability': 'full'
    },
    'max_steps': 10,
    'results_dir': 'results',
    'logs_dir': 'logs',
    'checkpoints_dir': 'checkpoints'
}

# Run experiment
runner = ExperimentRunner(test_config)
params = {}
result = runner.run_single_experiment(params, seed=999)

print(f"\n✓ Experiment completed without crashing Python")
print(f"\nResult:")
print(f"  model_load_failed: {result.get('model_load_failed', False)}")
print(f"  model_load_error: {result.get('model_load_error', 'N/A')[:100]}...")
print(f"  total_steps: {result.get('total_steps', 0)}")
print(f"  crashed: {result.get('crashed', False)}")
print(f"  crash_type: {result.get('crash_type', 'N/A')}")

# Verify
assert result['model_load_failed'] == True, "Should have model_load_failed=True"
assert result['total_steps'] == 0, "Should have 0 steps when model fails to load"
assert result['model_load_error'] is not None, "Should have error message"

print("\n" + "="*60)
print("✓ Test Passed!")
print("="*60)
print("\nModel failures are now properly recorded in results:")
print("  - model_load_failed: boolean flag")
print("  - model_load_error: error message")
print("  - total_steps: 0 (no simulation run)")
print("  - No fallback agent used")
print("\nThis allows you to filter out failed models in analysis:")
print("  valid_results = [r for r in results if not r['model_load_failed']]")
