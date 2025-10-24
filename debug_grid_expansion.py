#!/usr/bin/env python3
"""
Debug script to check grid expansion for Phase 1.
Run this to verify how many tasks are being generated.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import ConfigLoader
from cluster_utils import expand_grid_slurm


def main():
    print("=" * 70)
    print("VendoMini Phase 1 Grid Expansion Debug")
    print("=" * 70)
    
    # Load config
    config_path = 'configs/phases/phase1_core_hypothesis.yaml'
    print(f"\nLoading config: {config_path}")
    config = ConfigLoader.load_config(config_path)
    
    # Show grid parameters
    print("\n" + "=" * 70)
    print("Grid Parameters:")
    print("=" * 70)
    grid = config.get('grid', {})
    
    total_combos = 1
    for key, values in grid.items():
        if isinstance(values, list):
            print(f"  {key}: {len(values)} values")
            print(f"    Values: {values}")
            total_combos *= len(values)
        else:
            print(f"  {key}: 1 value")
            print(f"    Value: {values}")
    
    replications = config.get('experiment', {}).get('replications', 1)
    print(f"\nReplications: {replications}")
    
    expected_total = total_combos * replications
    print(f"\nExpected total tasks: {total_combos} combinations × {replications} reps = {expected_total}")
    
    # Expand grid
    print("\n" + "=" * 70)
    print("Expanding Grid...")
    print("=" * 70)
    
    params_list = expand_grid_slurm(config)
    actual_total = len(params_list)
    
    print(f"\nActual tasks generated: {actual_total}")
    print(f"Expected tasks: {expected_total}")
    print(f"Match: {'✅ YES' if actual_total == expected_total else '❌ NO'}")
    
    if actual_total != expected_total:
        print(f"\n⚠️  WARNING: Task count mismatch!")
        print(f"   Difference: {abs(actual_total - expected_total)} tasks")
        print(f"   Ratio: {actual_total / expected_total:.2%}")
    
    # Check distribution by model
    print("\n" + "=" * 70)
    print("Distribution by Model:")
    print("=" * 70)
    
    model_counts = {}
    for params in params_list:
        # Check different possible keys for model name
        model_name = None
        if 'model.name' in params:
            model_name = params['model.name']
        elif 'model' in params and isinstance(params['model'], dict):
            model_name = params['model'].get('name', 'unknown')
        else:
            model_name = 'unknown'
        
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    expected_per_model = expected_total // len(grid.get('model.name', ['unknown']))
    
    for model, count in sorted(model_counts.items()):
        match = "✅" if count == expected_per_model else "❌"
        print(f"  {match} {model}: {count} tasks (expected: {expected_per_model})")
    
    # Show sample tasks
    print("\n" + "=" * 70)
    print("Sample Tasks (first 5):")
    print("=" * 70)
    
    for i, params in enumerate(params_list[:5]):
        print(f"\nTask {i}:")
        for key, value in sorted(params.items()):
            print(f"  {key}: {value}")
    
    # Show task ID range
    print("\n" + "=" * 70)
    print("Task ID Range:")
    print("=" * 70)
    print(f"  Min task ID: 0")
    print(f"  Max task ID: {actual_total - 1}")
    print(f"  SLURM array should be: #SBATCH --array=0-{actual_total - 1}")
    
    current_array = "0-359"  # From run_phase1.sh
    expected_array = f"0-{actual_total - 1}"
    
    if current_array == expected_array:
        print(f"  ✅ SLURM array matches: --array={current_array}")
    else:
        print(f"  ❌ SLURM array mismatch!")
        print(f"     Current:  --array={current_array}")
        print(f"     Expected: --array={expected_array}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    if actual_total == expected_total:
        print("✅ Grid expansion is correct!")
        print(f"   {actual_total} tasks will be generated")
        print(f"   SLURM array: #SBATCH --array=0-{actual_total - 1}")
    else:
        print("❌ Grid expansion has issues!")
        print(f"   Expected {expected_total} tasks but got {actual_total}")
        print(f"   Investigate expand_grid_slurm() in src/cluster_utils.py")
    
    return 0 if actual_total == expected_total else 1


if __name__ == "__main__":
    sys.exit(main())
