"""Test cluster utilities and SLURM integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cluster_utils import (
    setup_cluster_paths,
    get_slurm_array_info,
    check_cluster_environment,
    set_seed
)
from config import ConfigLoader


def main():
    print("=" * 60)
    print("VendoMini Cluster Utilities Test")
    print("=" * 60)
    
    # Test 1: Check environment
    print("\n1. Checking environment...")
    env_info = check_cluster_environment()
    print(f"   Is SLURM: {env_info['is_slurm']}")
    print(f"   Hostname: {env_info['hostname']}")
    print(f"   User: {env_info['user']}")
    
    # Test 2: Setup paths
    print("\n2. Setting up paths...")
    try:
        paths = setup_cluster_paths()
        print(f"   Base: {paths['base']}")
        print(f"   Logs: {paths['logs']}")
        print(f"   Results: {paths['results']}")
        print("   ✅ Paths created successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: SLURM array info
    print("\n3. Getting SLURM array info...")
    try:
        slurm_info = get_slurm_array_info()
        print(f"   Job ID: {slurm_info['job_id']}")
        print(f"   Task ID: {slurm_info['task_id']}")
        print(f"   Node: {slurm_info['node_name']}")
        if slurm_info['job_id'] == 'local':
            print("   ⚠️  Not running in SLURM (expected for local testing)")
        else:
            print("   ✅ SLURM environment detected")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 4: Config loading and grid expansion
    print("\n4. Testing config loading and grid expansion...")
    try:
        config_file = Path(__file__).parent.parent / 'configs' / 'base.yaml'
        if not config_file.exists():
            print(f"   ⚠️  Config file not found: {config_file}")
            print("   Skipping config test")
        else:
            config = ConfigLoader.load_config(str(config_file))
            print(f"   ✅ Loaded config: {config.get('experiment', {}).get('name')}")
            
            # Test grid expansion
            expanded = ConfigLoader.expand_grid(config)
            print(f"   ✅ Grid expansion: {len(expanded)} runs")
            
            if len(expanded) > 0:
                run_id = expanded[0].get('experiment', {}).get('run_id', 'N/A')
                print(f"   Sample run ID: {run_id}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 5: Seed setting
    print("\n5. Testing seed management...")
    try:
        set_seed(42)
        import random
        val1 = random.random()
        set_seed(42)
        val2 = random.random()
        if val1 == val2:
            print("   ✅ Seed setting works correctly")
        else:
            print("   ❌ Seed setting failed")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 6: Task configuration
    print("\n6. Testing task configuration mapping...")
    try:
        # Create a sample config with grid
        test_config = {
            'experiment': {
                'name': 'test',
                'replications': 2,
                'seed': 100
            },
            'grid': {
                'param_a': [1, 2],
                'param_b': ['x', 'y']
            }
        }
        
        expanded = ConfigLoader.expand_grid(test_config)
        print(f"   Grid: 2 × 2 × 2 reps = {len(expanded)} tasks")
        
        if len(expanded) == 8:  # 2 × 2 × 2
            print(f"   ✅ Correct number of tasks")
            print(f"   Task 0 run_id: {expanded[0]['experiment']['run_id']}")
            print(f"   Task 7 run_id: {expanded[7]['experiment']['run_id']}")
        else:
            print(f"   ❌ Expected 8 tasks, got {len(expanded)}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All cluster utility tests passed!")
    print("=" * 60)
    print("\nReady for SLURM execution:")
    print("  1. Update slurm/run_phase1.sh with your email and paths")
    print("  2. Submit: sbatch slurm/run_phase1.sh")
    print("  3. Monitor: squeue -u $USER")
    print("  4. Aggregate: python scripts/aggregate_results.py ...")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
