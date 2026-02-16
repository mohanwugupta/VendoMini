"""VendoMini experiment runner - supports local and cluster execution."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import ConfigLoader
from experiment_runner import ExperimentRunner
from cluster_utils import get_slurm_array_info, setup_cluster_paths, check_cluster_environment, get_task_params_slurm


def main():
    parser = argparse.ArgumentParser(description="Run VendoMini experiments")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs (local mode)')
    parser.add_argument('--cluster', action='store_true', help='Run in cluster mode (SLURM array job)')
    parser.add_argument('--task-id', type=int, default=None, help='Task ID (overrides SLURM_ARRAY_TASK_ID)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VendoMini Experiment Runner")
    print("=" * 60)
    
    # Check environment
    env_info = check_cluster_environment()
    print(f"Environment: {'SLURM Cluster' if env_info['is_slurm'] else 'Local'}")
    print(f"Hostname: {env_info['hostname']}")
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = ConfigLoader.load_config(args.config)
    
    # Setup paths
    if args.cluster or env_info['is_slurm']:
        # Get project root directory (parent of script location)
        project_root = str(Path(__file__).parent.absolute())
        paths = setup_cluster_paths(base_dir=project_root)
        # Update config with cluster paths
        config['paths'] = paths
        print(f"Cluster paths setup: {paths['base']}")
    
    # Initialize runner
    runner = ExperimentRunner(config)
    
    # Determine execution mode
    if args.cluster or env_info['is_slurm']:
        # Cluster mode: Run single task from array job
        slurm_info = get_slurm_array_info()
        task_id = args.task_id if args.task_id is not None else slurm_info['task_id']
        
        print(f"\n[CLUSTER MODE]")
        print(f"Job ID: {slurm_info['job_id']}")
        print(f"Array Job ID: {slurm_info['array_job_id']}")
        print(f"Task ID: {task_id}")
        print(f"Node: {slurm_info['node_name']}")
        
        # Run single task
        params = get_task_params_slurm(config, task_id)
        print(f"\nRunning experiment with params: {params}")
        
        result = runner.run_single_experiment(params)
        
        print(f"\n{'='*60}")
        print(f"Task {task_id} Results:")
        print(f"{'='*60}")
        print(f"Total steps: {result.get('total_steps', 0)}")
        print(f"Crashed: {result.get('crashed', 'unknown')}")
        print(f"Crash type: {result.get('crash_type', 'N/A')}")
        print(f"Final budget: ${result.get('final_budget', 0):.2f}")
        print(f"{'='*60}")
        
    else:
        # Local mode: Run with joblib parallelization
        print(f"\n[LOCAL MODE]")
        print(f"Parallel jobs: {args.n_jobs}")
        
        # Expand grid to show what will be run
        all_configs = ConfigLoader.expand_grid(config)
        print(f"Total experiments: {len(all_configs)}")
        
        # Run experiments
        results = runner.run_parallel(n_jobs=args.n_jobs)
        
        print(f"\n✅ All experiments complete!")
        print(f"Total runs: {len(results)}")
        crashed_count = sum(1 for r in results if r.get('crashed', False))
        print(f"Crashed: {crashed_count}/{len(results)} ({100*crashed_count/len(results):.1f}%)")


if __name__ == "__main__":
    main()
