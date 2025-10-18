"""Cluster utilities for VendoMini SLURM array jobs."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import numpy as np


def setup_cluster_paths(base_dir: Optional[str] = None, models_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Setup paths for cluster execution.
    
    Args:
        base_dir: Base directory (defaults to SLURM_SUBMIT_DIR or cwd)
        models_dir: Directory for HuggingFace model cache (optional)
        
    Returns:
        Dictionary of paths
    """
    if base_dir is None:
        # Try SLURM environment variable first, then current directory
        base_dir = os.environ.get('SLURM_SUBMIT_DIR', os.getcwd())
    
    base_path = Path(base_dir)
    
    paths = {
        'base': str(base_path),
        'configs': str(base_path / 'configs'),
        'logs': str(base_path / 'logs'),
        'checkpoints': str(base_path / 'checkpoints'),
        'results': str(base_path / 'results'),
        'data': str(base_path / 'data')
    }
    
    # Set up models directory if provided
    if models_dir is None:
        # Try to use a models subdirectory in base_dir
        models_dir = str(base_path / 'models')
    
    paths['models'] = models_dir
    
    # Set HuggingFace cache environment variables to use local models directory
    # This prevents re-downloading models that are already on the cluster
    os.environ['HF_HOME'] = models_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = models_dir
    os.environ['TRANSFORMERS_CACHE'] = models_dir
    os.environ['HF_DATASETS_CACHE'] = models_dir
    
    print(f"[*] HuggingFace cache set to: {models_dir}")
    
    # Create directories if they don't exist
    for path in [paths['logs'], paths['checkpoints'], paths['results'], paths['data']]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Create models directory if it doesn't exist
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    return paths


def get_slurm_array_info() -> Dict[str, Any]:
    """
    Get SLURM array job information.
    
    Returns:
        Dictionary with job_id, array_id, task_id, node_name
    """
    return {
        'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID', 'local'),
        'task_id': int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)),
        'node_name': os.environ.get('SLURMD_NODENAME', 'localhost'),
        'num_tasks': int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: torch seed setting removed since we don't use torch


def get_task_config(task_id: int, all_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get configuration for a specific task ID.
    
    Args:
        task_id: Task ID from SLURM array
        all_configs: List of all run configurations
        
    Returns:
        Configuration for this task
    """
    if task_id >= len(all_configs):
        raise ValueError(f"Task ID {task_id} out of range (max: {len(all_configs)-1})")
    
    return all_configs[task_id]


def expand_grid_slurm(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand grid parameters for SLURM array jobs.
    
    Args:
        config: Configuration dictionary with grid parameters
        
    Returns:
        List of parameter dictionaries for each task
    """
    from itertools import product
    import copy
    
    grid = config.get('grid', {})
    replications = config.get('experiment', {}).get('replications', 1)
    base_seed = config.get('experiment', {}).get('seed', 42)
    
    if not grid:
        # No grid parameters, create configs based on replications
        params_list = []
        for rep in range(replications):
            params = {
                'replication_id': rep,
                'seed': base_seed + rep if base_seed else rep
            }
            params_list.append(params)
        return params_list
    
    # Generate all combinations of grid parameters
    grid_keys = list(grid.keys())
    grid_values = [grid[k] if isinstance(grid[k], list) else [grid[k]] 
                   for k in grid_keys]
    
    combinations = list(product(*grid_values))
    
    params_list = []
    for combo_idx, combo in enumerate(combinations):
        for rep in range(replications):
            params = dict(zip(grid_keys, combo))
            params['combination_id'] = combo_idx
            params['replication_id'] = rep
            params['seed'] = base_seed + combo_idx * replications + rep if base_seed else combo_idx * replications + rep
            params_list.append(params)
    
    return params_list


def get_task_params_slurm(config: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    """
    Get parameters for a specific SLURM task.
    
    Args:
        config: Configuration dictionary
        task_id: Task ID from SLURM_ARRAY_TASK_ID
        
    Returns:
        Parameter dictionary for this task
    """
    all_params = expand_grid_slurm(config)
    
    if task_id >= len(all_params):
        raise ValueError(f"Task ID {task_id} exceeds number of parameter combinations ({len(all_params)})")
    
    return all_params[task_id]


def save_task_results(
    task_id: int,
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = "task"
):
    """
    Save results for a specific task.
    
    Args:
        task_id: Task ID
        results: Results dictionary
        output_dir: Output directory
        prefix: File prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{prefix}_{task_id:04d}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved task {task_id} results to {output_file}")


def aggregate_task_results(
    input_dir: str,
    output_file: str,
    prefix: str = "task"
) -> List[Dict[str, Any]]:
    """
    Aggregate results from all task files.
    
    Args:
        input_dir: Directory containing task result files
        output_file: Output file for aggregated results
        prefix: File prefix to match
        
    Returns:
        List of all results
    """
    input_path = Path(input_dir)
    
    # Find all task result files
    task_files = sorted(input_path.glob(f"{prefix}_*.json"))
    
    if not task_files:
        print(f"⚠️ No task files found in {input_dir} with prefix '{prefix}'")
        return []
    
    print(f"📊 Found {len(task_files)} task result files")
    
    # Load all results
    all_results = []
    for task_file in task_files:
        try:
            with open(task_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"⚠️ Error loading {task_file}: {e}")
    
    # Save aggregated results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✅ Aggregated {len(all_results)} results to {output_path}")
    
    return all_results


def get_local_model_path(models_dir: str, model_name: str) -> Optional[str]:
    """
    Find the actual local model path in the models directory.
    Follows HuggingFace cache structure: models--org--name/snapshots/hash/
    
    Args:
        models_dir: Directory where models are cached
        model_name: Model name (e.g., "microsoft/phi-2")
        
    Returns:
        Path to local model if found, None otherwise
    """
    # Convert model name to HF cache format: models--org--name
    cache_name = f"models--{model_name.replace('/', '--')}"
    base_model_dir = os.path.join(models_dir, cache_name)
    
    if os.path.exists(base_model_dir):
        # Check for snapshots directory (HF cache structure)
        snapshots_dir = os.path.join(base_model_dir, "snapshots")
        if os.path.exists(snapshots_dir) and os.path.isdir(snapshots_dir):
            # Find the most recent snapshot
            snapshots = [d for d in os.listdir(snapshots_dir) 
                        if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                # Use the first snapshot (typically there's only one)
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"✅ Found local model: {snapshot_path}")
                return snapshot_path
        
        # If no snapshots, check if base directory has the files
        required_files = ['config.json']
        has_required = all(os.path.exists(os.path.join(base_model_dir, f)) for f in required_files)
        
        if has_required:
            print(f"✅ Found local model: {base_model_dir}")
            return base_model_dir
    
    print(f"⚠️  Local model not found for {model_name} in {models_dir}")
    print(f"   Will attempt to download from HuggingFace Hub")
    return None


def check_cluster_environment() -> Dict[str, Any]:
    """
    Check cluster environment and resources.
    
    Returns:
        Dictionary with environment info
    """
    env_info = {
        'is_slurm': 'SLURM_JOB_ID' in os.environ,
        'hostname': os.environ.get('HOSTNAME', 'unknown'),
        'user': os.environ.get('USER', 'unknown'),
        'home': os.environ.get('HOME', 'unknown'),
        'python_path': os.environ.get('PYTHONPATH', ''),
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none')
    }
    
    if env_info['is_slurm']:
        env_info.update({
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_ntasks': os.environ.get('SLURM_NTASKS'),
            'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
            'slurm_mem_per_cpu': os.environ.get('SLURM_MEM_PER_CPU'),
            'slurm_nodelist': os.environ.get('SLURM_NODELIST')
        })
    
    return env_info
