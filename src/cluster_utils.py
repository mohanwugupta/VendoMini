"""Cluster utilities for VendoMini SLURM array jobs."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import numpy as np


def setup_cluster_paths(base_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Setup paths for cluster execution.
    
    Args:
        base_dir: Base directory (defaults to SLURM_SUBMIT_DIR or cwd)
        
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
    
    # Create directories if they don't exist
    for path in [paths['logs'], paths['checkpoints'], paths['results'], paths['data']]:
        Path(path).mkdir(parents=True, exist_ok=True)
    
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
