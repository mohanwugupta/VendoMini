#!/usr/bin/env python3
"""
Setup script for VendoMini on cluster.

Run this before submitting SLURM jobs to:
1. Verify models are downloaded
2. Set up directory structure
3. Check environment
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cluster_utils import setup_cluster_paths, get_local_model_path, check_cluster_environment


def check_model_availability(models_dir: str, model_names: list):
    """Check if models are available locally."""
    print("\n" + "=" * 70)
    print("Checking Model Availability")
    print("=" * 70)
    
    available = []
    missing = []
    
    for model_name in model_names:
        local_path = get_local_model_path(models_dir, model_name)
        if local_path:
            available.append(model_name)
        else:
            missing.append(model_name)
    
    print(f"\n✅ Available locally: {len(available)}/{len(model_names)}")
    for model in available:
        print(f"   ✓ {model}")
    
    if missing:
        print(f"\n⚠️  Missing (will download on first use): {len(missing)}")
        for model in missing:
            print(f"   ✗ {model}")
        print("\nTo pre-download missing models:")
        print("```python")
        print("from transformers import AutoModel, AutoTokenizer")
        for model in missing:
            print(f"AutoTokenizer.from_pretrained('{model}')")
            print(f"AutoModel.from_pretrained('{model}')")
        print("```")
    
    return available, missing


def verify_cluster_setup(base_dir: str = None, models_dir: str = None):
    """Verify complete cluster setup."""
    
    print("\n" + "=" * 70)
    print("VendoMini Cluster Setup Verification")
    print("=" * 70)
    
    # Set up paths
    print("\n1. Setting up paths...")
    paths = setup_cluster_paths(base_dir, models_dir)
    
    print("\n   Paths configured:")
    for key, path in paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"   {exists} {key}: {path}")
    
    # Check environment
    print("\n2. Checking cluster environment...")
    env_info = check_cluster_environment()
    
    print(f"   On cluster: {'✓' if env_info['is_slurm'] else '✗'}")
    print(f"   Job ID: {env_info.get('job_id', 'N/A')}")
    print(f"   Node: {env_info.get('node_name', 'N/A')}")
    
    # Check models
    print("\n3. Checking model availability...")
    
    # Common models for VendoMini
    test_models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "microsoft/Phi-3-mini-4k-instruct",
    ]
    
    available, missing = check_model_availability(paths['models'], test_models)
    
    # Summary
    print("\n" + "=" * 70)
    print("Setup Summary")
    print("=" * 70)
    
    all_paths_exist = all(os.path.exists(p) for p in paths.values())
    
    if all_paths_exist and len(available) > 0:
        print("✅ Cluster setup is ready!")
        print(f"   - All directories created")
        print(f"   - {len(available)} model(s) available locally")
        print(f"   - HuggingFace cache: {paths['models']}")
        print("\nYou can now submit SLURM jobs:")
        print("   sbatch slurm/run_phase1.sh")
    else:
        print("⚠️  Setup incomplete:")
        if not all_paths_exist:
            print("   - Some directories missing")
        if len(available) == 0:
            print("   - No models available locally (will download on first use)")
        print("\nModels will be downloaded automatically on first use.")
        print("To pre-download, run the commands shown above.")
    
    print("\n" + "=" * 70)


def main():
    """Run cluster setup verification."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify VendoMini cluster setup")
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base directory (default: current directory or SLURM_SUBMIT_DIR)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Models directory (default: base_dir/models)')
    
    args = parser.parse_args()
    
    verify_cluster_setup(args.base_dir, args.models_dir)


if __name__ == '__main__':
    main()
