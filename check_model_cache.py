#!/usr/bin/env python3
"""
Check if models are properly cached for offline loading.
Run this on the cluster to verify model availability.
"""

import os
import sys
from pathlib import Path

def check_model_cache(models_dir: str, model_names: list):
    """Check if models are cached in the expected directory."""
    
    print("=" * 70)
    print("VendoMini Model Cache Verification")
    print("=" * 70)
    
    print(f"\nCache directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"❌ Cache directory does not exist!")
        return False
    
    print(f"✅ Cache directory exists\n")
    
    # List contents
    print("Contents of cache directory:")
    try:
        contents = os.listdir(models_dir)
        if not contents:
            print("  (empty)")
        else:
            for item in sorted(contents):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
    except Exception as e:
        print(f"  Error listing directory: {e}")
    
    print("\n" + "=" * 70)
    print("Checking Model Availability:")
    print("=" * 70)
    
    all_found = True
    
    for model_name in model_names:
        print(f"\n🔍 {model_name}")
        
        # HuggingFace cache structure: models--org--name/
        cache_name = f"models--{model_name.replace('/', '--')}"
        cache_path = os.path.join(models_dir, cache_name)
        
        if os.path.exists(cache_path):
            print(f"   ✅ Cache directory found: {cache_name}")
            
            # Check for snapshots
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) 
                           if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    print(f"   ✅ Found {len(snapshots)} snapshot(s)")
                    for snap in snapshots:
                        snap_path = os.path.join(snapshots_dir, snap)
                        # Check for essential files
                        has_config = os.path.exists(os.path.join(snap_path, "config.json"))
                        has_tokenizer = os.path.exists(os.path.join(snap_path, "tokenizer_config.json"))
                        has_model = any(f.endswith('.safetensors') or f.endswith('.bin') 
                                      for f in os.listdir(snap_path) if os.path.isfile(os.path.join(snap_path, f)))
                        
                        print(f"      Snapshot: {snap[:8]}...")
                        print(f"         config.json: {'✅' if has_config else '❌'}")
                        print(f"         tokenizer_config.json: {'✅' if has_tokenizer else '❌'}")
                        print(f"         model weights: {'✅' if has_model else '❌'}")
                        
                        if not (has_config and has_tokenizer and has_model):
                            all_found = False
                else:
                    print(f"   ❌ No snapshots found")
                    all_found = False
            else:
                print(f"   ⚠️  No snapshots directory - checking for direct files")
                has_config = os.path.exists(os.path.join(cache_path, "config.json"))
                has_tokenizer = os.path.exists(os.path.join(cache_path, "tokenizer_config.json"))
                if has_config and has_tokenizer:
                    print(f"   ✅ Found config and tokenizer files")
                else:
                    print(f"   ❌ Missing essential files")
                    all_found = False
        else:
            print(f"   ❌ Not found in cache")
            all_found = False
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    if all_found:
        print("✅ All models are properly cached and ready for offline use!")
        print("\nYou can safely run experiments with:")
        print("  export HF_HUB_OFFLINE=1")
        print("  export TRANSFORMERS_OFFLINE=1")
        return True
    else:
        print("❌ Some models are missing or incomplete!")
        print("\nTo download missing models, run from a login node with internet:")
        print("  python3 scripts/download_models_simple.py")
        return False


def main():
    # Models to check (from phase1_core_hypothesis.yaml)
    models_to_check = [
        'openai/gpt-oss-20b',
        'deepseek-ai/DeepSeek-V2.5',
        'meta-llama/Llama-3.3-70B-Instruct',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen3-32B',
        'deepseek-ai/deepseek-llm-7b-chat'
    ]
    
    # Check environment variable or use default
    models_dir = os.environ.get('HF_HOME', 
                 os.environ.get('TRANSFORMERS_CACHE',
                 '/scratch/gpfs/JORDANAT/mg9965/VendoMini/models'))
    
    success = check_model_cache(models_dir, models_to_check)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
