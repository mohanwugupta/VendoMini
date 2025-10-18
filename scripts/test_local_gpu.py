"""
Quick local GPU test script.

Tests VendoMini with a small HuggingFace model on your local GPU.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from config import ConfigLoader
from experiment_runner import ExperimentRunner


def check_gpu():
    """Check GPU availability."""
    print("=" * 60)
    print("GPU Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Current allocation: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available - will use CPU (slower)")
    print()


def test_model_loading():
    """Test loading a small HuggingFace model."""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'local_test.yaml'
    print(f"Loading config: {config_path}")
    
    config_loader = ConfigLoader(str(config_path))
    base_config = config_loader.config
    
    model_name = base_config['model']['name']
    print(f"Model: {model_name}")
    print()
    
    # Test agent initialization
    print("Initializing agent...")
    try:
        from agent import LLMAgent
        agent = LLMAgent(base_config)
        
        print(f"✓ Agent initialized")
        print(f"  Provider: {agent.provider}")
        print(f"  Model: {agent.model_name}")
        
        if agent.client is not None:
            print(f"  Client loaded: Yes")
            if agent.provider == 'huggingface':
                device = agent.client['device']
                print(f"  Device: {device}")
                
                if device == 'cuda':
                    print(f"  VRAM usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        else:
            print(f"  Client loaded: No (dependencies missing)")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error loading agent: {e}")
        print()
        return False


def run_quick_test():
    """Run a quick experiment."""
    print("=" * 60)
    print("Running Quick Test")
    print("=" * 60)
    
    config_path = Path(__file__).parent.parent / 'configs' / 'local_test.yaml'
    
    print("Loading configuration...")
    config_loader = ConfigLoader(str(config_path))
    configs = config_loader.expand_grid()
    
    print(f"Generated {len(configs)} experiment configs")
    print()
    
    # Run just the first config
    print("Running first experiment...")
    runner = ExperimentRunner(config_path)
    
    try:
        result = runner.run_single_experiment(configs[0])
        
        print()
        print("✓ Experiment completed!")
        print(f"  Steps: {result['total_steps']}")
        print(f"  Final budget: ${result['final_budget']:.2f}")
        print(f"  Crashed: {result['crashed']}")
        if result['crashed']:
            print(f"  Crash type: {result['crash_type']}")
            print(f"  Crash step: {result['crash_step']}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VendoMini Local GPU Testing")
    print("=" * 60)
    print()
    
    # Check GPU
    check_gpu()
    
    # Test model loading
    model_ok = test_model_loading()
    
    if not model_ok:
        print("⚠ Model loading failed. Install dependencies:")
        print("  pip install torch transformers accelerate")
        print()
        return
    
    # Run quick test
    test_ok = run_quick_test()
    
    if test_ok:
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run full local test:")
        print("   python run_experiment.py --config configs/local_test.yaml --n-jobs 1")
        print()
        print("2. Try different models in configs/local_test.yaml:")
        print("   - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2GB VRAM)")
        print("   - microsoft/phi-2 (~5GB VRAM)")
        print("   - microsoft/Phi-3-mini-4k-instruct (~7GB VRAM)")
        print()
        print("3. View results:")
        print("   python scripts/aggregate_results.py")
        print()


if __name__ == '__main__':
    main()
