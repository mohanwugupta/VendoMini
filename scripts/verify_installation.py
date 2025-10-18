"""
Quick verification script to test installation.

Usage:
    python scripts/verify_installation.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    
    required_modules = [
        ('yaml', 'pyyaml'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib'),
        ('pytest', 'pytest'),
    ]
    
    optional_modules = [
        ('ray', 'ray[default]'),
        ('openai', 'openai'),
        ('anthropic', 'anthropic'),
        ('lifelines', 'lifelines'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    
    missing_required = []
    missing_optional = []
    
    for module, package in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (install with: pip install {package})")
            missing_required.append(package)
    
    print("\nOptional modules:")
    for module, package in optional_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ○ {module} (optional, install with: pip install {package})")
            missing_optional.append(package)
    
    return missing_required, missing_optional


def check_src_modules():
    """Check that all src modules can be imported."""
    print("\nChecking src modules...")
    
    modules = [
        'src.config',
        'src.env',
        'src.pe_calculator',
        'src.crash_detector',
        'src.agent',
        'src.logging_utils',
        'src.experiment_runner',
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False
    
    return all_ok


def check_config_files():
    """Check that config files exist."""
    print("\nChecking configuration files...")
    
    config_files = [
        'configs/base.yaml',
        'configs/phases/phase1_core_hypothesis.yaml',
        'configs/phases/phase2_pe_type.yaml',
    ]
    
    all_exist = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            print(f"  ✓ {config_file}")
        else:
            print(f"  ✗ {config_file} (missing)")
            all_exist = False
    
    return all_exist


def run_mini_test():
    """Run a minimal test."""
    print("\nRunning mini test...")
    
    try:
        from src.config import ConfigLoader
        from src.env import VendoMiniEnv
        from src.pe_calculator import PECalculator
        from src.crash_detector import CrashDetector
        
        # Create minimal config
        config = {
            'experiment': {'seed': 42},
            'simulation': {
                'max_steps': 10,
                'complexity_level': 0,
                'initial_budget': 200,
                'pressure_level': 'low'
            },
            'pe_induction': {
                'p_shock': 0.0,
                'pe_mag': 'low',
                'pe_type_mix': 'realistic',
                'observability': 'full'
            },
            'measurement': {
                'crash_threshold': 'moderate',
                'pe_windows': [10]
            }
        }
        
        # Test environment
        env = VendoMiniEnv(config)
        obs = env.reset()
        action = {'tool': 'tool_check_budget', 'args': {}}
        next_obs, done = env.step(action)
        print("  ✓ Environment works")
        
        # Test PE calculator
        pe_calc = PECalculator()
        pes = pe_calc.compute_pe(None, {})
        pe_calc.update_accumulators(pes)
        print("  ✓ PE calculator works")
        
        # Test crash detector
        crash_detector = CrashDetector()
        history = [{'action': {}, 'observation': {}}]
        is_crashed, crash_type = crash_detector.update(history)
        print("  ✓ Crash detector works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    print("="*60)
    print("VendoMini Installation Verification")
    print("="*60)
    
    # Check imports
    missing_required, missing_optional = check_imports()
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing_required)}")
        sys.exit(1)
    
    # Check src modules
    src_ok = check_src_modules()
    
    if not src_ok:
        print("\n❌ Some src modules failed to import")
        sys.exit(1)
    
    # Check config files
    configs_ok = check_config_files()
    
    if not configs_ok:
        print("\n⚠ Some config files are missing")
    
    # Run mini test
    test_ok = run_mini_test()
    
    if not test_ok:
        print("\n❌ Mini test failed")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("✓ Installation verified successfully!")
    print("="*60)
    
    print("\nNext steps:")
    print("  1. Run tests: python scripts/run_tests.py")
    print("  2. Run experiment: python run_experiment.py --config configs/base.yaml --n-jobs 1")
    print("  3. See README.md for full documentation")
    
    if missing_optional:
        print("\nOptional packages not installed:")
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print("\nThese are only needed for specific features (Ray cluster, LLM calls, etc.)")


if __name__ == '__main__':
    main()
