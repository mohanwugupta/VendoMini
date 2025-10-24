"""
Run all tests with coverage reporting.

Usage:
    python scripts/run_tests.py
"""

import subprocess
import sys


def main():
    """Run pytest with coverage."""
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        'tests/',
        '-v',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-report=xml'
    ]
    
    print("Running tests with coverage...")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ All tests passed!")
        print("Coverage report generated in htmlcov/index.html")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
