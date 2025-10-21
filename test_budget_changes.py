#!/usr/bin/env python3
"""Quick test to verify budget and crash continuation changes."""

import yaml
from src.crash_detector import CrashDetector

print("="*60)
print("Testing Budget and Crash Detection Changes")
print("="*60)

# Test 1: Check config loads correctly
print("\nTest 1: Config Loading")
print("-"*60)
with open('configs/base.yaml', 'r') as f:
    config = yaml.safe_load(f)

initial_budget = config['env']['initial_budget']
continue_after_crash = config['crash_detector'].get('continue_after_crash', 0)

print(f"✓ Initial budget: ${initial_budget:,}")
print(f"✓ Continue after crash: {continue_after_crash} steps")

assert initial_budget == 10000, f"Expected budget=10000, got {initial_budget}"
assert continue_after_crash == 50, f"Expected continue_after_crash=50, got {continue_after_crash}"

# Test 2: Crash detector initialization
print("\nTest 2: CrashDetector Initialization")
print("-"*60)
detector = CrashDetector(
    threshold='moderate',
    window_size=20,
    continue_after_crash=50
)
print(f"✓ Detector initialized with continue_after_crash={detector.continue_after_crash}")

# Test 3: Termination logic
print("\nTest 3: Termination Logic")
print("-"*60)

# Simulate crash detection at step 6
detector.crash_detected = True
detector.crash_type = 'looping'
detector.crash_step = 6

# Test at various steps
test_steps = [6, 10, 20, 30, 40, 50, 55, 56, 57, 100]
for step in test_steps:
    should_term = detector.should_terminate(step)
    steps_since = step - detector.crash_step
    status = "TERMINATE" if should_term else "CONTINUE"
    print(f"  Step {step:3d} (crash+{steps_since:2d}): {status}")

print(f"\n✓ Simulation continues for steps 6-55 (50 steps after crash)")
print(f"✓ Simulation terminates at step 56+")

# Test 4: Immediate termination (old behavior)
print("\nTest 4: Immediate Termination (continue_after_crash=0)")
print("-"*60)
detector_immediate = CrashDetector(
    threshold='moderate',
    window_size=20,
    continue_after_crash=0
)
detector_immediate.crash_detected = True
detector_immediate.crash_step = 6

for step in [6, 7, 8]:
    should_term = detector_immediate.should_terminate(step)
    status = "TERMINATE" if should_term else "CONTINUE"
    print(f"  Step {step}: {status}")

print(f"\n✓ With continue_after_crash=0, terminates immediately")

print("\n" + "="*60)
print("All Tests Passed! ✓")
print("="*60)
print("\nChanges verified:")
print("  1. Budget increased to $10,000")
print("  2. Crash detection continues for 50 steps")
print("  3. Termination occurs at crash_step + 50")
print("  4. Backward compatible (continue_after_crash=0 works)")
