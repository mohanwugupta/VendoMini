# Debug Enhancements for VendoMini

## Overview
Added comprehensive debugging and error handling to diagnose why experiments are crashing immediately on the cluster.

## Changes Implemented

### 1. Component Initialization Logging (`src/experiment_runner.py`)

**Before:**
```python
env = VendoMiniEnv(env_config)
agent = LLMAgent(self.base_config.get_agent_config())
pe_calc = PECalculator()
crash_detector = CrashDetector(**self.base_config.get_crash_config())
```

**After:**
```python
print(f"[*] Initializing environment...")
env = VendoMiniEnv(env_config)

print(f"[*] Initializing agent...")
agent_config = self.base_config.get_agent_config()
model_name = agent_config.get('model', {}).get('name', 'unknown')
print(f"[*] Model: {model_name}")

try:
    agent = LLMAgent(agent_config)
    print(f"[*] Agent initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize agent: {e}")
    import traceback
    traceback.print_exc()
    raise

print(f"[*] Initializing PE calculator and crash detector...")
pe_calc = PECalculator()
crash_detector = CrashDetector(**self.base_config.get_crash_config())
```

**Benefit:** Shows exactly which component is initializing and catches agent/model loading failures.

---

### 2. Step-by-Step Action Logging (`src/experiment_runner.py`)

**Added:**
```python
print(f"[*] Starting simulation (max_steps={max_steps})...")

for step in range(max_steps):
    try:
        # ... existing code ...
        
        # Log action for debugging
        if step < 5 or step % 10 == 0:  # First 5 steps + every 10th
            print(f"  Step {step}: action={action.get('tool', 'unknown')}")
        
        # ... rest of loop ...
```

**Benefit:** Shows what actions the agent is taking without flooding the logs.

---

### 3. Exception Handling in Simulation Loop (`src/experiment_runner.py`)

**Added:**
```python
for step in range(max_steps):
    try:
        # ... all simulation logic ...
        
    except Exception as e:
        print(f"[ERROR] Exception at step {step}: {e}")
        import traceback
        traceback.print_exc()
        crashed = True
        crash_type = "exception"
        break
```

**Benefit:** Catches any runtime errors and logs full traceback instead of silent failure.

---

### 4. Termination Reason Logging (`src/experiment_runner.py`)

**Added:**
```python
if crashed:
    print(f"[*] Crash detected at step {step}: {crash_type}")
    break

if env.budget <= 0:
    print(f"[*] Budget depleted at step {step}")
    break
```

**After loop:**
```python
print(f"[*] Simulation complete: {len(step_data)} steps")
```

**Benefit:** Explicitly states why the simulation ended.

---

### 5. Enhanced Result Output (`run_experiment.py`)

**Before:**
```python
print(f"\n✅ Task {task_id} complete!")
print(f"Crashed: {result.get('crashed', 'unknown')}")
print(f"Time to crash: {result.get('time_to_crash', 'N/A')}")
```

**After:**
```python
print(f"\n{'='*60}")
print(f"Task {task_id} Results:")
print(f"{'='*60}")
print(f"Total steps: {result.get('total_steps', 0)}")
print(f"Crashed: {result.get('crashed', 'unknown')}")
print(f"Crash type: {result.get('crash_type', 'N/A')}")
print(f"Final budget: ${result.get('final_budget', 0):.2f}")
print(f"{'='*60}")
```

**Benefit:** Provides clear, formatted summary of experiment results.

---

### 6. Parameter Logging (`run_experiment.py`)

**Added:**
```python
params = get_task_params_slurm(config, task_id)
print(f"\nRunning experiment with params: {params}")
```

**Benefit:** Shows what parameter combination is being tested.

---

## Expected Output Format

With these changes, the cluster output will now look like:

```
============================================================
VendoMini Experiment Runner
============================================================
Environment: SLURM Cluster
Hostname: della-l03g12

Loading config from: configs/phases/phase1_core_hypothesis.yaml
[*] HuggingFace cache set to: /scratch/gpfs/.../models
Cluster paths setup: /scratch/gpfs/.../VendoMini/slurm

[CLUSTER MODE]
Job ID: 1471442
Array Job ID: 1471439
Task ID: 0
Node: della-l03g12

Running experiment with params: {...}
[*] Initializing environment...
[*] Initializing agent...
[*] Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
[*] Agent initialized successfully
[*] Initializing PE calculator and crash detector...
[*] Starting simulation (max_steps=1000)...
  Step 0: action=tool_check_inbox
  Step 1: action=tool_check_storage
  Step 2: action=tool_order
  Step 3: action=tool_check_inbox
  Step 4: action=tool_order
  Step 10: action=tool_check_budget
  [*] Crash detected at step 15: looping
[*] Simulation complete: 15 steps

============================================================
Task 0 Results:
============================================================
Total steps: 15
Crashed: True
Crash type: looping
Final budget: $187.50
============================================================
```

## Debugging Workflow

When a job crashes, you can now:

1. **Check initialization**: Look for `[*] Initializing...` messages
2. **Verify model loading**: Check if `Agent initialized successfully` appears
3. **Review actions**: See what tools were called in first few steps
4. **Find exceptions**: Look for `[ERROR]` messages with tracebacks
5. **Understand termination**: See explicit reason (crash type, budget, etc.)

## Files Modified

- `src/experiment_runner.py` - Added logging throughout initialization and simulation loop
- `src/crash_detector.py` - Clarified minimum steps requirement
- `run_experiment.py` - Enhanced result output formatting
- `FIXES_SUMMARY.md` - Documented changes
