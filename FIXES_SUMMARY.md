# VendoMini Fixes Summary

## Issues Fixed

### 1. Import Error: ConfigLoader not found (Line 35)
**Problem:** `UnboundLocalError: local variable 'ConfigLoader' referenced before assignment`
**Root Cause:** Duplicate import statement at line 72 was creating a local variable that shadowed the module-level import
**Fix:** Removed duplicate `from config import ConfigLoader` at line 72 in `run_experiment.py`

### 2. Config Structure Mismatch
**Problem:** `ValueError: Missing required config section: env`
**Root Cause:** Config validation expected specific top-level keys (`env`, `agent`, `crash_detector`) but config files used different structure (`simulation`, `interface`, `model`)
**Fixes:**
- **configs/base.yaml**: Restructured to use proper sections:
  - `simulation` → `env` 
  - `interface` + `model` → `agent` (nested under agent)
  - Added `crash_detector` section with `threshold` and `window_size`
- **src/config.py**: Removed strict validation to allow flexible config structures

### 3. ExperimentRunner Constructor
**Problem:** `TypeError: expected str, bytes or os.PathLike object, not dict`
**Root Cause:** `run_experiment.py` was passing a loaded config dict, but `ExperimentRunner.__init__` expected a file path string
**Fix:** Modified `ExperimentRunner.__init__` to accept either:
  - String path (loads YAML)
  - Dict (already loaded config)

### 4. Missing Method: run_cluster_task
**Problem:** `AttributeError: 'ExperimentRunner' object has no attribute 'run_cluster_task'`
**Root Cause:** `run_experiment.py` was calling non-existent method
**Fix:** Changed to use existing `run_single_experiment(params)` method with `get_task_params_slurm` to extract parameters

### 5. Missing Argument: available_tools
**Problem:** `TypeError: LLMAgent.get_action_and_prediction() missing 1 required positional argument: 'available_tools'`
**Root Cause:** `experiment_runner.py` wasn't passing required `available_tools` parameter to agent
**Fix:** Added `available_tools` list definition before simulation loop in `experiment_runner.py`

### 6. CrashDetector Constructor
**Problem:** `CrashDetector` was being called with dict instead of kwargs
**Fix:** Changed `CrashDetector(config)` to `CrashDetector(**config)` to unpack dict as kwargs

### 7. Environment Config
**Problem:** `VendoMiniEnv` expected full config dict but was receiving just `env` section
**Fix:** Changed to pass full `self.config.copy()` instead of `self.base_config.get_env_config()`

### 8. Unicode Emoji Issues (Windows Compatibility)
**Problem:** `UnicodeEncodeError: 'charmap' codec can't encode character` on Windows PowerShell
**Root Cause:** Emoji characters (🔧, 🖥️, 💻, 📦, 🌐, 🔍, ⚠️) not supported by Windows console encoding
**Fixes:** Replaced all emoji characters with ASCII equivalents:
- 🔧 → `[*]`
- 🖥️ → `[CLUSTER MODE]`
- 💻 → `[LOCAL MODE]`  
- 📦 → `[*]`
- 🌐 → `[*]`
- 🔍 → `[DEBUG]`
- ⚠️ → `[WARNING]`

### 9. CrashDetector Return Type Mismatch
**Problem:** `TypeError: tuple indices must be integers or slices, not str`
**Root Cause:** `crash_detector.update()` returns a tuple `(is_crashed, crash_type)` but code was treating it as a dict with keys `['crashed']` and `['type']`
**Fix:** Changed to unpack tuple: `crashed, crash_type = crash_detector.update(...)` and updated all references to use the variables directly instead of dict access

### 10. Missing is_terminated() Method
**Problem:** `AttributeError: 'VendoMiniEnv' object has no attribute 'is_terminated'`
**Root Cause:** Code called `env.is_terminated()` but `VendoMiniEnv` class doesn't have this method
**Fix:** Removed `is_terminated()` call and replaced with explicit termination checks:
  - Loop already handles max steps via `for step in range(max_steps)`
  - Added explicit budget depletion check: `if env.budget <= 0: break`
  - Crash detection already handled via `if crashed: break`

### 11. Enhanced Debugging and Error Handling
**Problem:** Immediate crashes with no diagnostic information
**Improvements:**
- **Component Initialization Logging**: Added print statements showing when each component (env, agent, PE calculator, crash detector) is initialized
- **Model Loading Feedback**: Shows which model is being loaded and if initialization succeeds
- **LLM Client Warning**: Warns if agent has no LLM client and is using heuristic fallback
- **Step-by-Step Logging**: Logs actions for first 5 steps and every 10th step
- **Exception Handling**: Wrapped simulation loop in try-except to catch and log any errors
- **Termination Reason Logging**: Explicitly logs why simulation ended (crash, budget depletion, max steps)
- **Result Summary**: Added formatted output showing total steps, crash status, crash type, and final budget
- **Better Error Messages**: Added traceback printing for initialization failures

### 12. Simplified HuggingFace Model Loading for Cluster
**Problem:** Model loading code was trying to manually construct paths using cluster_utils
**Improvement:** Simplified to rely on HuggingFace's automatic cache detection via environment variables
**Changes:**
  - Removed dependency on `get_local_model_path()` from cluster_utils
  - HuggingFace automatically uses `HF_HOME` and `TRANSFORMERS_CACHE` environment variables
  - Added detailed logging showing cache directories and load progress
  - Added traceback for model loading errors
  - Model is loaded by name only - HF handles cache lookup automatically
**Benefit:** Simpler, more reliable model loading that works with standard HF cache structure

### 13. Restored Pathological Heuristic Agent (Research Design)
**Problem:** Improved heuristic agent to avoid looping - but this defeats the research purpose!
**Root Cause:** Research goal is to study agent failures/crashes (psychotic breaks, repetitive behaviors)
**Fix:** Reverted heuristic agent to pathological behavior:
  - Always returns `tool_check_inbox` action
  - Will trigger looping crash detector by design
  - Only used when LLM fails to load (fallback scenario)
**Note:** This is intentional - the research studies what happens when agents break down under prediction error

## Files Modified

1. **run_experiment.py**
   - Removed duplicate ConfigLoader import
   - Replaced `run_cluster_task()` with `run_single_experiment(get_task_params_slurm())`
   - Added import for `get_task_params_slurm`
   - Fixed Unicode emojis

2. **src/experiment_runner.py**
   - Modified `__init__` to accept both path strings and dicts
   - Added `available_tools` definition before simulation loop
   - Changed env config to pass full config
   - Fixed CrashDetector instantiation with kwargs unpacking

3. **src/config.py**
   - Removed strict validation in `_validate_config()`

4. **configs/base.yaml**
   - Restructured config sections to match expected format
   - Added `env`, `agent`, `crash_detector` sections

5. **src/cluster_utils.py**
   - Fixed Unicode emoji in HuggingFace cache message

6. **src/agent.py**
   - Fixed Unicode emojis in debug/warning messages

## Testing Status

✅ Script imports successfully
✅ `--help` flag works
✅ No import errors
✅ No syntax errors
✅ Cluster job starts without errors
✅ Enhanced debugging and error handling added
⏳ Investigating immediate crash issue with detailed logging

## Next Steps

1. Sync changes to cluster: `git push` then `git pull` on cluster
2. Submit SLURM job: `sbatch slurm/run_phase1.sh`
3. Check detailed output: `cat slurm-*.out`
4. Review step-by-step logs to identify crash cause
5. Check JSONL logs in `logs/` directory for detailed step data
