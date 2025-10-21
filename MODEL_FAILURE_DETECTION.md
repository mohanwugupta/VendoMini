# Model Failure Detection Changes

## Summary

Removed the heuristic fallback agent and added proper error tracking for model loading failures. This ensures clean experimental data where you can identify which models failed to initialize.

## Changes Made (Oct 20, 2025)

### 1. Removed Heuristic Fallback Agent

**Before:**
- If LLM failed to load, system used a pathological "heuristic agent"
- Heuristic agent always called `tool_check_inbox` (intentionally bad)
- Created looping behavior that contaminated results
- No way to distinguish real LLM failures from working models

**After:**
- If LLM fails to load, experiment returns immediately with error info
- No simulation runs with broken models
- Clean separation between working and failed models

### 2. Added Error Tracking Fields

**New fields in result JSON:**
```json
{
  "model_load_failed": true,
  "model_load_error": "LLM client is None for model 'fake-model' (provider: huggingface)",
  "total_steps": 0,
  "crashed": false,
  "crash_type": null
}
```

**For successful runs:**
```json
{
  "model_load_failed": false,
  "model_load_error": null,
  "total_steps": 56,
  "crashed": true,
  "crash_type": "looping"
}
```

### 3. Files Modified

**`src/agent.py`:**
- Removed `_heuristic_agent()` method entirely
- Changed `get_action_and_prediction()` to raise `RuntimeError` if client is None
- Removed fallback logic for parsing errors

**`src/experiment_runner.py`:**
- Added `model_load_failed` and `model_load_error` tracking
- Check if `agent.client is None` after initialization
- Return early with error info if model fails
- Added separate exception handler for `RuntimeError` (model errors)
- Include error fields in all result summaries

## Benefits

### 1. Clean Data
```python
# Easy to filter valid results
valid_results = [r for r in results if not r['model_load_failed']]

# Analyze failure rates by model
import pandas as pd
df = pd.DataFrame(results)
failure_rate = df.groupby('model.name')['model_load_failed'].mean()
```

### 2. Clear Error Messages
Instead of mysterious looping crashes at step 6, you get:
```
[ERROR] LLM client is None for model 'phi-4' (provider: huggingface)
[ERROR] Experiment cannot run without a working LLM - aborting
```

### 3. No Contaminated Results
Before: Failed models ran with heuristic agent → all crashed at step 6 with looping
After: Failed models return immediately → total_steps=0, easy to identify

## Testing

Run the verification test:
```bash
python3 test_model_failure.py
```

Expected output:
```
✓ Test Passed!

Model failures are now properly recorded in results:
  - model_load_failed: boolean flag
  - model_load_error: error message
  - total_steps: 0 (no simulation run)
  - No fallback agent used
```

## Impact on Phase 4 Results

Looking at your Phase 4 data, models with 100% crash rate at step 6 were likely:
- `phi-4`: 22/22 crashed at step 6
- `qwen-2.5-7b`: 22/22 crashed at step 6
- `qwen-2.5-32b`: 24/24 crashed at step 6
- `deepseek-ai/deepseek-llm-7b-chat`: 15/15 crashed at step 6

These were probably model loading failures using the heuristic fallback.

**With new changes:**
```python
# Re-run Phase 4
results = aggregate_results()

# Identify failed models
failed = [r for r in results if r['model_load_failed']]
print(f"Models that failed to load: {len(failed)}")

for model in set(r['params']['model.name'] for r in failed):
    count = sum(1 for r in failed if r['params']['model.name'] == model)
    print(f"  {model}: {count} failures")

# Analyze only valid runs
valid = [r for r in results if not r['model_load_failed']]
print(f"\nValid runs: {len(valid)}")
# Now you have clean data!
```

## Backward Compatibility

**Breaking change:** Old results don't have `model_load_failed` field.

**Handling old data:**
```python
def is_valid_result(r):
    # Old results: assume valid if not explicitly failed
    return not r.get('model_load_failed', False)
```

## Files Summary

Files modified:
- `src/agent.py` - Removed heuristic fallback
- `src/experiment_runner.py` - Added error tracking
- `test_model_failure.py` - Verification test (NEW)
- `MODEL_FAILURE_DETECTION.md` - This documentation (NEW)

Related changes:
- Works with budget increase ($10,000)
- Works with crash continuation (50 steps)
- All three changes together enable clean, long-running experiments
