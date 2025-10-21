# Budget and Crash Detection Changes

## Summary

Based on analysis of Phase 4 results, two key changes were made to allow longer, more meaningful experiment runs:

### Changes Made (Oct 20, 2025)

#### 1. Increased Initial Budget: $200 → $10,000

**Rationale:**
- Previous runs showed 100% of "surviving" agents went bankrupt by step ~5
- Mean budget at crash was only $93.91 
- Agents need more capital to run realistic multi-step operations

**Change Location:** `configs/base.yaml`
```yaml
env:
  initial_budget: 10000  # was 200
```

#### 2. Continue Running After Crash Detection

**Rationale:**
- 71.8% of runs crashed at exactly step 6 due to looping detection
- Stopping immediately prevented observing:
  - Recovery attempts
  - Long-term behavior patterns
  - PE accumulation effects
  - Severity classification (soft vs hard crashes)

**New Behavior:**
- Crash is **detected** when patterns emerge (e.g., 4 repeated actions)
- Simulation **continues for 50 more steps** to observe:
  - Does the agent recover?
  - Does the crash persist?
  - How do PEs accumulate during crashed state?
  - What happens to budget/orders during crash?

**Change Locations:**

`configs/base.yaml`:
```yaml
crash_detector:
  threshold: moderate
  window_size: 20
  continue_after_crash: 50  # NEW: continue for N steps after crash
```

`src/crash_detector.py`:
- Added `continue_after_crash` parameter to `__init__`
- Added `should_terminate(current_step)` method to check if termination threshold reached
- Tracks `crash_step` to calculate steps since crash

`src/experiment_runner.py`:
- Changed from `if crashed: break` to `if crashed and crash_detector.should_terminate(step): break`
- Added logging to show when crash detected vs when simulation terminates

### Expected Impact

**Before:**
- Mean time to crash: 6.1 steps
- 71.8% crash due to looping at step 6
- 28.2% survive but go bankrupt at step ~5
- **Total observable behavior: ~6 steps**

**After:**
- Crash detected at step ~6 (same detection)
- Simulation continues for 50 more steps
- **Total observable behavior: ~56 steps minimum**
- Can measure:
  - Recovery patterns
  - Crash severity (soft vs hard)
  - PE accumulation trajectories
  - Long-term effects of different PE conditions

### Testing

To verify changes work:

```bash
# Run a quick test with local config
python run_experiment.py --config configs/base.yaml --n-jobs 1

# Should see output like:
# [*] Crash detected at step 6: looping (continuing for 50 more steps)
# [*] Terminating at step 56: looping (crash detected at step 6)
```

### Backward Compatibility

- Old behavior can be restored by setting `continue_after_crash: 0` in config
- If parameter is omitted, defaults to 0 (immediate termination)
- All existing configs will work without modification

### Next Steps

1. **Re-run Phase 4** with new settings to get meaningful data
2. **Analyze recovery patterns**: Do any models recover from crashes?
3. **Classify crash severity**: Soft (recovers) vs Hard (persists)
4. **Long-term PE effects**: How do PEs accumulate during crashed state?
5. **Budget dynamics**: What happens to spending during crashes?

## File Summary

Files modified:
- `configs/base.yaml` - Increased budget, added continue_after_crash parameter
- `src/crash_detector.py` - Added continuation logic
- `src/experiment_runner.py` - Updated termination condition
- `BUDGET_AND_CRASH_CHANGES.md` - This documentation (NEW)

Files to update for Phase 4 re-run:
- `configs/phases/phase4_model_arch.yaml` - Inherits from base.yaml, will automatically use new settings
