# Enhanced Logging and Aggregation - Summary

## What Was Done

Enhanced VendoMini's logging and aggregation system to capture and analyze **scratchpad/notepad usage** and additional behavioral metrics.

## Files Modified

### 1. `src/env.py`
**Change**: Added full scratchpad contents to state logging
```python
'scratchpad': self.scratchpad.copy()  # Include full scratchpad contents
```
**Why**: Previously only logged size, now logs actual key-value pairs agents write

### 2. `src/experiment_runner.py`
**Change**: Added scratchpad and order metrics to summary
```python
'final_scratchpad': env.scratchpad.copy(),
'scratchpad_final_size': len(env.scratchpad),
'fulfilled_orders': env.fulfilled_orders,
'total_orders_requested': env.total_orders_requested,
```
**Why**: Preserve agent's final memory state for post-hoc analysis

### 3. `scripts/aggregate_from_logs.py`
**Major enhancements**:

#### New Functions
- `load_step_logs(run_dir)`: Loads step-by-step JSONL logs
- `extract_detailed_metrics(run_dir, summary)`: Extracts detailed behavioral metrics

#### New Aggregated Metrics

**Scratchpad Usage:**
- `scratchpad_max_size`: Peak number of entries
- `scratchpad_writes`: Total write operations
- `scratchpad_reads`: Total read operations  
- `scratchpad_deletes`: Total delete operations
- `scratchpad_used`: Boolean flag

**Action Patterns:**
- `max_repeat_streak`: Longest sequence of same action
- `num_repeated_actions`: Count of repeated action sequences
- `action_diversity`: Number of unique tools used
- `most_common_action`: Most frequently used tool
- `most_common_action_count`: Frequency of most common tool

**Budget Trajectory:**
- `budget_initial`: Starting budget
- `budget_final`: Ending budget
- `budget_min`: Minimum reached
- `budget_max`: Maximum reached
- `went_bankrupt`: Boolean (budget <= 0)
- `budget_burn_rate`: Average $/day spent

**PE Finals:**
- `pe_causal_fast_final`: Final fast EWMA
- `pe_causal_med_final`: Final medium EWMA
- `pe_causal_slow_final`: Final slow EWMA

#### Enhanced Summary Statistics
Now prints detailed breakdowns for:
- Scratchpad usage (% used, read/write ratios)
- Action patterns (repeat streaks, diversity)
- Bankruptcy rates and burn rates

### 4. `docs/LOGGING_AND_AGGREGATION.md` (NEW)
Comprehensive documentation covering:
- Log file structure and formats
- Scratchpad system explanation
- How to use aggregation script
- Analysis examples with Python code
- Troubleshooting tips

## How to Use

### 1. Run Experiments (logs are automatically created)
```bash
python run_experiment.py --config configs/local_test.yaml --n-jobs 4
```

### 2. Aggregate Results
```bash
python scripts/aggregate_from_logs.py
```

### 3. Analyze Results
```python
import pandas as pd

# Load CSV
df = pd.read_csv('results/aggregated_results.csv')

# Compare scratchpad users vs non-users
print("Crash rate (with scratchpad):", 
      df[df['scratchpad_used'] == True]['crashed'].mean())
print("Crash rate (without scratchpad):", 
      df[df['scratchpad_used'] == False]['crashed'].mean())
```

## What's Now Captured

### Before
- ❌ Scratchpad size only (not contents)
- ❌ No action pattern analysis
- ❌ No budget trajectory metrics
- ❌ Basic summary stats only

### After
- ✅ **Full scratchpad contents** in step logs
- ✅ **Scratchpad read/write/delete counts**
- ✅ **Action repetition patterns**
- ✅ **Action diversity metrics**
- ✅ **Budget trajectory analysis**
- ✅ **Bankruptcy detection**
- ✅ **Budget burn rate**
- ✅ **Detailed summary statistics**

## Example Output

```
SUMMARY STATISTICS
==================
Total runs: 1
Crashed: 0 (0.0%)
Survived: 1 (100.0%)

Final budget:
  Mean: $-4.76
  Median: $-4.76
  Min: $-4.76
  Max: $-4.76

Action patterns:
  Max repeat streak (mean): 4.0
  Max repeat streak (max): 4
  Action diversity (mean): 5.0 unique tools
  Action diversity (max): 5 unique tools

Bankruptcy:
  Went bankrupt: 1/1 (100.0%)
  Budget burn rate (mean): $-10.78/day
  Budget burn rate (median): $-10.78/day
```

## Key Insights Enabled

1. **Memory Usage**: Can now analyze if/how agents use scratchpad to maintain state
2. **Behavioral Loops**: Detect repeated action patterns that indicate confusion
3. **Economic Viability**: Track budget burn rates and bankruptcy timing
4. **Action Strategy**: Measure tool diversity and identify over-reliance on specific tools
5. **Recovery Patterns**: See what agents write in scratchpad when encountering errors

## Cluster Compatibility

✅ All changes are cluster-compatible:
- Logs saved to `logs/` directory (not just `results/`)
- Aggregation script works on completed cluster runs
- No additional dependencies required
- JSON/CSV outputs ready for transfer and analysis

## Testing

Tested with existing `test_run_001` logs:
```bash
$ python scripts/aggregate_from_logs.py
Found 3 run directories
Processing test_run_001...
✅ Loaded 1 results with detailed metrics
📊 Saved JSON to results/aggregated_results.json
📊 Saved CSV to results/aggregated_results.csv
```

Verified new metrics present in output:
- `scratchpad_max_size`: 0
- `scratchpad_writes`: 0  
- `max_repeat_streak`: 4
- `action_diversity`: 5
- `budget_burn_rate`: -10.78
- `went_bankrupt`: true

## Next Steps

1. **Run experiments with scratchpad tools enabled** to see agents actually use memory
2. **Compare scratchpad users vs non-users** in crash rates
3. **Analyze scratchpad contents** to understand agent reasoning
4. **Correlate action patterns** with PE accumulation and crashes
5. **Study budget management** strategies across different models

## Files for Reference

- Implementation: `src/env.py`, `src/experiment_runner.py`, `scripts/aggregate_from_logs.py`
- Documentation: `docs/LOGGING_AND_AGGREGATION.md`
- Summary: This file
