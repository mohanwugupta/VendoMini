# VendoMini Logging and Aggregation System

## Overview

VendoMini has a comprehensive logging system that captures detailed step-by-step behavior and aggregates results for analysis. This document explains how logs are saved and how to extract insights from them.

## Log Structure

### Directory Organization

```
logs/
├── run_<timestamp>_<seed>/
│   ├── steps.jsonl          # Step-by-step log (one JSON object per line)
│   └── summary.json         # Run summary with final metrics
├── run_<timestamp>_<seed>/
│   ├── steps.jsonl
│   └── summary.json
...
```

Each run gets its own directory with:
- **steps.jsonl**: Line-delimited JSON with one entry per simulation step
- **summary.json**: Final summary with aggregated metrics

### Step Log Format (`steps.jsonl`)

Each line in `steps.jsonl` contains a JSON object with:

```json
{
  "step": 0,
  "day": 1,
  "observation": {...},
  "action": {
    "tool": "tool_check_storage",
    "args": {}
  },
  "prediction": {
    "tool": "tool_check_storage",
    "args": {},
    "expected_success": true,
    "prediction_text": "Will check inventory levels"
  },
  "result": {...},
  "pe": {
    "temporal": 0.0,
    "quantity": 0.0,
    "cost": 0.0,
    "causal": 0.0
  },
  "cumulative_pe": {
    "temporal": {"fast": 0.0, "med": 0.0, "slow": 0.0},
    "quantity": {"fast": 0.0, "med": 0.0, "slow": 0.0},
    "cost": {"fast": 0.0, "med": 0.0, "slow": 0.0},
    "causal": {"fast": 0.0, "med": 0.0, "slow": 0.0}
  },
  "state": {
    "day": 1,
    "budget": 10000.0,
    "storage": {...},
    "orders": {...},
    "inbox": [...],
    "fulfilled_orders": 0,
    "total_orders_requested": 0,
    "scratchpad_size": 0,
    "scratchpad": {}  // Full scratchpad contents
  }
}
```

**Key fields:**
- `action`: What tool the agent decided to use
- `prediction`: Agent's prediction card (what they expected)
- `result`: Actual outcome from the environment
- `pe`: Prediction errors for this step
- `cumulative_pe`: Running EWMA of PEs (fast/medium/slow timescales)
- `state`: Full environment state including **scratchpad contents**

### Summary Format (`summary.json`)

Final summary saved at the end of each run:

```json
{
  "run_id": "test_run_001",
  "start_time": "2025-10-17T22:57:49.502273",
  "end_time": "2025-10-17T23:00:35.646920",
  "params": {...},
  "seed": 42,
  "model_load_failed": false,
  "model_load_error": null,
  "total_steps": 20,
  "crashed": false,
  "crash_type": null,
  "final_budget": -4.76,
  "final_storage": {...},
  "final_scratchpad": {...},  // Agent's final scratchpad state
  "scratchpad_final_size": 0,
  "fulfilled_orders": 3,
  "total_orders_requested": 3,
  "cumulative_pe": {
    "temporal": {"fast": 0.0, "med": 0.0, "slow": 0.0},
    "quantity": {"fast": 0.0, "med": 0.0, "slow": 0.0},
    "cost": {"fast": 0.0, "med": 0.0, "slow": 0.0},
    "causal": {"fast": 0.0055, "med": 0.0777, "slow": 0.0342}
  }
}
```

## Scratchpad/Notepad System

### What is the Scratchpad?

The scratchpad is a key-value memory store that agents can use to take notes. It's designed to test whether agents can:
1. Maintain internal state across steps
2. Track goals and plans
3. Recover from errors by reviewing past notes

### Scratchpad Tools

Agents have access to three scratchpad tools:
- `tool_write_scratchpad(key, value)`: Store a note
- `tool_read_scratchpad(key)`: Retrieve a note
- `tool_delete_scratchpad(key)`: Delete a note

### What Gets Logged

**In step logs (`steps.jsonl`):**
- `state.scratchpad`: Full contents of scratchpad at each step
- `state.scratchpad_size`: Number of entries
- `action.tool`: Whether agent used scratchpad tools

**In summary (`summary.json`):**
- `final_scratchpad`: Scratchpad contents at end of run
- `scratchpad_final_size`: Final number of entries

## Aggregation Script

### Usage

```bash
cd /path/to/VendoMini
python scripts/aggregate_from_logs.py
```

This will:
1. Scan `logs/` directory for all run directories
2. Load summaries and step logs
3. Extract detailed metrics including scratchpad usage
4. Save aggregated results to `results/`

### Output Files

**`results/aggregated_results.json`**: Full detailed data in JSON format

**`results/aggregated_results.csv`**: Flattened data for analysis in Excel/pandas

### New Aggregated Metrics

The enhanced aggregation script now extracts:

#### Scratchpad Metrics
- `scratchpad_used`: Boolean - did agent use scratchpad?
- `scratchpad_max_size`: Max entries stored
- `scratchpad_writes`: Total write operations
- `scratchpad_reads`: Total read operations
- `scratchpad_deletes`: Total delete operations

#### Action Pattern Metrics
- `max_repeat_streak`: Longest sequence of repeated actions
- `num_repeated_actions`: How many times agent repeated actions
- `action_diversity`: Number of unique tools used
- `most_common_action`: Most frequently used tool
- `most_common_action_count`: How many times it was used

#### Budget Trajectory Metrics
- `budget_initial`: Starting budget
- `budget_final`: Ending budget
- `budget_min`: Minimum budget reached
- `budget_max`: Maximum budget reached
- `budget_burn_rate`: Average $/day spent
- `went_bankrupt`: Boolean - budget <= 0

#### PE Trajectory Metrics
- `pe_causal_fast_final`: Final fast EWMA of causal PE
- `pe_causal_med_final`: Final medium EWMA of causal PE
- `pe_causal_slow_final`: Final slow EWMA of causal PE

### Summary Statistics

The script prints detailed statistics:

```
SUMMARY STATISTICS
==================
Total runs: 20
Crashed: 15 (75.0%)
Survived: 5 (25.0%)

Crash types:
  looping: 10 (66.7%)
  budget_denial: 5 (33.3%)

Time to crash (steps):
  Mean: 12.3
  Median: 10.0
  Min: 6
  Max: 25

Final budget:
  Mean: $-124.56
  Median: $15.23
  Min: $-500.00
  Max: $1234.56

Scratchpad usage:
  Used scratchpad: 8/20 (40.0%)
  Max size (mean): 3.2 entries
  Max size (max): 7 entries
  Total writes: 45
  Total reads: 32
  Read/Write ratio: 0.71

Action patterns:
  Max repeat streak (mean): 4.5
  Max repeat streak (max): 12
  Action diversity (mean): 5.2 unique tools
  Action diversity (max): 6 unique tools

Bankruptcy:
  Went bankrupt: 12/20 (60.0%)
  Budget burn rate (mean): $-45.23/day
  Budget burn rate (median): $-38.50/day
```

## Analysis Examples

### Loading and Analyzing in Python

```python
import json
import pandas as pd
from pathlib import Path

# Load aggregated results
with open('results/aggregated_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Filter to scratchpad users
scratchpad_users = df[df['scratchpad_used'] == True]

# Compare crash rates
print("Crash rate (with scratchpad):", scratchpad_users['crashed'].mean())
print("Crash rate (without scratchpad):", df[df['scratchpad_used'] == False]['crashed'].mean())

# Analyze action patterns
print("\nMost common actions:")
print(df['most_common_action'].value_counts())

# Budget analysis
print("\nBudget burn by crash type:")
print(df.groupby('crash_type')['budget_burn_rate'].mean())
```

### Examining Individual Scratchpad Contents

```python
# Load a specific run's step logs
import json

steps = []
with open('logs/run_1234567_42/steps.jsonl', 'r') as f:
    for line in f:
        steps.append(json.loads(line))

# See what agent wrote to scratchpad
for step in steps:
    scratchpad = step['state'].get('scratchpad', {})
    if scratchpad:
        print(f"Step {step['step']}: {scratchpad}")

# Check final scratchpad
with open('logs/run_1234567_42/summary.json', 'r') as f:
    summary = json.load(f)
    print("Final scratchpad:", summary.get('final_scratchpad', {}))
```

## Cluster-Specific Notes

On the cluster:
- Logs are always saved to `logs/` directory (not `results/`)
- Each SLURM task creates its own `run_<timestamp>_<taskid>/` directory
- Use `aggregate_from_logs.py` to collect results after jobs complete
- Summary files are also saved to `results/` as `vendomini_task_<taskid>.json` for quick access

## Key Changes from Original

**Enhanced Scratchpad Logging:**
- Now saves full `scratchpad` contents in step logs (not just size)
- Saves `final_scratchpad` in summary
- Aggregation extracts write/read/delete counts

**New Aggregated Metrics:**
- Action pattern analysis (repeats, diversity)
- Budget trajectory analysis (burn rate, bankruptcy)
- Scratchpad usage statistics

**Better Summary Statistics:**
- Scratchpad usage breakdown
- Action pattern summaries
- Budget analysis by crash type

## Troubleshooting

**Missing scratchpad data?**
- Make sure you're running the latest version of `env.py` and `experiment_runner.py`
- Old logs won't have full scratchpad contents, only size

**Large log files?**
- Each step log is one JSON line - they can get big for long runs
- Consider compressing old logs: `gzip logs/*/steps.jsonl`

**Aggregation taking forever?**
- The script loads all step logs to extract detailed metrics
- For quick summaries, you can skip step analysis by modifying the script
