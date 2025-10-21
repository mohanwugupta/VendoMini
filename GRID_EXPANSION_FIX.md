# Grid Expansion Fix Summary

## Problem
Task ID 224 exceeded the parameter count (180 instead of expected 360 for Phase 1).

## Root Cause
The grid expansion function `expand_grid_slurm()` was creating params with dotted keys like:
```python
{'model.name': 'llama-3.1', 'pe_induction.p_shock': 0.2, ...}
```

But `experiment_runner.py` was applying them with `config.update(params)`, which treats them as flat keys instead of nested paths. This meant:
- `config['model.name'] = 'llama-3.1'`  # WRONG - creates flat key
- Instead of `config['model']['name'] = 'llama-3.1'`  # CORRECT - nested

Additionally, the LLMAgent was looking for `config['model']` at top level, but the base config has `config['agent']['model']`.

## Fixes Applied

### 1. Added `_set_nested()` helper function
**File:** `src/cluster_utils.py`

```python
def _set_nested(d: Dict, path: str, value: Any):
    """
    Set a value in a nested dict using dot notation.
    
    Args:
        d: Dictionary to modify
        path: Dot-separated path (e.g., 'model.name')
        value: Value to set
    """
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
```

### 2. Updated experiment runner to apply dotted params correctly
**File:** `src/experiment_runner.py`

Changed from:
```python
env_config = self.config.copy()
env_config.update(params)  # WRONG - treats 'model.name' as flat key
```

To:
```python
from src.cluster_utils import _set_nested
import copy

env_config = copy.deepcopy(self.config)

# Apply grid parameters using nested path notation
for key, value in params.items():
    if '.' in key and key not in ['combination_id', 'replication_id', 'seed']:
        # This is a dotted path like 'model.name' - apply as nested
        _set_nested(env_config, key, value)
    else:
        # Direct key - just set it
        env_config[key] = value
```

### 3. Fixed LLMAgent to handle both config structures
**File:** `src/agent.py`

Updated `__init__` to handle:
- Full config with `agent.model` and `agent.interface` sections
- Direct agent config with `model` and `interface` at top level

```python
# Handle both config structures
if 'agent' in config:
    # Full config structure - extract agent section
    agent_cfg = config['agent']
    model_cfg = agent_cfg.get('model', {})
    interface_cfg = agent_cfg.get('interface', {})
else:
    # Direct agent config structure
    model_cfg = config.get('model', {})
    interface_cfg = config.get('interface', {})
```

## Verification

Run this to verify the fix:
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
python3 debug_grid_expansion.py
```

Expected output:
```
Grid Parameters:
  pe_induction.p_shock: 3 values
  pe_induction.pe_mag: 2 values
  interface.prediction_mode: 2 values
  model.name: 6 values  # <-- Should show 6 models

Expected total tasks: 3 × 2 × 2 × 6 × 5 = 360

Actual tasks generated: 360
Expected tasks: 360
Match: ✅ YES

✅ Grid expansion is correct!
   360 tasks will be generated
   SLURM array: #SBATCH --array=0-359
```

## Grid Parameters Breakdown

### Phase 1: Core Hypothesis
- `pe_induction.p_shock`: [0.0, 0.10, 0.20] = 3 values
- `pe_induction.pe_mag`: [low, high] = 2 values  
- `interface.prediction_mode`: [required, optional] = 2 values
- `model.name`: 6 models
- Replications: 5

**Total:** 3 × 2 × 2 × 6 × 5 = **360 tasks** ✅

### Phase 2: PE Type Analysis
- `pe_induction.pe_type_mix`: 5 values
- `pe_induction.p_shock`: 2 values
- `pe_induction.observability`: 3 values
- `model.name`: 6 models
- Replications: 5

**Total:** 5 × 2 × 3 × 6 × 5 = **900 tasks** ✅

### Phase 3: Complexity Scaling
- `simulation.complexity_level`: 3 values
- `pe_induction.p_shock`: 4 values
- `interface.recovery_tools`: 2 values
- `model.name`: 6 models
- Replications: 5

**Total:** 3 × 4 × 2 × 6 × 5 = **720 tasks** ✅

### Phase 4: Model Architecture
- `pe_induction.p_shock`: 3 values
- `pe_induction.pe_mag`: 3 values
- `model.name`: 6 models
- Replications: 5

**Total:** 3 × 3 × 6 × 5 = **270 tasks** ✅

### Phase 5: Long Horizon
- `simulation.complexity_level`: 2 values
- `simulation.max_steps`: [2500, 5000] = 2 values (implicit)
- `pe_induction.p_shock`: 2 values
- `model.name`: 2 models (top/bottom from Phase 4)
- Replications: 10

**Total:** 2 × 2 × 2 × 10 = **80 tasks** (may vary based on config)

## All Changes Summary

1. ✅ **Offline mode enabled** in `src/agent.py`
2. ✅ **Cache paths fixed** in all SLURM scripts (phases 1-5)
3. ✅ **Grid expansion fixed** in `src/cluster_utils.py`
4. ✅ **Param application fixed** in `src/experiment_runner.py`
5. ✅ **Agent config handling fixed** in `src/agent.py`
6. ✅ **Debug script added** (`debug_grid_expansion.py`)

## Testing on Cluster

```bash
# 1. Verify grid expansion
python3 debug_grid_expansion.py

# 2. Test single task with offline mode
sbatch --array=0 slurm/run_phase1.sh

# 3. Check the output
tail logs/slurm-*_0.out

# Expected to see:
# [*] OFFLINE MODE: Models must be pre-cached locally
# [*] Loading HuggingFace model: <model_name>
# [*] Model loaded successfully!

# 4. If successful, run a small batch
sbatch --array=0-5 slurm/run_phase1.sh

# 5. Once verified, run full phase
sbatch slurm/run_phase1.sh
```
