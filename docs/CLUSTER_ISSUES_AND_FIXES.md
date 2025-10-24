# Cluster Issues and Fixes

**Date:** October 21, 2025  
**Cluster:** Princeton Della (SLURM)  
**Status:** Issues identified with solutions

---

## 🔴 Issue 1: HuggingFace Model Loading Failures

### Symptoms
```
Failed to resolve 'huggingface.co' ([Errno -2] Name or service not known)
Error loading HuggingFace model: We couldn't connect to 'https://huggingface.co'
[ERROR] LLM client is None for model 'meta-llama/Llama-2-7b-chat-hf'
[ERROR] Experiment cannot run without a working LLM - aborting
```

### Root Cause
- **Compute nodes have no internet access** (cannot reach huggingface.co)
- Code tries to download models from HuggingFace Hub
- Even though cache directories are set, transformers library still tries to check online first

### Solution 1: Enable Offline Mode (RECOMMENDED)

**File:** `src/agent.py`

Add offline mode before loading any models (line ~83):

```python
elif self.provider == 'huggingface':
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # FORCE OFFLINE MODE - Don't contact HuggingFace servers
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        print(f"[*] Loading HuggingFace model: {self.model_name}")
        print(f"[*] OFFLINE MODE: Models must be pre-cached")
        # ... rest of code
```

### Solution 2: Fix Cache Path Mismatch

**Issue:** SLURM scripts point to one directory, but models might be in another

**Current SLURM paths:**
```bash
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
```

**Recommended:** Use project-specific path
```bash
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
```

### Solution 3: Pre-download Models

**On login node with internet:**
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
conda activate vendomini

# Download all models for Phase 1
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['HF_HOME'] = '/scratch/gpfs/JORDANAT/mg9965/VendoMini/models'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gpfs/JORDANAT/mg9965/VendoMini/models'

models = [
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-V2.5',
    'meta-llama/Llama-3.3-70B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen3-32B',
    'deepseek-ai/deepseek-llm-7b-chat'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Don't download full model on login node (too big), just tokenizer
        print(f'✅ Downloaded tokenizer for {model_name}')
    except Exception as e:
        print(f'❌ Failed: {e}')
"
```

**Verify cache:**
```bash
ls -lh /scratch/gpfs/JORDANAT/mg9965/VendoMini/models/models--*/
```

---

## 🔴 Issue 2: Task ID Out of Range

### Symptoms
```
ValueError: Task ID 224 exceeds number of parameter combinations (180)
```

**Expected:** 360 tasks (0-359)  
**Actual:** 180 tasks (0-179)

### Root Cause

Phase 1 configuration has:
- 3 p_shock values
- 2 pe_mag values
- 2 prediction_mode values
- **6 models**
- 5 replications

Should generate: **3 × 2 × 2 × 6 × 5 = 360 tasks**

But `expand_grid_slurm()` only generates **180 tasks** (missing half).

**Why?** The grid expansion treats `model.name` as a dotted path but may not be expanding the list correctly.

### Debug Steps

**Check actual parameter count:**
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
python3 -c "
from src.config import ConfigLoader

config = ConfigLoader.load_config('configs/phases/phase1_core_hypothesis.yaml')
grid = config.get('grid', {})

print('Grid parameters:')
for k, v in grid.items():
    if isinstance(v, list):
        print(f'  {k}: {len(v)} values')
    else:
        print(f'  {k}: 1 value')

print(f\"\\nReplications: {config.get('experiment', {}).get('replications', 1)}\")

# Try expanding
from src.cluster_utils import expand_grid_slurm
params = expand_grid_slurm(config)
print(f\"\\nTotal tasks generated: {len(params)}\")
print(f\"Expected: {3 * 2 * 2 * 6 * 5} = 360\")

# Show first few tasks
print('\\nFirst 5 tasks:')
for i, p in enumerate(params[:5]):
    print(f'  Task {i}: {p}')
"
```

### Likely Fix: Check Grid Expansion Logic

The issue is in `src/cluster_utils.py` function `expand_grid_slurm()`.

Check if it's properly handling the nested path `model.name` and expanding all 6 models.

**Verify in config:**
```yaml
grid:
  model.name: [model1, model2, model3, model4, model5, model6]  # Should have 6 items
```

---

## 🔴 Issue 3: Model Name in Base Config vs Grid

### Potential Problem

**File:** `configs/base.yaml`
```yaml
agent:
  model:
    name: meta-llama/Llama-2-7b-chat-hf  # Base default
```

**File:** `configs/phases/phase1_core_hypothesis.yaml`
```yaml
grid:
  model.name: [openai/gpt-oss-20b, ...]  # 6 models in grid
```

If the grid expansion doesn't properly override the nested `agent.model.name`, you might get:
- Base config model used instead of grid models
- Only partial expansion

**Solution:** Ensure grid expansion properly sets nested paths

---

## ✅ Verification Steps

### Step 1: Check Grid Expansion
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
python3 -c "
from src.cluster_utils import expand_grid_slurm
from src.config import ConfigLoader

config = ConfigLoader.load_config('configs/phases/phase1_core_hypothesis.yaml')
params = expand_grid_slurm(config)

print(f'Total tasks: {len(params)}')
print(f'Expected: 360')
print(f'Match: {len(params) == 360}')

# Check model distribution
models = {}
for p in params:
    model = p.get('model.name', 'unknown')
    models[model] = models.get(model, 0) + 1

print(f'\\nTasks per model:')
for model, count in sorted(models.items()):
    print(f'  {model}: {count}')
print(f'\\nExpected per model: {360 / 6} = 60')
"
```

### Step 2: Test Offline Model Loading
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
python3 -c "
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gpfs/JORDANAT/mg9965/VendoMini/models'

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
print('✅ Offline loading works!')
"
```

### Step 3: Run Single Task Test
```bash
# Test with a known task ID that should work
sbatch --array=0 slurm/run_phase1.sh

# Check output
tail logs/slurm-*_0.out
```

---

## 📝 Summary of Required Changes

1. **Add offline mode to `src/agent.py`** (3 lines)
2. **Verify grid expansion generates 360 tasks** (debug script above)
3. **Fix SLURM cache paths** if needed (update all `slurm/run_phase*.sh`)
4. **Pre-download models to cluster** (run from login node)

---

## 🚀 Next Steps

1. **Immediate:** Add offline mode to prevent network errors
2. **Debug:** Run grid expansion test to find why only 180 tasks
3. **Verify:** Pre-download at least one model and test loading
4. **Test:** Run `sbatch --array=0-2 slurm/run_phase1.sh` with 3 tasks
5. **Full run:** Once verified, run full Phase 1 array

---

## 📞 Questions to Answer

1. **Are models already downloaded?**
   ```bash
   ls /scratch/gpfs/JORDANAT/mg9965/VendoMini/models/
   ls /scratch/gpfs/JORDANAT/mg9965/prompt_patching/models/
   ```

2. **Which path should be used?** (VendoMini or prompt_patching?)

3. **Can login nodes access HuggingFace?** (for pre-downloading)

4. **What's the actual task count?** (run grid expansion debug script)
