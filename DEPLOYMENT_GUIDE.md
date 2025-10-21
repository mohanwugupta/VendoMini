# VendoMini Cluster Fixes - Complete Implementation

**Date:** October 21, 2025  
**Status:** ✅ All issues fixed and ready for cluster deployment

---

## 🎯 Issues Fixed

### 1. ✅ HuggingFace Network Failures (Solutions 1A & 1C)
- **Problem:** Compute nodes couldn't reach huggingface.co
- **Fix:** Enabled offline mode and updated cache paths

### 2. ✅ Grid Expansion Generating Wrong Task Count
- **Problem:** 180 tasks instead of 360 for Phase 1
- **Fix:** Properly apply dotted parameter paths to nested config structure

### 3. ✅ Agent Configuration Compatibility
- **Problem:** LLMAgent expected different config structure
- **Fix:** Handle both full config and agent-only config structures

---

## 📝 Files Modified

### Core Code Changes

1. **`src/agent.py`** - Added offline mode and flexible config handling
   - Added `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
   - Added `local_files_only=True` to model loading
   - Handle both `config['agent']['model']` and `config['model']` structures

2. **`src/cluster_utils.py`** - Added nested path helper
   - New `_set_nested()` function for dotted paths like `model.name`

3. **`src/experiment_runner.py`** - Fixed parameter application
   - Use `_set_nested()` to properly apply grid params
   - Deep copy config to avoid mutations

### SLURM Scripts Updated

All 5 phase scripts updated with:
- Correct cache path: `/scratch/gpfs/JORDANAT/mg9965/VendoMini/models`
- Offline mode environment variables
- Clear comments

Files updated:
- `slurm/run_phase1.sh`
- `slurm/run_phase2.sh`
- `slurm/run_phase3.sh`
- `slurm/run_phase4.sh`
- `slurm/run_phase5.sh`

### New Utility Scripts

1. **`debug_grid_expansion.py`** - Verify grid expansion
2. **`check_model_cache.py`** - Verify models are cached
3. **`CLUSTER_ISSUES_AND_FIXES.md`** - Detailed issue documentation
4. **`GRID_EXPANSION_FIX.md`** - Grid expansion fix details

---

## 🔍 Pre-Flight Checklist

Before running experiments on the cluster, verify these steps:

### Step 1: Verify Models Are Cached

```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
python3 check_model_cache.py
```

**Expected output:**
```
✅ All models are properly cached and ready for offline use!
```

**If models are missing:** Run `scripts/download_models_simple.py` from a login node with internet.

### Step 2: Verify Grid Expansion

```bash
python3 debug_grid_expansion.py
```

**Expected output:**
```
Actual tasks generated: 360
Expected tasks: 360
Match: ✅ YES

✅ Grid expansion is correct!
```

### Step 3: Test Single Task

```bash
# Submit just one task to test
sbatch --array=0 slurm/run_phase1.sh

# Wait a few minutes, then check output
ls -lt logs/slurm-*.out | head -1  # Find newest log
tail -50 <newest_log_file>
```

**Expected in log:**
```
[*] OFFLINE MODE: Models must be pre-cached locally
[*] HF_HOME: /scratch/gpfs/JORDANAT/mg9965/VendoMini/models
[*] Loading tokenizer...
[*] Tokenizer loaded successfully
[*] Device: cuda
[*] Loading model weights...
[*] Model loaded successfully!
```

**Should NOT see:**
```
Failed to resolve 'huggingface.co'  # ❌ This means offline mode isn't working
```

---

## 🚀 Running Experiments

### Option A: Run All Phases Sequentially

```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
bash slurm/submit_all_phases.sh
```

This submits all 5 phases with dependencies (each phase waits for previous to complete).

### Option B: Run Individual Phases

```bash
# Phase 1: Core Hypothesis (360 tasks)
sbatch slurm/run_phase1.sh

# Phase 2: PE Type Analysis (900 tasks)
sbatch slurm/run_phase2.sh

# Phase 3: Complexity Scaling (720 tasks)
sbatch slurm/run_phase3.sh

# Phase 4: Model Architecture (270 tasks)
sbatch slurm/run_phase4.sh

# Phase 5: Long Horizon (240 tasks)
sbatch slurm/run_phase5.sh
```

### Option C: Test with Small Batch First

```bash
# Run just 10 tasks from Phase 1
sbatch --array=0-9 slurm/run_phase1.sh

# Monitor progress
watch -n 30 'squeue -u $USER'

# Check results
ls logs/slurm-*_*.out | wc -l  # Should show ~10 files
```

---

## 📊 Monitoring Jobs

### Check Job Status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j <JOB_ID>

# Detailed job info
scontrol show job <JOB_ID>
```

### Check Logs
```bash
# Latest logs
ls -lt logs/ | head -20

# Check specific task
cat logs/slurm-<JOB_ID>_<TASK_ID>.out

# Check for errors
grep -i error logs/slurm-*.out | head -20

# Check for model failures
grep "model_load_failed" logs/slurm-*.out
```

### Cancel Jobs
```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific job (all array tasks)
scancel <JOB_ID>

# Cancel specific array task
scancel <JOB_ID>_<TASK_ID>
```

---

## 🔧 Troubleshooting

### Problem: Still getting network errors

**Symptoms:**
```
Failed to resolve 'huggingface.co'
```

**Solution:**
1. Check environment variables are set in SLURM script:
   ```bash
   grep "HF_HUB_OFFLINE" slurm/run_phase1.sh
   ```
2. Verify models exist:
   ```bash
   ls /scratch/gpfs/JORDANAT/mg9965/VendoMini/models/
   ```

### Problem: Task ID exceeds parameter count

**Symptoms:**
```
ValueError: Task ID 224 exceeds number of parameter combinations (180)
```

**Solution:**
1. Run grid expansion debug:
   ```bash
   python3 debug_grid_expansion.py
   ```
2. If output shows wrong count, check your phase config:
   ```bash
   cat configs/phases/phase1_core_hypothesis.yaml
   ```

### Problem: Model loads but gets errors

**Symptoms:**
```
[*] Model loaded successfully!
[ERROR] ...some later error...
```

**Solution:**
1. Check if model is compatible with your GPU
2. Check memory allocation (may need more than 48GB for large models)
3. Try a smaller model first to verify setup works

---

## 📈 Expected Results

### Task Distribution per Phase

| Phase | Tasks | Models | Reps | Expected Time |
|-------|-------|--------|------|---------------|
| Phase 1 | 360 | 6 | 5 | ~720 GPU-hours |
| Phase 2 | 900 | 6 | 5 | ~1,800 GPU-hours |
| Phase 3 | 720 | 6 | 5 | ~2,880 GPU-hours |
| Phase 4 | 270 | 6 | 5 | ~810 GPU-hours |
| Phase 5 | 240 | 2 | 10 | ~1,920 GPU-hours |
| **Total** | **2,490** | - | - | **~8,130 GPU-hours** |

### Output Files

Each task generates:
- `logs/slurm-<JOB_ID>_<TASK_ID>.out` - Standard output
- `logs/slurm-<JOB_ID>_<TASK_ID>.err` - Error output (if any)
- `results/task_<TASK_ID>.json` - Experimental results

### Result Structure

Each result JSON contains:
```json
{
  "run_id": "1760801459_42",
  "params": {"model.name": "...", "pe_induction.p_shock": 0.2, ...},
  "seed": 162,
  "model_load_failed": false,
  "model_load_error": null,
  "total_steps": 56,
  "crashed": true,
  "crash_type": "looping",
  "crash_step": 6,
  "final_budget": 9850.00,
  "orders_fulfilled": 5,
  "cumulative_pe": {...}
}
```

---

## 🎉 Success Criteria

Your experiments are working correctly if you see:

1. ✅ No network errors in logs
2. ✅ Models loading successfully from cache
3. ✅ Correct number of tasks (360 for Phase 1)
4. ✅ Tasks completing with `total_steps > 0`
5. ✅ Result JSON files being created
6. ✅ Jobs not failing due to model loading

---

## 📞 Next Steps After This

1. **Immediate:** Run pre-flight checklist above
2. **Test:** Submit 10-task test batch
3. **Verify:** Check logs show offline mode working
4. **Deploy:** Run full Phase 1 (360 tasks)
5. **Monitor:** Watch for failures and check results
6. **Analyze:** Use `scripts/aggregate_results.py` when complete

---

## 🔑 Key Environment Variables

All SLURM scripts now set:
```bash
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

These ensure:
- Models are loaded from local cache
- No internet connection attempts
- All HuggingFace operations work offline

---

## 📚 Documentation References

- **PRD:** `PRD.md` - Experiment design and hypothesis
- **Budget Changes:** `BUDGET_AND_CRASH_CHANGES.md` - Recent updates
- **Model Failure:** `MODEL_FAILURE_DETECTION.md` - Error tracking
- **SLURM Config:** `SLURM_CONFIGURATION.md` - Job array details
- **Issue Details:** `CLUSTER_ISSUES_AND_FIXES.md` - This document's companion
- **Grid Fix:** `GRID_EXPANSION_FIX.md` - Technical details on grid fix

Good luck with your experiments! 🚀
