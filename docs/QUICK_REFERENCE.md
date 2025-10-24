# VendoMini Quick Reference - Cluster Deployment

## 🚀 Quick Start (Copy-Paste Commands)

```bash
# 1. Navigate to project
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini

# 2. Activate environment
conda activate vendomini

# 3. Verify models are cached
python3 check_model_cache.py

# 4. Verify grid expansion
python3 debug_grid_expansion.py

# 5. Test with single task from Phase 1a
sbatch --array=0 slurm/run_phase1a.sh

# 6. Check the test (wait 2-3 minutes)
ls -lt logs/slurm-*.out | head -1  # Find latest log
tail -50 logs/slurm-*.out          # Check output

# 7. If test passes, run small batch
sbatch --array=0-9 slurm/run_phase1a.sh

# 8. If all good, run all phases with splits (11 arrays, all under 1000)
bash slurm/submit_all_phases_split.sh
```

## ✅ What Should Work Now

1. **No network errors** - Offline mode enabled
2. **Correct task counts** - Grid expansion fixed  
   - Phase 1: 1,320 tasks (split into 2×660)
   - Phase 2: 4,950 tasks (split into 5×990)
   - Phase 3: 1,980 tasks (split into 2×990)
   - Phase 4: 990 tasks (no split needed)
   - Phase 5: 1,320 tasks (split into 2×660)
   - **All arrays under 1000 tasks** ✅
3. **Proper model loading** - Cache paths corrected
4. **Nested parameters** - `model.name` properly applied

## 🔍 Quick Checks

### Is offline mode working?
```bash
# Should see in logs:
grep "OFFLINE MODE" logs/slurm-*.out
# ✅ "OFFLINE MODE: Models must be pre-cached locally"
```

### Are models loading?
```bash
# Should NOT see:
grep "huggingface.co" logs/slurm-*.out
# ❌ If you see network errors, models aren't cached
```

### Is grid expansion correct?
```bash
python3 debug_grid_expansion.py | grep "Match:"
# ✅ "Match: ✅ YES"
```

## 📊 Monitor Jobs

```bash
# All your jobs
squeue -u $USER

# Count running
squeue -u $USER -t RUNNING | wc -l

# Count pending
squeue -u $USER -t PENDING | wc -l

# Watch in real-time
watch -n 10 'squeue -u $USER'
```

## 🛑 Emergency Stop

```bash
# Cancel ALL your jobs
scancel -u $USER

# Cancel specific phase
scancel <JOB_ID>
```

## 📁 Key Files Changed

- ✅ `src/agent.py` - Offline mode
- ✅ `src/cluster_utils.py` - Grid expansion
- ✅ `src/experiment_runner.py` - Param application
- ✅ `slurm/run_phase*.sh` (all 5) - Cache paths

## 🎯 Expected Log Output (Good Run)

```
🚀 Starting VendoMini Phase 1 Array Job
Task ID: 0
[*] HF_HOME: /scratch/gpfs/JORDANAT/mg9965/VendoMini/models
[*] OFFLINE MODE: Models must be pre-cached locally
[*] Loading tokenizer...
[*] Tokenizer loaded successfully
[*] Device: cuda
[*] Loading model weights...
[*] Model loaded successfully!
[*] Starting simulation (max_steps=500)...
[*] Crash detected at step 6: looping
✅ Task 0 complete!
```

## ❌ Expected Log Output (Bad Run)

```
Failed to resolve 'huggingface.co'  # ❌ Offline mode not working
ValueError: Task ID 224 exceeds...  # ❌ Grid expansion broken
LLM client is None                   # ❌ Model not loaded
```

## 🆘 If Something Goes Wrong

1. **Network errors?** → Check models are cached with `check_model_cache.py`
2. **Task ID errors?** → Run `debug_grid_expansion.py`
3. **Model won't load?** → Check you're on GPU node with `nvidia-smi`
4. **Job won't start?** → Check queue with `squeue -u $USER`

## 📈 Results Analysis (After Jobs Complete)

```bash
# Count completed tasks
ls results/task_*.json | wc -l

# Aggregate results
python3 scripts/aggregate_results.py

# Analyze
python3 scripts/analyze_results.py
```

## 💾 All Fixes Implemented

1. **Solution 1A: Offline Mode**
   - `HF_HUB_OFFLINE=1` in SLURM scripts
   - `local_files_only=True` in agent.py

2. **Solution 1C: Cache Paths**
   - All scripts use `/scratch/gpfs/JORDANAT/mg9965/VendoMini/models`

3. **Grid Expansion Fix**
   - `_set_nested()` helper function
   - Proper dotted path handling
   - Config structure compatibility

4. **Parameter Space Expansion**
   - p_shock expanded from 0-1 by 0.1 (11 values)
   - All 6 models tested across all phases

5. **SLURM Array Splits**
   - All phases split into arrays under 1000 tasks
   - 11 total arrays instead of 5
   - Proper dependency management between splits

---

**Ready to deploy!** 🎉

Start with the Quick Start commands above.
