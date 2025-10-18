# VendoMini - Quick Start Guide# VendoMini Quick Start Guide



## ✅ What's ImplementedThis guide will get you up and running with VendoMini in 10 minutes.



VendoMini is a **production-ready** simulation for studying prediction error-induced crashes in LLM agents, with **full SLURM cluster support** for massive parallelization.## Step 1: Install Dependencies (2 minutes)



### Core Features```powershell

- ✅ Warehouse simulation with SKUs, suppliers, orders# Activate your virtual environment if not already active

- ✅ Typed prediction error tracking (temporal, quantity, cost, causal)# .\venv\Scripts\activate

- ✅ Multi-scale EWMA accumulators (fast/med/slow)

- ✅ Six crash detection types (looping, invalid bursts, etc.)# Install required packages

- ✅ **SLURM array job parallelization** (NO Ray dependency!)pip install -r requirements.txt

- ✅ Configurable experiments with grid expansion```

- ✅ Comprehensive logging and aggregation

- ✅ Unit and integration tests## Step 2: Verify Installation (1 minute)



## 🚀 Getting Started```powershell

python scripts/verify_installation.py

### Windows/Local Testing```



```powershellYou should see:

# 1. Install dependencies```

pip install -r requirements.txt✓ Installation verified successfully!

```

# 2. Verify installation

python scripts\test_cluster.py## Step 3: Run Tests (2 minutes)



# 3. Run a small test (1 experiment)```powershell

python run_experiment.py --config configs\base.yaml --n-jobs 1python scripts/run_tests.py

```

# 4. Run with parallelization (4 jobs)

python run_experiment.py --config configs\phases\phase1_core_hypothesis.yaml --n-jobs 4All tests should pass (green).

```

## Step 4: Run Your First Experiment (5 minutes)

### Linux/Cluster (SLURM)

### Option A: Minimal Test Run

```bash```powershell

# 1. Setup environment on clusterpython run_experiment.py --config configs/base.yaml --n-jobs 1

module load anaconda3/2024.2```

conda create -n vendomini python=3.10

conda activate vendominiThis runs 1 replication and should complete in ~30 seconds.

pip install -r requirements.txt

### Option B: Small Grid Experiment

# 2. Update SLURM scripts```powershell

nano slurm/run_phase1.sh# Run with 4 parallel jobs

# Change:python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 4

#   - #SBATCH --mail-user=YOUR_EMAIL```

#   - cd /path/to/your/vendomini

This runs Phase 1 (180 runs) and takes ~10-20 minutes with 4 cores.

# 3. Test configuration

python scripts/test_cluster.py## Step 5: View Results



# 4. Submit Phase 1 (180 parallel tasks)After the experiment completes:

sbatch slurm/run_phase1.sh

```powershell

# 5. Monitor jobs# Check the results CSV

squeue -u $USERdir results\

tail -f slurm-JOBID_0.out

# Run analysis

# 6. After completion, aggregate resultspython scripts/analyze_results.py --results results/phase1_core_hypothesis_results.csv

python scripts/aggregate_results.py \```

    --input-dir results \

    --output results/phase1_all_results.csvPlots will be saved to `analysis/` directory.



# 7. Analyze## What's Happening?

python scripts/analyze_results.py --input results/phase1_all_results.csv

```1. **Configuration Loading**: System loads `phase1_core_hypothesis.yaml`

2. **Grid Expansion**: Expands parameter combinations (3×2×2×3 = 36 combos × 5 reps = 180 runs)

## 📊 Experiment Phases3. **Parallel Execution**: Distributes runs across CPU cores

4. **Simulation**: Each run simulates warehouse operations with PE injection

| Phase | Tasks | Time/Task | Parallel Time | Description |5. **Logging**: Step-by-step logs saved to `logs/<run_id>/`

|-------|-------|-----------|---------------|-------------|6. **Aggregation**: Results combined into CSV

| 1 | 180 | ~1h | **~1h** | Core hypothesis (p_shock dose-response) |

| 2 | 450 | ~2h | **~2h** | PE type × observability |## Next Steps

| 3 | ~200 | ~2h | **~2h** | Complexity scaling |

| 4 | 243 | ~2h | **~2h** | Model architecture sweep |### Run All Phases

| 5 | 80 | ~5h | **~5h** | Long horizon (5000 steps) |

```powershell

**Total: ~1,150 experiments in ~12 hours** (vs ~2,500 hours sequential!)# Phase 1: Core hypothesis (~180 runs)

python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 8

## 🔧 How SLURM Parallelization Works

# Phase 2: PE type analysis (~300 runs)

```pythonpython run_experiment.py --config configs/phases/phase2_pe_type.yaml --n-jobs 8

# 1. Grid expansion creates all parameter combinations

configs = ConfigLoader.expand_grid(phase1_config)# Phase 3: Complexity scaling (~400 runs)

# → 180 unique configurationspython run_experiment.py --config configs/phases/phase3_complexity.yaml --n-jobs 8



# 2. SLURM array creates 180 parallel tasks# Phase 4: Model sweep (~243 runs)

#SBATCH --array=0-179python run_experiment.py --config configs/phases/phase4_model_arch.yaml --n-jobs 8



# 3. Each task runs ONE experiment# Phase 5: Long horizon (~80 runs, takes longer)

task_id = SLURM_ARRAY_TASK_ID  # 0, 1, 2, ..., 179python run_experiment.py --config configs/phases/phase5_long_horizon.yaml --n-jobs 8

config = configs[task_id]```

result = run_experiment(config)

save_result(f"task_{task_id}.json")### Use Cluster Mode



# 4. Aggregate after completionIf you have Ray installed and a cluster available:

results = merge_all_task_files()

save_csv(results)```powershell

```# Start Ray (if not already running)

ray start --head

## 📁 Key Files

# Run with Ray

- `run_experiment.py` - Main entry point (local/cluster)python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 32 --cluster

- `src/experiment_runner.py` - Orchestration```

- `src/cluster_utils.py` - SLURM utilities

- `slurm/run_phase1.sh` - Phase 1 array job script### Customize Experiments

- `scripts/aggregate_results.py` - Merge results

1. **Copy a config:**

## 📈 Outputs```powershell

copy configs\phases\phase1_core_hypothesis.yaml configs\my_experiment.yaml

``````

results/

├── vendomini_task_0000.json  # Task 02. **Edit parameters:**

├── vendomini_task_0001.json  # Task 1```yaml

...grid:

└── phase1_all_results.csv    # Aggregated  pe_induction.p_shock: [0.0, 0.05, 0.10, 0.20, 0.35]  # Add more values

  model.name: [llama-3.1-70b, gpt-4, claude-3-opus]

Columns: run_id, p_shock, crashed, time_to_crash, PE metrics, etc.replications: 10  # More replications

``````



## 🧪 Testing3. **Run your experiment:**

```powershell

```bashpython run_experiment.py --config configs/my_experiment.yaml --n-jobs 8

pytest tests/ -v --cov=src```

python scripts/test_cluster.py

```## Troubleshooting



---### "Module not found" errors

```powershell

**You're ready to run massive experiments!** 🚀pip install -r requirements.txt

```

See `README.md` and `IMPLEMENTATION_SUMMARY.md` for full details.

### Tests fail
```powershell
# Make sure you're in the project root
cd VendoMini

# Check Python version (need 3.10+)
python --version

# Reinstall
pip install -e .
```

### Out of memory
```powershell
# Reduce parallel jobs
python run_experiment.py --config configs/base.yaml --n-jobs 2
```

### Ray issues
```powershell
# Don't use --cluster flag if Ray isn't set up
python run_experiment.py --config configs/base.yaml --n-jobs 8
```

## Understanding Output

### Log Files
```
logs/
  phase1_core_hypothesis_c0_r0/
    steps.jsonl          # One line per simulation step
    summary.json         # Run summary with metrics
```

### Results CSV
```
results/
  phase1_core_hypothesis_results.csv   # All runs aggregated
```

**Key columns:**
- `crashed`: Boolean, did the agent crash?
- `crash_type`: What type of crash?
- `time_to_crash`: Steps until crash (or max_steps if no crash)
- `config.p_shock`: PE injection probability
- `success_rate`: Orders fulfilled / orders requested

### Analysis Plots
```
analysis/
  crash_distribution.png        # Histogram of crash times
  crash_by_config.p_shock.png  # Crash rate vs shock probability
```

## Performance Tips

1. **Use more cores:** `--n-jobs 16` or `--n-jobs -1` (all cores)
2. **Test small first:** Set `replications: 1` in config for testing
3. **Use cluster:** `--cluster` flag if you have Ray
4. **Monitor:** Watch `logs/` directory to see progress
5. **Checkpoint:** Logs are written incrementally, safe to stop/resume

## Help

- Full documentation: See `README.md`
- API details: See docstrings in `src/` files
- Config examples: See `configs/phases/` directory
- Issues: Check test output and log files

Happy experimenting! 🚀
