# SLURM | Phase | Tasks | Grid Configu## Job Array Splits (Under 1000 Tasks Each)

To comply with SLURM array limits, phases are split into smaller arrays:

### 1. Phase 1: Core Hypothesis (1,320 tasks → 2 splits)
- **run_phase1a.sh**: `#SBATCH --array=0-659` (660 tasks)
- **run_phase1b.sh**: `#SBATCH --array=660-1319` (660 tasks)

### 2. Phase 2: PE Type Analysis (4,950 tasks → 5 splits)
- **run_phase2a.sh**: `#SBATCH --array=0-989` (990 tasks)
- **run_phase2b.sh**: `#SBATCH --array=990-1979` (990 tasks)
- **run_phase2c.sh**: `#SBATCH --array=1980-2969` (990 tasks)
- **run_phase2d.sh**: `#SBATCH --array=2970-3959` (990 tasks)
- **run_phase2e.sh**: `#SBATCH --array=3960-4949` (990 tasks)

### 3. Phase 3: Complexity Scaling (1,980 tasks → 2 splits)
- **run_phase3a.sh**: `#SBATCH --array=0-989` (990 tasks)
- **run_phase3b.sh**: `#SBATCH --array=990-1979` (990 tasks)

### 4. Phase 4: Model Architecture Sweep (990 tasks - no split)
- **run_phase4.sh**: `#SBATCH --array=0-989` (990 tasks)

### 5. Phase 5: Long Horizon Extremes (1,320 tasks → 2 splits)
- **run_phase5a.sh**: `#SBATCH --array=0-659` (660 tasks)
- **run_phase5b.sh**: `#SBATCH --array=660-1319` (660 tasks)

**Total: 11 array jobs, all under 1000 tasks**-------|-------|-------------------|---------|
| **Phase 1** | 1,320 | 11 p_shock × 2 pe_mag × 2 pred_mode × 5 reps | 6 models |
| **Phase 2** | 4,950 | 11 p_shock × 5 pe_type_mix × 3 observability × 5 reps | 6 models |
| **Phase 3** | 1,980 | 11 p_shock × 3 complexity × 2 recovery_tools × 5 reps | 6 models |
| **Phase 4** | 990 | 11 p_shock × 3 pe_mag × 5 reps | 6 models |
| **Phase 5** | 1,320 | 11 p_shock × 2 complexity × 10 reps | 6 models |
| **TOTAL** | **10,560** | | |ation for All Phases

## Summary

All SLURM scripts have been updated to run experiments with **all 6 models** across all phases. Each phase stays **under 1000 jobs** as required.

## Job Array Sizes

| Phase | Tasks | Grid Configuration | Models |
|-------|-------|-------------------|---------|
| **Phase 1** | 360 | 3 p_shock × 2 pe_mag × 2 pred_mode × 5 reps | 6 models |
| **Phase 2** | 900 | 5 pe_type_mix × 2 p_shock × 3 observability × 5 reps | 6 models |
| **Phase 3** | 720 | 3 complexity × 4 p_shock × 2 recovery_tools × 5 reps | 6 models |
| **Phase 4** | 270 | 3 p_shock × 3 pe_mag × 5 reps | 6 models |
| **Phase 5** | 240 | 2 complexity × 2 p_shock × 10 reps | 6 models |
| **TOTAL** | **2,490** | | |

✅ All phases are under the 1000 job limit per phase

## Models Being Tested

Based on `phase4_model_arch.yaml`, the 6 models are:
1. `Qwen/Qwen3-30B-A3B-Instruct-2507`
2. `deepseek-ai/DeepSeek-V2.5`
3. `meta-llama/Llama-3.1-8B-Instruct`
4. `allenai/OLMo-2-1124-13B-Instruct`
5. `Qwen/Qwen3-32B`
6. `deepseek-ai/deepseek-llm-7b-chat`

## Updated Files

### 1. `slurm/run_phase1.sh`
- Updated array size: `#SBATCH --array=0-359` (was 0-179)
- Now tests all 6 models with 360 total tasks

### 2. `slurm/run_phase2.sh`
- Updated array size: `#SBATCH --array=0-899` (was 0-299)
- Now tests all 6 models with 900 total tasks

### 3. `slurm/run_phase3.sh`
- Updated array size: `#SBATCH --array=0-719` (was 0-359)
- Now tests all 6 models with 720 total tasks

### 4. `slurm/run_phase4.sh`
- Array size remains: `#SBATCH --array=0-269` (already correct)
- Tests all 6 models with 270 total tasks

### 5. `slurm/run_phase5.sh`
- Array size updated: `#SBATCH --array=0-239` (was 0-159)
- Now tests all 6 models with 240 total tasks

### 6. `slurm/submit_all_phases.sh`
- Updated job counts in summary output
- Correct total: 2,490 tasks

## Key Settings Applied

All experiments will use:
- **Budget**: $10,000 (configured in `configs/base.yaml`)
- **Crash Continuation**: 50 steps after crash detection
- **No Heuristic Fallback**: Model failures are properly tracked
- **Model Failure Detection**: Experiments record `model_load_failed` and `model_load_error` fields

## How to Run

### Submit All Phases with Splits (Recommended)
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
bash slurm/submit_all_phases_split.sh
```

This submits **11 array jobs** (all under 1000 tasks) with dependencies:
- Phase 1: 2 arrays → Phase 2: 5 arrays → Phase 3: 2 arrays → Phase 4: 1 array → Phase 5: 2 arrays

### Submit Individual Phase Splits
```bash
# Phase 1 (2 splits)
sbatch slurm/run_phase1a.sh  # 660 jobs (tasks 0-659)
sbatch slurm/run_phase1b.sh  # 660 jobs (tasks 660-1319)

# Phase 2 (5 splits)
sbatch slurm/run_phase2a.sh  # 990 jobs (tasks 0-989)
sbatch slurm/run_phase2b.sh  # 990 jobs (tasks 990-1979)
sbatch slurm/run_phase2c.sh  # 990 jobs (tasks 1980-2969)
sbatch slurm/run_phase2d.sh  # 990 jobs (tasks 2970-3959)
sbatch slurm/run_phase2e.sh  # 990 jobs (tasks 3960-4949)

# Phase 3 (2 splits)
sbatch slurm/run_phase3a.sh  # 990 jobs (tasks 0-989)
sbatch slurm/run_phase3b.sh  # 990 jobs (tasks 990-1979)

# Phase 4 (no split needed)
sbatch slurm/run_phase4.sh   # 990 jobs (tasks 0-989)

# Phase 5 (2 splits)
sbatch slurm/run_phase5a.sh  # 660 jobs (tasks 0-659)
sbatch slurm/run_phase5b.sh  # 660 jobs (tasks 660-1319)
```

### Monitor Jobs
```bash
# Check your jobs
squeue -u $USER

# Check specific phase
squeue -j <job_id>

# Cancel all jobs
scancel -u $USER

# Cancel specific phase
scancel <job_id>
```

## Resource Allocation

Each task requests:
- **Memory**: 48GB (Phase 1-3), 64GB (Phase 4-5 for larger models)
- **CPUs**: 1 per task
- **GPU**: 1 GPU per task
- **Time**: 2-8 hours depending on phase complexity

## Expected Runtime

- **Phase 1**: ~2 hours per task × 1,320 tasks = ~2,640 GPU-hours
- **Phase 2**: ~2 hours per task × 4,950 tasks = ~9,900 GPU-hours ⚠️ **EXCEEDS 1000 JOB LIMIT**
- **Phase 3**: ~4 hours per task × 1,980 tasks = ~7,920 GPU-hours
- **Phase 4**: ~3 hours per task × 990 tasks = ~2,970 GPU-hours
- **Phase 5**: ~8 hours per task × 1,320 tasks = ~10,560 GPU-hours

**Total**: ~34,000 GPU-hours (with parallelization, wall-clock time depends on cluster capacity)

## Output Files

Results are saved to:
- **Individual logs**: `logs/phase{N}_{JOB_ID}_{TASK_ID}.out` and `.err`
- **Results JSON**: Per-task results in `results/` directory
- **Aggregated results**: Use `scripts/aggregate_results.py` after completion

## Notes

1. All phases are configured to handle model loading failures gracefully
2. The experiment will continue 50 steps after crash detection to observe recovery patterns
3. Each task is independent and can be rerun individually if it fails
4. All 6 models are tested across all phases for comprehensive comparison
5. ✅ All phases split into arrays under 1000 tasks for SLURM compliance
