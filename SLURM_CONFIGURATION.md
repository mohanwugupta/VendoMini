# SLURM Configuration for All Phases

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
1. `openai/gpt-oss-20b`
2. `deepseek-ai/DeepSeek-V2.5`
3. `meta-llama/Llama-3.3-70B-Instruct`
4. `Qwen/Qwen2.5-72B-Instruct`
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

### Submit All Phases (Sequential with Dependencies)
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini
bash slurm/submit_all_phases.sh
```

This will submit all phases with dependencies:
- Phase 2 waits for Phase 1 to complete
- Phase 3 waits for Phase 2 to complete
- Phase 4 waits for Phase 3 to complete
- Phase 5 waits for Phase 4 to complete

### Submit Individual Phases
```bash
sbatch slurm/run_phase1.sh  # 360 jobs
sbatch slurm/run_phase2.sh  # 900 jobs
sbatch slurm/run_phase3.sh  # 720 jobs
sbatch slurm/run_phase4.sh  # 270 jobs
sbatch slurm/run_phase5.sh  # 240 jobs
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

- **Phase 1**: ~2 hours per task × 360 tasks = ~720 GPU-hours
- **Phase 2**: ~2 hours per task × 900 tasks = ~1,800 GPU-hours
- **Phase 3**: ~4 hours per task × 720 tasks = ~2,880 GPU-hours
- **Phase 4**: ~3 hours per task × 270 tasks = ~810 GPU-hours
- **Phase 5**: ~8 hours per task × 240 tasks = ~1,920 GPU-hours

**Total**: ~8,130 GPU-hours (with parallelization, wall-clock time depends on cluster capacity)

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
