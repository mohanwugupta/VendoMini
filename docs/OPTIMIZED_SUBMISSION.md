# Optimized Resource Allocation for Multi-Model Experiments

## Overview

VendoMini now uses **separate job configurations** for small and large models to optimize cluster resource usage and reduce queue times.

## Model Categories

### Small Models (1 GPU Required)
- **deepseek-ai/deepseek-llm-7b-chat** (7B params) ≈ 14GB VRAM
- **openai/gpt-oss-20b** (20B params) ≈ 40GB VRAM

**Resources**: 1 GPU, 48GB RAM, 2 CPUs, ~2 hour runtime

### Large Models (2 GPUs Required)
- **Qwen/Qwen3-32B** (32B params) ≈ 64GB VRAM
- **meta-llama/Llama-3.3-70B-Instruct** (70B params) ≈ 140GB VRAM
- **Qwen/Qwen2.5-72B-Instruct** (72B params) ≈ 144GB VRAM
- **deepseek-ai/DeepSeek-V2.5** (236B params) ≈ 500GB VRAM

**Resources**: 2 GPUs, 128GB RAM, 4 CPUs, ~6 hour runtime (includes CPU offloading)

## File Structure

Each phase now has **3 configuration files**:

1. `configs/phases/phase1_core_hypothesis.yaml` - **Original** (all 6 models)
2. `configs/phases/phase1_small_models.yaml` - **New** (2 small models)
3. `configs/phases/phase1_large_models.yaml` - **New** (4 large models)

And **3 SLURM scripts**:

1. `slurm/run_phase1.sh` - **Original** (all models, 2 GPUs for all)
2. `slurm/run_phase1_small.sh` - **New** (1 GPU, fast queue)
3. `slurm/run_phase1_large.sh` - **New** (2 GPUs, slower queue)

## Submission Options

### Option 1: Test First (Recommended)
```bash
# Test with small models only (fast results)
./slurm/submit_all_optimized.sh test
```

### Option 2: Submit All Phases
```bash
# Submit everything (small jobs go to fast queue, large to slow queue)
./slurm/submit_all_optimized.sh all
```

### Option 3: Small Models Only
```bash
# Get results for 7B and 20B models quickly
./slurm/submit_all_optimized.sh small
```

### Option 4: Large Models Only
```bash
# Run large models separately (longer queue time)
./slurm/submit_all_optimized.sh large
```

### Option 5: Specific Phase
```bash
# Submit both small and large for one phase
./slurm/submit_all_optimized.sh phase1
```

### Option 6: Manual Submission
```bash
# Submit individual jobs manually
sbatch slurm/run_phase1_small.sh   # Just 7B and 20B models
sbatch slurm/run_phase1_large.sh   # Just 32B-236B models
```

## Resource Comparison

### Before Optimization (All Models Together)
```
#SBATCH --gres=gpu:2          # Always request 2 GPUs
#SBATCH --mem-per-cpu=32G     # 128GB RAM
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
```
- **Total tasks per phase**: ~1,320-4,950 jobs
- **Problem**: Small models waste GPU resources
- **Queue time**: Longer (2-GPU jobs wait more)

### After Optimization (Split by Model Size)

#### Small Model Jobs
```
#SBATCH --gres=gpu:1          # Only 1 GPU needed
#SBATCH --mem-per-cpu=24G     # 48GB RAM
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
```
- **Tasks per phase**: ~440 jobs (2/6 models × total)
- **Benefits**: 
  - ✅ Faster queue times (1-GPU jobs more available)
  - ✅ Lower resource waste
  - ✅ Quicker results for initial analysis

#### Large Model Jobs
```
#SBATCH --gres=gpu:2          # 2 GPUs required
#SBATCH --mem-per-cpu=32G     # 128GB RAM
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
```
- **Tasks per phase**: ~880 jobs (4/6 models × total)
- **Benefits**: 
  - ✅ Longer runtime for CPU offloading
  - ✅ Only request 2 GPUs when actually needed

## Job Count Breakdown

### Phase 1: Core Hypothesis
- **Original**: 1,320 jobs (all models)
- **Small models**: 440 jobs (7B, 20B)
- **Large models**: 880 jobs (32B, 70B, 72B, 236B)

### Phase 2: PE Type Analysis
- **Original**: 4,950 jobs
- **Small models**: 1,650 jobs
- **Large models**: 3,300 jobs

### Phase 3: Complexity Scaling
- **Original**: 1,980 jobs
- **Small models**: 660 jobs
- **Large models**: 1,320 jobs

### Phase 4: Model Architecture
- **Original**: 660 jobs
- **Small models**: 220 jobs
- **Large models**: 440 jobs

### Phase 5: Long Horizon Planning
- **Original**: 990 jobs
- **Small models**: 330 jobs
- **Large models**: 660 jobs

## Expected Queue Times

Based on typical cluster usage:

| Job Type | GPU Req | Typical Wait | Reason |
|----------|---------|--------------|--------|
| Small (1 GPU) | 1 × 40GB | 5-30 min | More nodes available |
| Large (2 GPU) | 2 × 40GB | 30 min - 2 hrs | Fewer nodes with 2+ GPUs |

**Strategy**: Submit small models first to get early results while large models wait in queue.

## Monitoring Jobs

### Check queue status
```bash
squeue -u $USER
```

### Check running jobs by size
```bash
# Small model jobs
squeue -u $USER | grep "small"

# Large model jobs  
squeue -u $USER | grep "large"
```

### Monitor GPU usage on running node
```bash
# Get node name from squeue output
ssh <node-name>
nvidia-smi
```

### Check logs
```bash
# Small model logs
ls logs/*small*/

# Large model logs
ls logs/*large*/
```

## Cost Analysis

### Resource-Hours Comparison

**Before** (all jobs request 2 GPUs):
- Phase 1: 1,320 tasks × 4 hours × 2 GPUs = **10,560 GPU-hours**

**After** (optimized):
- Phase 1 small: 440 tasks × 2 hours × 1 GPU = **880 GPU-hours**
- Phase 1 large: 880 tasks × 6 hours × 2 GPUs = **10,560 GPU-hours**
- **Total**: **11,440 GPU-hours**

**Savings**: Despite longer runtime for large models, you save queue time and get small model results **2x faster**.

## Troubleshooting

### Small models getting OOM errors
This shouldn't happen. If it does:
1. Check if model loaded correctly: `grep "Model loaded" logs/*/slurm-*.out`
2. Verify GPU allocation: `grep "GPU 0:" logs/*/slurm-*.out`
3. Increase memory: Change `--mem-per-cpu=24G` to `32G` in small scripts

### Large models too slow
This is expected for 70B+ models with CPU offloading. Options:
1. **Enable 8-bit quantization** (see `docs/MULTI_GPU_SETUP.md`)
2. **Request 4 GPUs** for 70B+ models (change `--gres=gpu:2` to `gpu:4`)
3. **Skip 236B model** if too slow (remove from config)

### Jobs stuck in queue
Check priority and available resources:
```bash
squeue -u $USER --start  # Shows estimated start times
sinfo -N                 # Shows available nodes
```

## Recommended Workflow

1. **Test setup** (5-10 minutes):
   ```bash
   ./slurm/submit_all_optimized.sh test
   ```
   Wait for one small job to complete, check logs

2. **Run small models** (hours to days):
   ```bash
   ./slurm/submit_all_optimized.sh small
   ```
   Get initial results while large models queue

3. **Analyze small results** (while large models run):
   ```bash
   python scripts/aggregate_results.py
   python scripts/analyze_results.py
   ```

4. **Run large models** (days to weeks):
   ```bash
   ./slurm/submit_all_optimized.sh large
   ```

5. **Combine results**:
   ```bash
   python scripts/aggregate_results.py
   ```

## Migration from Old Scripts

If you've already submitted jobs with the old scripts (`run_phase1.sh`):

1. **Cancel old jobs**:
   ```bash
   scancel -u $USER  # Cancel all your jobs
   # Or selectively:
   scancel <job-id>
   ```

2. **Submit with new optimized scripts**:
   ```bash
   ./slurm/submit_all_optimized.sh all
   ```

The results will be compatible - just check logs to see which model was used.

## Summary

✅ **Small models**: Fast queue, quick results, 1 GPU
✅ **Large models**: Proper resources, CPU offloading, 2 GPUs  
✅ **Better cluster utilization**: Don't request 2 GPUs when 1 is enough
✅ **Faster time-to-first-results**: Small jobs complete first

Use `./slurm/submit_all_optimized.sh test` to get started! 🚀
