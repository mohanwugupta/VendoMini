# Quick Reference: Optimized Job Submission

## TL;DR

**Small models** (7B, 20B) now use **1 GPU** → faster queue times ⚡
**Large models** (32B-236B) use **2 GPUs** → proper resources 🚀

## Quick Start

### Test Setup (Recommended First)
```bash
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/
./slurm/submit_all_optimized.sh test
```
Runs Phase 1 with small models only (~10 min queue + 30 min run)

### Submit Everything
```bash
./slurm/submit_all_optimized.sh all
```
Submits all phases, both small and large models

### Small Models Only (Fast Results)
```bash
./slurm/submit_all_optimized.sh small
```
7B and 20B models only - results in hours, not days

### Large Models Only
```bash
./slurm/submit_all_optimized.sh large
```
32B-236B models - longer queue and runtime

## File Changes

### New Config Files (Per Phase)
- `configs/phases/phase1_small_models.yaml` - 2 models (7B, 20B)
- `configs/phases/phase1_large_models.yaml` - 4 models (32B-236B)

### New SLURM Scripts (Per Phase)
- `slurm/run_phase1_small.sh` - 1 GPU, 48GB RAM, fast queue
- `slurm/run_phase1_large.sh` - 2 GPUs, 128GB RAM, slow queue

### Old Files (Still Work)
- `configs/phases/phase1_core_hypothesis.yaml` - All 6 models
- `slurm/run_phase1.sh` - Requests 2 GPUs for all jobs (wasteful)

## Resource Comparison

| Type | GPUs | RAM | CPUs | Time | Queue | Models |
|------|------|-----|------|------|-------|--------|
| **Small** | 1 | 48GB | 2 | 2h | Fast ⚡ | 7B, 20B |
| **Large** | 2 | 128GB | 4 | 6h | Slow 🐌 | 32B-236B |
| Old (all) | 2 | 128GB | 4 | 4h | Slow 🐌 | All 6 |

## Job Counts Per Phase

| Phase | Total | Small | Large |
|-------|-------|-------|-------|
| Phase 1 | 1,320 | 440 | 880 |
| Phase 2 | 4,950 | 1,650 | 3,300 |
| Phase 3 | 1,980 | 660 | 1,320 |
| Phase 4 | 660 | 220 | 440 |
| Phase 5 | 990 | 330 | 660 |

## Monitor Jobs

```bash
# Check all your jobs
squeue -u $USER

# Check small model jobs
squeue -u $USER | grep small

# Check large model jobs
squeue -u $USER | grep large

# View logs
ls logs/
tail -f logs/*/slurm-*.out
```

## Typical Timeline

### Small Models (1 GPU)
- Queue: 5-30 minutes
- Runtime: 30 min - 2 hours per job
- **Total for all phases**: ~1-3 days

### Large Models (2 GPUs)
- Queue: 30 min - 2 hours
- Runtime: 2-6 hours per job (70B+ models slow with CPU offloading)
- **Total for all phases**: ~3-7 days

### Strategy
1. Submit small models first → get early results
2. Submit large models while analyzing small results
3. Combine all results when complete

## Common Commands

```bash
# Test
./slurm/submit_all_optimized.sh test

# Submit all
./slurm/submit_all_optimized.sh all

# Submit only small
./slurm/submit_all_optimized.sh small

# Submit only large  
./slurm/submit_all_optimized.sh large

# Submit one phase
./slurm/submit_all_optimized.sh phase1

# Cancel all jobs
scancel -u $USER

# Check queue position
squeue -u $USER --start

# GPU usage on running node
ssh <node-name>
nvidia-smi
```

## Troubleshooting

### OOM on small models
Shouldn't happen. If it does:
```bash
# Edit slurm/run_phase*_small.sh
# Change: --mem-per-cpu=24G
# To:     --mem-per-cpu=32G
```

### Large models too slow
Expected for 70B+ with CPU offloading. Options:
1. Enable 8-bit quantization (edit `src/agent.py`)
2. Request 4 GPUs (change `--gres=gpu:2` to `gpu:4`)
3. Skip 236B model (remove from large configs)

### Jobs stuck in queue
```bash
# Check available resources
sinfo -N

# Check estimated start time
squeue -u $USER --start

# Consider lowering resource requests
```

## Why This Is Better

✅ **Faster results**: Small models finish in hours, not days
✅ **Better resource use**: Don't waste GPUs on 7B models
✅ **Shorter queue times**: 1-GPU jobs more available
✅ **Flexibility**: Can run small/large separately or together

## Migration from Old Setup

Already submitted with old scripts? Cancel and resubmit:

```bash
# Cancel old jobs
scancel -u $USER

# Submit optimized
./slurm/submit_all_optimized.sh all
```

Results are compatible - just check model name in logs.

---

**See full documentation**: `docs/OPTIMIZED_SUBMISSION.md`
