# VendoMini Phase Splits Summary

All phases have been split to keep SLURM array jobs under 1000 tasks.

## Split Configuration

### Phase 1: Core Hypothesis (1,320 tasks → 2 splits)
```bash
run_phase1a.sh  #SBATCH --array=0-659      (660 tasks)
run_phase1b.sh  #SBATCH --array=660-1319   (660 tasks)
```
**Config**: `configs/phases/phase1_core_hypothesis.yaml`  
**Parameters**: 11 p_shock × 2 pe_mag × 2 pred_mode × 6 models × 5 reps

---

### Phase 2: PE Type Analysis (4,950 tasks → 5 splits)
```bash
run_phase2a.sh  #SBATCH --array=0-989      (990 tasks)
run_phase2b.sh  #SBATCH --array=990-1979   (990 tasks)
run_phase2c.sh  #SBATCH --array=1980-2969  (990 tasks)
run_phase2d.sh  #SBATCH --array=2970-3959  (990 tasks)
run_phase2e.sh  #SBATCH --array=3960-4949  (990 tasks)
```
**Config**: `configs/phases/phase2_pe_type.yaml`  
**Parameters**: 11 p_shock × 5 pe_type_mix × 3 observability × 6 models × 5 reps

---

### Phase 3: Complexity Scaling (1,980 tasks → 2 splits)
```bash
run_phase3a.sh  #SBATCH --array=0-989      (990 tasks)
run_phase3b.sh  #SBATCH --array=990-1979   (990 tasks)
```
**Config**: `configs/phases/phase3_complexity.yaml`  
**Parameters**: 11 p_shock × 3 complexity × 2 recovery × 6 models × 5 reps

---

### Phase 4: Model Architecture Sweep (990 tasks - no split)
```bash
run_phase4.sh   #SBATCH --array=0-989      (990 tasks)
```
**Config**: `configs/phases/phase4_model_arch.yaml`  
**Parameters**: 11 p_shock × 3 pe_mag × 6 models × 5 reps

---

### Phase 5: Long Horizon Extremes (1,320 tasks → 2 splits)
```bash
run_phase5a.sh  #SBATCH --array=0-659      (660 tasks)
run_phase5b.sh  #SBATCH --array=660-1319   (660 tasks)
```
**Config**: `configs/phases/phase5_long_horizon.yaml`  
**Parameters**: 11 p_shock × 2 complexity × 6 models × 10 reps

---

## Total Summary

- **Total Tasks**: 10,560
- **Total Arrays**: 11 (down from 5)
- **Max Array Size**: 990 (all under 1000) ✅
- **Split Scripts**: 13 (11 split + 2 unsplit originals + Phase 4)

## How to Submit

### Option 1: Use the split submission script (Recommended)
```bash
bash slurm/submit_all_phases_split.sh
```

This will submit all 11 arrays with proper dependencies:
- Phase 1a, 1b run in parallel
- Phase 2a-e wait for Phase 1, run in parallel
- Phase 3a-b wait for Phase 2, run in parallel  
- Phase 4 waits for Phase 3
- Phase 5a-b wait for Phase 4, run in parallel

### Option 2: Submit splits individually
```bash
# Phase 1
sbatch slurm/run_phase1a.sh
sbatch slurm/run_phase1b.sh

# Phase 2
sbatch slurm/run_phase2a.sh
sbatch slurm/run_phase2b.sh
sbatch slurm/run_phase2c.sh
sbatch slurm/run_phase2d.sh
sbatch slurm/run_phase2e.sh

# Phase 3
sbatch slurm/run_phase3a.sh
sbatch slurm/run_phase3b.sh

# Phase 4
sbatch slurm/run_phase4.sh

# Phase 5
sbatch slurm/run_phase5a.sh
sbatch slurm/run_phase5b.sh
```

## Verification

All task IDs are correctly assigned:
- No gaps in task ranges
- No overlaps between splits
- Each split covers its portion of the parameter space
- All splits use the same config file (task ID determines which parameters)

## Key Points

1. ✅ All arrays are under 1000 tasks (SLURM compliant)
2. ✅ Task IDs are continuous within each phase
3. ✅ All splits use the same config file per phase
4. ✅ Dependencies ensure phases run in order
5. ✅ Parallel execution within each phase group maximizes throughput

## Original vs Split Comparison

| Phase | Original | Split Into | Largest Array |
|-------|----------|------------|---------------|
| 1     | 1,320    | 2 × 660    | 660           |
| 2     | 4,950 ❌ | 5 × 990    | 990           |
| 3     | 1,980    | 2 × 990    | 990           |
| 4     | 990 ✅   | 1 × 990    | 990           |
| 5     | 1,320    | 2 × 660    | 660           |

**Before**: 3 phases exceeded 1000 limit ❌  
**After**: All phases under 1000 limit ✅
