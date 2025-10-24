#!/bin/bash
#
# Submit all VendoMini phases with splits to stay under 1000 jobs per array
# This script submits 11 array jobs total (instead of 5), each under 1000 tasks
#
# Usage: bash slurm/submit_all_phases_split.sh
#

set -e  # Exit on error

echo "============================================"
echo "VendoMini Experiment Suite - Split Submission"
echo "============================================"
echo ""
echo "📊 Total Tasks: 10,560"
echo "📦 Total Arrays: 11 (all under 1000 tasks)"
echo ""

# Create output directories
mkdir -p logs checkpoints results data

# ============================================
# PHASE 1: Core Hypothesis (1,320 tasks → 2 splits)
# ============================================
echo "📊 Phase 1: Core Hypothesis (1,320 tasks)"
echo "   Submitting 2 splits of 660 tasks each..."

phase1a_job_id=$(sbatch --parsable slurm/run_phase1a.sh)
echo "   ✅ Phase 1a: Job $phase1a_job_id (tasks 0-659)"

phase1b_job_id=$(sbatch --parsable slurm/run_phase1b.sh)
echo "   ✅ Phase 1b: Job $phase1b_job_id (tasks 660-1319)"

# Wait for both Phase 1 splits to complete
phase1_deps="afterok:$phase1a_job_id:$phase1b_job_id"

echo ""

# ============================================
# PHASE 2: PE Type Analysis (4,950 tasks → 5 splits)
# ============================================
echo "🔧 Phase 2: PE Type Analysis (4,950 tasks)"
echo "   Submitting 5 splits of 990 tasks each..."

phase2a_job_id=$(sbatch --parsable --dependency=$phase1_deps slurm/run_phase2a.sh)
echo "   ✅ Phase 2a: Job $phase2a_job_id (tasks 0-989)"

phase2b_job_id=$(sbatch --parsable --dependency=$phase1_deps slurm/run_phase2b.sh)
echo "   ✅ Phase 2b: Job $phase2b_job_id (tasks 990-1979)"

phase2c_job_id=$(sbatch --parsable --dependency=$phase1_deps slurm/run_phase2c.sh)
echo "   ✅ Phase 2c: Job $phase2c_job_id (tasks 1980-2969)"

phase2d_job_id=$(sbatch --parsable --dependency=$phase1_deps slurm/run_phase2d.sh)
echo "   ✅ Phase 2d: Job $phase2d_job_id (tasks 2970-3959)"

phase2e_job_id=$(sbatch --parsable --dependency=$phase1_deps slurm/run_phase2e.sh)
echo "   ✅ Phase 2e: Job $phase2e_job_id (tasks 3960-4949)"

# Wait for all Phase 2 splits to complete
phase2_deps="afterok:$phase2a_job_id:$phase2b_job_id:$phase2c_job_id:$phase2d_job_id:$phase2e_job_id"

echo ""

# ============================================
# PHASE 3: Complexity Scaling (1,980 tasks → 2 splits)
# ============================================
echo "🔍 Phase 3: Complexity Scaling (1,980 tasks)"
echo "   Submitting 2 splits of 990 tasks each..."

phase3a_job_id=$(sbatch --parsable --dependency=$phase2_deps slurm/run_phase3a.sh)
echo "   ✅ Phase 3a: Job $phase3a_job_id (tasks 0-989)"

phase3b_job_id=$(sbatch --parsable --dependency=$phase2_deps slurm/run_phase3b.sh)
echo "   ✅ Phase 3b: Job $phase3b_job_id (tasks 990-1979)"

# Wait for both Phase 3 splits to complete
phase3_deps="afterok:$phase3a_job_id:$phase3b_job_id"

echo ""

# ============================================
# PHASE 4: Model Architecture Sweep (990 tasks - no split)
# ============================================
echo "🤖 Phase 4: Model Architecture Sweep (990 tasks)"
echo "   Submitting 1 array (under 1000)..."

phase4_job_id=$(sbatch --parsable --dependency=$phase3_deps slurm/run_phase4.sh)
echo "   ✅ Phase 4: Job $phase4_job_id (tasks 0-989)"

echo ""

# ============================================
# PHASE 5: Long Horizon Extremes (1,320 tasks → 2 splits)
# ============================================
echo "⏱️  Phase 5: Long Horizon Extremes (1,320 tasks)"
echo "   Submitting 2 splits of 660 tasks each..."
echo "   ⚠️  NOTE: Update phase5 config with TOP/BOTTOM models from Phase 4 before running"

phase5a_job_id=$(sbatch --parsable --dependency=afterok:$phase4_job_id slurm/run_phase5a.sh)
echo "   ✅ Phase 5a: Job $phase5a_job_id (tasks 0-659)"

phase5b_job_id=$(sbatch --parsable --dependency=afterok:$phase4_job_id slurm/run_phase5b.sh)
echo "   ✅ Phase 5b: Job $phase5b_job_id (tasks 660-1319)"

echo ""
echo "============================================"
echo "✅ All 11 array jobs submitted!"
echo "============================================"
echo ""
echo "📊 Job Summary:"
echo "   Phase 1: $phase1a_job_id, $phase1b_job_id (2×660 = 1,320 tasks, ~2h)"
echo "   Phase 2: $phase2a_job_id, $phase2b_job_id, $phase2c_job_id, $phase2d_job_id, $phase2e_job_id (5×990 = 4,950 tasks, ~2h)"
echo "   Phase 3: $phase3a_job_id, $phase3b_job_id (2×990 = 1,980 tasks, ~4h)"
echo "   Phase 4: $phase4_job_id (990 tasks, ~3h)"
echo "   Phase 5: $phase5a_job_id, $phase5b_job_id (2×660 = 1,320 tasks, ~8h)"
echo "   ══════════════════════════════════════"
echo "   TOTAL: 10,560 tasks across 11 arrays"
echo ""
echo "🔍 Monitor jobs:"
echo "   squeue -u \$USER"
echo "   watch -n 10 'squeue -u \$USER'"
echo ""
echo "🛑 Cancel all jobs:"
all_jobs="$phase1a_job_id,$phase1b_job_id,$phase2a_job_id,$phase2b_job_id,$phase2c_job_id,$phase2d_job_id,$phase2e_job_id,$phase3a_job_id,$phase3b_job_id,$phase4_job_id,$phase5a_job_id,$phase5b_job_id"
echo "   scancel $all_jobs"
echo ""
echo "📁 Results location: results/ directory"
echo "🔄 After completion: python scripts/aggregate_results.py"
echo ""
