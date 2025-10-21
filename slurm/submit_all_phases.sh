#!/bin/bash

# Master script to submit all VendoMini phases to SLURM
# Usage: bash submit_all_phases.sh

echo "🚀 Submitting all VendoMini experiments to SLURM"

# Check if we're in the right directory
if [ ! -f "run_experiment.py" ]; then
    echo "❌ Please run this script from the VendoMini root directory"
    exit 1
fi

# Create output directories
mkdir -p logs checkpoints results data

# Submit Phase 1: Core Hypothesis (360 tasks)
echo "📊 Submitting Phase 1 (Core Hypothesis)..."
phase1_job_id=$(sbatch --parsable slurm/run_phase1.sh)
echo "   Job ID: $phase1_job_id"

# Submit Phase 2: PE Type Analysis (900 tasks) - depends on Phase 1
echo "🔧 Submitting Phase 2 (PE Type Analysis)..."
phase2_job_id=$(sbatch --parsable --dependency=afterok:$phase1_job_id slurm/run_phase2.sh)
echo "   Job ID: $phase2_job_id (after Phase 1)"

# Submit Phase 3: Complexity Scaling (720 tasks) - depends on Phase 2
echo "🔍 Submitting Phase 3 (Complexity Scaling)..."
phase3_job_id=$(sbatch --parsable --dependency=afterok:$phase2_job_id slurm/run_phase3.sh)
echo "   Job ID: $phase3_job_id (after Phase 2)"

# Submit Phase 4: Model Architecture Sweep (270 tasks) - depends on Phase 3
echo "🤖 Submitting Phase 4 (Model Architecture Sweep)..."
phase4_job_id=$(sbatch --parsable --dependency=afterok:$phase3_job_id slurm/run_phase4.sh)
echo "   Job ID: $phase4_job_id (after Phase 3)"

# Submit Phase 5: Long Horizon Extremes (240 tasks) - depends on Phase 4
echo "⏱️  Submitting Phase 5 (Long Horizon Extremes)..."
echo "   ⚠️  NOTE: Update phase5 config with TOP/BOTTOM models from Phase 4 before running"
phase5_job_id=$(sbatch --parsable --dependency=afterok:$phase4_job_id slurm/run_phase5.sh)
echo "   Job ID: $phase5_job_id (after Phase 4)"

echo ""
echo "✅ All jobs submitted!"
echo ""
echo "📊 Job Summary:"
echo "  Phase 1: $phase1_job_id (360 tasks, ~500 steps, 2h)"
echo "  Phase 2: $phase2_job_id (900 tasks, ~1000 steps, 2h)"
echo "  Phase 3: $phase3_job_id (720 tasks, 500-2500 steps, 4h)"
echo "  Phase 4: $phase4_job_id (270 tasks, ~1000 steps, 3h)"
echo "  Phase 5: $phase5_job_id (240 tasks, ~5000 steps, 8h)"
echo "  TOTAL: 2,490 tasks"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel $phase1_job_id $phase2_job_id $phase3_job_id $phase4_job_id $phase5_job_id"
echo "📋 Job Summary:"
echo "   - Phase 1: $phase1_job_id (180 tasks, ~1h each)"
echo "   - Phase 2: $phase2_job_id (450 tasks, ~2h each)"
echo ""
echo "🔍 Monitor jobs with:"
echo "   squeue -u \$USER"
echo "   squeue -j $phase1_job_id,$phase2_job_id"
echo ""
echo "📁 Results will be saved in results/ directory with task IDs"
echo "🔄 Use scripts/aggregate_results.py after jobs complete to merge results"
