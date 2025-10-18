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

# Submit Phase 1
echo "📊 Submitting Phase 1 (Core Hypothesis)..."
phase1_job_id=$(sbatch --parsable slurm/run_phase1.sh)
echo "   Job ID: $phase1_job_id"

# Submit Phase 2
echo "🔧 Submitting Phase 2 (PE Type Analysis)..."
phase2_job_id=$(sbatch --parsable slurm/run_phase2.sh)
echo "   Job ID: $phase2_job_id"

# Submit Phase 3 (add more as needed)
# echo "🔍 Submitting Phase 3 (Complexity Scaling)..."
# phase3_job_id=$(sbatch --parsable slurm/run_phase3.sh)
# echo "   Job ID: $phase3_job_id"

echo ""
echo "✅ All jobs submitted!"
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
