#!/bin/bash
#SBATCH --job-name=vendomini-phase1
#SBATCH --array=0-179        # 3 p_shock × 2 pe_mag × 2 pred_mode × 3 models × 5 reps = 180 tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=1:00:00

# VendoMini Phase 1: Core Hypothesis
# Parallelizes across all parameter combinations via SLURM array jobs

echo "🚀 Starting VendoMini Phase 1 Array Job"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Change to project directory
cd $SLURM_SUBMIT_DIR || cd /path/to/vendomini

# Load conda/python environment
module load anaconda3/2024.2 || module load python/3.10

# Activate environment (adjust to your setup)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate vendomini
elif [ -f ~/.conda/envs/vendomini/bin/activate ]; then
    source ~/.conda/envs/vendomini/bin/activate
else
    source activate vendomini
fi

echo "🔍 Running VendoMini Phase 1 experiment..."
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# Run the experiment with cluster flag
python run_experiment.py \
    --config configs/phases/phase1_core_hypothesis.yaml \
    --cluster

echo "✅ Task $SLURM_ARRAY_TASK_ID complete!"
echo "🏁 Job finished at $(date)"
