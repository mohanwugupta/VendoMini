#!/bin/bash
#SBATCH --job-name=vendomini-phase2
#SBATCH --array=0-299        # 5 pe_type × 2 p_shock × 3 obs × 3 models × 5 reps = 450 tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=2:00:00

# VendoMini Phase 2: PE Type Analysis
# Parallelizes across all parameter combinations via SLURM array jobs

echo "🚀 Starting VendoMini Phase 2 Array Job"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

cd $SLURM_SUBMIT_DIR || cd /path/to/vendomini

module load anaconda3/2024.2 || module load python/3.10

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate vendomini
elif [ -f ~/.conda/envs/vendomini/bin/activate ]; then
    source ~/.conda/envs/vendomini/bin/activate
else
    source activate vendomini
fi

echo "🔍 Running VendoMini Phase 2 experiment..."
python run_experiment.py \
    --config configs/phases/phase2_pe_type.yaml \
    --cluster

echo "✅ Task $SLURM_ARRAY_TASK_ID complete!"
echo "🏁 Job finished at $(date)"
