#!/bin/bash
#SBATCH --job-name=vendomini-phase3
#SBATCH --array=0-1979        # 11 p_shock × 3 complexity × 2 recovery × 6 models × 5 reps = 1,980 tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=2:00:00       # Longer time for complexity levels 2-3

# VendoMini Phase 3: Complexity Scaling
# Tests how complexity level and recovery tools affect crash behavior

echo "🚀 Starting VendoMini Phase 3 Array Job"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Change to project directory
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/

# Load conda/python environment
module load anaconda3/2024.2

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate vendomini
elif [ -f ~/.conda/envs/vendomini/bin/activate ]; then
    source ~/.conda/envs/vendomini/bin/activate
else
    source activate vendomini
fi

# Set up environment for HF models - using VendoMini project directory
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models

# Force offline mode - compute nodes have no internet access
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "🔍 Running VendoMini Phase 3 experiment..."
python run_experiment.py \
    --config configs/phases/phase3_complexity.yaml \
    --cluster

echo "✅ Task $SLURM_ARRAY_TASK_ID complete!"
echo "🏁 Job finished at $(date)"
