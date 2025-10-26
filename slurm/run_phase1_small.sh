#!/bin/bash
#SBATCH --job-name=vendomini-phase1-small
#SBATCH --array=0-439         # 11 p_shock × 2 pe_mag × 2 pred_mode × 2 models × 5 reps = 440 tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2     # Fewer CPUs needed for small models
#SBATCH --mem-per-cpu=24G     # 48GB total RAM (enough for 7B-20B models)
#SBATCH --gres=gpu:1          # Single GPU sufficient for 7B and 20B models
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=2:00:00        # Faster loading for small models

# VendoMini Phase 1: Core Hypothesis (Small Models: 7B, 20B)
# Parallelizes across all parameter combinations via SLURM array jobs

echo "🚀 Starting VendoMini Phase 1 Array Job (SMALL MODELS)"
echo "Models: deepseek-llm-7b-chat, gpt-oss-20b"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Change to project directory
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/

# Load conda/python environment
module load anaconda3/2024.2

# Activate environment (adjust to your setup)
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

echo "🔍 Running VendoMini Phase 1 experiment (small models)..."
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# Run the experiment with cluster flag
python run_experiment.py \
    --config configs/phases/phase1_small_models.yaml \
    --cluster

echo "✅ Phase 1 Small Models task $SLURM_ARRAY_TASK_ID completed at $(date)"
