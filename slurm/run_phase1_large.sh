#!/bin/bash
#SBATCH --job-name=vendomini-phase1-large
#SBATCH --array=0-879         # 11 p_shock × 2 pe_mag × 2 pred_mode × 4 models × 5 reps = 880 tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4     # More CPUs for data loading
#SBATCH --mem-per-cpu=32G     # 128GB total RAM
#SBATCH --gres=gpu:2          # Request 2 GPUs for large models (80GB total VRAM)
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=6:00:00        # Longer time for large model loading and CPU offloading

# VendoMini Phase 1: Core Hypothesis (Large Models: 32B-236B)
# Parallelizes across all parameter combinations via SLURM array jobs

echo "🚀 Starting VendoMini Phase 1 Array Job (LARGE MODELS)"
echo "Models: Qwen3-32B, Llama-3.3-70B, Qwen2.5-72B, DeepSeek-V2.5"
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

echo "🔍 Running VendoMini Phase 1 experiment (large models)..."
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# Run the experiment with cluster flag
python run_experiment.py \
    --config configs/phases/phase1_large_models.yaml \
    --cluster

echo "✅ Phase 1 Large Models task $SLURM_ARRAY_TASK_ID completed at $(date)"
