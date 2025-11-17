#!/bin/bash
#SBATCH --job-name=vendomini-phase1-large
#SBATCH --array=0-659         # 11 p_shock × 2 pe_mag × 2 pred_mode × 3 models × 5 reps = 660 tasks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8     # For parallel PDF processing
#SBATCH --mem=64G
#SBATCH --gres=gpu:4         # Request 4 GPUs for 30B models (160GB total VRAM)
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=2:00:00        # Time for large model loading and inference

# VendoMini Phase 1: Core Hypothesis (Large Models: 32B-236B)
# Parallelizes across all parameter combinations via SLURM array jobs

echo "🚀 Starting VendoMini Phase 1 Array Job (LARGE MODELS)"
echo "Models: Qwen3-30B-A3B-Instruct-2507, Qwen3-32B, DeepSeek-V2.5"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Change to project directory
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/

# Load conda/python environment
module load anaconda3/2025.6

# Activate environment (adjust to your setup)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate godsreach2  # Adjust environment name
elif [ -f ~/.conda/envs/godsreach2/bin/activate ]; then
    source ~/.conda/envs/godsreach2/bin/activate
else
    source activate godsreach2
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
