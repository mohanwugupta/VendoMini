#!/bin/bash
#SBATCH --job-name=vendomini-phase2-large
#SBATCH --array=1650-2474     # Part 3 of 3: 825 tasks (11×5×3×3×5 = 2475 total, split to stay under 1000)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G     # 128GB total RAM
#SBATCH --gres=gpu:2          # 2 GPUs for large models
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=6:00:00

# VendoMini phase2: PE Type Analysis (Large Models)
echo "🚀 Starting VendoMini phase2 (LARGE MODELS)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/
module load anaconda3/2024.2

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate vendomini
else
    source activate vendomini
fi

export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python run_experiment.py \
    --config configs/phases/phase2_large_models.yaml \
    --cluster

echo "✅ Task completed"
