#!/bin/bash
#SBATCH --job-name=vendomini-phase2e
#SBATCH --array=3960-4949     # Phase 2 Split 5/5: tasks 3960-4949 (990 tasks)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=2:00:00

# VendoMini Phase 2e: PE Type Analysis (Split 5/5)

echo "🚀 Starting VendoMini Phase 2e Array Job (Split 5/5)"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/
module load anaconda3/2024.2

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate vendomini
elif [ -f ~/.conda/envs/vendomini/bin/activate ]; then
    source ~/.conda/envs/vendomini/bin/activate
else
    source activate vendomini
fi

export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/VendoMini/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "🔍 Running VendoMini Phase 2e experiment..."
python run_experiment.py \
    --config configs/phases/phase2_pe_type.yaml \
    --cluster

echo "✅ Task $SLURM_ARRAY_TASK_ID complete!"
echo "🏁 Job finished at $(date)"
