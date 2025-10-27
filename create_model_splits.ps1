# PowerShell script to create small/large model splits for all phases

$phases = @(
    @{name="phase2"; desc="PE Type Analysis"; grid="11 p_shock × 5 pe_type × 3 obs"; total=4950},
    @{name="phase3"; desc="Complexity Scaling"; grid="11 p_shock × 3 complexity × 2 recovery"; total=1980},
    @{name="phase4"; desc="Model Architecture Effects"; grid="11 p_shock × 2 arch_sensitivity"; total=660},
    @{name="phase5"; desc="Long Horizon Planning"; grid="11 p_shock × 3 horizon"; total=990}
)

foreach ($phase in $phases) {
    $phaseName = $phase.name
    $phaseDesc = $phase.desc
    $gridDesc = $phase.grid
    $totalJobs = $phase.total
    
    # Calculate job counts
    $smallJobs = [math]::Floor($totalJobs * 2 / 6) - 1  # 2 models out of 6
    $largeJobs = [math]::Floor($totalJobs * 4 / 6) - 1  # 4 models out of 6
    
    Write-Host "Processing $phaseName..."
    
    # Read original config
    $configPath = "configs\phases\${phaseName}_*.yaml"
    $originalConfig = Get-ChildItem $configPath | Select-Object -First 1
    
    if ($originalConfig) {
        $content = Get-Content $originalConfig.FullName -Raw
        
        # Create small models config
        $smallContent = $content -replace 'agent\.model\.name: \[.*\]', 'agent.model.name: [deepseek-ai/deepseek-llm-7b-chat, openai/gpt-oss-20b]'
        $smallContent = $smallContent -replace 'name: ".*"', "name: `"${phaseName}_small_models`""
        $smallContent = $smallContent -replace 'description: ".*"', "description: `"$phaseDesc (Small models: 7B, 20B)`""
        
        Set-Content -Path "configs\phases\${phaseName}_small_models.yaml" -Value $smallContent
        
        # Create large models config
        $largeContent = $content -replace 'agent\.model\.name: \[.*\]', 'agent.model.name: [Qwen/Qwen3-32B, meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen2.5-72B-Instruct, deepseek-ai/DeepSeek-V2.5]'
        $largeContent = $largeContent -replace 'name: ".*"', "name: `"${phaseName}_large_models`""
        $largeContent = $largeContent -replace 'description: ".*"', "description: `"$phaseDesc (Large models: 32B-236B)`""
        
        Set-Content -Path "configs\phases\${phaseName}_large_models.yaml" -Value $largeContent
        
        Write-Host "  Created config files for $phaseName"
    }
    
    # Create small models SLURM script
    $smallScript = @"
#!/bin/bash
#SBATCH --job-name=vendomini-${phaseName}-small
#SBATCH --array=0-${smallJobs}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=24G     # 48GB total RAM
#SBATCH --gres=gpu:1          # Single GPU for small models
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=2:00:00

# VendoMini ${phaseName}: ${phaseDesc} (Small Models)
echo "🚀 Starting VendoMini ${phaseName} (SMALL MODELS)"
echo "Task ID: `$SLURM_ARRAY_TASK_ID"

cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/
module load anaconda3/2024.2

if command -v conda &> /dev/null; then
    eval "`$(conda shell.bash hook)"
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
    --config configs/phases/${phaseName}_small_models.yaml \
    --cluster

echo "✅ Task completed"
"@
    
    Set-Content -Path "slurm\run_${phaseName}_small.sh" -Value $smallScript
    
    # Create large models SLURM script  
    $largeScript = @"
#!/bin/bash
#SBATCH --job-name=vendomini-${phaseName}-large
#SBATCH --array=0-${largeJobs}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G     # 128GB total RAM
#SBATCH --gres=gpu:2          # 2 GPUs for large models
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=your-email@domain.edu
#SBATCH --time=6:00:00

# VendoMini ${phaseName}: ${phaseDesc} (Large Models)
echo "🚀 Starting VendoMini ${phaseName} (LARGE MODELS)"
echo "Task ID: `$SLURM_ARRAY_TASK_ID"

cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/
module load anaconda3/2024.2

if command -v conda &> /dev/null; then
    eval "`$(conda shell.bash hook)"
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
    --config configs/phases/${phaseName}_large_models.yaml \
    --cluster

echo "✅ Task completed"
"@
    
    Set-Content -Path "slurm\run_${phaseName}_large.sh" -Value $largeScript
    
    Write-Host "  Created SLURM scripts for $phaseName"
}

Write-Host "`n✅ Done! Created small/large splits for all phases."
Write-Host "`nUsage:"
Write-Host "  Small models (fast queue): sbatch slurm/run_phase1_small.sh"
Write-Host "  Large models (slower queue): sbatch slurm/run_phase1_large.sh"
