# PowerShell script to update SLURM configuration for multi-GPU support

$slurmFiles = Get-ChildItem -Path "slurm\run_phase*.sh" -File

foreach ($file in $slurmFiles) {
    if ($file.Name -eq "run_phase1.sh" -or $file.Name -eq "run_phase2.sh") {
        Write-Host "Skipping $($file.Name) - already updated"
        continue
    }
    
    Write-Host "Updating $($file.Name)..."
    
    $content = Get-Content $file.FullName -Raw
    
    # Update cpus-per-task from 1 to 4
    $content = $content -replace '#SBATCH --cpus-per-task=1', '#SBATCH --cpus-per-task=4     # More CPUs for data loading'
    
    # Update mem-per-cpu from 128G to 32G (maintaining 128GB total)
    $content = $content -replace '#SBATCH --mem-per-cpu=128G.*', '#SBATCH --mem-per-cpu=32G     # 128GB total RAM'
    
    # Update GPU count from 1 to 2
    $content = $content -replace '#SBATCH --gres=gpu:1', '#SBATCH --gres=gpu:2          # Request 2 GPUs for large models (80GB total VRAM)'
    
    # Update time from 3:00:00 to 4:00:00
    $content = $content -replace '#SBATCH --time=3:00:00', '#SBATCH --time=4:00:00        # Increased time for large model loading'
    
    Set-Content -Path $file.FullName -Value $content -NoNewline
}

Write-Host "`nDone! Updated SLURM scripts for multi-GPU support."
