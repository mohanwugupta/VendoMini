# VendoMini Cluster Deployment Guide

This guide shows how to deploy VendoMini on a SLURM cluster using pre-downloaded HuggingFace models.

## Quick Start

### 1. Transfer Code to Cluster

```bash
# From your local machine
scp -r VendoMini/ username@cluster.university.edu:/scratch/gpfs/username/
```

### 2. Set Up on Cluster

```bash
# SSH to cluster
ssh username@cluster.university.edu

# Navigate to project
cd /scratch/gpfs/username/VendoMini

# Create conda environment
conda create -n vendomini python=3.10 -y
conda activate vendomini

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Models (One-Time Setup)

VendoMini follows your DRM workflow: **download models once, use many times**.

```bash
# Activate environment
conda activate vendomini

# Run Python to download models
python << 'EOF'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set cache directory
models_dir = "/scratch/gpfs/username/VendoMini/models"
os.makedirs(models_dir, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = models_dir
os.environ['HF_HOME'] = models_dir

# Download models (choose one or more)
models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 2GB - fast testing
    "microsoft/phi-2",                      # 5GB - good quality
    "microsoft/Phi-3-mini-4k-instruct",     # 7GB - best quality
]

for model_name in models:
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    print(f"✓ {model_name} downloaded")

print(f"\nModels saved to: {models_dir}")
EOF
```

This downloads models to `VendoMini/models/` just like your DRM setup downloads to `prompt_patching/models/`.

### 4. Verify Setup

```bash
python scripts/setup_cluster.py
```

Expected output:
```
✅ Cluster setup is ready!
   - All directories created
   - 1 model(s) available locally
   - HuggingFace cache: /scratch/gpfs/username/VendoMini/models
```

### 5. Update SLURM Scripts

Edit `slurm/run_phase1.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=vendomini_p1
#SBATCH --output=logs/vendomini_p1_%A_%a.out
#SBATCH --error=logs/vendomini_p1_%A_%a.err
#SBATCH --array=0-179           # 180 tasks for Phase 1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@university.edu  # ← UPDATE THIS

# Navigate to project directory
cd /scratch/gpfs/YOUR_USERNAME/VendoMini  # ← UPDATE THIS

# Activate conda environment
module load anaconda3/2023.3
conda activate vendomini

# Set model cache (points to pre-downloaded models)
export TRANSFORMERS_CACHE=/scratch/gpfs/YOUR_USERNAME/VendoMini/models  # ← UPDATE THIS
export HF_HOME=$TRANSFORMERS_CACHE

# Run experiment
python run_experiment.py \
    --config configs/phases/phase1_core_hypothesis.yaml \
    --cluster \
    --task-id $SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID completed"
```

### 6. Submit Job

```bash
# Submit Phase 1
sbatch slurm/run_phase1.sh

# Check status
squeue -u $USER

# View output
tail -f logs/vendomini_p1_*.out
```

## How It Works

### Model Loading (DRM Pattern)

VendoMini now uses your proven DRM model loading approach:

1. **Set HF cache environment variables** → Points to local models directory
2. **Check for local model** → Looks in `models/models--org--name/snapshots/`
3. **Load from local path** → Uses pre-downloaded model
4. **Fallback to download** → Only if local model not found

```python
# In agent.py (following drm_cluster_utils.py pattern)
models_dir = os.getenv('TRANSFORMERS_CACHE')
local_path = get_local_model_path(models_dir, model_name)
if local_path:
    model = AutoModelForCausalLM.from_pretrained(local_path)  # Local
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)  # Download
```

### Directory Structure

```
/scratch/gpfs/username/VendoMini/
├── models/                          # ← Pre-downloaded models (like DRM)
│   └── models--microsoft--phi-2/
│       └── snapshots/
│           └── abc123.../          # Actual model files
├── logs/                           # Job outputs
├── results/                        # Experiment results
│   ├── vendomini_task_0000.json
│   ├── vendomini_task_0001.json
│   └── ...
├── configs/
│   └── phases/
│       ├── phase1_core_hypothesis.yaml
│       └── ...
└── src/
    ├── agent.py                    # Uses local models
    ├── cluster_utils.py            # Model path utilities
    └── ...
```

## Testing

### Test Single Task Locally

```bash
# Test task 0 without SLURM
python run_experiment.py \
    --config configs/phases/phase1_core_hypothesis.yaml \
    --task-id 0 \
    --cluster
```

### Test on Cluster (Single Task)

```bash
# Submit 1 task as a test
sbatch --array=0 slurm/run_phase1.sh

# Check output
tail logs/vendomini_p1_*.out
```

### Full Phase 1 (180 Tasks)

```bash
sbatch slurm/run_phase1.sh
```

## Aggregating Results

After all tasks complete:

```bash
python scripts/aggregate_results.py \
    --input-dir results \
    --output results/phase1_all.csv
```

This creates:
- `results/phase1_all.json` - All results
- `results/phase1_all.csv` - Flattened CSV
- Console output with crash statistics

## Model Comparison

| Model | Size | VRAM | Quality | Speed | Local Path Check |
|-------|------|------|---------|-------|-----------------|
| TinyLlama | 1.1B | ~2GB | ⭐⭐ | ⚡⚡⚡ | ✅ |
| Phi-2 | 2.7B | ~5GB | ⭐⭐⭐⭐ | ⚡⚡ | ✅ |
| Phi-3 Mini | 3.8B | ~7GB | ⭐⭐⭐⭐⭐ | ⚡ | ✅ |

All models now check for local copies first (no re-downloading)!

## Troubleshooting

### Models not found locally

```bash
# Check if models are downloaded
ls -la models/

# Should see: models--microsoft--phi-2/
# If not, re-run download script above
```

### SLURM job fails immediately

```bash
# Check error log
cat logs/vendomini_p1_*_0.err

# Common issues:
# 1. Wrong paths in run_phase1.sh
# 2. Conda environment not activated
# 3. Missing dependencies
```

### Model downloads during job (slow)

This means local model wasn't found. Ensure:

```bash
# 1. Environment variables are set in SLURM script
export TRANSFORMERS_CACHE=/path/to/VendoMini/models
export HF_HOME=$TRANSFORMERS_CACHE

# 2. Models were downloaded to correct location
python scripts/setup_cluster.py
```

## Comparison with DRM Workflow

| Aspect | DRM | VendoMini |
|--------|-----|-----------|
| Model loading | `load_model_cluster()` | `_initialize_client()` with local path check |
| Cache location | `/scratch/.../prompt_patching/models` | `/scratch/.../VendoMini/models` |
| Cache format | HF snapshots | HF snapshots (same) |
| Environment vars | `HF_HOME`, `TRANSFORMERS_CACHE` | Same ✅ |
| Local path check | `get_local_model_path()` | `get_local_model_path()` (adapted) |
| Fallback | Download from hub | Download from hub (same) |
| Array jobs | `--array=0-N` | `--array=0-N` ✅ |
| Task mapping | `get_params_for_task(task_id)` | Config grid expansion ✅ |

**Same proven pattern, adapted for VendoMini!**

## Next Steps

1. ✅ Download models once (see step 3 above)
2. ✅ Update SLURM scripts with your paths
3. ✅ Test single task
4. ✅ Submit full phase
5. ✅ Aggregate results
6. ✅ Analyze crash patterns

Your VendoMini cluster setup now mirrors your successful DRM workflow! 🚀
