# Cluster Installation Instructions

## Quick Setup

Run these commands on the cluster to install all dependencies:

```bash
# 1. Navigate to project directory
cd /scratch/gpfs/JORDANAT/mg9965/VendoMini/

# 2. Load Python/Conda
module load anaconda3/2024.2

# 3. Create/activate conda environment
conda create -n vendomini python=3.10 -y
conda activate vendomini

# 4. Install all requirements
pip install -r requirements-cluster.txt

# 5. Verify installation
python scripts/verify_installation.py
```

## Individual Package Installation

If the above fails, install packages individually:

```bash
# Core dependencies
pip install pyyaml numpy pandas scikit-learn joblib pytest matplotlib seaborn tqdm

# PyTorch (CUDA 11.8 - adjust for your cluster)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# HuggingFace
pip install transformers>=4.35.0 accelerate>=0.24.0

# CRITICAL: Protobuf (fixes the error you encountered)
pip install protobuf>=3.20.0

# Tokenizer support
pip install sentencepiece>=0.1.99

# Optional but recommended
pip install safetensors einops
```

## Common Errors & Fixes

### Error: "requires the protobuf library"
**Solution:**
```bash
pip install protobuf>=3.20.0
```

### Error: "DynamicCache object has no attribute seen_tokens"
**Solution:** Already fixed in the code with `use_cache=False`

### Error: Model not found
**Solution:** Download models first:
```bash
python scripts/download_models_simple.py
```

## Verify Setup

```bash
# Check Python packages
python -c "import torch, transformers, protobuf; print('✅ All packages installed')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check models
ls /scratch/gpfs/JORDANAT/mg9965/VendoMini/models/
```

## Running Experiments

Once setup is complete:

```bash
# Test single run
python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml

# Submit to SLURM
sbatch slurm/run_phase1a.sh
```
