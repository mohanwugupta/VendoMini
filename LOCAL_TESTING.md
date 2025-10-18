# Local GPU Testing Guide

This guide helps you test VendoMini on your local machine with a small HuggingFace LLM.

## Prerequisites

- **GPU**: NVIDIA GPU with 8GB VRAM (or CPU fallback)
- **Python**: 3.10 or 3.11 (avoid 3.13 - torch may not support it yet)
- **OS**: Windows, Linux, or macOS

## Step 1: Install Dependencies

### Core dependencies
```powershell
pip install -r requirements.txt
```

### GPU dependencies (required for local LLM testing)
```powershell
# Install PyTorch with CUDA support (for GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install HuggingFace libraries
pip install transformers accelerate
```

**Note**: If you don't have a GPU, PyTorch will use CPU (much slower but works).

## Step 2: Choose a Model

Edit `configs/local_test.yaml` and uncomment ONE model:

### Option 1: TinyLlama (Smallest - ~2GB VRAM)
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```
- **Pros**: Tiny, fits easily in 8GB
- **Cons**: Lower quality responses
- **Best for**: Quick testing, debugging

### Option 2: Phi-2 (Balanced - ~5GB VRAM) ⭐ RECOMMENDED
```yaml
model:
  name: "microsoft/phi-2"
```
- **Pros**: Good quality, still fits in 8GB
- **Cons**: Takes a few minutes to download
- **Best for**: Realistic testing

### Option 3: Phi-3 Mini (Best Quality - ~7GB VRAM)
```yaml
model:
  name: "microsoft/Phi-3-mini-4k-instruct"
```
- **Pros**: Best quality for the size
- **Cons**: Uses most of your 8GB VRAM
- **Best for**: Production-like testing

## Step 3: Run Quick Test

Test that everything works:

```powershell
python scripts\test_local_gpu.py
```

This will:
1. Check your GPU
2. Load the model
3. Run a single short experiment (50 steps)

Expected output:
```
==============================================================
VendoMini Local GPU Testing
==============================================================

GPU Check
----------------------------------------------------------
✓ CUDA available
  Device: NVIDIA GeForce RTX 3070
  Memory: 8.0 GB
  ...

Testing Model Loading
----------------------------------------------------------
Loading config: configs\local_test.yaml
Model: microsoft/phi-2
Initializing agent...
✓ Agent initialized
  ...

Running Quick Test
----------------------------------------------------------
✓ Experiment completed!
  Steps: 50
  Final budget: $845.23
  Crashed: False
```

## Step 4: Run Full Local Test

Run multiple experiments in parallel (uses joblib, not GPU):

```powershell
# Run 2 experiments (configured in local_test.yaml)
python run_experiment.py --config configs\local_test.yaml --n-jobs 1
```

**Important**: Use `--n-jobs 1` when running LLMs locally to avoid GPU memory issues!

## Step 5: View Results

Aggregate and view results:

```powershell
python scripts\aggregate_results.py --input-dir results --output results\local_test.csv
```

This creates:
- `results/local_test.json` - All results
- `results/local_test.csv` - CSV format
- Console output with crash statistics

## Troubleshooting

### "CUDA out of memory"
- Use a smaller model (TinyLlama)
- Close other programs using GPU
- Restart Python and try again

### "torch not available"
Try installing with CPU support:
```powershell
pip install torch torchvision torchaudio
```

### Model takes forever to download
First download is slow (downloads ~2-7GB model files). Subsequent runs use cached models.

Cache location:
- Windows: `C:\Users\<username>\.cache\huggingface`
- Linux: `~/.cache/huggingface`

### Import errors
Make sure you're in the VendoMini directory:
```powershell
cd C:\Users\sheik\Box\ResearchProjects\VendoMini
python scripts\test_local_gpu.py
```

## Performance Tips

### Speed up testing
1. **Reduce max_steps**: Edit `configs/local_test.yaml`:
   ```yaml
   environment:
     max_steps: 20  # Instead of 50
   ```

2. **Reduce grid size**: 
   ```yaml
   grid:
     runs_per_config: 1  # Instead of 2
   ```

3. **Use smaller model**: TinyLlama is 5-10x faster than Phi-3

### GPU memory management
```python
# If you want to free GPU memory between runs:
import torch
torch.cuda.empty_cache()
```

## Next Steps

Once local testing works:

1. **Test different configurations**: Modify `configs/local_test.yaml`
2. **Implement actual LLM logic**: Update `src/agent.py` to use LLM for actions
3. **Run longer experiments**: Increase `max_steps` to 100-200
4. **Scale to cluster**: Use SLURM scripts for large-scale experiments

## Model Comparison

| Model | Size | VRAM | Quality | Speed | Download |
|-------|------|------|---------|-------|----------|
| TinyLlama | 1.1B | ~2GB | ⭐⭐ | ⚡⚡⚡ | 2 GB |
| Phi-2 | 2.7B | ~5GB | ⭐⭐⭐⭐ | ⚡⚡ | 5 GB |
| Phi-3 Mini | 3.8B | ~7GB | ⭐⭐⭐⭐⭐ | ⚡ | 7 GB |

## Example: Full Local Workflow

```powershell
# 1. Install dependencies
pip install torch transformers accelerate

# 2. Test GPU
python scripts\test_local_gpu.py

# 3. Run experiments
python run_experiment.py --config configs\local_test.yaml --n-jobs 1

# 4. Aggregate results
python scripts\aggregate_results.py

# 5. View results
type results\local_test.csv
```

That's it! You now have VendoMini running with a real LLM on your local GPU. 🚀
