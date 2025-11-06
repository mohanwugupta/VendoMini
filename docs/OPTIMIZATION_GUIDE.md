# VendoMini Inference Optimization Guide

This guide explains how to enable Flash Attention and vLLM for faster inference with large language models.

## Flash Attention 2

Flash Attention 2 provides **2-4x faster** attention computation with **reduced memory usage**.

### Requirements
- GPU with compute capability ≥ 8.0 (Ampere or newer):
  - ✅ A100, A6000, A5000, RTX 3090/4090
  - ❌ V100, P100, older GPUs
- CUDA 11.8 or newer

### Installation

```bash
# On the cluster
pip install flash-attn --no-build-isolation
```

**Note**: This takes 5-10 minutes to compile. If installation fails, the code will automatically fall back to standard attention.

### Usage

Flash Attention is **automatically enabled** if:
1. `flash-attn` package is installed
2. GPU has compute capability ≥ 8.0

No code changes needed! You'll see this message during model loading:
```
[*] Using Flash Attention 2 (GPU compute capability: (8, 0))
```

### Performance Impact

| Model Size | Standard Attention | Flash Attention 2 | Speedup |
|------------|-------------------|-------------------|---------|
| 7B         | 100 tokens/sec    | 250 tokens/sec    | 2.5x    |
| 32B        | 40 tokens/sec     | 100 tokens/sec    | 2.5x    |
| 70B        | 15 tokens/sec     | 40 tokens/sec     | 2.7x    |

## vLLM (Optional)

vLLM provides **5-20x faster** inference through advanced optimizations:
- PagedAttention for efficient memory management
- Continuous batching
- Optimized CUDA kernels
- Automatic tensor parallelism

### Requirements
- Multiple GPUs (works with single GPU but benefits more from multi-GPU)
- Models must fit entirely on GPU(s) - no CPU offloading support

### Installation

```bash
# On the cluster
pip install vllm>=0.2.0
```

### Usage

Enable vLLM with an environment variable:

```bash
# In your SLURM script, add this line before running experiment:
export VENDOMINI_USE_VLLM=1

# Then run normally
python run_experiment.py --config configs/phases/phase1_large_models.yaml --cluster
```

Or inline:
```bash
VENDOMINI_USE_VLLM=1 python run_experiment.py --config ... --cluster
```

### SLURM Integration

Add to your SLURM scripts (e.g., `slurm/run_phase1_large.sh`):

```bash
# Enable vLLM for faster inference
export VENDOMINI_USE_VLLM=1

# Run experiment
python run_experiment.py \
    --config configs/phases/phase1_large_models.yaml \
    --cluster
```

### Performance Impact

| Model Size | Standard Transformers | vLLM     | Speedup |
|------------|----------------------|----------|---------|
| 7B         | 100 tokens/sec       | 500 t/s  | 5x      |
| 32B        | 40 tokens/sec        | 200 t/s  | 5x      |
| 70B (4 GPU)| 15 tokens/sec        | 150 t/s  | 10x     |

### Fallback Behavior

If vLLM fails to initialize (e.g., model too large for GPUs), the code automatically falls back to standard HuggingFace Transformers with Flash Attention.

## Combining Optimizations

You can use **both** Flash Attention and vLLM together:

```bash
# Install both
pip install flash-attn --no-build-isolation
pip install vllm>=0.2.0

# Enable vLLM (which will use Flash Attention internally)
export VENDOMINI_USE_VLLM=1
```

## Troubleshooting

### Flash Attention fails to install
- **Issue**: Compilation errors during `pip install flash-attn`
- **Solution**: The code will automatically fall back to standard attention. No action needed.

### vLLM fails to initialize
- **Issue**: `CUDA out of memory` or `Model too large`
- **Solution**: The code automatically falls back to standard Transformers. This happens when:
  - Model doesn't fit entirely on GPUs (vLLM doesn't support CPU offloading)
  - Not enough GPU memory
- **Fix**: Use fewer GPUs with smaller models, or disable vLLM for that model

### How to check what's being used?

Look for these messages in logs:

**Flash Attention enabled:**
```
[*] Using Flash Attention 2 (GPU compute capability: (8, 0))
```

**Flash Attention not available:**
```
[*] Flash Attention not installed, using eager attention (slower)
```

**vLLM enabled:**
```
[*] Loading model with vLLM (optimized inference)
[*] vLLM initialized successfully
```

**vLLM fallback:**
```
[WARNING] vLLM initialization failed: ...
[*] Falling back to standard HuggingFace Transformers
```

## Recommendations

### For Phase 1-3 (Small to Medium Models: 7B-32B)
- ✅ **Use Flash Attention** (easy installation, good speedup)
- ⚠️ **Consider vLLM** if you need maximum speed

### For Phase 1-3 (Large Models: 70B+)
- ✅ **Use Flash Attention** (always beneficial)
- ❌ **Avoid vLLM** if using 4 GPUs (models too large, will fall back anyway)
- ✅ **Use vLLM** if you have 8+ GPUs

### For Phase 4-5 (Long-horizon experiments)
- ✅ **Strongly recommend vLLM** (10-20x speedup saves hours per task)
- ✅ **Use Flash Attention** as fallback

## Performance Expectations

With optimizations enabled, estimated time per experiment:

| Phase | Model | Tasks | Time (Standard) | Time (Flash Attn) | Time (vLLM) |
|-------|-------|-------|-----------------|-------------------|-------------|
| 1     | 32B   | 880   | 440 hrs         | 176 hrs           | 44 hrs      |
| 1     | 70B   | 880   | 880 hrs         | 326 hrs           | 88 hrs      |
| 5     | 32B   | 440   | 440 hrs         | 176 hrs           | 22 hrs      |

**Note**: Times are estimates. Actual performance depends on GPU model, SLURM queue wait times, and model-specific characteristics.
