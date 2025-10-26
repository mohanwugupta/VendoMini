# Multi-GPU Configuration for Large Language Models

## Problem Summary

Your VendoMini experiment uses 6 different language models ranging from 7B to 236B parameters:

1. `openai/gpt-oss-20b` (20B params) ≈ 40GB
2. `deepseek-ai/DeepSeek-V2.5` (236B) ≈ 500GB  
3. `meta-llama/Llama-3.3-70B-Instruct` (70B) ≈ 140GB
4. `Qwen/Qwen2.5-72B-Instruct` (72B) ≈ 144GB
5. `Qwen/Qwen3-32B` (32B) ≈ 64GB
6. `deepseek-ai/deepseek-llm-7b-chat` (7B) ≈ 14GB

**Previous setup**: 1 GPU with 40GB VRAM
**Problem**: Models >20B parameters cannot fit on a single 40GB GPU

## Solutions Implemented

### 1. Multi-GPU Support in SLURM Scripts

All phase scripts now request **2 GPUs** instead of 1:

```bash
#SBATCH --gres=gpu:2          # Request 2 GPUs (80GB total VRAM)
#SBATCH --cpus-per-task=4     # More CPUs for data loading
#SBATCH --mem-per-cpu=32G     # 128GB total RAM
```

**Benefits**:
- 80GB total VRAM (2 × 40GB GPUs)
- Can fit models up to ~70B parameters
- Better parallelism for data loading

### 2. Enhanced Model Loading in `src/agent.py`

#### Device Map Strategy
Changed from `device_map="sequential"` to `device_map="auto"`:
- **Auto mode**: Optimally distributes model layers across GPUs and CPU
- **Smart allocation**: Places frequently-used layers on GPU, overflow on CPU
- **Memory efficient**: Uses CPU RAM when GPU VRAM is full

#### CPU Offloading
Added CPU offloading for layers that don't fit on GPUs:

```python
max_memory = {
    0: "35GB",  # GPU 0: 90% of 40GB
    1: "35GB",  # GPU 1: 90% of 40GB  
    "cpu": "120GB"  # CPU RAM for overflow
}

model_kwargs = {
    "device_map": "auto",
    "max_memory": max_memory,
    "offload_folder": "./offload",  # Disk offload for very large models
    "offload_state_dict": True  # Reduces peak memory during loading
}
```

#### Memory Management Improvements
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Reduces memory fragmentation
- `torch.cuda.empty_cache()`: Clears GPU cache before loading
- **Dynamic allocation**: Uses 90% of actual GPU memory (not hardcoded 75GB)
- **Multi-GPU detection**: Automatically detects and uses all available GPUs

### 3. Model Size Compatibility

With 2 × 40GB GPUs (80GB total VRAM) + CPU offloading:

| Model | Params | Memory | Status |
|-------|--------|--------|--------|
| deepseek-llm-7b-chat | 7B | 14GB | ✅ Fits easily |
| openai/gpt-oss-20b | 20B | 40GB | ✅ Fits on 1-2 GPUs |
| Qwen3-32B | 32B | 64GB | ✅ Fits with CPU offload |
| Llama-3.3-70B | 70B | 140GB | ⚠️ Needs CPU offload |
| Qwen2.5-72B | 72B | 144GB | ⚠️ Needs CPU offload |
| DeepSeek-V2.5 | 236B | 500GB | ⚠️ Requires extensive CPU/disk offload (will be slow) |

**Note**: Models marked ⚠️ will have some layers on CPU, which will slow down inference but should work.

## Testing Recommendations

### Quick Test (7B model)
```bash
sbatch slurm/run_phase1.sh  # Uses deepseek-llm-7b-chat
```
Check logs for:
- `[*] GPU 0: ... GB total memory`
- `[*] GPU 1: ... GB total memory`
- `[*] Setting max_memory: {0: '35GB', 1: '35GB', 'cpu': '120GB'}`
- `[*] Model loaded successfully!`

### Medium Test (20B-32B models)
Once 7B works, try `openai/gpt-oss-20b` or `Qwen3-32B`:
- Should see model distributed across both GPUs
- Check with `nvidia-smi` during run: both GPUs should show memory usage

### Large Model Test (70B+ models)
For `Llama-3.3-70B` or `Qwen2.5-72B`:
- Expect slower loading (layers being offloaded to CPU)
- Watch for warnings about CPU offloading
- **First generation will be slow** as weights transfer between GPU/CPU

### Very Large Model (236B)
`DeepSeek-V2.5` will be **very slow**:
- Most layers will be on CPU or disk
- Consider skipping this model if inference is too slow
- Alternative: Use 8-bit quantization (add `load_in_8bit=True` to model_kwargs)

## Alternative: 8-Bit Quantization

If CPU offloading is too slow, you can enable 8-bit quantization for large models:

Edit `src/agent.py` around line 195:
```python
if "236B" in model_to_load or "72B" in model_to_load or "70B" in model_to_load:
    model_kwargs["load_in_8bit"] = True  # ~50% memory reduction
    print(f"[*] Enabling 8-bit quantization for large model")
```

This reduces memory by ~50%:
- 70B model: 140GB → 70GB (fits on 2 GPUs!)
- 236B model: 500GB → 250GB (still needs CPU offload but faster)

## Monitoring GPU Usage

While job is running, SSH to compute node and run:
```bash
watch -n 1 nvidia-smi
```

You should see:
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     12345      C   python                          35000MiB |
|    1   N/A  N/A     12345      C   python                          32000MiB |
+-----------------------------------------------------------------------------+
```

## Troubleshooting

### Still getting OOM errors
1. Check actual GPU memory: `nvidia-smi`
2. Reduce max_memory percentage in `agent.py` from 0.9 to 0.85
3. Enable 8-bit quantization for large models

### Models loading too slowly
1. CPU offloading is happening - check logs for device_map
2. Consider using smaller models for initial testing
3. Use nodes with more GPUs if available

### "offload folder not found" error
The script will create `./offload/` automatically, but ensure write permissions:
```bash
mkdir -p ./offload
chmod 755 ./offload
```

## Files Modified

1. **All SLURM scripts** (`slurm/run_phase*.sh`):
   - Changed `--gres=gpu:1` → `--gres=gpu:2`
   - Changed `--cpus-per-task=1` → `--cpus-per-task=4`
   - Changed `--mem-per-cpu=128G` → `--mem-per-cpu=32G`

2. **`src/agent.py`**:
   - Changed `device_map="sequential"` → `device_map="auto"`
   - Added CPU offloading: `max_memory["cpu"] = "120GB"`
   - Added disk offloading: `offload_folder="./offload"`
   - Added `offload_state_dict=True` for lower peak memory

## Next Steps

1. **Test with small model first**: Run phase1 with `deepseek-llm-7b-chat`
2. **Monitor resource usage**: Check GPU memory, CPU usage, and inference speed
3. **Adjust if needed**: 
   - If too slow: Enable 8-bit quantization
   - If still OOM: Increase CPU RAM allocation in SLURM
   - If working well: Run full experiment suite!

Good luck! 🚀
