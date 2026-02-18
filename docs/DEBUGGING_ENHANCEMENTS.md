# Debugging Enhancements for Large Model Performance Issues

## Problem

Large models (70B+ parameters) appear to load successfully but then hang during inference. The log shows:
- ✅ Model loads successfully
- ✅ Test inference completes
- ❌ Then appears to hang (no further output)

## Debugging Added

### 1. HuggingFace Inference Timing (`src/agent.py`)

Added comprehensive timing and progress tracking to `_call_llm()` for HuggingFace models:

**What's tracked:**
- Tokenization time and token count
- Input device placement
- GPU memory before/after generation
- Generation time (with warnings if >5 minutes)
- Decoding time
- Total inference time
- Response length

**Example output:**
```
[DEBUG] Starting HuggingFace inference...
[DEBUG] Retrieved tokenizer, model, device (0.01s)
[DEBUG] Tokenizing input (prompt length: 1234 chars)...
[DEBUG] Tokenization complete (0.05s, 456 tokens)
[DEBUG] Moving inputs to device...
[DEBUG] First model parameter is on: cuda:0
[DEBUG] Inputs moved to cuda:0 (0.02s)
[DEBUG] GPU 0 before generation: 25.3GB allocated, 28.1GB reserved
[DEBUG] GPU 1 before generation: 18.7GB allocated, 20.2GB reserved
[DEBUG] Starting model.generate() with max_new_tokens=512...
[DEBUG] Generation complete (45.2s, 0.8 min)  # <-- This is where it might hang
[DEBUG] GPU 0 after generation: 26.1GB allocated, 28.9GB reserved
[DEBUG] GPU 1 after generation: 19.2GB allocated, 20.8GB reserved
[DEBUG] Decoding response...
[DEBUG] Decoding complete (0.01s)
[DEBUG] Total inference time: 45.3s (0.8 min)
[DEBUG] Response length: 127 chars
```

### 2. Experiment Runner Step Tracking (`src/experiment_runner.py`)

Added detailed logging for each simulation step:

**What's tracked:**
- Observation retrieval
- Agent decision time (with timing)
- Action execution
- Step progress indicators

**Example output:**
```
[*] Starting simulation (max_steps=500)...
[DEBUG] Step 0: Getting observation...
[DEBUG] Step 0: Observation received (day=0, budget=$10000.00)
[DEBUG] Step 0: Calling agent.get_action_and_prediction()...
[DEBUG] Starting HuggingFace inference...  # <-- From agent.py
... (HuggingFace debugging from above) ...
[DEBUG] Step 0: Agent decision received (47.5s, 0.8 min)
[DEBUG] Step 0: Action: tool_check_storage
  Step 0: action=tool_check_storage
[DEBUG] Step 0: Executing action in environment...
[DEBUG] Step 0: Action executed, result received
```

### 3. Safety Limits

Added safety measures to prevent infinite hangs:

**Changes:**
- **Max tokens cap**: Limited to 512 tokens (was 2000) for initial debugging
  - Reduces generation time for testing
  - Can increase after verifying it works
- **Truncation**: Added `max_length=2048` to tokenization
  - Prevents extremely long prompts from causing issues

## What to Look For in Logs

### ✅ Good Signs
```
[DEBUG] Generation complete (X.Xs, Y.Y min)  # Any time under 2 min is good
[DEBUG] Step 0: Agent decision received (X.Xs, Y.Y min)
[DEBUG] Step 0: Action: tool_check_storage
```

### ⚠️ Warning Signs
```
[DEBUG] Starting model.generate() with max_new_tokens=512...
# <-- If this is the last line, it's hanging during generation
```

### ❌ Problem Indicators

1. **Hangs at `model.generate()`**: Last line is "Starting model.generate()"
   - **Cause**: Insufficient GPU memory causing extreme slowdown
   - **Solution**: Reduce max_new_tokens further (try 256 or 128)

2. **Very long generation times** (>5 minutes per step):
   - **Cause**: Heavy CPU offloading (model too large for GPUs)
   - **Solution**: Use smaller model or request more GPUs

3. **OOM errors** during generation:
   - **Cause**: Memory spikes during generation (KV cache)
   - **Solution**: Reduce batch size, use gradient checkpointing

## Testing Strategy

### 1. Quick Test (Small Tokens)
```bash
# Already done: max_new_tokens=512
sbatch --array=0 slurm/run_phase1_large.sh
```

Check log for:
- How long does first generation take?
- Does it complete or hang?

### 2. If Still Hangs

Try reducing tokens further by editing `src/agent.py` line ~445:
```python
max_new_tokens=min(self.max_tokens, 128),  # Even smaller
```

### 3. If Completes But Very Slow (>2 min per step)

Options:
1. **Accept slowness**: 70B model with CPU offload is inherently slow
2. **Use quantization**: Add `load_in_8bit=True` to reduce memory
3. **Skip largest models**: Remove 70B+ from experiments
4. **Request more resources**: Change to 4 GPUs in SLURM script

## Memory Analysis

If you see in logs:
```
[DEBUG] GPU 0 before generation: 35.0GB allocated, 37.0GB reserved
```

This means GPU is nearly full (37GB/40GB). Generation will need extra memory for:
- **KV cache**: ~2-5GB for 70B model
- **Activation buffers**: ~1-3GB
- **Total headroom needed**: ~8-10GB

**If GPU is >90% full before generation, it will be VERY slow or OOM.**

## Expected Performance

Based on model size and setup:

| Model | GPU Usage | Expected Gen Time | Status |
|-------|-----------|-------------------|--------|
| **7B** | 1 GPU, ~14GB | 2-5s per step | ✅ Fast |
| **20B** | 1-2 GPUs, ~40GB | 5-15s per step | ✅ Good |
| **32B** | 2 GPUs, ~64GB | 15-30s per step | ⚠️ Acceptable |
| **70B** | 2 GPUs + CPU offload | 1-3 min per step | ⚠️ Slow |
| **236B** | Mostly CPU/disk | 5-15 min per step | ❌ Too slow |

## Next Steps

1. **Check latest log** for where it hangs:
   ```bash
   tail -50 logs/slurm-1707070_5.out
   ```

2. **Look for last DEBUG line** - this tells you where it got stuck

3. **If hanging at generation:**
   - Model is too large for available resources
   - Try with smaller max_new_tokens (edit code to 128)
   - Or skip 70B+ models for now

4. **If completing but slow:**
   - This is expected for 70B+ models
   - Consider limiting experiments to 7B, 20B, 32B models only
   - Or increase timeout in SLURM script (currently 6 hours)

## Quick Fixes

### Fix 1: Reduce Token Limit (Already Applied)
Location: `src/agent.py` line ~445
```python
max_new_tokens=min(self.max_tokens, 512),  # Was 2000
```

### Fix 2: If Still Slow, Try Greedy Decode Only
Location: `src/agent.py` line ~447
```python
temperature=0.0,           # Force greedy (no sampling)
do_sample=False,           # Disable sampling
```

### Fix 3: Enable 8-bit Quantization (Reduces Memory 50%)
Location: `src/agent.py` line ~250 (in model loading)
```python
model_kwargs = {
    "torch_dtype": dtype,
    "load_in_8bit": True,  # ADD THIS LINE
    "low_cpu_mem_usage": True,
    ...
}
```

This will reduce memory by ~50% but slightly reduce quality.
