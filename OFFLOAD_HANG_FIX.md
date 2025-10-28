# Critical Fix: Model Initialization Hang

## TL;DR
**Problem**: Large models (70B+) hang inside `AutoModelForCausalLM.from_pretrained()` after loading checkpoint shards  
**Root Cause**: `offload_state_dict=True` creates meta device parameters that never initialize properly  
**Solution**: Remove `offload_folder` and `offload_state_dict` parameters - let accelerate handle offloading automatically

## The Hang

**What we saw in logs**:
```
[DEBUG] About to call AutoModelForCausalLM.from_pretrained()...
Loading checkpoint shards: 100%|██████████| 30/30 [01:59<00:00,  3.98s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
The following generation flags are not valid and may be ignored: ['temperature', 'top_p'].
<HANGS FOREVER - never returns from from_pretrained()>
```

**What SHOULD happen**:
```
[DEBUG] About to call AutoModelForCausalLM.from_pretrained()...
Loading checkpoint shards: 100%|██████████| 30/30 [01:59<00:00,  3.98s/it]
[DEBUG] AutoModelForCausalLM.from_pretrained() returned!  <-- THIS NEVER APPEARED
[*] Model loaded successfully!
```

## Changes Made

### 1. Removed Aggressive Offloading (src/agent.py)
```diff
- model_kwargs["offload_folder"] = "./offload"
- model_kwargs["offload_state_dict"] = True
```

**Result**: CPU offloading still works (via `device_map="auto"`) but doesn't create meta device parameters

### 2. Increased GPU Memory Allocation (src/agent.py)
```diff
- max_memory = {i: f"{int(gpu_memory[i] * 0.85)}GB" for i in range(num_gpus)}
- max_memory["cpu"] = "120GB"
+ max_memory = {i: f"{int(gpu_memory[i] * 0.90)}GB" for i in range(num_gpus)}
+ max_memory["cpu"] = "150GB"
```

**Result**: 
- 79GB GPUs → 71GB allocated per GPU (was 67GB)
- Less CPU offloading needed → simpler device mapping
- Still 8GB headroom for activations/KV cache

### 3. Simplified Generation Config (src/agent.py)
```diff
- from transformers import GenerationConfig
- model.generation_config = GenerationConfig.from_pretrained(model_to_load, local_files_only=True)
+ # Set generation config directly instead of loading from pretrained
+ if hasattr(model, 'generation_config'):
+     if model.generation_config.pad_token_id is None:
+         model.generation_config.pad_token_id = tokenizer.eos_token_id
```

**Result**: No network calls, no conflicts with meta device parameters

## Why It Was Hanging

1. **`offload_state_dict=True`** tells accelerate to keep model state_dict on disk/CPU during loading
2. This creates **"meta" device placeholders** for parameters
3. After checkpoint shards load, accelerate tries to **move meta parameters to real devices**
4. But with complex multi-GPU + CPU setup, this **synchronization gets stuck**
5. The function never returns, no error is raised, just infinite wait

## Expected Behavior Now

```
[*] Max memory allocation: {0: '71GB', 1: '71GB', 'cpu': '150GB'}
[*] Using auto device mapping with max_memory constraints
[*] CPU offloading will be used automatically if model doesn't fit on GPUs
[DEBUG] About to call AutoModelForCausalLM.from_pretrained()...
Loading checkpoint shards: 100%|██████████| 30/30 [01:59<00:00,  3.98s/it]
[DEBUG] AutoModelForCausalLM.from_pretrained() returned!  ✅ NOW THIS APPEARS
[*] Model loaded successfully!
[DEBUG] Model class: LlamaForCausalLM
[*] GPU 0 memory: XX.XXGB allocated, XX.XXGB reserved, 79.25GB total
[*] GPU 1 memory: XX.XXGB allocated, XX.XXGB reserved, 79.25GB total
[*] Testing model with small inference...
[*] Test inference successful!
```

## Test Command

```bash
cd /Users/mg9965/Library/CloudStorage/Box-Box/ResearchProjects/VendoMini
python run_experiment.py --config configs/local_test.yaml
```

Or on cluster:
```bash
sbatch slurm/submit_phase1_large.sh
```

## Key Insight

**You don't need `offload_state_dict=True` for CPU offloading to work!**

Just using `device_map="auto"` with `max_memory` constraints is enough:
- Accelerate will automatically put layers on CPU if they don't fit on GPU
- But it will initialize them properly, not as meta device placeholders
- This avoids the synchronization deadlock

## If This Doesn't Work

If it still hangs, try these progressively simpler approaches:

1. **Remove max_memory entirely** - let accelerate figure it out:
   ```python
   model_kwargs["device_map"] = "auto"
   # Remove max_memory parameter
   ```

2. **Use sequential device mapping** instead of auto:
   ```python
   model_kwargs["device_map"] = "sequential"
   ```

3. **Load in 8-bit** to reduce memory needs:
   ```python
   model_kwargs["load_in_8bit"] = True
   ```

But based on the symptoms, removing `offload_state_dict=True` should fix it!
