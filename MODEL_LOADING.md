# HuggingFace Model Loading on Cluster

## Overview

VendoMini uses HuggingFace Transformers models loaded from a centralized cache directory on the cluster. The system is designed to work with **one model at a time** that you pre-download to the cluster.

## Cluster Setup

### Environment Variables (set in `slurm/run_phase1.sh`):

```bash
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
export HUGGINGFACE_HUB_CACHE=/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
```

All variables point to the same directory: `/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models`

## How Model Loading Works

### 1. Model Detection (`src/agent.py`)

```python
def _detect_provider(self) -> str:
    """Detect which LLM provider to use based on model name."""
    model_lower = self.model_name.lower()
    
    if 'gpt' in model_lower or 'o1' in model_lower:
        return 'openai'
    elif 'claude' in model_lower:
        return 'anthropic'
    elif '/' in self.model_name:  # HuggingFace format: org/model
        return 'huggingface'
    else:
        return 'openai'  # Default to OpenAI-compatible
```

**Your models** (e.g., `llama-3.1-70b`) don't have `/` in the name, so they'll be detected as 'openai' provider by default unless they contain 'llama', 'mistral', etc.

### 2. HuggingFace Model Loading

When `provider == 'huggingface'`:

```python
# HuggingFace will automatically use cache if available
model_to_load = self.model_name  # e.g., "meta-llama/Llama-3.1-70B"

tokenizer = AutoTokenizer.from_pretrained(
    model_to_load,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_to_load,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
```

**Key Points:**
- No manual path construction needed
- HuggingFace automatically checks `TRANSFORMERS_CACHE` → `HF_HOME` → default cache
- Will use cached model if found, otherwise downloads (if network available)

### 3. Diagnostic Output

The code now provides detailed logging:

```
[*] Loading HuggingFace model: meta-llama/Llama-3.1-70B
[*] HF_HOME: /scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
[*] TRANSFORMERS_CACHE: /scratch/gpfs/JORDANAT/mg9965/prompt_patching/models
[*] Loading tokenizer...
[*] Tokenizer loaded successfully
[*] Device: cuda
[*] Loading model weights...
[*] Model loaded successfully!
```

Or if it fails:

```
[ERROR] Failed to load HuggingFace model: <error>
<full traceback>
```

## Configuration

### Config File (`configs/base.yaml`):

```yaml
agent:
  model:
    name: meta-llama/Llama-3.1-70B  # Use HuggingFace format: org/model
    context_length: 32000
    temperature: 0.3
    max_tokens_per_call: 2000
```

**Important:** Use the **HuggingFace model identifier** (with `/`) so the provider detection works correctly.

### Common Model Names:

- `meta-llama/Llama-3.1-70B`
- `meta-llama/Llama-3.2-3B`
- `mistralai/Mistral-7B-v0.1`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Pre-downloading Models

To pre-download a model to the cluster cache:

```bash
# SSH to cluster
ssh della.princeton.edu

# Activate environment
conda activate vendomini

# Set cache directory
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/prompt_patching/models

# Download model (Python)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'meta-llama/Llama-3.1-70B'
print(f'Downloading {model_name}...')
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='float16')
print('Download complete!')
"
```

Or use HuggingFace CLI:

```bash
huggingface-cli download meta-llama/Llama-3.1-70B
```

## Fallback Behavior

If the model fails to load (missing dependencies, wrong name, etc.):

```python
if agent.client is None:
    print(f"[WARNING] LLM client is None - agent will use pathological heuristic fallback!")
    print(f"[WARNING] This means NO actual LLM is being used for decision-making")
```

The agent will fall back to a **pathological heuristic** that:
- Always calls `tool_check_inbox`
- Will trigger the looping crash detector after 5 steps
- This is **intentional** for the research - you want to see crashes!

## Troubleshooting

### Model Not Loading?

1. **Check model name format:**
   ```yaml
   # ✅ Correct (HuggingFace format)
   name: meta-llama/Llama-3.1-70B
   
   # ❌ Wrong (not detected as HuggingFace)
   name: llama-3.1-70b
   ```

2. **Check environment variables in SLURM output:**
   ```
   [*] HF_HOME: /scratch/gpfs/.../models
   [*] TRANSFORMERS_CACHE: /scratch/gpfs/.../models
   ```

3. **Check if model exists in cache:**
   ```bash
   ls -la /scratch/gpfs/JORDANAT/mg9965/prompt_patching/models/models--*/
   ```

4. **Check for error messages:**
   ```
   [ERROR] Failed to load HuggingFace model: ...
   ```

### Still Using Heuristic?

If you see:
```
[WARNING] LLM client is None - agent will use pathological heuristic fallback!
```

Then the model didn't load. Check:
- Model name in config
- HuggingFace cache directory has the model
- GPU availability (`Device: cuda`)
- Dependencies installed (`transformers`, `torch`, `accelerate`)

## Research Design Note

**The goal is to study agent failures under prediction error**, so:
- ✅ Crashes are expected and desired
- ✅ Looping behavior is a valid research outcome
- ✅ The heuristic fallback should be pathological (not smart)
- ❌ Don't "fix" crashes - that defeats the research purpose!

The experiments measure:
- Time to crash
- Crash type (looping, invalid actions, budget issues, etc.)
- Prediction error accumulation before crash
- Different crash patterns under different PE induction conditions
