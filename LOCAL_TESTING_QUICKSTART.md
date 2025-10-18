# VendoMini Local GPU Testing - Quick Start

## ✅ Your System is Working!

You just successfully ran a **full 20-step VendoMini simulation** with a HuggingFace LLM making real decisions!

### Test Results

- ✅ **GPU**: NVIDIA GeForce RTX 3060 (12.9 GB)
- ✅ **Model**: TinyLlama-1.1B loaded successfully
- ✅ **VRAM Usage**: 2.21 GB (plenty of room for bigger models!)
- ✅ **Simulation**: 20 steps completed
- ✅ **LLM Decisions**: Model made tool choices and predictions
- ✅ **Orders Placed**: 3 successful orders
- ✅ **Logs Saved**: `logs/test_run_001_steps.jsonl` and `logs/test_run_001_summary.json`

## What Just Happened?

The LLM agent:
1. ✅ Checked inventory and budget
2. ✅ Made ordering decisions (tool_order with supplier_id, sku, quantity)
3. ✅ Got price quotes (tool_quote)
4. ✅ Monitored inbox messages
5. ✅ Generated predictions for each action
6. ✅ Tracked prediction errors

## Quick Test Command

```powershell
# Run the full test
chcp 65001 > $null; python scripts\test_huggingface.py
```

## Try Better Models

You have **12.9 GB VRAM**, so you can use much better models!

### Edit `scripts\test_huggingface.py` line 67:

```python
# Current (smallest):
'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # 2GB VRAM

# Better quality options:
'name': 'microsoft/phi-2',  # 5GB VRAM ⭐ RECOMMENDED
'name': 'microsoft/Phi-3-mini-4k-instruct',  # 7GB VRAM
```

## Customize Your Test

### Longer Simulation
Edit line 52 in `scripts\test_huggingface.py`:
```python
'max_steps': 50,  # Instead of 20
```

### Different Budget
Edit line 53:
```python
'initial_budget': 5000,  # Instead of 1000
```

### Enable More Shocks
Edit line 70:
```python
'frequency': 0.5,  # 50% chance per step (currently 30%)
```

## View Your Results

```powershell
# View step-by-step log
Get-Content logs\test_run_001_steps.jsonl | Select-Object -First 10

# View summary
Get-Content logs\test_run_001_summary.json | ConvertFrom-Json
```

## Summary

🎉 **You now have a fully functional VendoMini simulation running locally with a real LLM!**

The system successfully:
- Loads HuggingFace models on your GPU
- Runs complete simulations  
- Makes LLM-driven decisions
- Tracks prediction errors
- Saves detailed logs

**Ready for real research!** 🚀

See `LOCAL_TESTING.md` for detailed documentation.
