# VendoMini — Prediction Error Crash Simulation

[![Tests](https://github.com/mohanwugupta/VendoMini/actions/workflows/ci.yml/badge.svg)](https://github.com/mohanwugupta/VendoMini/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Topics:** `llm-safety` · `agent-evaluation` · `prediction-error` · `crash-detection` · `survival-analysis` · `warehouse-simulation` · `slurm` · `joblib`

---

## What This Is

**VendoMini** is a controlled simulation framework for studying how accumulated **prediction errors (PEs)** cause LLM-based agents to catastrophically fail ("crash"). An agent manages a warehouse — ordering inventory, fulfilling customer orders, tracking budgets — while the environment silently injects stochastic shocks that corrupt the agent's beliefs about the world. The central question:

> *Does accumulated mismatch between an LLM's world model and ground truth causally induce failure, independent of context length?*

This is a safety-relevant question. Long-horizon agent deployments regularly show vivid derailments — looping, contradictory actions, budget denial — that do **not** align with context windows becoming full. VendoMini isolates the PE mechanism with direct causal control so it can be measured rigorously.

---

## Why It Matters for Safety

Current LLM agent evaluations mostly measure *task success at episode end*. They rarely ask *when and why* agents break down during execution, or what internal belief states precede failure. VendoMini contributes:

1. **Causal control of PE induction** — shock frequency, magnitude, type, and observability are independent variables, not confounds.
2. **Falsifiable agent beliefs** — agents must submit a *prediction card* before every action. Their expectations are logged and scored against actual outcomes.
3. **Typed crash taxonomy** — six distinct failure modes (looping, invalid-action burst, budget denial, decoupling, exploration collapse, slow divergence) are detected with windowed heuristics.
4. **Survival-curve framing** — primary metric is *time-to-crash*, enabling dose–response and model-sweep comparisons using standard survival statistics.

---

## Current Status

> **Brutally honest accounting — as of May 2026**

| Component | Status | Notes |
|---|---|---|
| Simulation environment (`env.py`) | ✅ Done | 6 shock types, customer orders, budget tracking |
| Prediction card protocol | ✅ Done | Structured JSON; required before every action |
| PE calculator — EWMA accumulators | ✅ Done | Typed (temporal, quantity, cost, causal); fast/med/slow scales |
| Crash detector — 6 failure modes | ✅ Done | Windowed heuristics; soft/hard/mission-abandon severity |
| LLM agent interface | ✅ Done | OpenAI, Anthropic, HuggingFace, Mock |
| Constrained decoding (vLLM guided JSON) | ✅ Done | Enforces action schema at token level |
| Config system — YAML + grid expansion | ✅ Done | Inheritance, dot-notation overrides |
| Parallelism — Joblib local | ✅ Done | `--n-jobs N` |
| SLURM cluster orchestration | ✅ Done | Array jobs, phase splits, checkpoint/resume |
| Test suite | ✅ Done | `pytest tests/` — unit + integration |
| CI (GitHub Actions) | ✅ Done | Python 3.10 & 3.11 |
| Phase 1–5 experiment configs | ✅ Done | ~10,560 total runs defined across 5 phases |
| Data collection (Phases 1–5) | 🔄 In progress | Running on cluster |
| Survival analysis pipeline | 🔄 In progress | Scripts exist; pending full data |
| Visualization dashboard | ❌ Planned | Streamlit; not started |
| Paper / preprint | ❌ Planned | Pending analysis |

---

## Experimental Design at a Glance

The full design space has five phases (~10,560 runs across 6 models, 5 random seeds each):

| Phase | Focus | Key factors |
|---|---|---|
| 1 | Dose–response & prediction-mode ablation | `p_shock` × `pe_mag` × `prediction_mode` |
| 2 | PE-type × observability | `pe_type_mix` × `observability` |
| 3 | Complexity scaling + recovery tools | `complexity_level` × `recovery_tools` |
| 4 | Model architecture sweep | 6 models × `pe_mag` |
| 5 | Long-horizon extremes | `max_steps` up to 5000 |

**Controllable factors (selection):**

```
pe_induction:
  p_shock:       [0, 0.05, 0.10, 0.20, 0.35]   # shock injection rate
  pe_mag:        [low, medium, high]
  pe_type_mix:   [realistic, temporal_only, quantity_only, causal_only, uniform]
  observability: [full, delayed, partial, hidden]

agent.interface:
  prediction_mode:   [required, optional, required+confidence, required+full]
  prediction_format: [minimal, structured, rich]
  recovery_tools:    [none, reset, audit, help, all]
```

---

## How It Works

```
Configure (YAML grid)
       │
       ▼
ExperimentRunner ──── expands grid ──── N run configs
       │
       ▼  (parallel via Joblib / SLURM array)
  Per-run loop
       │
       ├─ Environment step: shocks injected, state updated
       ├─ Agent: reads state → submits prediction card → calls tool
       ├─ PE Calculator: scores prediction vs. actual; updates EWMA
       ├─ Crash Detector: checks 6 windowed heuristics
       └─ Logger: writes steps.jsonl + summary.json
       │
       ▼
Aggregation scripts → CSV → survival plots
```

**Prediction card example (structured format):**

```json
{
  "tool": "tool_order",
  "args": { "supplier_id": "S1", "sku": "keyboard", "quantity": 10 },
  "expected_delivery_day": 45,
  "expected_quantity": 10,
  "expected_cost": 150.0,
  "expected_storage_after": 50,
  "expected_budget_after": 350.0,
  "expected_success": true,
  "prediction_text": "S1 delivers keyboards in ~3 days at $15/unit"
}
```

The PE calculator scores this against the actual environment outcome and updates fast (α=0.3), medium (α=0.1), and slow (α=0.01) EWMA accumulators per error type.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/mohanwugupta/VendoMini.git
cd VendoMini
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the demo (mock model, no API key needed, ~15 steps)
python run_experiment.py --config demo/demo_config.yaml --n-jobs 1
```

Or in one command (Linux/macOS):

```bash
bash demo/run_demo.sh
```

**Analyze results:**

```bash
python scripts/aggregate_results.py --input-dir results/demo --output results/demo.csv
python scripts/analyze_results.py --results results/demo.csv
```

---

## Repository Layout

```
src/
├── env.py               # Warehouse simulation & state machine
├── agent.py             # LLM interface (OpenAI, Anthropic, HuggingFace, Mock)
├── pe_calculator.py     # Typed PE computation + multi-scale EWMA
├── crash_detector.py    # Six failure-mode detectors
├── experiment_runner.py # Grid expansion, Joblib/SLURM orchestration
├── config.py            # YAML loader with inheritance & grid expansion
└── logging_utils.py     # Structured JSONL logging

configs/
├── base.yaml            # Default hyperparameters
├── local_test.yaml      # Quick local run
└── phases/              # Phase 1–5 experiment grids (~10k runs)

tests/                   # pytest unit + integration tests
scripts/                 # Result aggregation & survival analysis
docs/                    # Design docs (PRD, phase summaries, cluster guides)
```

---

## Code Quality

Formatting is enforced automatically:

```bash
black src/ tests/ run_experiment.py       # PEP 8, 88-char lines
ruff check src/ tests/ run_experiment.py  # Linting + import sorting
npx prettier --write "configs/**/*.yaml"  # YAML normalization
```

Configuration lives in [`pyproject.toml`](pyproject.toml).

---

## Citation

```bibtex
@software{gupta2025vendomini,
  author  = {Gupta, Mohan},
  title   = {{VendoMini}: Prediction Error Crash Simulation},
  year    = {2025},
  url     = {https://github.com/mohanwugupta/VendoMini}
}
```

---

## License

MIT. Author: Mohan Gupta
