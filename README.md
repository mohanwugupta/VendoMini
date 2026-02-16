# VendoMini — Prediction Error Crash Simulation

Abstract
--------
 **VendoMini** is a reproducible simulation framework to study how accumulated prediction errors (PEs) induce catastrophic failures in autonomous agents. In a simulated warehouse environment, agents manage inventory and logistics while encountering stochastic shocks. The project systematically injects prediction errors, tracks them using multi-scale EWMA, and analyzes the "crash" points where agents fail to recover. This codebase serves as an end-to-end engineering demonstration of a reliable, parallelized experimental pipeline using Joblib, Ray, and SLURM.

Key contributions
-----------------
- **Configurable Simulator**: Modular environment for systematic PE induction and stress testing.
- **Robust Engineering**: Full test coverage, type-hinted codebase, and CI-ready workflow.
- **Scalable Orchestration**: Seamless switching between local debugging, multi-core processing, and cluster deployment (SLURM/Ray).
- **Data Pipeline**: Structured logging (JSONL traces) and automated analysis scripts for survival statistics.

Quick facts 
--------------------------------------
- **Language**: Python 3.10+ | **Tests**: `pytest`
- **Architecture**: Modular (Env, Agent, CrashDetector, Runner)
- **Demo**: One-command reproduction (see below)
- **Files to inspect**: 
  - `src/experiment_runner.py` (Orchestration & Parallelism)
  - `src/env.py` (Simulation Logic)
  - `src/config.py` (Configuration Management)

Badges
---------------------------
![Build Status](https://github.com/mohanwugupta/VendoMini/actions/workflows/ci.yml/badge.svg) 
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Quick start (One-command Demo)
-------------------------------------
Run the included demo script to set up a virtual environment, install dependencies, and run a minimal experiment with a mock model.

```bash
# Mac / Linux
bash demo/run_demo.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Run a tiny deterministic test sweep (local, 1 job)
python run_experiment.py --config demo/demo_config.yaml --n-jobs 1 
```

Architecture & Workflow
-------------------------
The system follows a clean "Configure -> Simulate -> Analyze" pipeline:

1. **Configuration**: YAML files define experimental grids (parameters, models, shock intensities).
2. **Orchestration**: `experiment_runner.py` expands the grid and schedules jobs (Local/Ray/SLURM).
3. **Simulation**: `env.py` runs the step-by-step warehouse logic; `agent.py` bridges to LLMs.
4. **Analysis**: Scripts in `scripts/` aggregate JSONL logs into CSV summaries and generate survival plots.

### Key Components

- **`src/`**: Core library code.
  - `config.py` — YAML loader & grid expansion
  - `env.py` — Simulation environment & state machine
  - `agent.py` — LLM interface (OpenAI, Anthropic, HuggingFace, Mock)
  - `pe_calculator.py` — Prediction Error tracking (EWMA)
  - `crash_detector.py` — Failure mode detection logic
- **`configs/`**: Experiment definitions.
- **`tests/`**: Unit and integration tests (`pytest`).

Reproducibility & Evaluation
----------------------------
Each run creates a self-contained output folder with:
- `config_used.yaml`: Exact parameters used.
- `experiment.log`: Runtime logs.
- `summary.json`: High-level metrics (steps survived, total cost).
- `steps.jsonl`: Full per-step trace of state, action, and prediction error.

**Example evaluation:**
```bash
python scripts/aggregate_results.py --input-dir results/demo --output results/demo.csv
python scripts/analyze_results.py --results results/demo.csv
```

What to add next
-----------------------------------------
- **Visualization Dashboard**: A Streamlit app to visualize the step-by-step agent decisions.
- **More Crash Types**: Extending `crash_detector.py` to handle more subtle failure modes.

Citation
--------
If you use VendoMini: Gupta, M. (2025). VendoMini: Prediction Error Crash Simulation.

License
-----------------
MIT. Author: Mohan Gupta
