# VendoMini: Prediction Error Crash Simulation# VendoMini: Prediction Error Crash Simulation



A controlled warehouse/vending simulation to study how accumulated prediction errors (PEs) cause catastrophic failures in LLM agents.**What this is:** A complete, production-ready simulation system to study how prediction errors (PEs) induce catastrophic failures in LLM agents operating in a warehouse/vending environment.



## Overview## Overview



VendoMini implements a warehouse simulation where LLM agents manage inventory, place orders, and make predictions about outcomes. The simulation systematically injects prediction errors to test the hypothesis that accumulated PEs between an agent's world model and reality causally induce crashes ("psychotic breaks").VendoMini implements a controlled environment where LLM agents manage inventory, place orders, and make operational decisions while we systematically inject prediction errors. The system logs multi-scale cumulative PEs and detects various crash types (looping, budget denial, exploration collapse, etc.).



**Key Features:****Key Features:**

- Multi-scale PE tracking (temporal, quantity, cost, causal)- Configurable PE injection (frequency, magnitude, type, observability)

- Multiple crash detection patterns (looping, invalid bursts, budget denial, etc.)- Multi-scale EWMA PE tracking (fast/medium/slow)

- Configurable experiment phases with grid expansion- Multiple crash type detection with severity levels

- **Parallel execution via joblib (local) or SLURM array jobs (cluster)**- Grid expansion for parameter sweeps

- Comprehensive logging and survival analysis- Parallel execution (joblib + Ray for cluster scaling)

- Comprehensive logging and analysis tools

## Quick Start- Full unit test coverage



### Local Installation## Installation



```bash### Prerequisites

# Clone repository- Python 3.10 or higher

git clone https://github.com/mohanwugupta/VendoMini.git- pip or conda

cd VendoMini- (Optional) Ray cluster for distributed execution



# Install dependencies### Setup

pip install -r requirements.txt

1. **Clone or navigate to the repository:**

# Verify installation```bash

python scripts/verify_installation.pycd VendoMini

```

# Run a simple test

python run_experiment.py --config configs/base.yaml --n-jobs 12. **Create and activate a virtual environment:**

``````bash

# Using venv

### Running Experimentspython -m venv venv

.\venv\Scripts\activate  # Windows

**Local execution (small scale):**source venv/bin/activate  # Linux/Mac

```bash

# Run Phase 1 with 4 parallel jobs# Or using conda

python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 4conda create -n vendomini python=3.10

conda activate vendomini

# Results saved to results/phase1_core_hypothesis_results.csv```

```

3. **Install dependencies:**

**Cluster execution (SLURM - massive parallelization):**```bash

```bashpip install -r requirements.txt

# Submit Phase 1 as array job (180 tasks run in parallel)

sbatch slurm/run_phase1.sh# Or install in development mode

pip install -e .

# Submit all phases```

bash slurm/submit_all_phases.sh

4. **(Optional) Set up LLM API keys:**

# Monitor jobs```bash

squeue -u $USER# For OpenAI

$env:OPENAI_API_KEY="your-key-here"  # Windows PowerShell

# After completion, aggregate resultsexport OPENAI_API_KEY="your-key-here"  # Linux/Mac

python scripts/aggregate_results.py --input-dir results --output results/phase1_all.csv

```# For Anthropic

$env:ANTHROPIC_API_KEY="your-key-here"  # Windows PowerShell

## Cluster Usage (SLURM)export ANTHROPIC_API_KEY="your-key-here"  # Linux/Mac

```

VendoMini uses **SLURM array jobs** for massive parallelization, similar to your DRM experiments:

## Quick Start

### Setup

### 1. Run Tests

1. **Create conda environment on cluster:**

```bashVerify installation by running the test suite:

module load anaconda3/2024.2

conda create -n vendomini python=3.10```bash

conda activate vendomini# Run all tests

pip install -r requirements.txtpython scripts/run_tests.py

```

# Or run directly with pytest

2. **Update SLURM scripts (`slurm/run_phase1.sh`):**pytest tests/ -v

```bash```

#SBATCH --mail-user=your-email@university.edu  # Update email

#SBATCH --array=0-179  # 180 parallel tasks for Phase 1### 2. Run a Small Experiment



# Update pathsTest the system with a minimal configuration:

cd /scratch/gpfs/username/vendomini  # Your cluster path

``````bash

python run_experiment.py --config configs/base.yaml --n-jobs 1

3. **Submit jobs:**```

```bash

# Single phase### 3. Run Phase 1 Experiment

sbatch slurm/run_phase1.sh

Run the first phase (dose-response analysis):

# All phases at once

bash slurm/submit_all_phases.sh```bash

```# Sequential (1 job)

python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 1

### How Array Jobs Work

# Parallel (8 jobs)

- Each SLURM script defines `--array=0-N` where N = (total parameter combinations) - 1python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 8

- Each array task runs ONE experiment (one config combo + replication)  

- Tasks run **in parallel** across cluster nodes# Parallel on cluster (32 jobs with Ray)

- Results saved independently: `results/vendomini_task_0000.json`, `task_0001.json`, etc.python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 32 --cluster

```

**Example: Phase 1**

- Grid: 3 p_shock × 2 pe_mag × 2 pred_mode × 3 models × 5 reps = 180 combinations### 4. Analyze Results

- SLURM: `--array=0-179` creates 180 parallel tasks

- Each task: ~1 hour```bash

- Total wall time: ~1 hour (vs 180 hours sequential!)python scripts/analyze_results.py --results results/phase1_core_hypothesis_results.csv

```

### Aggregating Results

## Project Structure

After SLURM jobs complete:

```

```bashVendoMini/

python scripts/aggregate_results.py \├── src/                      # Source code

    --input-dir results \│   ├── __init__.py

    --output results/phase1_all_results.csv \│   ├── agent.py             # LLM agent interface

    --prefix vendomini_task│   ├── config.py            # Configuration loader with grid expansion

```│   ├── crash_detector.py   # Crash detection logic

│   ├── env.py               # VendoMini simulation environment

This merges all `vendomini_task_XXXX.json` files into one CSV with summary statistics.│   ├── experiment_runner.py # Parallel experiment execution

│   ├── logging_utils.py     # Logging and result aggregation

## Configuration│   └── pe_calculator.py     # Prediction error calculation

├── tests/                    # Unit and integration tests

Configurations use YAML with inheritance and grid expansion:│   ├── test_config.py

│   ├── test_crash_detector.py

```yaml│   ├── test_env.py

inherit: ../base.yaml  # Inherit from parent config│   ├── test_integration.py

│   └── test_pe_calculator.py

experiment:├── scripts/                  # Utility scripts

  name: "phase1_core_hypothesis"│   ├── run_tests.py         # Test runner with coverage

  replications: 5│   └── analyze_results.py   # Result analysis and plotting

├── configs/                  # Configuration files

grid:  # Parameters to sweep (creates cross-product)│   ├── base.yaml            # Base configuration

  pe_induction.p_shock: [0.0, 0.10, 0.20]│   └── phases/              # Phase-specific configs

  pe_induction.pe_mag: [low, high]│       ├── phase1_core_hypothesis.yaml

  interface.prediction_mode: [required, optional]│       ├── phase2_pe_type.yaml

  model.name: [llama-3.2-3b, llama-3.1-70b]│       ├── phase3_complexity.yaml

│       ├── phase4_model_arch.yaml

fixed:  # Parameters held constant│       └── phase5_long_horizon.yaml

  simulation.max_steps: 500├── run_experiment.py         # Main experiment runner

  simulation.complexity_level: 1├── requirements.txt          # Python dependencies

```├── setup.py                  # Package setup

├── pytest.ini               # Pytest configuration

Grid expansion: 3 × 2 × 2 × 2 × 5 = **120 runs** → 120 SLURM array tasks├── PRD.md                   # Product requirements document

└── README.md                # This file

## Experiment Phases```



| Phase | Description | Tasks | Time/Task | Total Parallelized Time |## Usage

|-------|-------------|-------|-----------|------------------------|

| 1 | Core hypothesis (dose-response) | 1,320 | ~1h | ~1h |### Running Experiments

| 2 | PE type × observability | 4,950 | ~2h | ~2h ⚠️ **EXCEEDS 1000 JOB LIMIT** |

| 3 | Complexity scaling | 1,980 | ~2h | ~2h |#### Basic Usage

| 4 | Model architecture sweep | 990 | ~2h | ~2h |

| 5 | Long horizon (5000 steps) | 1,320 | ~5h | ~5h |```bash

python run_experiment.py --config <config_file> [options]

**Total:** ~10,560 experiments, ~34,000 GPU-hours wall time with sufficient cluster resources```



## Project Structure**Options:**

- `--config`: Path to YAML configuration file (required)

```- `--n-jobs`: Number of parallel jobs (default: 1)

VendoMini/- `--cluster`: Enable Ray for distributed execution

├── configs/- `--output-dir`: Override output directory

│   ├── base.yaml                 # Base configuration

│   └── phases/#### Examples

│       ├── phase1_core_hypothesis.yaml

│       ├── phase2_pe_type.yaml```bash

│       └── ...# Single run for testing

├── src/python run_experiment.py --config configs/base.yaml --n-jobs 1

│   ├── env.py                    # VendoMini environment

│   ├── pe_calculator.py          # PE calculation & EWMA# Phase 1: 8 parallel jobs

│   ├── crash_detector.py         # Crash detectionpython run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 8

│   ├── agent.py                  # LLM agent interface

│   ├── experiment_runner.py      # Orchestration (local/cluster)# Phase 2: 32 jobs on cluster

│   ├── cluster_utils.py          # SLURM utilitiespython run_experiment.py --config configs/phases/phase2_pe_type.yaml --n-jobs 32 --cluster

│   └── ...

├── slurm/# Custom output directory

│   ├── run_phase1.sh             # SLURM array job for Phase 1python run_experiment.py --config configs/base.yaml --output-dir my_results

│   ├── run_phase2.sh             # SLURM array job for Phase 2```

│   └── submit_all_phases.sh      # Submit all phases

├── scripts/### Configuration

│   ├── aggregate_results.py      # Merge SLURM task results

│   ├── analyze_results.py        # Survival analysis & plots#### Config File Structure

│   └── ...

├── run_experiment.py             # Main entry point```yaml

└── requirements.txtexperiment:

```  name: "experiment_name"

  replications: 5           # Number of replications per condition

## Key Parameters  seed: 42                  # Base random seed



### PE Induction# Grid expansion (Cartesian product of all values)

- `p_shock`: Shock probability (0.0 - 0.35)grid:

- `pe_mag`: Magnitude (low, medium, high)  pe_induction.p_shock: [0.0, 0.10, 0.20]

- `pe_type_mix`: realistic, temporal_only, quantity_only, causal_only, uniform  model.name: [llama-3.1-70b, gpt-4]

- `observability`: full, delayed, partial, hidden

# Fixed parameters (applied to all runs)

### Measurementfixed:

- `crash_threshold`: strict, moderate, lenient  simulation.max_steps: 500

- `success_metric`: time_to_crash (primary), survival_rate, orders_fulfilled

# Main configuration sections

## Outputssimulation:

  max_steps: 1000

### Per-Run Logs (local mode)  complexity_level: 1       # 0-4 (affects SKUs, suppliers)

```  initial_budget: 200

logs/phase1_c0_r0/  pressure_level: medium    # low | medium | high

  steps.jsonl        # Step-by-step trace

  summary.json       # Run summarype_induction:

```  p_shock: 0.10            # Probability of shock per step

  pe_mag: medium           # low | medium | high

### SLURM Task Results  pe_type_mix: realistic   # realistic | temporal_only | quantity_only | causal_only | uniform

```  observability: full      # full | delayed | partial | hidden

results/

  vendomini_task_0000.json  # Task 0 resultinterface:

  vendomini_task_0001.json  # Task 1 result  prediction_mode: required           # required | optional | required+confidence

  ...  prediction_format: structured       # minimal | structured | rich

  phase1_all_results.csv    # Aggregated (after merge)  memory_tools: full                  # none | basic | full

```  recovery_tools: none                # none | reset | audit | help | all



## Developmentmodel:

  name: llama-3.1-70b

### Run Tests  context_length: 32000

```bash  temperature: 0.3

pytest tests/ -v --cov=src

```measurement:

  crash_threshold: moderate  # strict | moderate | lenient

### Add New Phase  pe_windows: [10, 100, 500]

1. Create `configs/phases/phase6_new.yaml````

2. Create `slurm/run_phase6.sh` (update `--array` size)

3. Submit: `sbatch slurm/run_phase6.sh`#### Grid Expansion



## TroubleshootingThe system automatically expands grid parameters into individual runs:



**SLURM jobs failing:**```yaml

- Check logs: `slurm-JOBID_TASKID.out`grid:

- Verify conda activation in SLURM script  pe_induction.p_shock: [0.0, 0.10, 0.20]  # 3 values

- Check paths: `cd $SLURM_SUBMIT_DIR`  model.name: [llama, gpt]                  # 2 values

replications: 5                             # 5 replications

**Import errors:**

- Add to SLURM script: `export PYTHONPATH=/path/to/VendoMini:$PYTHONPATH`# Results in: 3 × 2 × 5 = 30 total runs

```

**Out of memory:**

- Increase `#SBATCH --mem-per-cpu=` in SLURM scripts### Parallel Execution



## Citation#### Local Parallelism (joblib)



```bibtex```bash

@software{vendomini2025,# Use all available cores

  title={VendoMini: Prediction Error Crash Simulation},python run_experiment.py --config configs/phase1.yaml --n-jobs -1

  author={Gupta, Mohan},

  year={2025},# Use specific number of cores

  url={https://github.com/mohanwugupta/VendoMini}python run_experiment.py --config configs/phase1.yaml --n-jobs 8

}```

```

#### Cluster Parallelism (Ray)

## License

1. **Start Ray cluster:**

MIT License

```bash

## Contact# On head node

ray start --head --port=6379

- **Author:** Mohan Gupta

- **Repository:** https://github.com/mohanwugupta/VendoMini# On worker nodes

ray start --address=<head-node-ip>:6379
```

2. **Run experiment:**

```bash
python run_experiment.py --config configs/phase1.yaml --n-jobs 64 --cluster
```

### Analyzing Results

#### Generate Summary Statistics

```bash
python scripts/analyze_results.py --results results/phase1_core_hypothesis_results.csv
```

This generates:
- Summary statistics (crash rates, time-to-crash distributions)
- Crash distribution plots
- Crash rate by parameter plots
- Saved to `analysis/` directory

#### Manual Analysis

Results are saved as CSV files with one row per run:

```python
import pandas as pd

# Load results
df = pd.read_csv('results/phase1_core_hypothesis_results.csv')

# Analyze crash rates
crash_rate = df.groupby('config.p_shock')['crashed'].mean()

# Plot survival curves
import matplotlib.pyplot as plt
df[df['crashed']].hist('time_to_crash', bins=30)
plt.show()
```

## Configuration Guide

### Complexity Levels

| Level | SKUs | Suppliers | Tools Available |
|-------|------|-----------|-----------------|
| 0     | 5    | 2         | Basic           |
| 1     | 10   | 3         | Basic           |
| 2     | 15   | 4         | + Expedite      |
| 3     | 20   | 5         | + All           |
| 4     | 25   | 6         | + All           |

### PE Type Mix

- **realistic**: Weighted mix (40% temporal, 30% quantity, 20% causal, 10% rule)
- **temporal_only**: Only temporal delays
- **quantity_only**: Only quantity variations
- **causal_only**: Only causal/rule violations
- **uniform**: Equal probability of all types

### Crash Types

1. **Looping**: Repeated identical actions with no state change
2. **Invalid Burst**: High rate of failed tool calls
3. **Budget Denial**: Ordering while bankrupt
4. **Decoupling**: Actions contradict predictions
5. **Exploration Collapse**: Low tool diversity
6. **Slow Divergence**: Incoherent state summaries

## Outputs

### Log Files

For each run, logs are saved to `logs/<run_id>/`:

- **steps.jsonl**: One JSON object per step containing:
  - Observation, prediction, action, result
  - PE values by type
  - Cumulative PE accumulators
  - Crash detection flags

- **summary.json**: Run summary with:
  - Crash information
  - Final statistics
  - PE accumulator values
  - Configuration parameters

### Results CSV

Aggregated results in `results/<experiment_name>_results.csv`:

- One row per run
- All configuration parameters (flattened)
- Summary metrics (time_to_crash, success_rate, etc.)
- PE statistics

## Testing

### Run All Tests

```bash
python scripts/run_tests.py
```

### Run Specific Tests

```bash
# Test environment
pytest tests/test_env.py -v

# Test PE calculator
pytest tests/test_pe_calculator.py -v

# Test with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

View coverage report:
```bash
# Generate and open HTML report
python scripts/run_tests.py
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

## Extending the System

### Adding New Tools

1. Add tool implementation to `src/env.py`:

```python
def _tool_my_new_tool(self, arg1: str, arg2: int) -> Dict[str, Any]:
    """My new tool."""
    # Implementation
    return {'success': True, 'result': 'value'}
```

2. Register in `_execute_action`:

```python
elif tool == 'tool_my_new_tool':
    return self._tool_my_new_tool(**args)
```

3. Add to available tools in experiment runner

### Adding New Crash Types

1. Add to `CrashType` in `src/crash_detector.py`
2. Implement detection logic in `CrashDetector.update()`
3. Add unit tests

### Custom Analysis Scripts

Create custom analysis in `scripts/`:

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv('results/my_experiment_results.csv')

# Your analysis here
```

## Phases Overview

### Phase 1: Core Hypothesis
- **Goal**: Dose-response of p_shock; prediction-mode ablation
- **Runs**: 1,320 (11×2×2×6 conditions × 5 reps)
- **Duration**: ~2-4 hours on 32 cores

### Phase 2: PE Type Analysis
- **Goal**: PE type × observability interactions
- **Runs**: 4,950 (5×11×3×6 × 5 reps) - Split into 5 arrays
- **Duration**: ~4-6 hours on 32 cores

### Phase 3: Complexity Scaling
- **Goal**: Test across complexity levels + recovery tools
- **Runs**: 1,980 (3×11×2×6 × 5 reps)
- **Duration**: ~6-8 hours on 32 cores

### Phase 4: Model Architecture Sweep
- **Goal**: Compare different LLM models
- **Runs**: 990 (6×11×3 × 5 reps)
- **Duration**: ~8-12 hours on 64 cores

### Phase 5: Long Horizon
- **Goal**: Extended runs (5000 steps) for rare events
- **Runs**: 1,320 (2×11×6 × 10 reps)
- **Duration**: ~12-24 hours on 64 cores

## Troubleshooting

### Import Errors

If you see import errors for `openai`, `anthropic`, or `ray`:

```bash
# These are optional dependencies
# Install only if needed:
pip install openai anthropic ray[default]
```

### Ray Connection Issues

```bash
# Check Ray status
ray status

# Restart Ray
ray stop
ray start --head
```

### Out of Memory

- Reduce `n_jobs`
- Decrease `max_steps` in config
- Use cluster mode instead of local

### Slow Execution

- Increase `n_jobs`
- Use `--cluster` flag
- Reduce `replications` for testing

## Contributing

When adding features:

1. Write unit tests in `tests/`
2. Update documentation in README
3. Follow existing code style
4. Run test suite before committing

## License

See project documentation for license information.

## Citation

If you use VendoMini in your research, please cite:

```bibtex
@software{vendomini2025,
  title={VendoMini: Prediction Error Crash Simulation},
  author={Gupta, Mohan},
  year={2025}
}
```
