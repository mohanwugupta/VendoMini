# VendoMini Starter Pack

**What this is:** A minimal, production-ready bundle (PRD + configs) so an agent coder can spin up the simulation and run Phase 1–5 experiments.

## Quick Start
1. Read `PRD.md` (10 min).
2. Pick a phase config from `configs/phases/*.yaml`.
3. Implement the engine skeletons: `VendoMiniEnv`, `PECalculator`, `CrashDetector`, `ExperimentRunner` (names referenced inside configs).
4. Run your first grid:
```bash
python run_experiment.py --config configs/phases/phase1_core_hypothesis.yaml --n-jobs 8
```

## Config Conventions
- **Grid** keys are dot‑paths (e.g., `pe_induction.p_shock`).
- **fixed** applies to all runs expanded from `grid`.
- **replications** repeats each condition with different seeds.
- Add new factors by extending the schema (keep names stable).

## Outputs
- `logs/<run_id>/{steps.jsonl,summary.json}`
- `results/<experiment_name>_results.csv`

## Phases Included
- Phase 1–5 YAMLs mirror the PRD and can be extended safely.
