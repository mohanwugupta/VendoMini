#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VendoMini Experiment Runner
===========================
Handles grid expansion, parallel execution, and result aggregation for VendoMini simulations.
Supports both local (joblib) and cluster (SLURM array jobs) execution modes.
"""

import os
import sys
import json
import yaml
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import argparse

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.env import VendoMiniEnv
from src.agent import LLMAgent
from src.pe_calculator import PECalculator
from src.crash_detector import CrashDetector
from src.logging_utils import setup_logging, log_step, save_summary
from src.cluster_utils import expand_grid_slurm, get_task_params_slurm

class ExperimentRunner:
    """Manages the execution of VendoMini experiments with grid expansion and parallelization."""

    def __init__(self, config, cluster_mode: bool = False):
        """Initialize experiment runner with configuration."""
        self.cluster_mode = cluster_mode

        # Handle config input (either path or dict)
        if isinstance(config, str):
            # Config is a path - load it
            self.config_path = config
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Config is already loaded dict
            self.config_path = None
            self.config = config

        # Set up directories
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / self.config.get('results_dir', 'results')
        self.logs_dir = self.project_root / self.config.get('logs_dir', 'logs')
        self.checkpoints_dir = self.project_root / self.config.get('checkpoints_dir', 'checkpoints')

        # Create directories
        for dir_path in [self.results_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        setup_logging(self.logs_dir)

        # Initialize components
        self.base_config = Config(self.config)

    def expand_grid(self) -> List[Dict[str, Any]]:
        """Expand parameter grid for all combinations."""
        if self.cluster_mode:
            return expand_grid_slurm(self.config)
        else:
            # Local expansion logic
            grid_params = self.config.get('grid', {})
            return self._expand_local_grid(grid_params)

    def _expand_local_grid(self, grid_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand grid for local execution."""
        from itertools import product

        # Get all parameter combinations
        keys = list(grid_params.keys())
        values = [grid_params[key] for key in keys]

        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def run_single_experiment(self, params: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
        """Run a single experiment with given parameters."""
        # Set random seed
        np.random.seed(seed)

        # Create unique run ID
        run_id = f"{int(time.time())}_{seed}"

        # Set up logging for this run
        run_logs_dir = self.logs_dir / f"run_{run_id}"
        run_logs_dir.mkdir(exist_ok=True)

        # Initialize components with grid parameters properly applied
        from src.cluster_utils import _set_nested
        import copy
        
        env_config = copy.deepcopy(self.config)
        
        # Apply grid parameters using nested path notation
        for key, value in params.items():
            if '.' in key and key not in ['combination_id', 'replication_id', 'seed']:
                # This is a dotted path like 'model.name' - apply as nested
                _set_nested(env_config, key, value)
            else:
                # Direct key - just set it
                env_config[key] = value

        print(f"[*] Initializing environment...")
        env = VendoMiniEnv(env_config)
        
        print(f"[*] Initializing agent...")
        # Agent needs full config (it extracts model/interface sections internally)
        # Get model name for logging
        model_name = env_config.get('agent', {}).get('model', {}).get('name', 
                     env_config.get('model', {}).get('name', 'unknown'))
        print(f"[*] Model: {model_name}")
        
        # Track if model initialization failed
        model_load_failed = False
        model_load_error = None
        
        try:
            agent = LLMAgent(env_config)
            print(f"[*] Agent initialized successfully")
            
            # Check if client actually loaded
            if agent.client is None:
                model_load_failed = True
                model_load_error = f"LLM client is None for model '{model_name}' (provider: {agent.provider})"
                print(f"[ERROR] {model_load_error}")
                print(f"[ERROR] Experiment cannot run without a working LLM - aborting")
                
                # Return early with error info
                return {
                    'run_id': run_id,
                    'params': params,
                    'seed': seed,
                    'model_load_failed': True,
                    'model_load_error': model_load_error,
                    'total_steps': 0,
                    'crashed': False,
                    'crash_type': None,
                    'final_budget': env.budget,
                    'final_storage': env.storage.copy(),
                    'cumulative_pe': {
                        'temporal': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
                        'quantity': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
                        'cost': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
                        'causal': {'fast': 0.0, 'med': 0.0, 'slow': 0.0}
                    }
                }
            else:
                print(f"[*] LLM client loaded: provider={agent.provider}")
        except Exception as e:
            model_load_failed = True
            model_load_error = str(e)
            print(f"[ERROR] Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            
            # Return early with error info
            return {
                'run_id': run_id,
                'params': params,
                'seed': seed,
                'model_load_failed': True,
                'model_load_error': model_load_error,
                'total_steps': 0,
                'crashed': False,
                'crash_type': None,
                'final_budget': env.budget,
                'final_storage': env.storage.copy(),
                'cumulative_pe': {
                    'temporal': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
                    'quantity': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
                    'cost': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
                    'causal': {'fast': 0.0, 'med': 0.0, 'slow': 0.0}
                }
            }
        
        print(f"[*] Initializing PE calculator and crash detector...")
        pe_calc = PECalculator()
        crash_detector = CrashDetector(**self.base_config.get_crash_config())

        # Run simulation
        max_steps = self.config.get('max_steps', 100)
        step_data = []

        # Define available tools based on interface config
        available_tools = [
            'tool_check_inbox',
            'tool_check_storage', 
            'tool_check_budget',
            'tool_order',
            'tool_quote',
            'tool_cancel_order',
            'tool_ship_customer_order',
        ]
        
        print(f"[*] Starting simulation (max_steps={max_steps})...")
        
        for step in range(max_steps):
            try:
                # Get current observation
                print(f"[DEBUG] Step {step}: Getting observation...")
                observation = env.get_observation()
                print(f"[DEBUG] Step {step}: Observation received (day={observation.get('day', 0)}, budget=${observation.get('budget', 0):.2f})")

                # Agent makes decision
                print(f"[DEBUG] Step {step}: Calling agent.get_action_and_prediction()...")
                action_start = time.time()
                action, prediction = agent.get_action_and_prediction(observation, available_tools)
                action_time = time.time() - action_start
                print(f"[DEBUG] Step {step}: Agent decision received ({action_time:.2f}s, {action_time/60:.1f} min)")
                print(f"[DEBUG] Step {step}: Action: {action.get('tool', 'unknown')}")
                
                # Log action for debugging
                if step < 5 or step % 10 == 0:  # Log first 5 steps and every 10th step
                    print(f"  Step {step}: action={action.get('tool', 'unknown')}")

                # Execute action — env.step() returns (next_observation, done_flag)
                print(f"[DEBUG] Step {step}: Executing action in environment...")
                step_result = env.step(action)
                if isinstance(step_result, tuple):
                    result, env_done = step_result
                else:
                    result, env_done = step_result, False
                print(f"[DEBUG] Step {step}: Action executed, result received")

                # Calculate prediction errors
                pe = pe_calc.compute_pe(prediction, result)
                pe_calc.update_accumulators(pe)

                # Check for crashes
                crashed, crash_type = crash_detector.update(step_data[-self.base_config.get_crash_config().get('window_size', 10):] if step_data else [])

                # Log step
                step_info = {
                    'step': step,
                    'observation': observation,
                    'action': action,
                    'prediction': prediction,
                    'result': result,
                    'pe': pe,
                    'cumulative_pe': pe_calc.get_cumulative_pes(),
                    'crash_detected': crashed,
                    'crash_type': crash_type if crashed else None
                }
                step_data.append(step_info)
                log_step(run_logs_dir / 'steps.jsonl', step_info)

                # Check termination conditions
                if crashed and crash_detector.should_terminate(step):
                    print(f"[*] Terminating at step {step}: {crash_type} (crash detected at step {crash_detector.crash_step})")
                    break
                elif crashed:
                    # Crash detected but continuing to observe behavior
                    if step == crash_detector.crash_step:
                        print(f"[*] Crash detected at step {step}: {crash_type} (continuing for {crash_detector.continue_after_crash} more steps)")
                
                # Check if budget depleted (terminal condition)
                if env.budget <= 0:
                    print(f"[*] Budget depleted at step {step}")
                    break

                # Check environment done flag (covers max_steps, budget < -100,
                # and customer_orders_failed >= max_failures)
                if env_done:
                    reason = "max_steps" if env.current_day >= env.max_steps else \
                             "budget_depleted" if env.budget < -100 else \
                             "customer_order_failures"
                    print(f"[*] Environment signalled done at step {step}: {reason}")
                    break
                    
            except RuntimeError as e:
                # This is likely from the agent failing to initialize or call LLM
                error_msg = str(e)
                print(f"[ERROR] RuntimeError at step {step}: {error_msg}")
                
                # Return with error status
                return {
                    'run_id': run_id,
                    'params': params,
                    'seed': seed,
                    'model_load_failed': True,
                    'model_load_error': error_msg,
                    'total_steps': step,
                    'crashed': True,
                    'crash_type': 'model_error',
                    'final_budget': env.budget,
                    'final_storage': env.storage.copy(),
                    'cumulative_pe': pe_calc.get_cumulative_pes()
                }
                    
            except Exception as e:
                print(f"[ERROR] Exception at step {step}: {e}")
                import traceback
                traceback.print_exc()
                crashed = True
                crash_type = "exception"
                break

        print(f"[*] Simulation complete: {len(step_data)} steps")

        # Save summary
        summary = {
            'run_id': run_id,
            'params': params,
            'seed': seed,
            'model_load_failed': model_load_failed,
            'model_load_error': model_load_error,
            'total_steps': len(step_data),
            'crashed': crashed,
            'crash_type': crash_type if crashed else None,
            'final_budget': env.budget,
            'final_storage': env.storage,
            'final_scratchpad': env.scratchpad.copy(),  # Save final scratchpad state
            'scratchpad_final_size': len(env.scratchpad),  # Add size for quick reference
            'fulfilled_orders': env.fulfilled_orders,
            'total_orders_requested': env.total_orders_requested,
            'revenue': env.revenue,
            'customer_orders_shipped': env.customer_orders_shipped,
            'customer_orders_failed': env.customer_orders_failed,
            'cumulative_pe': pe_calc.get_cumulative_pes()
        }
        save_summary(run_logs_dir / 'summary.json', summary)
        
        # Also save to results directory for cluster aggregation
        if self.cluster_mode:
            task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', seed))
            task_result_file = self.results_dir / f'vendomini_task_{task_id:04d}.json'
            save_summary(task_result_file, summary)

        return summary

    def run_parallel(self, n_jobs: int = 1) -> List[Dict[str, Any]]:
        """Run experiments in parallel."""
        if self.cluster_mode:
            # SLURM array job mode
            task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
            params = get_task_params_slurm(self.config, task_id)
            seed = task_id  # Use task ID as seed for reproducibility

            result = self.run_single_experiment(params, seed)
            return [result]
        else:
            # Local parallel mode
            grid = self.expand_grid()
            seeds = list(range(len(grid)))

            results = Parallel(n_jobs=n_jobs)(
                delayed(self.run_single_experiment)(params, seed)
                for params, seed in zip(grid, seeds)
            )

            return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Aggregate results from multiple runs."""
        df = pd.DataFrame(results)

        # Save aggregated results
        output_path = self.results_dir / 'aggregated_results.csv'
        df.to_csv(output_path, index=False)

        return df

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description='Run VendoMini experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs (local mode)')
    parser.add_argument('--cluster', action='store_true', help='Run in cluster mode (SLURM)')

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(args.config, args.cluster)

    # Run experiments
    results = runner.run_parallel(args.n_jobs)

    # Aggregate results (only in local mode)
    if not args.cluster:
        df = runner.aggregate_results(results)
        print(f"Results saved to {runner.results_dir / 'aggregated_results.csv'}")
        print(f"Total runs: {len(results)}")

if __name__ == '__main__':
    main()