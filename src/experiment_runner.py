"""Experiment runner for VendoMini with local and cluster support.""""""Experiment runner with parallel execution support."""



import sysimport random

import jsonfrom typing import Dict, Any, List, Optional

from pathlib import Pathfrom pathlib import Path

from typing import Dict, Any, List, Optionalimport json

from tqdm import tqdmfrom tqdm import tqdm



# Add src to pathfrom src.env import VendoMiniEnv

sys.path.insert(0, str(Path(__file__).parent))from src.pe_calculator import PECalculator

from src.crash_detector import CrashDetector

from config import ConfigLoaderfrom src.agent import LLMAgent

from env import VendoMiniEnvfrom src.logging_utils import Logger, ResultsAggregator

from pe_calculator import PECalculatorfrom src.config import ConfigLoader

from crash_detector import CrashDetector

from agent import LLMAgent

from logging_utils import Logger, ResultsAggregatorclass ExperimentRunner:

from cluster_utils import set_seed    """

    Run VendoMini experiments with grid expansion and parallel execution.

    """

class ExperimentRunner:    

    """    def __init__(self, config: Dict[str, Any]):

    Run VendoMini experiments with support for parallel execution.        """

            Initialize experiment runner.

    Supports two modes:        

    1. Local mode: Use joblib for parallel execution        Args:

    2. Cluster mode: Use SLURM array jobs for massive parallelization            config: Experiment configuration

    """        """

            self.config = config

    def __init__(self, config: Dict[str, Any]):        self.experiment_name = config.get('experiment', {}).get('name', 'experiment')

        """        

        Initialize experiment runner.    def run_parallel(self, n_jobs: int = 1, use_ray: bool = False):

                """

        Args:        Run experiments in parallel.

            config: Configuration dictionary        

        """        Args:

        self.config = config            n_jobs: Number of parallel jobs

                    use_ray: Whether to use Ray for distributed execution

    def run_single_experiment(self, run_config: Dict[str, Any]) -> Dict[str, Any]:        """

        """        # Expand grid

        Run a single experiment instance.        run_configs = ConfigLoader.expand_grid(self.config)

                

        Args:        print(f"Running {len(run_configs)} experiments with {n_jobs} parallel jobs...")

            run_config: Configuration for this run        

                    if use_ray and n_jobs > 1:

        Returns:            results = self._run_with_ray(run_configs, n_jobs)

            Summary results dictionary        elif n_jobs > 1:

        """            results = self._run_with_joblib(run_configs, n_jobs)

        # Get run metadata        else:

        run_id = run_config['experiment']['run_id']            results = self._run_sequential(run_configs)

        seed = run_config['experiment']['seed']        

                # Aggregate results

        # Set seed for reproducibility        self._aggregate_results(results)

        set_seed(seed)        

                print(f"Experiment complete! Results saved to results/{self.experiment_name}_results.csv")

        # Initialize logger        

        logs_dir = run_config.get('paths', {}).get('logs_dir', 'logs')        return results

        logger = Logger(run_id, logs_dir)    

            def _run_sequential(self, run_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        print(f"🚀 Starting run: {run_id} (seed={seed})")        """Run experiments sequentially."""

                results = []

        try:        for config in tqdm(run_configs, desc="Running experiments"):

            # Initialize components            result = run_single_experiment(config)

            env = VendoMiniEnv(run_config, seed=seed)            results.append(result)

            pe_calc = PECalculator(        return results

                windows=run_config.get('measurement', {}).get('pe_windows', [10, 100, 500])    

            )    def _run_with_joblib(self, run_configs: List[Dict[str, Any]], n_jobs: int) -> List[Dict[str, Any]]:

            crash_detector = CrashDetector(        """Run experiments in parallel using joblib."""

                threshold=run_config.get('measurement', {}).get('crash_threshold', 'moderate'),        from joblib import Parallel, delayed

                window_size=20        

            )        results = Parallel(n_jobs=n_jobs, verbose=10)(

            agent = LLMAgent(run_config)            delayed(run_single_experiment)(config)

                        for config in run_configs

            # Reset environment        )

            observation = env.reset()        

                    return results

            # Initialize tracking    

            step_history = []    def _run_with_ray(self, run_configs: List[Dict[str, Any]], n_jobs: int) -> List[Dict[str, Any]]:

            crashed = False        """Run experiments in parallel using Ray."""

            crash_step = None        import ray

            crash_type = None        

                    # Initialize Ray if not already initialized

            # Run simulation loop        if not ray.is_initialized():

            max_steps = env.max_steps            ray.init(num_cpus=n_jobs)

            prediction_mode = run_config.get('interface', {}).get('prediction_mode', 'required')        

                    # Convert function to Ray remote

            for step in range(max_steps):        remote_experiment = ray.remote(run_single_experiment)

                # Get action and prediction from agent        

                action, prediction_card = agent.get_action_and_prediction(        # Submit all jobs

                    observation,        futures = [remote_experiment.remote(config) for config in run_configs]

                    available_tools=['tool_order', 'tool_check_inbox', 'tool_check_storage',         

                                   'tool_check_budget', 'tool_cancel_order', 'tool_quote']        # Collect results with progress bar

                )        results = []

                        with tqdm(total=len(futures), desc="Running experiments") as pbar:

                # Execute step            while futures:

                new_observation, done = env.step(action, prediction_card)                # Wait for at least one to complete

                                done, futures = ray.wait(futures, num_returns=1)

                # Get actual outcome for PE calculation                results.extend(ray.get(done))

                action_result = env.action_history[-1]['result']                pbar.update(len(done))

                        

                # Compute PE        return results

                actual_outcome = {    

                    'actual_success': action_result.get('success', False),    def _aggregate_results(self, results: List[Dict[str, Any]]):

                    'actual_cost': action_result.get('price', 0) if 'price' in action_result else None,        """Aggregate and save results."""

                    'actual_quantity': action_result.get('quantity', 0) if 'quantity' in action_result else None,        results_dir = Path(self.config.get('paths', {}).get('results_dir', 'results'))

                    'actual_delivery_day': action_result.get('eta_day', 0) if 'eta_day' in action_result else None        results_path = results_dir / f"{self.experiment_name}_results.csv"

                }        

                        ResultsAggregator.aggregate_to_csv(results, str(results_path))

                pe_dict = pe_calc.compute_pe(prediction_card, actual_outcome)

                pe_calc.update_accumulators(pe_dict)

                def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:

                # Record step    """

                step_record = {    Run a single experiment.

                    'step': step,    

                    'day': env.current_day,    Args:

                    'observation': observation,        config: Run configuration

                    'action': action,        

                    'prediction': prediction_card,    Returns:

                    'result': action_result,        Summary dictionary

                    'pe': pe_dict,    """

                    'cumulative_pe': pe_calc.get_cumulative_pes(),    run_id = config['experiment']['run_id']

                    'state': env.get_full_state()    seed = config['experiment']['seed']

                }    

                    # Initialize components

                step_history.append(step_record)    env = VendoMiniEnv(config, seed=seed)

                    pe_calc = PECalculator(windows=config.get('measurement', {}).get('pe_windows', [10, 100, 500]))

                # Log step    crash_detector = CrashDetector(

                logger.log_step(step_record)        threshold=config.get('measurement', {}).get('crash_threshold', 'moderate'),

                        window_size=20

                # Check for crash    )

                is_crashed, detected_type = crash_detector.update(step_history)    agent = LLMAgent(config)

                if is_crashed and not crashed:    

                    crashed = True    # Initialize logger

                    crash_step = step    logs_dir = config.get('paths', {}).get('logs_dir', 'logs')

                    crash_type = detected_type    logger = Logger(run_id, logs_dir)

                    print(f"  💥 Crash detected at step {step}: {crash_type}")    

                    break    # Run simulation

                    observation = env.reset()

                # Update observation    done = False

                observation = new_observation    step_count = 0

                    history = []

                # Check if environment signals done    

                if done:    available_tools = [

                    print(f"  🏁 Environment terminated at step {step}")        'tool_order', 'tool_check_inbox', 'tool_check_storage',

                    break        'tool_check_budget', 'tool_cancel_order', 'tool_quote'

                ]

            # Compute summary statistics    

            summary = self._compute_summary(    # Add complexity-dependent tools

                run_config, env, pe_calc, crash_detector,     if env.complexity_level >= 2:

                step_history, crashed, crash_step, crash_type        available_tools.append('tool_expedite')

            )    

                # Add memory tools

            # Log summary    memory_tools = config.get('interface', {}).get('memory_tools', 'none')

            logger.log_summary(summary)    if memory_tools in ['basic', 'full']:

            logger.close()        available_tools.extend(['tool_write_scratchpad', 'tool_read_scratchpad'])

                if memory_tools == 'full':

            print(f"✅ Completed run: {run_id} - {'Crashed' if crashed else 'Survived'}")        available_tools.append('tool_delete_scratchpad')

                

            return summary    while not done and step_count < env.max_steps:

                    # Get action and prediction from agent

        except Exception as e:        action, prediction_card = agent.get_action_and_prediction(observation, available_tools)

            print(f"❌ Error in run {run_id}: {e}")        

            import traceback        # Execute step

            traceback.print_exc()        next_observation, done = env.step(action, prediction_card)

                    

            # Log error summary        # Get actual outcome from action result

            error_summary = {        action_result = env.action_history[-1]['result'] if env.action_history else {}

                'run_id': run_id,        

                'error': str(e),        # Compute PE

                'crashed': True,        actual_outcome = {

                'crash_type': 'error',            'actual_success': action_result.get('success', False),

                'time_to_crash': 0            'actual_cost': action_result.get('price', 0) if action.get('tool') == 'tool_order' else 0,

            }            'actual_delivery_day': action_result.get('eta_day', 0) if action.get('tool') == 'tool_order' else 0,

            logger.log_summary(error_summary)            'actual_quantity': action.get('args', {}).get('quantity', 0)

            logger.close()        }

                    

            return error_summary        pes = pe_calc.compute_pe(prediction_card, actual_outcome)

            pe_calc.update_accumulators(pes)

    def _compute_summary(        

        self,        # Update history for crash detection

        run_config: Dict[str, Any],        step_record = {

        env: VendoMiniEnv,            'step': step_count,

        pe_calc: PECalculator,            'action': action,

        crash_detector: CrashDetector,            'prediction': prediction_card,

        step_history: List[Dict[str, Any]],            'result': action_result,

        crashed: bool,            'observation': observation

        crash_step: Optional[int],        }

        crash_type: Optional[str]        history.append(step_record)

    ) -> Dict[str, Any]:        

        """Compute summary statistics for a run."""        # Check for crash

                is_crashed, crash_type = crash_detector.update(history)

        # Extract config parameters for logging        

        config_params = {        # Log step

            'p_shock': run_config.get('pe_induction', {}).get('p_shock'),        cumulative_pes = pe_calc.get_cumulative_pes()

            'pe_mag': run_config.get('pe_induction', {}).get('pe_mag'),        logger.log_step({

            'pe_type_mix': run_config.get('pe_induction', {}).get('pe_type_mix'),            'step': step_count,

            'observability': run_config.get('pe_induction', {}).get('observability'),            'day': env.current_day,

            'prediction_mode': run_config.get('interface', {}).get('prediction_mode'),            'observation': observation,

            'prediction_format': run_config.get('interface', {}).get('prediction_format'),            'prediction': prediction_card,

            'model_name': run_config.get('model', {}).get('name'),            'action': action,

            'complexity_level': run_config.get('simulation', {}).get('complexity_level'),            'result': action_result,

            'crash_threshold': run_config.get('measurement', {}).get('crash_threshold')            'pes': pes,

        }            'cumulative_pes': cumulative_pes,

                    'crash_detected': is_crashed,

        # Get final cumulative PEs            'crash_type': crash_type

        final_pes = pe_calc.get_cumulative_pes()        })

        pe_stats = pe_calc.get_windowed_stats()        

                # Check if crashed

        # Compute outcomes        if is_crashed:

        time_to_crash = crash_step if crashed else len(step_history)            done = True

        survival_rate = 1.0 if not crashed else 0.0        

                observation = next_observation

        summary = {        step_count += 1

            'run_id': run_config['experiment']['run_id'],    

            'combination_id': run_config['experiment'].get('combination_id'),    # Compute summary statistics

            'replication_id': run_config['experiment'].get('replication_id'),    crash_info = crash_detector.get_crash_info()

            'seed': run_config['experiment']['seed'],    final_state = env.get_full_state()

                pe_stats = pe_calc.get_windowed_stats()

            # Config parameters    

            **config_params,    summary = {

                    'run_id': run_id,

            # Primary outcome        'combination_id': config['experiment']['combination_id'],

            'crashed': crashed,        'replication_id': config['experiment']['replication_id'],

            'crash_type': crash_type,        'seed': seed,

            'crash_step': crash_step,        'crashed': crash_info['crashed'],

            'time_to_crash': time_to_crash,        'crash_type': crash_info['type'],

            'survival_rate': survival_rate,        'time_to_crash': crash_info['step'] if crash_info['crashed'] else step_count,

                    'total_steps': step_count,

            # Secondary outcomes        'final_budget': final_state['budget'],

            'total_steps': len(step_history),        'fulfilled_orders': final_state['fulfilled_orders'],

            'orders_fulfilled': env.fulfilled_orders,        'total_orders_requested': final_state['total_orders_requested'],

            'total_orders_requested': env.total_orders_requested,        'success_rate': final_state['fulfilled_orders'] / max(final_state['total_orders_requested'], 1),

            'fulfillment_rate': env.fulfilled_orders / max(env.total_orders_requested, 1),        'pe_stats': pe_stats,

            'final_budget': env.budget,        'cumulative_pes': pe_calc.get_cumulative_pes(),

            'final_storage': sum(env.storage.values()),        'config': {

                        'p_shock': config.get('pe_induction', {}).get('p_shock'),

            # PE metrics            'pe_mag': config.get('pe_induction', {}).get('pe_mag'),

            'final_pe_temporal_fast': final_pes['temporal']['fast'],            'pe_type_mix': config.get('pe_induction', {}).get('pe_type_mix'),

            'final_pe_temporal_med': final_pes['temporal']['med'],            'observability': config.get('pe_induction', {}).get('observability'),

            'final_pe_temporal_slow': final_pes['temporal']['slow'],            'prediction_mode': config.get('interface', {}).get('prediction_mode'),

            'final_pe_quantity_fast': final_pes['quantity']['fast'],            'model_name': config.get('model', {}).get('name'),

            'final_pe_quantity_med': final_pes['quantity']['med'],            'complexity_level': config.get('simulation', {}).get('complexity_level')

            'final_pe_quantity_slow': final_pes['quantity']['slow'],        }

            'final_pe_cost_fast': final_pes['cost']['fast'],    }

            'final_pe_cost_med': final_pes['cost']['med'],    

            'final_pe_cost_slow': final_pes['cost']['slow'],    # Log summary

            'final_pe_causal_fast': final_pes['causal']['fast'],    logger.log_summary(summary)

            'final_pe_causal_med': final_pes['causal']['med'],    logger.close()

            'final_pe_causal_slow': final_pes['causal']['slow'],    

                return summary

            # PE statistics
            'pe_temporal_mean': pe_stats.get('temporal', {}).get('mean', 0),
            'pe_quantity_mean': pe_stats.get('quantity', {}).get('mean', 0),
            'pe_cost_mean': pe_stats.get('cost', {}).get('mean', 0),
            'pe_causal_mean': pe_stats.get('causal', {}).get('mean', 0),
            
            # Shock tracking
            'num_shocks': len(env.shock_history)
        }
        
        return summary
    
    def run_local_parallel(self, n_jobs: int = 1) -> List[Dict[str, Any]]:
        """
        Run experiments in parallel using joblib (local execution).
        
        Args:
            n_jobs: Number of parallel jobs
            
        Returns:
            List of summary results
        """
        from joblib import Parallel, delayed
        
        # Expand grid to get all run configurations
        all_configs = ConfigLoader.expand_grid(self.config)
        
        print(f"🔧 Running {len(all_configs)} experiments with {n_jobs} parallel jobs")
        
        # Run in parallel
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.run_single_experiment)(config) 
            for config in all_configs
        )
        
        # Aggregate results
        self._save_aggregated_results(results)
        
        return results
    
    def run_cluster_task(self, task_id: int) -> Dict[str, Any]:
        """
        Run a single task from SLURM array job.
        
        Args:
            task_id: Task ID from SLURM_ARRAY_TASK_ID
            
        Returns:
            Summary results
        """
        # Expand grid to get all run configurations
        all_configs = ConfigLoader.expand_grid(self.config)
        
        if task_id >= len(all_configs):
            raise ValueError(f"Task ID {task_id} exceeds number of configs ({len(all_configs)})")
        
        # Get config for this task
        task_config = all_configs[task_id]
        
        print(f"📋 Running task {task_id}/{len(all_configs)-1}")
        
        # Run experiment
        result = self.run_single_experiment(task_config)
        
        # Save task-specific result
        from cluster_utils import save_task_results
        results_dir = self.config.get('paths', {}).get('results_dir', 'results')
        save_task_results(task_id, result, results_dir, prefix="vendomini_task")
        
        return result
    
    def _save_aggregated_results(self, results: List[Dict[str, Any]]):
        """Save aggregated results to CSV."""
        experiment_name = self.config.get('experiment', {}).get('name', 'experiment')
        results_dir = self.config.get('paths', {}).get('results_dir', 'results')
        
        output_file = Path(results_dir) / f"{experiment_name}_results.csv"
        
        ResultsAggregator.aggregate_to_csv(results, str(output_file))
        
        print(f"📊 Saved aggregated results to {output_file}")
