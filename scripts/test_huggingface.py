"""
Full VendoMini simulation test with HuggingFace LLM.

Runs a complete simulation with the LLM making real decisions.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import json


def run_full_simulation():
    """Run a complete VendoMini simulation with LLM."""
    
    print("\n" + "=" * 70)
    print("VendoMini Full Simulation Test with HuggingFace LLM")
    print("=" * 70)
    print()
    
    # Check GPU
    print("1. GPU Check")
    print("-" * 70)
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device_name = "GPU"
    else:
        print("⚠ CUDA not available - using CPU (slower)")
        device_name = "CPU"
    print()
    
    # Configuration
    print("2. Loading Configuration")
    print("-" * 70)
    
    # Use a simple test config
    config = {
        'experiment': {
            'name': 'llm_test',
            'run_id': 'test_run_001',
            'seed': 42
        },
        'paths': {
            'logs_dir': 'logs',
            'results_dir': 'results'
        },
        'environment': {
            'max_steps': 20,  # Short test
            'initial_budget': 1000,
            'initial_storage': {
                'snacks': 15,
                'drinks': 10,
                'candy': 8
            },
            'storage_capacity': 100,
            'delivery_cost': 50,
            'delivery_delay': 2,
            'tool_costs': {
                'tool_check_storage': 0,
                'tool_check_inbox': 0,
                'tool_order_delivery': 0
            },
            'shock_config': {
                'enabled': True,
                'frequency': 0.3,
                'types': {
                    'demand_spike': 0.5,
                    'supply_delay': 0.3,
                    'budget_cut': 0.2
                },
                'intensity_range': [0.5, 1.5]
            }
        },
        'model': {
            'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'temperature': 0.7,
            'max_tokens_per_call': 300,
            'context_length': 2048
        },
        'interface': {
            'prediction_mode': 'required',
            'prediction_format': 'structured',
            'memory_tools': 'none',
            'recovery_tools': 'none'
        },
        'pe_tracking': {
            'enabled': True,
            'pe_types': ['tool_outcome', 'delivery_arrival', 'shock_occurrence'],
            'ewma': {
                'enabled': True,
                'alpha': 0.3,
                'scales': [1, 5, 10]
            },
            'prediction_horizon': 5
        },
        'crash_detection': {
            'enabled': True,
            'thresholds': {
                'budget_depletion': 50,
                'inventory_depletion': 5,
                'excessive_ordering': 500,
                'tool_failure_rate': 0.5,
                'repeated_errors': 5,
                'pe_explosion': 3.0
            }
        },
        'logging': {
            'save_step_logs': True,
            'save_summary': True,
            'save_interval': 5
        },
        'measurement': {
            'pe_windows': [5, 10],
            'crash_threshold': 'moderate'
        },
        'simulation': {
            'complexity_level': 1
        }
    }
    
    print(f"✓ Configuration loaded")
    print(f"  Max steps: {config['environment']['max_steps']}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Initial budget: ${config['environment']['initial_budget']}")
    print()
    
    # Load components
    print("3. Initializing Components")
    print("-" * 70)
    
    try:
        from env import VendoMiniEnv
        from agent import LLMAgent
        from pe_calculator import PECalculator
        from crash_detector import CrashDetector
        from logging_utils import Logger
        
        print("Loading LLM model...")
        agent = LLMAgent(config)
        
        if agent.client is None:
            print("✗ Failed to load LLM client")
            print("  Install dependencies: pip install torch transformers accelerate")
            return False
        
        print(f"✓ LLM loaded on {device_name}")
        if agent.provider == 'huggingface' and torch.cuda.is_available():
            print(f"  VRAM used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        env = VendoMiniEnv(config, seed=42)
        pe_calc = PECalculator(windows=[5, 10])
        crash_detector = CrashDetector(threshold='moderate', window_size=10)
        logger = Logger('test_run_001', 'logs')
        
        print(f"✓ All components initialized")
        print()
        
    except Exception as e:
        print(f"✗ Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run simulation
    print("4. Running Simulation")
    print("-" * 70)
    print(f"Running {config['environment']['max_steps']} steps with LLM making decisions...")
    print()
    
    observation = env.reset()
    crashed = False
    crash_type = None
    
    available_tools = [
        'tool_check_inbox',
        'tool_check_storage',
        'tool_order',
        'tool_check_budget',
        'tool_quote'
    ]
    
    history = []
    
    try:
        for step in range(config['environment']['max_steps']):
            print(f"Step {step + 1}/{config['environment']['max_steps']}:")
            print(f"  Budget: ${observation['budget']:.2f}")
            print(f"  Storage: {observation['storage']}")
            
            # Get action from LLM
            action, prediction = agent.get_action_and_prediction(observation, available_tools)
            
            print(f"  🤖 LLM Action: {action['tool']}")
            if action.get('args'):
                print(f"     Args: {action['args']}")
            if prediction and prediction.get('prediction_text'):
                print(f"     Prediction: {prediction['prediction_text']}")
            
            # Execute action
            next_observation, done = env.step(action, prediction)
            
            # Get result
            if env.action_history:
                result = env.action_history[-1]['result']
                success = result.get('success', False)
                print(f"  {'✓' if success else '✗'} Result: {result.get('message', 'No message')}")
            
            # Compute PE
            if env.action_history:
                action_result = env.action_history[-1]['result']
                actual_outcome = {
                    'actual_success': action_result.get('success', False),
                    'actual_cost': action_result.get('price', 0) if 'price' in action_result else None,
                    'actual_quantity': action_result.get('quantity', 0) if 'quantity' in action_result else None,
                    'actual_delivery_day': action_result.get('eta_day', 0) if 'eta_day' in action_result else None
                }
                
                pe_dict = pe_calc.compute_pe(prediction, actual_outcome)
                pe_calc.update_accumulators(pe_dict)
                
                if pe_dict:
                    print(f"  📊 Prediction Error: {pe_dict}")
            
            # Log step
            step_record = {
                'step': step,
                'day': env.current_day,
                'observation': observation,
                'action': action,
                'prediction': prediction,
                'result': env.action_history[-1]['result'] if env.action_history else {},
                'pe': pe_dict if 'pe_dict' in locals() else {},
                'cumulative_pe': pe_calc.get_cumulative_pes(),
                'state': env.get_full_state()
            }
            history.append(step_record)
            logger.log_step(step_record)
            
            # Check for crash
            is_crashed, detected_type = crash_detector.update(history)
            if is_crashed:
                crashed = True
                crash_type = detected_type
                print(f"\n  💥 CRASH DETECTED: {crash_type}")
                print(f"     Simulation ended at step {step + 1}")
                break
            
            observation = next_observation
            
            if done:
                print(f"\n  🏁 Environment signaled completion at step {step + 1}")
                break
            
            print()
        
        print()
        print("=" * 70)
        print("5. Results Summary")
        print("=" * 70)
        
        final_state = env.get_full_state()
        cumulative_pes = pe_calc.get_cumulative_pes()
        
        print(f"Simulation completed: {len(history)} steps")
        print(f"Final budget: ${final_state['budget']:.2f}")
        print(f"Final storage: {final_state['storage']}")
        print(f"Orders fulfilled: {final_state['fulfilled_orders']}/{final_state['total_orders_requested']}")
        print(f"Crashed: {crashed}")
        if crashed:
            print(f"Crash type: {crash_type}")
        print()
        
        print("Cumulative Prediction Errors:")
        for pe_type, scales in cumulative_pes.items():
            if scales:
                print(f"  {pe_type}:")
                for scale, value in scales.items():
                    print(f"    {scale}: {value:.4f}")
        print()
        
        # Save summary
        summary = {
            'run_id': 'test_run_001',
            'total_steps': len(history),
            'crashed': crashed,
            'crash_type': crash_type,
            'final_budget': final_state['budget'],
            'final_storage': final_state['storage'],
            'fulfilled_orders': final_state['fulfilled_orders'],
            'cumulative_pes': cumulative_pes
        }
        
        logger.log_summary(summary)
        logger.close()
        
        print("=" * 70)
        print("✓ Full simulation test completed successfully!")
        print("=" * 70)
        print()
        print(f"Logs saved to: logs/test_run_001_steps.jsonl")
        print(f"Summary saved to: logs/test_run_001_summary.json")
        print()
        
        if agent.provider == 'huggingface' and torch.cuda.is_available():
            print(f"Final VRAM usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
            print()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the full test."""
    success = run_full_simulation()
    
    if success:
        print("🎉 Success! Your VendoMini setup is fully functional.")
        print()
        print("Next steps:")
        print("1. Try different models (edit model name in this script)")
        print("2. Run longer simulations (increase max_steps)")
        print("3. Use configs/local_test.yaml for grid experiments")
        print("4. See LOCAL_TESTING.md for more options")
    else:
        print("❌ Test failed. Check error messages above.")
    
    return success


if __name__ == '__main__':
    main()
