"""
Convert VendoMini logs (run_*/steps.jsonl) into the legacy JSON format 
expected by VendBench analysis scripts.

This enables reusing:
- extract_eval_data_clean.py
- detect_model_breakdowns_sentiment.py
- extract_model_beliefs.py
"""

import json
import logging
from pathlib import Path
import argparse
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_singe_run(run_dir: Path) -> Dict[str, Any]:
    """Convert a single run directory into legacy JSON format."""
    steps_file = run_dir / 'steps.jsonl'
    if not steps_file.exists():
        logger.warning(f"Skipping {run_dir}, no steps.jsonl found.")
        return None

    # Load summary for metadata
    summary_file = run_dir / 'summary.json'
    summary = {}
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        except Exception:
            pass

    # Basic metadata structure
    legacy_data = {
        "metadata": {
            "run_id": run_dir.name,
            "model": summary.get('params', {}).get('agent', {}).get('model', {}).get('name', 'unknown'),
            "crashed": summary.get('crashed', False),
            "crash_type": summary.get('crash_type', None),
            # Add other relevant metadata if needed
        },
        "logs": []
    }

    steps = []
    try:
        with open(steps_file, 'r') as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading {steps_file}: {e}")
        return None

    # Convert steps to event logs
    logs = []
    
    for step in steps:
        step_num = step.get('step', 0)
        state = step.get('state', {})
        action = step.get('action', {})
        observation = step.get('observation', {})
        
        # 1. State events (budget, storage)
        if 'budget' in state:
            logs.append({
                "relative_timestamp": step_num,
                "event": "money_balance",
                "payload": state['budget']
            })
            # Approximation of net worth if we don't have inventory value easily calculated here
            logs.append({
                "relative_timestamp": step_num,
                "event": "net_worth", 
                "payload": state['budget']  # simplistic mapping for now
            })
            
        # 2. Action events (tool calls)
        tool_name = action.get('tool')
        if tool_name:
            logs.append({
                "relative_timestamp": step_num,
                "event": "tool_calls",
                "payload": tool_name
            })
            
            # Message content (for sentiment analysis)
            # In VendoMini, the action usually has a 'reasoning' or 'thought' field, or we extract it from scratchpad?
            # It seems 'scratchpad' or 'thought' is not always explicitly logged as a string in the top-level action dict 
            # unless it's a specific tool.
            # Assuming 'action' might have 'reasoning'
            reasoning = action.get('reasoning', '')
            if reasoning:
                logs.append({
                    "relative_timestamp": step_num,
                    "event": "model_message",
                    "payload": {
                        "content": reasoning,
                        "role": "assistant"
                    }
                })
        
        # 3. Observation events (user messages/feedback)
        # Not strictly required for all scripts but helpful for context
        pass

    legacy_data['logs'] = logs
    return legacy_data

def main():
    parser = argparse.ArgumentParser(description='Convert VendoMini logs to legacy format')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Path to VendoMini logs directory')
    parser.add_argument('--output-dir', type=str, default='processed/legacy_format', help='Directory to save converted JSONs')
    
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all run directories
    if not logs_dir.exists():
        # Try fallback logic similar to other scripts
        repo_root = Path(__file__).parent.parent.parent
        logs_dir = repo_root / 'logs'
        
    if not logs_dir.exists():
        logger.error(f"Logs directory not found at {args.logs_dir} or {logs_dir}")
        return

    run_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and (d.name.startswith('run_') or d.name.startswith('test_'))]
    logger.info(f"Found {len(run_dirs)} runs to process.")
    
    count = 0
    for run_dir in run_dirs:
        try:
            converted_data = convert_singe_run(run_dir)
            if converted_data:
                output_file = output_dir / f"{run_dir.name}.json"
                with open(output_file, 'w') as f:
                    json.dump(converted_data, f, indent=2)
                count += 1
        except Exception as e:
            logger.error(f"Failed to convert {run_dir.name}: {e}")
            
    logger.info(f"Successfully converted {count} runs to {output_dir}")

if __name__ == "__main__":
    main()
