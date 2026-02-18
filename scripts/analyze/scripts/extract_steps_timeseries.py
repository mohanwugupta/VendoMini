"""
Extract time-series data from steps.jsonl files into a tidy CSV format.
"""
import argparse
import csv
from pathlib import Path
import sys

# Import adapter
try:
    from steps_adapter import iter_steps_jsonl, extract_assistant_text
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from steps_adapter import iter_steps_jsonl, extract_assistant_text

def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def process_single_file(steps_jsonl, run_id):
    rows = []
    for s in iter_steps_jsonl(steps_jsonl):
        obs = s.get("observation") or {}
        # In VendoMini, 'state' often holds the true environment state at that step
        state = s.get("state") or {}
        
        # Determine day/budget/storage from state if available, else observation
        day = state.get("day") if state.get("day") is not None else obs.get("day")
        budget = state.get("budget") if state.get("budget") is not None else obs.get("budget")
        
        # Try to get detailed storage breakdown
        storage_dict = state.get("storage") or obs.get("storage") or {}
        total_storage = sum(storage_dict.values()) if isinstance(storage_dict, dict) else 0

        rows.append({
            "run_id": run_id,
            "step": s.get("step"),
            "day": day,
            "budget": budget,
            "total_storage": total_storage,
            
            "tool": safe_get(s, "action", "tool"),
            "crash_detected": s.get("crash_detected"),
            "crash_type": s.get("crash_type"),

            # PE channels (Prediction Error)
            # Flatten out all PE metrics for time-series analysis
            "pe_temporal": safe_get(s, "pe", "temporal", default=0.0),
            "pe_quantity": safe_get(s, "pe", "quantity", default=0.0),
            "pe_cost": safe_get(s, "pe", "cost", default=0.0),
            "pe_causal": safe_get(s, "pe", "causal", default=0.0),
            
            "cum_temporal_fast": safe_get(s, "cumulative_pe", "temporal", "fast", default=0.0),
            "cum_quantity_fast": safe_get(s, "cumulative_pe", "quantity", "fast", default=0.0),
            "cum_cost_fast": safe_get(s, "cumulative_pe", "cost", "fast", default=0.0),
            "cum_causal_fast": safe_get(s, "cumulative_pe", "causal", "fast", default=0.0),
            
            "scratchpad_len": len(extract_assistant_text(s))
        })
    return rows

def main():
    parser = argparse.ArgumentParser(description='Extract time series from steps.jsonl')
    parser.add_argument('--logs-dir', type=str, help='Path to logs directory', default='logs')
    parser.add_argument('--output', type=str, default='processed/steps_timeseries.csv', help='Output CSV path')
    args = parser.parse_args()
    
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
         logs_dir = Path(__file__).parents[3] / 'logs'
    
    if not logs_dir.exists():
        print(f"Logs directory not found at {args.logs_dir} or {logs_dir}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    run_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and (d.name.startswith('run_') or d.name.startswith('test_'))]
    print(f"Found {len(run_dirs)} runs in {logs_dir}")
    
    all_rows = []
    for run_dir in run_dirs:
        steps_file = run_dir / 'steps.jsonl'
        if steps_file.exists():
            try:
                rows = process_single_file(str(steps_file), run_dir.name)
                all_rows.extend(rows)
            except Exception as e:
                print(f"Error processing {run_dir.name}: {e}")

    if all_rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} rows to {output_path}")
    else:
        print("No data found to extract.")

if __name__ == "__main__":
    main()
