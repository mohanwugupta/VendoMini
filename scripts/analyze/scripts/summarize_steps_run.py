"""
Summarize runs based on steps.jsonl content.
Calculates crash onset, dominant crash type, tool usage stats.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
import sys

# Import adapter
try:
    from steps_adapter import iter_steps_jsonl
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from steps_adapter import iter_steps_jsonl

def summarize_run(steps_jsonl_path: str, run_id: str):
    steps = list(iter_steps_jsonl(steps_jsonl_path))
    if not steps:
        return None
        
    # Crash analysis
    crash_steps = [s.get("step", 0) for s in steps if s.get("crash_detected")]
    crash_onset = min(crash_steps) if crash_steps else None
    
    # Crash type from detected steps
    crash_types = [s.get("crash_type") for s in steps if s.get("crash_detected") and s.get("crash_type")]
    crash_type_first = crash_types[0] if crash_types else None
    crash_type_mode = Counter(crash_types).most_common(1)[0][0] if crash_types else None

    # Tool analysis
    tools = [(s.get("action") or {}).get("tool") for s in steps]
    # Filter out Nones
    tools = [t for t in tools if t]
    tool_counts = Counter(tools)
    
    summary = {
        "run_id": run_id,
        "n_steps": len(steps),
        "crashed": crash_onset is not None,
        "crash_onset_step": crash_onset if crash_onset else "",
        "crash_type_first": crash_type_first if crash_type_first else "",
        "crash_type_mode": crash_type_mode if crash_type_mode else "",
        "unique_tools": len(tool_counts),
        "most_common_tool": tool_counts.most_common(1)[0][0] if tool_counts else "",
        "most_common_tool_count": tool_counts.most_common(1)[0][1] if tool_counts else 0,
        "final_budget": (steps[-1].get("state") or {}).get("budget", "")
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description='Summarize runs from steps.jsonl')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Path to logs directory')
    parser.add_argument('--output', type=str, default='processed/run_summaries_from_steps.csv', help='Output CSV path')
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
    print(f"Found {len(run_dirs)} runs to summarize.")
    
    summaries = []
    for run_dir in run_dirs:
        steps_file = run_dir / 'steps.jsonl'
        if steps_file.exists():
            try:
                s = summarize_run(str(steps_file), run_dir.name)
                if s:
                    summaries.append(s)
            except Exception as e:
                print(f"Error summarizing {run_dir.name}: {e}")
                
    if summaries:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Saved summaries to {output_path}")
    else:
        print("No summaries generated.")

if __name__ == "__main__":
    main()
