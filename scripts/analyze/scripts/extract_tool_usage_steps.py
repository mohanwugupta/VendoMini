"""
Extract tool usage patterns from steps.jsonl files.
Adapted for VendoMini structure.
"""
import argparse
import csv
import pandas as pd
from pathlib import Path
from collections import Counter
import sys

# Import adapter
try:
    from steps_adapter import iter_steps_jsonl
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from steps_adapter import iter_steps_jsonl

def extract_tool_usage_from_steps(steps_jsonl_path: str, run_id: str):
    rows = []
    for s in iter_steps_jsonl(steps_jsonl_path):
        action = s.get("action") or {}
        tool = action.get("tool")
        args = action.get("args")
        
        # Also check observation/result for crash info
        # VendoMini logs crash info in the step state usually
        state = s.get("state") or {}
        
        rows.append({
            "run_id": run_id,
            "step": s.get("step"),
            "tool_name": tool,
            "args": str(args) if args else "",
            # Some logs might have crash info in top level, some in state
            "crash_detected": s.get("crash_detected") or state.get("crash_detected", False),
            "crash_type": s.get("crash_type") or state.get("crash_type", ""),
        })
    return rows

def analyze_tool_patterns(tool_rows):
    """Compute aggregate patterns from raw tool rows"""
    if not tool_rows:
        return {}
        
    tools = [r['tool_name'] for r in tool_rows if r['tool_name']]
    if not tools:
        return {}
        
    # 1. Tool counts
    tool_counts = Counter(tools)
    most_common = tool_counts.most_common(1)[0]
    
    # 2. Tool switching rate
    switches = 0
    if len(tools) > 1:
        for i in range(1, len(tools)):
            if tools[i] != tools[i-1]:
                switches += 1
    switching_rate = switches / (len(tools) - 1) if len(tools) > 1 else 0
    
    # 3. Loops (max streak of same tool)
    max_streak = 0
    current_streak = 0
    last_tool = None
    for t in tools:
        if t == last_tool:
            current_streak += 1
        else:
            current_streak = 1
            last_tool = t
        max_streak = max(max_streak, current_streak)

    return {
        "n_tool_calls": len(tools),
        "unique_tools": len(tool_counts),
        "most_common_tool": most_common[0],
        "most_common_tool_pct": most_common[1] / len(tools),
        "switching_rate": switching_rate,
        "max_repetition_streak": max_streak
    }

def main():
    parser = argparse.ArgumentParser(description='Extract tool usage from steps.jsonl')
    parser.add_argument('--results', type=str, default='results/aggregated_results.csv', help='Path to aggregated results CSV')
    parser.add_argument('--output', type=str, default='processed/tool_usage_stats.csv', help='Output CSV path')
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        results_path = Path(__file__).parents[3] / 'results' / 'aggregated_results.csv'
    
    if not results_path.exists():
        print(f"Results not found at {results_path}")
        return

    print(f"Loading runs from {results_path}...")
    df = pd.read_csv(results_path)
    logs_root = results_path.parent.parent / 'logs'
    
    all_stats = []
    
    print(f"Extracting tool usage for {len(df)} runs...")
    for _, row in df.iterrows():
        run_id = row.get('run_id')
        if not run_id:
            continue
            
        # Find steps.jsonl with robust path checking
        steps_file = logs_root / str(run_id) / 'steps.jsonl'
        if not steps_file.exists():
            steps_file = logs_root / f"run_{run_id}" / 'steps.jsonl'
            
        if not steps_file.exists():
             # Try searching for directory that contains the run_id
            found = False
            for d in logs_root.iterdir():
                if d.is_dir() and str(run_id) in d.name:
                    steps_file = d / 'steps.jsonl'
                    if steps_file.exists():
                        found = True
                        break
            if not found:
                continue
            
        # Extract raw rows
        tool_rows = extract_tool_usage_from_steps(str(steps_file), str(run_id))
        
        # Analyze patterns
        stats = analyze_tool_patterns(tool_rows)
        if stats:
            stats['run_id'] = run_id
            stats['crashed'] = row.get('crashed')
            stats['crash_type'] = row.get('crash_type')
            all_stats.append(stats)
            
    if all_stats:
        out_df = pd.DataFrame(all_stats)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved tool usage stats to {out_path}")
        print("\nTool Usage Stats by Crash Status:")
        print(out_df.groupby('crashed')[['switching_rate', 'max_repetition_streak', 'unique_tools']].mean())

if __name__ == "__main__":
    main()
