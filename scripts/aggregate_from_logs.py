"""Aggregate results from logs directory (for already-completed runs)."""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'logs'))

from logging_utils import ResultsAggregator


def load_step_logs(run_dir: Path) -> list:
    """Load step-by-step logs from steps.jsonl file."""
    steps_file = run_dir / 'steps.jsonl'
    steps = []
    
    if steps_file.exists():
        try:
            with open(steps_file, 'r') as f:
                for line in f:
                    if line.strip():
                        steps.append(json.loads(line))
        except Exception as e:
            print(f"⚠️ Error loading steps from {steps_file}: {e}")
    
    return steps


def extract_detailed_metrics(run_dir: Path, summary: dict) -> dict:
    """Extract detailed metrics from step logs including scratchpad usage."""
    steps = load_step_logs(run_dir)
    
    if not steps:
        return {}
    
    # Track scratchpad usage over time
    scratchpad_sizes = []
    max_scratchpad = 0
    scratchpad_writes = 0
    scratchpad_reads = 0
    scratchpad_deletes = 0
    
    # Track action patterns
    action_counts = {}
    repeated_actions = 0
    last_action = None
    repeat_streak = 0
    max_repeat_streak = 0
    
    # Track PE accumulation
    final_cumulative_pe = None
    pe_trajectory = {
        'temporal_fast': [],
        'temporal_med': [],
        'temporal_slow': [],
        'quantity_fast': [],
        'quantity_med': [],
        'quantity_slow': [],
        'cost_fast': [],
        'cost_med': [],
        'cost_slow': [],
        'causal_fast': [],
        'causal_med': [],
        'causal_slow': []
    }
    
    # Track budget trajectory
    budget_trajectory = []
    
    for step in steps:
        # Scratchpad metrics
        state = step.get('state', {})
        scratchpad_size = state.get('scratchpad_size', 0)
        scratchpad_sizes.append(scratchpad_size)
        max_scratchpad = max(max_scratchpad, scratchpad_size)
        
        # Count scratchpad operations
        action = step.get('action', {})
        tool = action.get('tool', '')
        if tool == 'tool_write_scratchpad':
            scratchpad_writes += 1
        elif tool == 'tool_read_scratchpad':
            scratchpad_reads += 1
        elif tool == 'tool_delete_scratchpad':
            scratchpad_deletes += 1
        
        # Track action patterns
        action_counts[tool] = action_counts.get(tool, 0) + 1
        
        if tool == last_action:
            repeat_streak += 1
            max_repeat_streak = max(max_repeat_streak, repeat_streak)
        else:
            if repeat_streak >= 2:  # Count as repeated if 2+ in a row
                repeated_actions += 1
            repeat_streak = 1
            last_action = tool
        
        # Track PE accumulation
        cumulative_pe = step.get('cumulative_pe', {})
        if cumulative_pe:
            for pe_type in ['temporal', 'quantity', 'cost', 'causal']:
                if pe_type in cumulative_pe:
                    for scale in ['fast', 'med', 'slow']:
                        key = f'{pe_type}_{scale}'
                        if scale in cumulative_pe[pe_type]:
                            pe_trajectory[key].append(cumulative_pe[pe_type][scale])
            final_cumulative_pe = cumulative_pe
        
        # Budget trajectory
        if 'day' in state and 'budget' in state:
            budget_trajectory.append({
                'day': state['day'],
                'budget': state['budget']
            })
    
    # Calculate additional metrics
    metrics = {
        # Scratchpad metrics
        'scratchpad_max_size': max_scratchpad,
        'scratchpad_writes': scratchpad_writes,
        'scratchpad_reads': scratchpad_reads,
        'scratchpad_deletes': scratchpad_deletes,
        'scratchpad_used': max_scratchpad > 0,
        
        # Action pattern metrics
        'max_repeat_streak': max_repeat_streak,
        'num_repeated_actions': repeated_actions,
        'action_diversity': len(action_counts),
        'most_common_action': max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else None,
        'most_common_action_count': max(action_counts.values()) if action_counts else 0,
        
        # PE trajectory summaries (final values already in summary, but add rate of change)
        'pe_causal_fast_final': final_cumulative_pe.get('causal', {}).get('fast', 0) if final_cumulative_pe else 0,
        'pe_causal_med_final': final_cumulative_pe.get('causal', {}).get('med', 0) if final_cumulative_pe else 0,
        'pe_causal_slow_final': final_cumulative_pe.get('causal', {}).get('slow', 0) if final_cumulative_pe else 0,
        
        # Budget metrics
        'budget_initial': budget_trajectory[0]['budget'] if budget_trajectory else 0,
        'budget_final': budget_trajectory[-1]['budget'] if budget_trajectory else 0,
        'budget_min': min([b['budget'] for b in budget_trajectory]) if budget_trajectory else 0,
        'budget_max': max([b['budget'] for b in budget_trajectory]) if budget_trajectory else 0,
        'went_bankrupt': budget_trajectory[-1]['budget'] <= 0 if budget_trajectory else False,
    }
    
    # Calculate rate of budget burn
    if len(budget_trajectory) >= 2:
        total_days = budget_trajectory[-1]['day'] - budget_trajectory[0]['day']
        if total_days > 0:
            budget_change = budget_trajectory[-1]['budget'] - budget_trajectory[0]['budget']
            metrics['budget_burn_rate'] = budget_change / total_days
        else:
            metrics['budget_burn_rate'] = 0
    else:
        metrics['budget_burn_rate'] = 0
    
    return metrics


def main():
    # Prefer repo-root logs dir (works when running script from anywhere).
    # Path(__file__) is this script, .parent is scripts/, .parent.parent is project root
    logs_dir = Path(__file__).parent.parent / 'logs'
    
    # Fallback to cluster absolute path if repo-relative logs not present.
    if not logs_dir.exists():
        fallback = Path('/scratch/gpfs/JORDANAT/mg9965/VendoMini/logs')
        if fallback.exists():
            logs_dir = fallback
        else:
            print(f"⚠️ Logs directory not found at {logs_dir} or {fallback}")
            return
    
    print("=" * 60)
    print("VendoMini Results Aggregator (from logs)")
    print("=" * 60)
    
    # Find all run directories (starting with 'run_' or 'test_')
    run_dirs = sorted([
        d for d in logs_dir.iterdir() 
        if d.is_dir() and (d.name.startswith('run_') or d.name.startswith('test_'))
    ])
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Load all summaries and detailed metrics
    all_results = []
    for run_dir in run_dirs:
        summary_file = run_dir / 'summary.json'
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    result = json.load(f)
                    
                    # Extract detailed metrics from step logs
                    print(f"  Processing {run_dir.name}...")
                    detailed_metrics = extract_detailed_metrics(run_dir, result)
                    result.update(detailed_metrics)
                    
                    all_results.append(result)
            except Exception as e:
                print(f"⚠️ Error loading {summary_file}: {e}")
    
    if not all_results:
        print("⚠️ No results found!")
        return
    
    print(f"\n✅ Loaded {len(all_results)} results with detailed metrics")
    
    # Save aggregated results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_output = output_dir / 'aggregated_results.json'
    with open(json_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"📊 Saved JSON to {json_output}")
    
    # Save as CSV
    csv_output = output_dir / 'aggregated_results.csv'
    ResultsAggregator.aggregate_to_csv(all_results, csv_output)
    print(f"📊 Saved CSV to {csv_output}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total runs: {len(all_results)}")
    
    crashed_count = sum(1 for r in all_results if r.get('crashed', False))
    print(f"Crashed: {crashed_count} ({100*crashed_count/len(all_results):.1f}%)")
    print(f"Survived: {len(all_results) - crashed_count} ({100*(len(all_results)-crashed_count)/len(all_results):.1f}%)")
    
    # Crash type breakdown
    crash_types = {}
    for r in all_results:
        if r.get('crashed', False):
            crash_type = r.get('crash_type', 'unknown')
            crash_types[crash_type] = crash_types.get(crash_type, 0) + 1
    
    if crash_types:
        print("\nCrash types:")
        for crash_type, count in sorted(crash_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {crash_type}: {count} ({100*count/crashed_count:.1f}%)")
    
    # Time to crash stats
    times_to_crash = [r.get('total_steps', 0) for r in all_results if r.get('crashed', False)]
    if times_to_crash:
        print(f"\nTime to crash (steps):")
        print(f"  Mean: {np.mean(times_to_crash):.1f}")
        print(f"  Median: {np.median(times_to_crash):.1f}")
        print(f"  Min: {np.min(times_to_crash):.0f}")
        print(f"  Max: {np.max(times_to_crash):.0f}")
    
    # Budget analysis
    final_budgets = [r.get('final_budget', 0) for r in all_results]
    if final_budgets:
        print(f"\nFinal budget:")
        print(f"  Mean: ${np.mean(final_budgets):.2f}")
        print(f"  Median: ${np.median(final_budgets):.2f}")
        print(f"  Min: ${np.min(final_budgets):.2f}")
        print(f"  Max: ${np.max(final_budgets):.2f}")
    
    # Scratchpad usage analysis
    scratchpad_users = [r for r in all_results if r.get('scratchpad_used', False)]
    if scratchpad_users:
        print(f"\nScratchpad usage:")
        print(f"  Used scratchpad: {len(scratchpad_users)}/{len(all_results)} ({100*len(scratchpad_users)/len(all_results):.1f}%)")
        
        max_sizes = [r.get('scratchpad_max_size', 0) for r in scratchpad_users]
        if max_sizes:
            print(f"  Max size (mean): {np.mean(max_sizes):.1f} entries")
            print(f"  Max size (max): {np.max(max_sizes):.0f} entries")
        
        total_writes = sum(r.get('scratchpad_writes', 0) for r in scratchpad_users)
        total_reads = sum(r.get('scratchpad_reads', 0) for r in scratchpad_users)
        print(f"  Total writes: {total_writes}")
        print(f"  Total reads: {total_reads}")
        print(f"  Read/Write ratio: {total_reads/total_writes:.2f}" if total_writes > 0 else "  Read/Write ratio: N/A")
    
    # Action pattern analysis
    print(f"\nAction patterns:")
    max_streaks = [r.get('max_repeat_streak', 0) for r in all_results]
    if max_streaks:
        print(f"  Max repeat streak (mean): {np.mean(max_streaks):.1f}")
        print(f"  Max repeat streak (max): {np.max(max_streaks):.0f}")
    
    action_diversity = [r.get('action_diversity', 0) for r in all_results]
    if action_diversity:
        print(f"  Action diversity (mean): {np.mean(action_diversity):.1f} unique tools")
        print(f"  Action diversity (max): {np.max(action_diversity):.0f} unique tools")
    
    # Budget trajectory analysis
    went_bankrupt = sum(1 for r in all_results if r.get('went_bankrupt', False))
    if went_bankrupt > 0:
        print(f"\nBankruptcy:")
        print(f"  Went bankrupt: {went_bankrupt}/{len(all_results)} ({100*went_bankrupt/len(all_results):.1f}%)")
    
    burn_rates = [r.get('budget_burn_rate', 0) for r in all_results if 'budget_burn_rate' in r]
    if burn_rates:
        print(f"  Budget burn rate (mean): ${np.mean(burn_rates):.2f}/day")
        print(f"  Budget burn rate (median): ${np.median(burn_rates):.2f}/day")
    
    print("\n✅ Aggregation complete!")


if __name__ == "__main__":
    main()
