"""Aggregate results from logs directory (for already-completed runs)."""

import json
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from logging_utils import ResultsAggregator


def main():
    logs_dir = Path('logs')
    
    print("=" * 60)
    print("VendoMini Results Aggregator (from logs)")
    print("=" * 60)
    
    # Find all run directories
    run_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    print(f"Found {len(run_dirs)} run directories")
    
    # Load all summaries
    all_results = []
    for run_dir in run_dirs:
        summary_file = run_dir / 'summary.json'
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
            except Exception as e:
                print(f"⚠️ Error loading {summary_file}: {e}")
    
    if not all_results:
        print("⚠️ No results found!")
        return
    
    print(f"\n✅ Loaded {len(all_results)} results")
    
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
        import numpy as np
        print(f"\nTime to crash (steps):")
        print(f"  Mean: {np.mean(times_to_crash):.1f}")
        print(f"  Median: {np.median(times_to_crash):.1f}")
        print(f"  Min: {np.min(times_to_crash):.0f}")
        print(f"  Max: {np.max(times_to_crash):.0f}")
    
    # Budget analysis
    final_budgets = [r.get('final_budget', 0) for r in all_results]
    if final_budgets:
        import numpy as np
        print(f"\nFinal budget:")
        print(f"  Mean: ${np.mean(final_budgets):.2f}")
        print(f"  Median: ${np.median(final_budgets):.2f}")
        print(f"  Min: ${np.min(final_budgets):.2f}")
        print(f"  Max: ${np.max(final_budgets):.2f}")
    
    print("\n✅ Aggregation complete!")


if __name__ == "__main__":
    main()
