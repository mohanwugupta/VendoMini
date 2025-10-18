"""Aggregate results from SLURM array jobs."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cluster_utils import aggregate_task_results
from logging_utils import ResultsAggregator


def main():
    parser = argparse.ArgumentParser(description="Aggregate VendoMini SLURM task results")
    parser.add_argument('--input-dir', default='results', help='Directory containing task result files')
    parser.add_argument('--output', required=True, help='Output file (JSON or CSV)')
    parser.add_argument('--prefix', default='vendomini_task', help='Task file prefix')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VendoMini Results Aggregator")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print(f"Prefix: {args.prefix}")
    
    # Aggregate task results
    all_results = aggregate_task_results(
        args.input_dir,
        args.output,
        prefix=args.prefix
    )
    
    if not all_results:
        print("⚠️ No results found!")
        return
    
    # If output is CSV, also create it
    if args.output.endswith('.csv'):
        ResultsAggregator.aggregate_to_csv(all_results, args.output)
    elif args.output.endswith('.json'):
        # Also create CSV version
        csv_output = args.output.replace('.json', '.csv')
        ResultsAggregator.aggregate_to_csv(all_results, csv_output)
        print(f"📊 Also saved CSV version to {csv_output}")
    
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
    
    # Average time to crash
    times_to_crash = [r['time_to_crash'] for r in all_results if 'time_to_crash' in r]
    if times_to_crash:
        import numpy as np
        print(f"\nTime to crash:")
        print(f"  Mean: {np.mean(times_to_crash):.1f}")
        print(f"  Median: {np.median(times_to_crash):.1f}")
        print(f"  Min: {np.min(times_to_crash):.0f}")
        print(f"  Max: {np.max(times_to_crash):.0f}")
    
    print("\n✅ Aggregation complete!")


if __name__ == "__main__":
    main()
