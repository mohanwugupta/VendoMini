"""
Analyze experiment results and generate summary statistics.

Usage:
    python scripts/analyze_results.py --results results/phase1_core_hypothesis_results.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_path: str) -> pd.DataFrame:
    """Load results CSV."""
    return pd.read_csv(results_path)


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Crashed runs: {df['crashed'].sum()} ({df['crashed'].mean()*100:.1f}%)")
    
    if 'crash_type' in df.columns:
        print("\nCrash types:")
        crash_types = df[df['crashed']]['crash_type'].value_counts()
        for crash_type, count in crash_types.items():
            print(f"  {crash_type}: {count}")
    
    print(f"\nTime to crash (crashed runs):")
    crashed = df[df['crashed']]
    
    # Use 'total_steps' as time_to_crash if time_to_crash is not explicitly present
    time_col = 'time_to_crash' if 'time_to_crash' in df.columns else 'total_steps'
    
    if len(crashed) > 0 and time_col in crashed.columns:
        print(f"  Mean: {crashed[time_col].mean():.1f} steps")
        print(f"  Median: {crashed[time_col].median():.1f} steps")
        print(f"  Std: {crashed[time_col].std():.1f} steps")
    
    print(f"\nSuccess metrics:")
    if 'success_rate' in df.columns:
        print(f"  Mean success rate: {df['success_rate'].mean()*100:.1f}%")
    else:
        # Calculate from crashed column if success_rate is missing
        print(f"  Mean success rate: {(~df['crashed']).mean()*100:.1f}%")

    if 'final_budget' in df.columns:
        print(f"  Mean final budget: ${df['final_budget'].mean():.2f}")
    
    if 'fulfilled_orders' in df.columns:
        print(f"  Mean fulfilled orders: {df['fulfilled_orders'].mean():.1f}")


def plot_crash_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot crash time distribution."""
    plt.figure(figsize=(10, 6))
    
    crashed = df[df['crashed']]
    time_col = 'time_to_crash' if 'time_to_crash' in df.columns else 'total_steps'
    
    if len(crashed) > 0 and time_col in crashed.columns:
        sns.histplot(data=crashed, x=time_col, bins=30)
        plt.xlabel('Time to Crash (steps)')
        plt.ylabel('Count')
        plt.title('Distribution of Time to Crash')
        
        output_path = output_dir / 'crash_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
        plt.close()


def plot_crash_by_parameter(df: pd.DataFrame, param: str, output_dir: Path):
    """Plot crash rate by parameter."""
    if param not in df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    
    crash_rate = df.groupby(param)['crashed'].mean()
    crash_rate.plot(kind='bar')
    plt.xlabel(param)
    plt.ylabel('Crash Rate')
    plt.title(f'Crash Rate by {param}')
    plt.xticks(rotation=45, ha='right')
    
    output_path = output_dir / f'crash_by_{param}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    df = load_results(args.results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Print summary
    print_summary_stats(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_crash_distribution(df, output_dir)
    
    # Plot by key parameters if they exist
    for param in ['config.p_shock', 'config.pe_mag', 'config.prediction_mode', 
                  'config.complexity_level', 'config.model_name']:
        plot_crash_by_parameter(df, param, output_dir)
    
    print(f"\nAnalysis complete! Plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
