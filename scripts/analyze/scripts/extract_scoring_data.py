#!/usr/bin/env python3
"""
Extract scoring data from all .eval files in the data folder.

This script processes .eval files (which are ZIP archives) and extracts:
- Overall scores from header.json (reduced across epochs)
- Per-sample scores from reductions.json
- Model metadata and run information

Output: processed/scoring_data.csv with scoring information for each run
"""

import argparse
import json
import zipfile
from pathlib import Path
import csv
from typing import Dict, List, Optional
import re

def extract_run_metadata(file_path: Path) -> Dict:
    """Extract run metadata from eval file path and header"""
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            header_data = json.loads(zf.read("header.json").decode('utf-8'))
            eval_info = header_data.get("eval", {})
            
            # Extract metadata from header
            metadata = {
                'eval_file': str(file_path),
                'model_folder': file_path.parent.name,
                'run_id': eval_info.get('run_id', ''),
                'task_id': eval_info.get('task_id', ''),
                'task': eval_info.get('task', ''),
                'model': eval_info.get('model', ''),
                'created': eval_info.get('created', ''),
                'epochs': eval_info.get('config', {}).get('epochs', 0),
                'total_samples': header_data.get('results', {}).get('total_samples', 0),
                'completed_samples': header_data.get('results', {}).get('completed_samples', 0),
                'status': header_data.get('status', ''),
            }
            
            # Extract model usage stats if available
            stats = header_data.get('stats', {})
            model_usage = stats.get('model_usage', {})
            if model_usage:
                # Get the first (and usually only) model's usage
                for model_name, usage in model_usage.items():
                    metadata.update({
                        'input_tokens': usage.get('input_tokens', 0),
                        'output_tokens': usage.get('output_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0),
                    })
                    break  # Just take the first model
                    
            # Calculate duration if timestamps available
            if 'started_at' in stats and 'completed_at' in stats:
                from datetime import datetime
                try:
                    start = datetime.fromisoformat(stats['started_at'].replace('+01-00', '+01:00'))
                    end = datetime.fromisoformat(stats['completed_at'].replace('+01-00', '+01:00'))
                    duration_seconds = (end - start).total_seconds()
                    metadata['duration_seconds'] = duration_seconds
                    metadata['duration_hours'] = duration_seconds / 3600
                except Exception:
                    metadata['duration_seconds'] = None
                    metadata['duration_hours'] = None
            
            return metadata
            
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return {
            'eval_file': str(file_path),
            'model_folder': file_path.parent.name,
            'run_id': '',
            'task_id': '',
            'task': '',
            'model': '',
            'created': '',
            'epochs': 0,
            'total_samples': 0,
            'completed_samples': 0,
            'status': 'error',
        }

def extract_overall_scores(file_path: Path) -> Dict:
    """Extract overall scores from header.json"""
    scores = {}
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            header_data = json.loads(zf.read("header.json").decode('utf-8'))
            results = header_data.get('results', {})
            
            # Extract scores from results
            for score_entry in results.get('scores', []):
                score_name = score_entry.get('name', '')
                metrics = score_entry.get('metrics', {})
                
                # Handle different metric types
                if 'bool_score' in metrics:
                    scores[f"{score_name}_bool"] = metrics['bool_score'].get('value', None)
                elif 'num_score' in metrics:
                    scores[f"{score_name}_num"] = metrics['num_score'].get('value', None)
                    
    except Exception as e:
        print(f"Error reading overall scores from {file_path}: {e}")
        
    return scores

def extract_individual_epoch_scores(file_path: Path) -> List[Dict]:
    """Extract scores from each individual epoch's sample file"""
    epoch_scores = []
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Find all sample files (each represents an epoch)
            sample_files = [name for name in zf.namelist() if name.startswith('samples/') and name.endswith('.json')]
            
            for sample_file in sample_files:
                try:
                    sample_data = json.loads(zf.read(sample_file).decode('utf-8'))
                    
                    # Extract basic information
                    epoch = sample_data.get('epoch', 0)
                    sample_id = sample_data.get('id', 1)
                    
                    # Extract scores if they exist
                    epoch_data = {
                        'epoch': epoch,
                        'sample_id': sample_id,
                        'sample_file': sample_file
                    }
                    
                    if 'scores' in sample_data:
                        scores = sample_data['scores']
                        
                        # Extract boolean metrics
                        if 'boolean_metrics' in scores and 'value' in scores['boolean_metrics']:
                            bool_metrics = scores['boolean_metrics']['value']
                            for metric, value in bool_metrics.items():
                                epoch_data[f"{metric}_bool"] = value
                        
                        # Extract numeric metrics
                        if 'numeric_metrics' in scores and 'value' in scores['numeric_metrics']:
                            num_metrics = scores['numeric_metrics']['value']
                            for metric, value in num_metrics.items():
                                epoch_data[f"{metric}_num"] = value
                    
                    # Extract additional store metrics if available
                    if 'store' in sample_data:
                        store = sample_data['store']
                        # These might provide additional context
                        epoch_data['store_money_balance'] = store.get('money_balance')
                        epoch_data['relative_time'] = store.get('relative_time')
                        
                        # Calculate net worth from inventory if not in scores
                        if f"net_worth_num" not in epoch_data:
                            # Try to calculate from store data if possible
                            pass
                    
                    # Extract model usage for this epoch
                    if 'model_usage' in sample_data:
                        model_usage = sample_data['model_usage']
                        for model_name, usage in model_usage.items():
                            epoch_data[f'epoch_input_tokens'] = usage.get('input_tokens', 0)
                            epoch_data[f'epoch_output_tokens'] = usage.get('output_tokens', 0)
                            epoch_data[f'epoch_total_tokens'] = usage.get('total_tokens', 0)
                            break  # Just take the first model
                    
                    epoch_scores.append(epoch_data)
                    
                except Exception as e:
                    print(f"Error processing sample file {sample_file}: {e}")
                    continue
            
    except Exception as e:
        print(f"Error reading epoch scores from {file_path}: {e}")
        
    return epoch_scores

def extract_sample_scores(file_path: Path) -> List[Dict]:
    """Extract per-sample scores from reductions.json (kept for backward compatibility)"""
    sample_scores = []
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            reductions_data = json.loads(zf.read("reductions.json").decode('utf-8'))
            
            # Combine all samples across different scorers
            samples_by_id = {}
            
            for scorer_data in reductions_data:
                scorer_name = scorer_data.get('scorer', '')
                samples = scorer_data.get('samples', [])
                
                for sample in samples:
                    sample_id = sample.get('sample_id', 1)
                    value_dict = sample.get('value', {})
                    
                    if sample_id not in samples_by_id:
                        samples_by_id[sample_id] = {'sample_id': sample_id}
                    
                    # Add scores with scorer type suffix
                    if scorer_name == 'boolean_metrics':
                        for metric, value in value_dict.items():
                            samples_by_id[sample_id][f"{metric}_bool"] = value
                    elif scorer_name == 'numeric_metrics':
                        for metric, value in value_dict.items():
                            samples_by_id[sample_id][f"{metric}_num"] = value
                    else:
                        # Generic scorer
                        for metric, value in value_dict.items():
                            samples_by_id[sample_id][f"{metric}_{scorer_name}"] = value
            
            sample_scores = list(samples_by_id.values())
            
    except Exception as e:
        print(f"Error reading sample scores from {file_path}: {e}")
        
    return sample_scores

def main():
    parser = argparse.ArgumentParser(description="Extract scoring data from .eval files")
    parser.add_argument("--input", "-i", default="data", help="Input directory containing .eval files")
    parser.add_argument("--output", "-o", default="processed/scoring_data.csv", help="Output CSV file")
    parser.add_argument("--output-samples", default="processed/scoring_data_samples.csv", help="Output CSV file for per-sample data")
    parser.add_argument("--output-epochs", default="processed/scoring_data_epochs.csv", help="Output CSV file for per-epoch data")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_samples_path = Path(args.output_samples)
    output_epochs_path = Path(args.output_epochs)
    
    # Create output directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_samples_path.parent.mkdir(parents=True, exist_ok=True)
    output_epochs_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all .eval files
    eval_files = list(input_dir.rglob("*.eval"))
    print(f"Found {len(eval_files)} .eval files")
    
    if not eval_files:
        print("No .eval files found!")
        return
    
    # Process each eval file
    all_runs = []
    all_sample_data = []
    all_epoch_data = []
    
    for eval_file in eval_files:
        print(f"Processing: {eval_file}")
        
        # Extract metadata and overall scores
        metadata = extract_run_metadata(eval_file)
        overall_scores = extract_overall_scores(eval_file)
        
        # Combine metadata and overall scores
        run_data = {**metadata, **overall_scores}
        all_runs.append(run_data)
        
        # Extract per-sample scores (from reductions.json)
        sample_scores = extract_sample_scores(eval_file)
        for sample_score in sample_scores:
            sample_data = {**metadata, **sample_score}
            all_sample_data.append(sample_data)
        
        # Extract individual epoch scores (NEW - from each sample file)
        epoch_scores = extract_individual_epoch_scores(eval_file)
        for epoch_score in epoch_scores:
            epoch_data = {**metadata, **epoch_score}
            all_epoch_data.append(epoch_data)
    
    # Write overall scoring data
    if all_runs:
        # Get all unique field names
        all_fields = set()
        for run in all_runs:
            all_fields.update(run.keys())
        
        # Sort fields for consistent output
        fieldnames = sorted(list(all_fields))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_runs)
        
        print(f"\nOverall scoring data written to: {output_path}")
        print(f"Processed {len(all_runs)} runs")
    
    # Write per-sample scoring data
    if all_sample_data:
        # Get all unique field names for samples
        all_sample_fields = set()
        for sample in all_sample_data:
            all_sample_fields.update(sample.keys())
        
        # Sort fields for consistent output
        sample_fieldnames = sorted(list(all_sample_fields))
        
        with open(output_samples_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sample_fieldnames)
            writer.writeheader()
            writer.writerows(all_sample_data)
        
        print(f"Per-sample scoring data written to: {output_samples_path}")
        print(f"Processed {len(all_sample_data)} samples")
    
    # Write per-epoch scoring data (NEW)
    if all_epoch_data:
        # Get all unique field names for epochs
        all_epoch_fields = set()
        for epoch in all_epoch_data:
            all_epoch_fields.update(epoch.keys())
        
        # Sort fields for consistent output
        epoch_fieldnames = sorted(list(all_epoch_fields))
        
        with open(output_epochs_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=epoch_fieldnames)
            writer.writeheader()
            writer.writerows(all_epoch_data)
        
        print(f"Per-epoch scoring data written to: {output_epochs_path}")
        print(f"Processed {len(all_epoch_data)} individual epochs")
    
    # Print summary statistics
    if all_runs:
        print(f"\nSummary:")
        print(f"- Total runs: {len(all_runs)}")
        print(f"- Total epochs: {len(all_epoch_data)}")
        
        # Count by model folder
        model_counts = {}
        epoch_counts = {}
        for run in all_runs:
            model_folder = run.get('model_folder', 'unknown')
            model_counts[model_folder] = model_counts.get(model_folder, 0) + 1
        
        for epoch in all_epoch_data:
            model_folder = epoch.get('model_folder', 'unknown')
            epoch_counts[model_folder] = epoch_counts.get(model_folder, 0) + 1
        
        print(f"- Runs by model folder:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count} runs ({epoch_counts.get(model, 0)} epochs)")
        
        # Count successful runs
        successful_runs = [r for r in all_runs if r.get('status') == 'success']
        print(f"- Successful runs: {len(successful_runs)}")
        
        # Print example metrics available
        if all_epoch_data:
            example_epoch = all_epoch_data[0]
            score_fields = [k for k in example_epoch.keys() if k.endswith('_bool') or k.endswith('_num')]
            if score_fields:
                print(f"- Available metrics per epoch: {', '.join(sorted(score_fields))}")

if __name__ == "__main__":
    main()
