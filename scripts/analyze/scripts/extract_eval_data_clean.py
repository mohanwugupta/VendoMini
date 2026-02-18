#!/usr/bin/env python3
"""
Extract time-series data from JSON log files into a tidy JSON format.

This script processes JSON files in model folders and extracts:
- Time-series data (money_balance, net_worth, units_sold) 
- Tool usage events with timestamps

Output: processed/timeseries_data.json with time-series data for each run
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import re
import csv

def extract_timeseries_from_json(json_file_path):
    """Extract time-series data from a JSON log file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        logs = data.get('logs', [])
        
        # Initialize time-series data
        timeseries_events = []
        tool_events = []
        
        # Process each log entry
        for log_entry in logs:
            timestamp = log_entry.get('relative_timestamp', 0)
            event = log_entry.get('event', '')
            payload = log_entry.get('payload')
            
            if event in ['money_balance', 'net_worth', 'units_sold']:
                timeseries_events.append({
                    'timestamp': timestamp,
                    'event_type': event,
                    'value': payload
                })
            elif event == 'tool_calls':
                tool_events.append({
                    'timestamp': timestamp,
                    'tool_name': payload
                })
        
        return metadata, timeseries_events, tool_events
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return None, [], []


def extract_run_id_from_filename(filename):
    """Extract run ID from filename"""
    # Try to extract run ID from various filename patterns
    patterns = [
        r'_([A-Za-z0-9]{22})_\d+\.json$',  # Pattern with run ID and epoch number
        r'_([A-Za-z0-9]{22})\.json$',      # Pattern like _TSeFRGu8PTSBx4WjChiKXS.json
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    # Fallback: use filename stem
    return Path(filename).stem


def main():
    parser = argparse.ArgumentParser(description="Extract time-series data from JSON log files")
    parser.add_argument("--input", "-i", default="data", help="Input directory containing model folders")
    parser.add_argument("--output", "-o", default="processed/timeseries_data.json", help="Output JSON file")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_runs = []
    
    # Find all JSON files in model folders
    for model_folder in input_dir.iterdir():
        if model_folder.is_dir() and not model_folder.name.startswith('.') and model_folder.name != 'processed':
            print(f"Processing model folder: {model_folder.name}")
            
            for json_file in model_folder.glob("*.json"):
                print(f"  Processing: {json_file.name}")
                
                # Extract epoch number from filename
                epoch_num = None
                if json_file.name.endswith('.json'):
                    try:
                        # Extract the last number before .json (e.g., "..._3.json" -> epoch 3)
                        epoch_num = int(json_file.stem.split('_')[-1])
                    except (ValueError, IndexError):
                        print(f"    Could not extract epoch from filename: {json_file.name}")
                
                # Extract run ID from filename
                run_id = extract_run_id_from_filename(json_file.name)
                
                metadata, timeseries_events, tool_events = extract_timeseries_from_json(json_file)
                if metadata:
                    run_data = {
                        'file': str(json_file),
                        'model_folder': model_folder.name,
                        'epoch': epoch_num,
                        'run_id': run_id,
                        'metadata': metadata,
                        'timeseries': timeseries_events,
                        'tools': tool_events
                    }
                    all_runs.append(run_data)
    
    # Write results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_runs, f, ensure_ascii=False, indent=2)
    
    print(f"\nExtracted time-series data for {len(all_runs)} runs")
    print(f"Output written to: {output_path}")
    
    # Also export a tidy CSV with one row per event (timeseries and tool events)
    output_csv_path = output_path.with_suffix('.csv')
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['file', 'model_folder', 'epoch', 'run_id', 'timestamp', 'event_type', 'value', 'metadata', 'event_index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for run in all_runs:
            metadata_str = json.dumps(run.get('metadata', {}), ensure_ascii=False)
            # Combine timeseries and tool events, normalize to common format
            combined = []
            for e in run.get('timeseries', []):
                combined.append({
                    'timestamp': e.get('timestamp', 0),
                    'event_type': e.get('event_type'),
                    'value': e.get('value')
                })
            for t in run.get('tools', []):
                combined.append({
                    'timestamp': t.get('timestamp', 0),
                    'event_type': 'tool_call',
                    'value': t.get('tool_name')
                })

            # Sort by timestamp to preserve time-series order, then enumerate
            combined.sort(key=lambda x: (x.get('timestamp', 0)))
            for idx, e in enumerate(combined):
                writer.writerow({
                    'file': run.get('file'),
                    'model_folder': run.get('model_folder'),
                    'epoch': run.get('epoch'),
                    'run_id': run.get('run_id'),
                    'timestamp': e.get('timestamp'),
                    'event_type': e.get('event_type'),
                    'value': e.get('value'),
                    'metadata': metadata_str,
                    'event_index': idx
                })

    print(f"CSV written to: {output_csv_path}")
    
    # Print summary statistics
    total_timeseries = defaultdict(int)
    total_tools = 0
    for run in all_runs:
        for event in run['timeseries']:
            total_timeseries[event['event_type']] += 1
        total_tools += len(run['tools'])
    
    print(f"\nSummary:")
    for event_type, count in total_timeseries.items():
        print(f"- {event_type}: {count} events")
    print(f"- tool_calls: {total_tools} events")


if __name__ == "__main__":
    main()
