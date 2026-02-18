#!/usr/bin/env python3
"""
Extract model's perceived metrics from conversation logs in JSON files.

This script parses the model's messages to extract what the model thinks
the metrics are at each step, enabling calculation of divergence scores
between actual and perceived performance.

Outputs a tidy CSV with model beliefs that can be compared against actual metrics.
"""

import json
import csv
import re
import argparse
from pathlib import Path
import pandas as pd

def extract_numeric_from_text(text, metric_name):
    """Extract numeric values that the model mentions for specific metrics"""
    patterns = {
        'money_balance': [
            r'(?:current\s+)?(?:money\s+)?balance[:\s]*\$?(\d+(?:\.\d+)?)',
            r'(?:have|got|remaining)\s+\$(\d+(?:\.\d+)?)',
            r'money[:\s]*\$?(\d+(?:\.\d+)?)',
            r'cash[:\s]*\$?(\d+(?:\.\d+)?)',
            r'\$(\d+(?:\.\d+)?)\s+(?:left|remaining|available)',
            r'wallet[:\s]*\$?(\d+(?:\.\d+)?)',
            r'funds?[:\s]*\$?(\d+(?:\.\d+)?)'
        ],
        'net_worth': [
            r'net\s*worth[:\s]*\$?(\d+(?:\.\d+)?)',
            r'total\s*(?:value|worth)[:\s]*\$?(\d+(?:\.\d+)?)',
            r'worth[:\s]*\$?(\d+(?:\.\d+)?)',
            r'total\s*assets?[:\s]*\$?(\d+(?:\.\d+)?)'
        ],
        'units_sold': [
            r'(?:total\s+)?units?\s+sold[:\s]*(\d+)',
            r'sold\s+(\d+)\s+units?',
            r'(\d+)\s+units?\s+sold',
            r'(?:have\s+)?sold[:\s]*(\d+)',
            r'sales?[:\s]*(\d+)\s+units?',
            r'items?\s+sold[:\s]*(\d+)',
            r'products?\s+sold[:\s]*(\d+)'
        ],
        'revenue': [
            r'(?:total\s+)?revenue[:\s]*\$?(\d+(?:\.\d+)?)',
            r'sales?\s+revenue[:\s]*\$?(\d+(?:\.\d+)?)',
            r'earned[:\s]*\$?(\d+(?:\.\d+)?)',
            r'made[:\s]*\$?(\d+(?:\.\d+)?)',
            r'income[:\s]*\$?(\d+(?:\.\d+)?)',
            r'gross[:\s]*\$?(\d+(?:\.\d+)?)'
        ],
        'profit': [
            r'profit[:\s]*\$?(\d+(?:\.\d+)?)',
            r'(?:net\s+)?profit[:\s]*\$?(\d+(?:\.\d+)?)',
            r'profit\s+margin[:\s]*\$?(\d+(?:\.\d+)?)'
        ]
    }
    
    if metric_name not in patterns:
        return []
    
    values = []
    for pattern in patterns[metric_name]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                values.append(float(match))
            except ValueError:
                continue
    
    return values

def extract_model_beliefs_from_messages(messages, epoch):
    """Extract what the model believes about metrics from its messages"""
    beliefs = []
    
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", "")
        
        # Handle different content formats
        if isinstance(content, list):
            # Some models have content as a list of objects
            content_text = ""
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    content_text += item["text"] + " "
                elif isinstance(item, str):
                    content_text += item + " "
            content = content_text.strip()
        elif not isinstance(content, str):
            content = str(content)
        
        if not content:
            continue
        
        # Extract step number from context (look at surrounding tool calls)
        step = i  # Use message index as step approximation
        
        # Look for the model's statements about metrics
        metrics_to_extract = ['money_balance', 'net_worth', 'units_sold', 'revenue', 'profit']
        
        for metric in metrics_to_extract:
            values = extract_numeric_from_text(content, metric)
            
            if values:
                # Always create a full context entry with all values
                full_context = content[:500] + "..." if len(content) > 500 else content
                
                # Create individual rows for each value found (for detailed analysis)
                for j, value in enumerate(values):
                    beliefs.append({
                        "epoch": epoch,
                        "step": step,
                        "name": metric,
                        "value": value,
                        "metric_type": "individual",
                        "message_index": i,
                        "value_index": j,
                        "full_context": full_context,
                        "summary_value": None,  # Will be filled below
                        "all_values": str(values)
                    })
                
                # Create a summary row with a single representative value
                # Use the most salient value (highest for money/revenue, latest mentioned, etc.)
                if metric in ['money_balance', 'net_worth', 'revenue', 'profit']:
                    # For monetary values, use the highest value (most salient)
                    summary_value = max(values)
                elif metric == 'units_sold':
                    # For units sold, use the sum if multiple values, or the highest
                    summary_value = sum(values) if len(values) > 1 else max(values)
                else:
                    # Default: use the last mentioned value
                    summary_value = values[-1]
                
                # Create summary row
                beliefs.append({
                    "epoch": epoch,
                    "step": step,
                    "name": metric,
                    "value": summary_value,
                    "metric_type": "summary",
                    "message_index": i,
                    "value_index": -1,  # -1 indicates summary
                    "full_context": full_context,
                    "summary_value": summary_value,
                    "all_values": str(values)
                })
                
                # Update individual rows with summary value for easy filtering
                for belief in beliefs:
                    if (belief["epoch"] == epoch and belief["step"] == step and 
                        belief["name"] == metric and belief["metric_type"] == "individual" and
                        belief["summary_value"] is None):
                        belief["summary_value"] = summary_value
    
    return beliefs

def extract_model_reasoning_patterns(messages):
    """Extract reasoning patterns and confidence indicators"""
    patterns = []
    
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
            
        content = msg.get("content", "")
        
        # Handle different content formats
        if isinstance(content, list):
            # Some models have content as a list of objects
            content_text = ""
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    content_text += item["text"] + " "
                elif isinstance(item, str):
                    content_text += item + " "
            content = content_text.strip()
        elif not isinstance(content, str):
            content = str(content)
        
        content_lower = content.lower()
        
        # Look for confidence indicators
        confidence_patterns = [
            r'(?:i\s+)?(?:think|believe|estimate|guess|assume)',
            r'(?:probably|likely|maybe|perhaps|possibly)',
            r'(?:approximately|about|around|roughly)',
            r'(?:should be|might be|could be)',
            r'(?:not sure|uncertain|unclear)'
        ]
        
        confidence_level = "high"  # default
        for pattern in confidence_patterns:
            if re.search(pattern, content_lower):
                confidence_level = "low"
                break
        
        # Look for calculation attempts
        has_calculation = bool(re.search(r'[\+\-\*\/\=]|\d+\s*[\+\-\*\/]\s*\d+', content))
        
        # Look for explicit metric tracking
        tracks_metrics = bool(re.search(r'(?:track|monitor|keep track|record|note)', content_lower))
        
        patterns.append({
            "message_index": i,
            "confidence_level": confidence_level,
            "has_calculation": has_calculation,
            "tracks_metrics": tracks_metrics,
            "message_length": len(content)
        })
    
    return patterns

def process_json_file(file_path):
    """Process a single JSON file and extract model beliefs"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        logs = data.get('logs', [])
        
        # Extract model information from filename or metadata
        filename = file_path.name
        model = "unknown"
        if "_openai_" in filename:
            model_part = filename.split("_openai_")[1].split("_")[0]
            model = f"openai_{model_part}"
        elif "_google_" in filename:
            model_part = filename.split("_google_")[1].split("_")[0]
            model = f"google_{model_part}"
        elif "_bedrock_" in filename:
            model_part = filename.split("_bedrock_")[1].split("_")[0]
            model = f"bedrock_{model_part}"
        elif "_human_" in filename:
            model = "human"
        
        # Extract epoch from filename
        epoch = 1
        if filename.endswith('.json'):
            try:
                epoch = int(filename.split('_')[-1].replace('.json', ''))
            except (ValueError, IndexError):
                pass
        
        # Extract run ID from filename
        run_id = file_path.stem
        patterns = [
            r'_([A-Za-z0-9]{22})_\d+\.json$',
            r'_([A-Za-z0-9]{22})\.json$',
        ]
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                run_id = match.group(1)
                break
        
        task = metadata.get('task', 'market_simulation')
        
        # Find assistant messages in logs
        messages = []
        for log_entry in logs:
            event = log_entry.get('event', '')
            if event == 'output.message.text':
                payload = log_entry.get('payload')
                if payload:  # payload is the text content directly
                    messages.append({
                        'role': 'assistant',
                        'content': payload,
                        'timestamp': log_entry.get('relative_timestamp', 0)
                    })
        
        # Extract model beliefs and patterns
        all_beliefs = extract_model_beliefs_from_messages(messages, epoch)
        all_patterns = extract_model_reasoning_patterns(messages)
        
        # Add metadata to beliefs
        for belief in all_beliefs:
            belief["run_id"] = run_id
            belief["model"] = model
            belief["task"] = task
            belief["file"] = str(file_path)
        
        # Add metadata to patterns
        for pattern in all_patterns:
            pattern["run_id"] = run_id
            pattern["model"] = model
            pattern["epoch"] = epoch
            pattern["file"] = str(file_path)
        
        return all_beliefs, all_patterns
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

def main():
    parser = argparse.ArgumentParser(description="Extract model beliefs from JSON log files")
    parser.add_argument("--input", "-i", default="data", help="Input directory containing model folders with JSON files")
    parser.add_argument("--output", "-o", default="processed", help="Output directory for CSV files")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in model folders
    json_files = []
    for model_folder in input_dir.iterdir():
        if model_folder.is_dir() and not model_folder.name.startswith('.') and model_folder.name != 'processed':
            json_files.extend(list(model_folder.glob("*.json")))
    
    print(f"Found {len(json_files)} JSON files")
    
    if not json_files:
        print("No JSON files found!")
        return
    
    # Collect all data
    all_beliefs = []
    all_patterns = []
    
    for i, file_path in enumerate(json_files, 1):
        print(f"Processing {i}/{len(json_files)}: {file_path.name}")
        
        beliefs, patterns = process_json_file(file_path)
        all_beliefs.extend(beliefs)
        all_patterns.extend(patterns)
    
    # Write beliefs to CSV
    beliefs_file = output_dir / "model_beliefs.csv"
    print(f"Writing {len(all_beliefs)} belief records to {beliefs_file}")
    
    if all_beliefs:
        fieldnames = ["run_id", "model", "task", "epoch", "step", "name", "value", 
                     "metric_type", "message_index", "value_index", "full_context", 
                     "summary_value", "all_values", "file"]
        
        with open(beliefs_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_beliefs)
    
    # Write reasoning patterns to CSV
    patterns_file = output_dir / "model_reasoning_patterns.csv"
    print(f"Writing {len(all_patterns)} pattern records to {patterns_file}")
    
    if all_patterns:
        pattern_fieldnames = ["run_id", "model", "epoch", "message_index", 
                             "confidence_level", "has_calculation", "tracks_metrics", "message_length", "file"]
        
        with open(patterns_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=pattern_fieldnames)
            writer.writeheader()
            writer.writerows(all_patterns)
    
    print("Done!")
    
    # Print summary
    if all_beliefs:
        beliefs_df = pd.DataFrame(all_beliefs)
        unique_runs = beliefs_df['run_id'].nunique()
        unique_models = beliefs_df['model'].nunique()
        
        print(f"\nBelief Extraction Summary:")
        print(f"- Total belief records: {len(all_beliefs)}")
        print(f"- Unique runs: {unique_runs}")
        print(f"- Unique models: {unique_models}")
        
        if 'name' in beliefs_df.columns:
            print(f"- Belief metrics extracted:")
            for metric, count in beliefs_df['name'].value_counts().items():
                print(f"  {metric}: {count} mentions")
        
        if 'metric_type' in beliefs_df.columns:
            print(f"- Record types:")
            for record_type, count in beliefs_df['metric_type'].value_counts().items():
                print(f"  {record_type}: {count} records")
        
        # Summary statistics for summary values only
        summary_df = beliefs_df[beliefs_df['metric_type'] == 'summary']
        if not summary_df.empty:
            print(f"\nSummary Value Statistics:")
            for metric in summary_df['name'].unique():
                metric_data = summary_df[summary_df['name'] == metric]['value']
                print(f"  {metric}: mean={metric_data.mean():.2f}, std={metric_data.std():.2f}, "
                      f"min={metric_data.min():.2f}, max={metric_data.max():.2f}, count={len(metric_data)}")
    
    if all_patterns:
        patterns_df = pd.DataFrame(all_patterns)
        print(f"\nReasoning Pattern Summary:")
        print(f"- Total pattern records: {len(all_patterns)}")
        
        if 'confidence_level' in patterns_df.columns:
            print(f"- Confidence levels:")
            for level, count in patterns_df['confidence_level'].value_counts().items():
                print(f"  {level}: {count} messages")
        
        if 'has_calculation' in patterns_df.columns:
            calc_count = patterns_df['has_calculation'].sum()
            print(f"- Messages with calculations: {calc_count}")
        
        if 'tracks_metrics' in patterns_df.columns:
            track_count = patterns_df['tracks_metrics'].sum()
            print(f"- Messages tracking metrics: {track_count}")
    
    # Create a simple summary file for easy downstream use
    if all_beliefs:
        summary_only_df = beliefs_df[beliefs_df['metric_type'] == 'summary']
        summary_file = output_dir / "model_beliefs_summary_only.csv"
        print(f"\nCreating summary-only file: {summary_file}")
        summary_only_df.to_csv(summary_file, index=False)
        print(f"Summary file contains {len(summary_only_df)} records (one per metric mention)")

if __name__ == "__main__":
    main()
