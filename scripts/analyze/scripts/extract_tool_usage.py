#!/usr/bin/env python3
"""
Extract tool usage data from individual epochs in .eval files.

This script processes .eval files and extracts detailed tool usage information:
- Tool usage counts per epoch
- Tool categories and patterns
- Tool usage timing and frequency
- Tool success/failure rates if available

Output: processed/tool_usage_epochs.csv with detailed tool usage data for each epoch
"""

import argparse
import json
import zipfile
from pathlib import Path
import csv
from typing import Dict, List, Optional, Tuple
import re
from collections import Counter, defaultdict

# Define the complete list of valid tools based on the actual tool specifications
VALID_TOOLS = {
    'get_money_balance',
    'get_machine_inventory', 
    'list_storage_products',
    'check_storage_quantities',
    'read_email_inbox',
    'read_email',
    'send_email',
    'ai_web_search',
    'wait_for_next_day',
    'read_scratchpad',
    'write_scratchpad',
    'erase_scratchpad',
    'get_kw_value',
    'set_kw_value',
    'delete_kw_value',
    'add_to_vector_db',
    'search_vector_db',
    'chat_with_sub_agent',
    'run_sub_agent'
}

def is_valid_tool(tool_name: str) -> bool:
    """Check if a tool name is valid (not hallucinated)"""
    return tool_name in VALID_TOOLS

def categorize_tool(tool_name: str) -> str:
    """Categorize tools into functional groups"""
    tool_name = tool_name.lower()
    
    # Define tool categories based on function
    categories = {
        'financial': ['get_money_balance'],
        'inventory_query': ['get_machine_inventory', 'list_storage_products', 'check_storage_quantities'],
        'communication': ['read_email_inbox', 'read_email', 'send_email'],
        'search': ['ai_web_search'],
        'timing': ['wait_for_next_day'],
        'utility': ['read_scratchpad', 'write_scratchpad', 'erase_scratchpad', 
                   'get_kw_value', 'set_kw_value', 'delete_kw_value',
                   'add_to_vector_db', 'search_vector_db']
    }
    
    # Check each category
    for category, tools in categories.items():
        if tool_name in tools:
            return category
    
    return 'other'

def extract_tool_usage_from_messages(messages: List[Dict]) -> Tuple[List[Dict], Dict[str, int], Dict[str, int]]:
    """Extract tool usage from conversation messages, separating valid from hallucinated tools"""
    tool_usage_events = []
    valid_tool_counts = Counter()
    hallucinated_tool_counts = Counter()
    
    message_index = 0
    for msg in messages:
        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
            tool_calls = msg.get('tool_calls', [])
            for i, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, dict):
                    # Handle different formats of function field
                    function_info = tool_call.get('function', 'unknown')
                    if isinstance(function_info, str):
                        # Function is directly a string (tool name)
                        tool_name = function_info
                    elif isinstance(function_info, dict):
                        # Function is a dict with name field
                        tool_name = function_info.get('name', 'unknown')
                    else:
                        tool_name = str(function_info)
                    
                    # Skip empty or invalid tool names
                    if not tool_name or tool_name in ['unknown', '']:
                        continue
                    
                    # Determine if this is a valid or hallucinated tool
                    is_valid = is_valid_tool(tool_name)
                    
                    tool_event = {
                        'source': 'messages',
                        'message_index': message_index,
                        'tool_call_index': i,
                        'tool_name': tool_name,
                        'tool_category': categorize_tool(tool_name) if is_valid else 'hallucinated',
                        'is_valid': is_valid,
                        'tool_id': tool_call.get('id', ''),
                        'arguments': str(tool_call.get('arguments', {}))
                    }
                    tool_usage_events.append(tool_event)
                    
                    # Count separately
                    if is_valid:
                        valid_tool_counts[tool_name] += 1
                    else:
                        hallucinated_tool_counts[tool_name] += 1
        
        message_index += 1
    
    return tool_usage_events, dict(valid_tool_counts), dict(hallucinated_tool_counts)

def extract_tool_usage_from_events(events: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """Extract tool usage from event logs"""
    tool_usage_events = []
    tool_counts = Counter()
    
    for i, event in enumerate(events):
        event_type = event.get('event', '')
        
        # Look for direct tool events
        if event_type == 'tool':
            tool_name = event.get('function', '')
            if tool_name:
                tool_event = {
                    'source': 'events',
                    'event_index': i,
                    'timestamp': event.get('timestamp', ''),
                    'tool_name': tool_name,
                    'tool_category': categorize_tool(tool_name),
                    'event_type': event_type,
                    'tool_id': event.get('id', ''),
                    'result_length': len(str(event.get('result', '')))
                }
                tool_usage_events.append(tool_event)
                tool_counts[tool_name] += 1
        
        # Look for other tool-related events
        elif 'tool' in event_type.lower():
            payload = event.get('payload', '')
            if payload:
                tool_event = {
                    'source': 'events',
                    'event_index': i,
                    'timestamp': event.get('timestamp', event.get('relative_timestamp', i)),
                    'tool_name': str(payload),
                    'tool_category': categorize_tool(str(payload)),
                    'event_type': event_type
                }
                tool_usage_events.append(tool_event)
                tool_counts[str(payload)] += 1
        
        # Also check if there are nested tool calls in state changes
        elif event_type == 'state' and isinstance(event.get('payload'), dict):
            payload = event.get('payload', {})
            if 'messages' in payload:
                # Handle nested messages in state events
                nested_messages = payload['messages']
                if isinstance(nested_messages, list):
                    for msg in nested_messages:
                        if isinstance(msg, dict) and msg.get('role') == 'assistant' and 'tool_calls' in msg:
                            tool_calls = msg.get('tool_calls', [])
                            for tool_call in tool_calls:
                                if isinstance(tool_call, dict):
                                    # Handle different formats of function field
                                    function_info = tool_call.get('function', 'unknown')
                                    if isinstance(function_info, str):
                                        tool_name = function_info
                                    elif isinstance(function_info, dict):
                                        tool_name = function_info.get('name', 'unknown')
                                    else:
                                        tool_name = str(function_info)
                                    
                                    # Skip empty or invalid tool names
                                    if not tool_name or tool_name in ['unknown', '']:
                                        continue
                                    
                                    tool_event = {
                                        'source': 'events_nested',
                                        'event_index': i,
                                        'timestamp': event.get('timestamp', event.get('relative_timestamp', i)),
                                        'tool_name': tool_name,
                                        'tool_category': categorize_tool(tool_name),
                                        'event_type': 'nested_tool_call'
                                    }
                                    tool_usage_events.append(tool_event)
                                    tool_counts[tool_name] += 1
    
    return tool_usage_events, dict(tool_counts)

def extract_run_metadata(file_path: Path) -> Dict:
    """Extract basic run metadata from eval file"""
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            header_data = json.loads(zf.read("header.json").decode('utf-8'))
            eval_info = header_data.get("eval", {})
            
            return {
                'eval_file': str(file_path),
                'model_folder': file_path.parent.name,
                'run_id': eval_info.get('run_id', ''),
                'task_id': eval_info.get('task_id', ''),
                'model': eval_info.get('model', ''),
                'created': eval_info.get('created', ''),
                'epochs': eval_info.get('config', {}).get('epochs', 0),
            }
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return {
            'eval_file': str(file_path),
            'model_folder': file_path.parent.name,
            'run_id': '',
            'task_id': '',
            'model': '',
            'created': '',
            'epochs': 0,
        }

def extract_epoch_tool_usage(file_path: Path) -> List[Dict]:
    """Extract tool usage data from each epoch in an eval file"""
    epoch_tool_data = []
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Get basic metadata
            metadata = extract_run_metadata(file_path)
            
            # Find all sample files (each represents an epoch)
            sample_files = [name for name in zf.namelist() if name.startswith('samples/') and name.endswith('.json')]
            
            for sample_file in sample_files:
                try:
                    sample_data = json.loads(zf.read(sample_file).decode('utf-8'))
                    
                    epoch = sample_data.get('epoch', 0)
                    sample_id = sample_data.get('id', 1)
                    
                    # Extract tool usage from messages (now returns valid and hallucinated separately)
                    messages = sample_data.get('messages', [])
                    message_tools, valid_message_counts, hallucinated_message_counts = extract_tool_usage_from_messages(messages)
                    
                    # Extract tool usage from events (these should all be valid)
                    events = sample_data.get('events', [])
                    event_tools, event_counts = extract_tool_usage_from_events(events)
                    
                    # Calculate totals
                    total_valid_tools = sum(valid_message_counts.values()) + sum(event_counts.values())
                    total_hallucinated_tools = sum(hallucinated_message_counts.values())
                    total_tools = total_valid_tools + total_hallucinated_tools
                    
                    # Calculate hallucination rate
                    hallucination_rate = (total_hallucinated_tools / total_tools) if total_tools > 0 else 0
                    
                    # Combine all valid tool counts (from messages and events)
                    all_valid_tool_counts = Counter()
                    all_valid_tool_counts.update(valid_message_counts)
                    all_valid_tool_counts.update(event_counts)
                    
                    # Categorize valid tool usage
                    category_counts = Counter()
                    for tool_name, count in all_valid_tool_counts.items():
                        category = categorize_tool(tool_name)
                        category_counts[category] += count
                    
                    # Also categorize hallucinated tools
                    hallucinated_category_counts = Counter()
                    for tool_name, count in hallucinated_message_counts.items():
                        hallucinated_category_counts['hallucinated'] += count
                    
                    # Create epoch summary record
                    epoch_record = {
                        **metadata,
                        'epoch': epoch,
                        'sample_id': sample_id,
                        'sample_file': sample_file,
                        'total_tools_used': total_tools,
                        'total_valid_tools': total_valid_tools,
                        'total_hallucinated_tools': total_hallucinated_tools,
                        'hallucination_rate': round(hallucination_rate, 4),
                        'unique_valid_tools': len(all_valid_tool_counts),
                        'unique_hallucinated_tools': len(hallucinated_message_counts),
                        'tools_from_messages_valid': sum(valid_message_counts.values()),
                        'tools_from_messages_hallucinated': sum(hallucinated_message_counts.values()),
                        'tools_from_events': sum(event_counts.values()),
                    }
                    
                    # Add individual valid tool counts
                    for tool_name, count in all_valid_tool_counts.most_common():
                        epoch_record[f'tool_{tool_name}'] = count
                    
                    # Add individual hallucinated tool counts
                    for tool_name, count in hallucinated_message_counts.items():
                        epoch_record[f'hallucinated_{tool_name}'] = count
                    
                    # Add category counts (valid tools)
                    for category, count in category_counts.items():
                        epoch_record[f'category_{category}'] = count
                    
                    # Add hallucinated category count
                    for category, count in hallucinated_category_counts.items():
                        epoch_record[f'category_{category}'] = count
                    
                    # Add some derived metrics
                    if all_valid_tool_counts:
                        most_used_tool = all_valid_tool_counts.most_common(1)[0]
                        epoch_record['most_used_valid_tool'] = most_used_tool[0]
                        epoch_record['most_used_valid_tool_count'] = most_used_tool[1]
                    
                    if hallucinated_message_counts:
                        most_hallucinated = Counter(hallucinated_message_counts).most_common(1)[0]
                        epoch_record['most_hallucinated_tool'] = most_hallucinated[0]
                        epoch_record['most_hallucinated_tool_count'] = most_hallucinated[1]
                    
                    epoch_tool_data.append(epoch_record)
                    
                except Exception as e:
                    print(f"Error processing sample file {sample_file}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading tool usage from {file_path}: {e}")
    
    return epoch_tool_data

def main():
    parser = argparse.ArgumentParser(description="Extract tool usage data from .eval files")
    parser.add_argument("--input", "-i", default="data", help="Input directory containing .eval files")
    parser.add_argument("--output", "-o", default="processed/tool_usage_epochs.csv", help="Output CSV file")
    parser.add_argument("--output-detailed", default="processed/tool_usage_detailed.csv", help="Output CSV file for detailed tool events")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_detailed_path = Path(args.output_detailed)
    
    # Create output directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_detailed_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all .eval files
    eval_files = list(input_dir.rglob("*.eval"))
    print(f"Found {len(eval_files)} .eval files")
    
    if not eval_files:
        print("No .eval files found!")
        return
    
    # Process each eval file
    all_epoch_data = []
    all_detailed_events = []
    
    for eval_file in eval_files:
        print(f"Processing: {eval_file}")
        
        # Extract tool usage for all epochs in this file
        epoch_data = extract_epoch_tool_usage(eval_file)
        all_epoch_data.extend(epoch_data)
    
    # Write epoch-level tool usage data
    if all_epoch_data:
        # Get all unique field names
        all_fields = set()
        for record in all_epoch_data:
            all_fields.update(record.keys())
        
        # Sort fields for consistent output
        fieldnames = sorted(list(all_fields))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_epoch_data)
        
        print(f"\nTool usage data written to: {output_path}")
        print(f"Processed {len(all_epoch_data)} epochs")
    
    # Print summary statistics
    if all_epoch_data:
        print(f"\nSummary:")
        print(f"- Total epochs with tool usage: {len(all_epoch_data)}")
        
        # Calculate overall statistics
        total_tools = sum(record.get('total_tools_used', 0) for record in all_epoch_data)
        total_valid_tools = sum(record.get('total_valid_tools', 0) for record in all_epoch_data)
        total_hallucinated_tools = sum(record.get('total_hallucinated_tools', 0) for record in all_epoch_data)
        avg_tools_per_epoch = total_tools / len(all_epoch_data) if all_epoch_data else 0
        overall_hallucination_rate = (total_hallucinated_tools / total_tools) if total_tools > 0 else 0
        
        print(f"- Total tool calls across all epochs: {total_tools}")
        print(f"- Total valid tool calls: {total_valid_tools}")
        print(f"- Total hallucinated tool calls: {total_hallucinated_tools}")
        print(f"- Overall hallucination rate: {overall_hallucination_rate:.1%}")
        print(f"- Average tool calls per epoch: {avg_tools_per_epoch:.1f}")
        
        # Find most common valid tools across all epochs
        all_tool_fields = [field for field in fieldnames if field.startswith('tool_') and not field.startswith('tool_category')]
        tool_totals = defaultdict(int)
        
        for record in all_epoch_data:
            for field in all_tool_fields:
                if field in record and record[field]:
                    tool_name = field.replace('tool_', '')
                    tool_totals[tool_name] += record[field]
        
        print(f"- Most used valid tools overall:")
        for tool, count in sorted(tool_totals.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {tool}: {count}")
        
        # Find most common hallucinated tools
        hallucinated_tool_fields = [field for field in fieldnames if field.startswith('hallucinated_')]
        hallucinated_totals = defaultdict(int)
        
        for record in all_epoch_data:
            for field in hallucinated_tool_fields:
                if field in record and record[field]:
                    tool_name = field.replace('hallucinated_', '')
                    hallucinated_totals[tool_name] += record[field]
        
        if hallucinated_totals:
            print(f"- Most common hallucinated tools:")
            for tool, count in sorted(hallucinated_totals.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {tool}: {count}")
        else:
            print(f"- No hallucinated tools found!")
        
        # Category analysis
        category_fields = [field for field in fieldnames if field.startswith('category_')]
        category_totals = defaultdict(int)
        
        for record in all_epoch_data:
            for field in category_fields:
                if field in record and record[field]:
                    category_name = field.replace('category_', '')
                    category_totals[category_name] += record[field]
        
        print(f"- Tool usage by category:")
        for category, count in sorted(category_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")

if __name__ == "__main__":
    main()
