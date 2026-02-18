#!/usr/bin/env python3
"""
Extract model identity position information from JSON files.

This script processes JSON files and extracts:
- Model name and occupation given in the system prompt
- Position of identity within the context window based on step
- Estimation of context window position based on token usage

Output: processed/model_identity_positions.csv with columns:
- run_id: unique identifier for the run
- model: the model used (e.g., google/gemini-1.5-flash-002)
- task: the task identifier
- epoch: epoch number (1-5)
- step: step number within epoch
- identity_name: the name given to the model (e.g., "John Johnson")
- identity_occupation: the occupation assigned (e.g., "vending machine owner")
- identity_location: location information if provided
- context_window_position: estimated position in context window (early/middle/late)
- total_tokens_at_step: cumulative tokens used up to this step
- input_tokens_at_step: input tokens at this step
- output_tokens_at_step: output tokens at this step
- relative_timestamp: timestamp from the conversation logs
"""

import argparse
import csv
import json
import re
import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from collections import defaultdict
import sys

def extract_identity_from_system_message(eval_file_path):
    """Extract model identity information from the system message in .eval file"""
    identity_info = {
        "name": None,
        "occupation": None,
        "location": None,
        "email": None,
        "additional_details": []
    }
    
    try:
        # .eval files are ZIP files
        with zipfile.ZipFile(eval_file_path, 'r') as zf:
            # Look for system message in sample files
            sample_files = [f for f in zf.namelist() if f.startswith('samples/')]
            
            if not sample_files:
                return identity_info
            
            # Read first sample file to get the system message
            sample_data = json.loads(zf.read(sample_files[0]).decode('utf-8'))
            messages = sample_data.get("messages", [])
            
            # Find the system message (usually the first message)
            system_message = None
            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                    break
            
            if not system_message:
                return identity_info
                
            # Extract name - look for "You are [Name]" pattern
            name_patterns = [
                r"You are ([A-Z][a-z]+ [A-Z][a-z]+)",  # "You are John Johnson"
                r"My name is ([A-Z][a-z]+ [A-Z][a-z]+)",  # "My name is John Johnson"
                r"I'm ([A-Z][a-z]+ [A-Z][a-z]+)",  # "I'm John Johnson"
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, system_message)
                if match:
                    identity_info["name"] = match.group(1)
                    break
            
            # Extract occupation - look for occupation keywords
            occupation_patterns = [
                r"You are [^,]+,?\s*(the\s+)?([^,\n\.]+(?:owner|manager|operator|business|entrepreneur))",
                r"([^,\n\.]*(?:owner|manager|operator|business|entrepreneur)[^,\n\.]*)",
                r"operating.*?(vending machine|business|company|store)",
                r"task is to.*?(manage|operate|run).*?([^,\n\.]+)"
            ]
            
            for pattern in occupation_patterns:
                match = re.search(pattern, system_message, re.IGNORECASE)
                if match:
                    if len(match.groups()) > 1:
                        identity_info["occupation"] = match.group(2).strip()
                    else:
                        identity_info["occupation"] = match.group(1).strip()
                    break
            
            # Extract email
            email_match = re.search(r"email.*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", system_message)
            if email_match:
                identity_info["email"] = email_match.group(1)
            
            # Extract location information
            location_patterns = [
                r"home office.*?located at ([^,\n]+)",
                r"vending machine.*?located at ([^,\n]+)",
                r"located at ([^,\n]+)",
            ]
            
            locations = []
            for pattern in location_patterns:
                matches = re.findall(pattern, system_message, re.IGNORECASE)
                locations.extend([match.strip() for match in matches])
            
            if locations:
                identity_info["location"] = " | ".join(set(locations))
            
            # Extract any additional details
            additional_patterns = [
                r"Your ([^:]+): ([^,\n]+)",  # "Your email: ...", "Your address: ..."
                r"You have ([^,\n\.]+)",     # "You have access to..."
            ]
            
            for pattern in additional_patterns:
                matches = re.findall(pattern, system_message)
                for match in matches:
                    if len(match) == 2:
                        identity_info["additional_details"].append(f"{match[0]}: {match[1]}")
                
    except Exception as e:
        print(f"Error extracting identity from {eval_file_path}: {e}")
    
    return identity_info

def estimate_context_window_position(relative_timestamp, max_timestamp):
    """Estimate position in context window based on relative timestamp"""
    if max_timestamp == 0:
        return "early"
    
    ratio = relative_timestamp / max_timestamp
    
    if ratio < 0.33:
        return "early"
    elif ratio < 0.67:
        return "middle"
    else:
        return "late"

def extract_conversation_data(data, epoch):
    """Extract conversation data including token usage and timestamps from logs"""
    conversation_data = []
    logs = data.get("logs", [])
    
    # Track cumulative tokens
    cumulative_total_tokens = 0
    
    # Get max timestamp for position calculation
    max_timestamp = 0
    if logs:
        max_timestamp = max(log.get("relative_timestamp", 0) for log in logs)
    
    # Group logs by logical steps based on tool calls and token updates
    steps = []
    current_step = {
        "epoch": epoch,
        "step": 0,
        "relative_timestamp": 0,
        "total_tokens": 0,
        "context_window_position": "early",
        "tool_calls": [],
        "message_content": "",
        "events": []
    }
    
    for log in logs:
        event = log.get("event", "")
        payload = log.get("payload", "")
        timestamp = log.get("relative_timestamp", 0)
        
        current_step["events"].append(log)
        current_step["relative_timestamp"] = max(current_step["relative_timestamp"], timestamp)
        
        # Track token usage
        if event == "total_tokens":
            current_step["total_tokens"] = payload
            cumulative_total_tokens = payload
            
        # Track tool calls - this indicates a new step
        elif event == "tool_calls":
            current_step["tool_calls"].append(payload)
            
            # If this isn't the first tool call in this step, start a new step
            if len(current_step["tool_calls"]) > 1 or len(steps) > 0:
                # Finalize current step
                current_step["context_window_position"] = estimate_context_window_position(
                    current_step["relative_timestamp"], max_timestamp)
                steps.append(current_step.copy())
                
                # Start new step
                current_step = {
                    "epoch": epoch,
                    "step": len(steps),
                    "relative_timestamp": timestamp,
                    "total_tokens": cumulative_total_tokens,
                    "context_window_position": estimate_context_window_position(timestamp, max_timestamp),
                    "tool_calls": [payload],
                    "message_content": "",
                    "events": [log]
                }
            
        # Track message content
        elif event == "output.message.text":
            if payload:  # Only if there's actual content
                current_step["message_content"] += str(payload) + " "
    
    # Add the final step if it has any content
    if current_step["tool_calls"] or current_step["message_content"].strip():
        current_step["context_window_position"] = estimate_context_window_position(
            current_step["relative_timestamp"], max_timestamp)
        steps.append(current_step)
    
    return steps

def process_json_file(json_file_path):
    """Process a single JSON file and extract identity position data"""
    try:
        # Find the corresponding .eval file
        eval_file_path = find_corresponding_eval_file(json_file_path)
        if not eval_file_path:
            print(f"Warning: No corresponding .eval file found for {json_file_path}")
            return []
        
        # Extract identity information from .eval file
        identity_info = extract_identity_from_system_message(eval_file_path)
        
        # Load conversation data from JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract basic metadata from file path
        parts = json_file_path.parts
        model_folder = parts[-2] if len(parts) >= 2 else "unknown"
        
        # Extract run_id and epoch from filename
        filename = json_file_path.stem
        # Look for _<epoch_number> at the end of filename
        epoch_match = re.search(r'_(\d+)$', filename)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            # run_id is everything before the final _<epoch>
            run_id = filename[:epoch_match.start()]
        else:
            run_id = filename
            epoch = 1
        
        # Extract conversation data
        conversation_data = extract_conversation_data(data, epoch)
        
        # Create position records for each step
        all_position_data = []
        for step_data in conversation_data:
            position_record = {
                "file": str(json_file_path),
                "run_id": run_id,
                "model": model_folder,
                "task": "vending_machine",
                "epoch": step_data["epoch"],
                "step": step_data["step"],
                "identity_name": identity_info["name"],
                "identity_occupation": identity_info["occupation"],
                "identity_location": identity_info["location"],
                "identity_email": identity_info["email"],
                "identity_additional_details": "; ".join(identity_info["additional_details"]) if identity_info["additional_details"] else None,
                "context_window_position": step_data["context_window_position"],
                "total_tokens_at_step": step_data["total_tokens"],
                "relative_timestamp": step_data["relative_timestamp"],
                "tool_calls_at_step": "; ".join(step_data["tool_calls"]) if step_data["tool_calls"] else None,
                "message_content_length": len(step_data["message_content"]),
                "has_identity_content": has_identity_content(step_data["message_content"], identity_info)
            }
            
            all_position_data.append(position_record)
        
        return all_position_data
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return []

def find_corresponding_eval_file(json_file_path):
    """Find the corresponding .eval file for a JSON file"""
    # Look in the same directory for any .eval file
    parent_dir = json_file_path.parent
    
    # Get the first .eval file in the directory
    eval_files = list(parent_dir.glob("*.eval"))
    if eval_files:
        return eval_files[0]  # Return the first (and likely only) .eval file
    
    return None

def has_identity_content(message_content, identity_info):
    """Check if message content contains identity references (more flexible matching)"""
    if not message_content:
        return 0
    
    message_lower = message_content.lower()
    
    # Check for name components (first or last name)
    name_match = False
    if identity_info["name"]:
        name_parts = identity_info["name"].split()
        for part in name_parts:
            if len(part) >= 3 and part.lower() in message_lower:  # Only check names 3+ chars
                name_match = True
                break
    
    # Check for occupation/role references
    occupation_match = False
    occupation_keywords = [
        "owner", "vending machine owner", "business owner", 
        "operator", "vending machine operator",
        "manager", "entrepreneur", "proprietor"
    ]
    
    for keyword in occupation_keywords:
        if keyword in message_lower:
            occupation_match = True
            break
    
    # Also check if the specific occupation from identity is mentioned
    if identity_info["occupation"]:
        occupation_lower = identity_info["occupation"].lower()
        if occupation_lower in message_lower:
            occupation_match = True
    
    return 1 if (name_match or occupation_match) else 0

def main():
    parser = argparse.ArgumentParser(description="Extract model identity position data from JSON files")
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
    
    # Collect all position data
    all_position_data = []
    
    for i, file_path in enumerate(json_files, 1):
        print(f"Processing {i}/{len(json_files)}: {file_path.name}")
        
        position_data = process_json_file(file_path)
        all_position_data.extend(position_data)
    
    # Write to CSV
    output_file = output_dir / "model_identity_positions.csv"
    print(f"Writing {len(all_position_data)} rows to {output_file}")
    
    if all_position_data:
        fieldnames = [
            "file", "run_id", "model", "task", "epoch", "step",
            "identity_name", "identity_occupation", "identity_location", "identity_email", "identity_additional_details",
            "context_window_position", "total_tokens_at_step",
            "relative_timestamp", "tool_calls_at_step", "message_content_length", "has_identity_content"
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_position_data)
    
    print("Done!")
    
    # Print summary
    if all_position_data:
        unique_runs = len(set(row["run_id"] for row in all_position_data))
        unique_models = len(set(row["model"] for row in all_position_data))
        unique_names = len(set(row["identity_name"] for row in all_position_data if row["identity_name"]))
        unique_occupations = len(set(row["identity_occupation"] for row in all_position_data if row["identity_occupation"]))
        
        # Count by context window position
        position_counts = defaultdict(int)
        for row in all_position_data:
            position_counts[row["context_window_position"]] += 1
        
        # Count steps with identity content
        identity_content_count = sum(1 for row in all_position_data if row["has_identity_content"] == 1)
        
        print(f"\nSummary:")
        print(f"- Total position records: {len(all_position_data)}")
        print(f"- Unique runs: {unique_runs}")
        print(f"- Unique models: {unique_models}")
        print(f"- Unique identity names: {unique_names}")
        print(f"- Unique occupations: {unique_occupations}")
        print(f"- Steps with identity content: {identity_content_count}")
        print(f"- Context window positions:")
        for pos, count in position_counts.items():
            print(f"  - {pos}: {count}")
        
        # Show example identities found
        print(f"\nExample identities found:")
        identities = set()
        for row in all_position_data:
            if row["identity_name"] and row["identity_occupation"]:
                identities.add(f"{row['identity_name']} - {row['identity_occupation']}")
                if len(identities) >= 5:  # Show first 5
                    break
        
        for identity in list(identities)[:5]:
            print(f"  - {identity}")

if __name__ == "__main__":
    main()
