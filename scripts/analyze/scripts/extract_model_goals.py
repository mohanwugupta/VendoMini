#!/usr/bin/env python3
"""
Extract model goal information from JSON files.

This script processes JSON files and extracts:
- Model goals given in the system prompt (e.g., "turn a profit")
- Goal-related behavior tracking throughout conversation
- Achievement metrics and goal-oriented actions

Output: processed/model_goals.csv with columns:
- run_id: unique identifier for the run
- model: the model used (e.g., google/gemini-1.5-flash-002)
- task: the task identifier
- epoch: epoch number (1-5)
- step: step number within epoch
- primary_goal: the main goal extracted from system prompt
- goal_keywords: specific goal-related terms found
- goal_metrics: measurable objectives (e.g., profit targets)
- context_window_position: estimated position in context window (early/middle/late)
- has_goal_content: whether this step contains goal-related content
- goal_action_type: type of goal-oriented action taken
- relative_timestamp: timestamp from the conversation logs
"""

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from collections import defaultdict
import sys

def extract_goals_from_system_message(eval_file_path):
    """Extract model goal information from the system message in .eval file"""
    goal_info = {
        "primary_goal": None,
        "goal_keywords": [],
        "goal_metrics": [],
        "success_criteria": [],
        "constraints": [],
        "strategies": []
    }
    
    try:
        # .eval files are ZIP files
        with zipfile.ZipFile(eval_file_path, 'r') as zf:
            # Look for system message in sample files
            sample_files = [f for f in zf.namelist() if f.startswith('samples/')]
            
            if not sample_files:
                return goal_info
            
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
                return goal_info
                
            # Extract primary goal - look for goal-related patterns
            goal_patterns = [
                r"goal is to ([^,\n\.]+)",  # "goal is to turn a profit"
                r"objective is to ([^,\n\.]+)",  # "objective is to maximize revenue"
                r"aim is to ([^,\n\.]+)",  # "aim is to be profitable"
                r"task is to ([^,\n\.]+)",  # "task is to grow the business"
                r"you should ([^,\n\.]*(?:profit|money|revenue|income|earn|maximize|optimize)[^,\n\.]*)",  # profit-related goals
                r"you need to ([^,\n\.]*(?:profit|money|revenue|income|earn|maximize|optimize)[^,\n\.]*)",
                r"your purpose is to ([^,\n\.]+)",
                r"you must ([^,\n\.]*(?:profit|money|revenue|income|earn|maximize|optimize)[^,\n\.]*)"
            ]
            
            for pattern in goal_patterns:
                match = re.search(pattern, system_message, re.IGNORECASE)
                if match:
                    goal_info["primary_goal"] = match.group(1).strip()
                    break
            
            # Extract goal keywords
            goal_keywords = [
                "profit", "profitable", "profitability", "turn a profit", "make money",
                "revenue", "income", "earnings", "sales", "maximize", "optimize",
                "efficiency", "cost-effective", "ROI", "return on investment",
                "business growth", "expand", "scale", "success", "competitive",
                "market share", "customer satisfaction", "demand", "supply"
            ]
            
            found_keywords = []
            for keyword in goal_keywords:
                if keyword.lower() in system_message.lower():
                    found_keywords.append(keyword)
            goal_info["goal_keywords"] = found_keywords
            
            # Extract numerical goals/metrics
            metric_patterns = [
                r"(\$[\d,]+)",  # Dollar amounts
                r"(\d+%)",  # Percentages
                r"(\d+\s*(?:dollars?|cents?))",  # Dollar/cent amounts
                r"(\d+\s*(?:units?|items?|products?))",  # Quantity targets
                r"(?:within|by|after)\s+(\d+\s*(?:days?|weeks?|months?))",  # Time targets
                r"(?:at least|minimum of|more than)\s+(\$?[\d,]+)",  # Minimum targets
                r"(?:target|goal|aim)\s+(?:of\s+)?(\$?[\d,]+)"  # Target amounts
            ]
            
            metrics = []
            for pattern in metric_patterns:
                matches = re.findall(pattern, system_message, re.IGNORECASE)
                metrics.extend(matches)
            goal_info["goal_metrics"] = list(set(metrics))  # Remove duplicates
            
            # Extract success criteria
            success_patterns = [
                r"success(?:ful)?\s+(?:is|means|when|if)\s+([^,\n\.]+)",
                r"consider(?:ed)?\s+successful\s+(?:if|when)\s+([^,\n\.]+)",
                r"achieve\s+success\s+by\s+([^,\n\.]+)",
                r"winning\s+(?:means|is)\s+([^,\n\.]+)"
            ]
            
            success_criteria = []
            for pattern in success_patterns:
                matches = re.findall(pattern, system_message, re.IGNORECASE)
                success_criteria.extend(matches)
            goal_info["success_criteria"] = [s.strip() for s in success_criteria]
            
            # Extract constraints
            constraint_patterns = [
                r"(?:cannot|must not|should not|avoid|don't)\s+([^,\n\.]+)",
                r"(?:limit|constraint|restriction)(?:s)?\s+(?:is|are|include)\s+([^,\n\.]+)",
                r"(?:within|under)\s+(?:budget|limit)\s+(?:of\s+)?([^,\n\.]+)"
            ]
            
            constraints = []
            for pattern in constraint_patterns:
                matches = re.findall(pattern, system_message, re.IGNORECASE)
                constraints.extend(matches)
            goal_info["constraints"] = [c.strip() for c in constraints]
            
            # Extract strategies
            strategy_patterns = [
                r"(?:strategy|approach|method|way)\s+(?:is|should be|to)\s+([^,\n\.]+)",
                r"(?:by|through)\s+([^,\n\.]*(?:stocking|pricing|marketing|advertising|customer)[^,\n\.]*)",
                r"(?:focus on|concentrate on|prioritize)\s+([^,\n\.]+)"
            ]
            
            strategies = []
            for pattern in strategy_patterns:
                matches = re.findall(pattern, system_message, re.IGNORECASE)
                strategies.extend(matches)
            goal_info["strategies"] = [s.strip() for s in strategies]
                
    except Exception as e:
        print(f"Error extracting goals from {eval_file_path}: {e}")
    
    return goal_info

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

def categorize_goal_action(tool_calls, message_content):
    """Categorize the type of goal-oriented action based on tool calls and content"""
    if not tool_calls and not message_content:
        return "none"
    
    content_lower = message_content.lower() if message_content else ""
    tools_str = " ".join(tool_calls).lower() if tool_calls else ""
    
    # Financial monitoring actions
    if "get_money_balance" in tools_str or any(word in content_lower for word in ["balance", "money", "funds", "budget"]):
        return "financial_monitoring"
    
    # Inventory management actions
    if any(tool in tools_str for tool in ["get_machine_inventory", "check_storage_quantities", "list_storage_products"]):
        return "inventory_management"
    
    # Sales/pricing actions
    if any(word in content_lower for word in ["price", "pricing", "cost", "sell", "sales", "revenue"]):
        return "pricing_strategy"
    
    # Market research actions
    if "ai_web_search" in tools_str or any(word in content_lower for word in ["research", "market", "competitor", "demand"]):
        return "market_research"
    
    # Customer communication actions
    if any(tool in tools_str for tool in ["read_email", "send_email", "read_email_inbox"]):
        return "customer_communication"
    
    # Planning/strategy actions
    if any(tool in tools_str for tool in ["write_scratchpad", "read_scratchpad"]) or any(word in content_lower for word in ["plan", "strategy", "goal", "objective"]):
        return "strategic_planning"
    
    # Operational actions
    if "wait_for_next_day" in tools_str:
        return "operational"
    
    # Analysis actions
    if any(word in content_lower for word in ["analyze", "analysis", "evaluate", "assess", "performance"]):
        return "analysis"
    
    return "other"

def extract_conversation_data(data, epoch, goal_info):
    """Extract conversation data including goal-related content from logs"""
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
    "events": [],
    "message_goal_count": 0
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
                text = str(payload)
                current_step["message_content"] += text + " "
                # determine if this individual message contains goal-related content
                try:
                    msg_has_goal = has_goal_content(text, goal_info)
                except Exception:
                    msg_has_goal = 0
                # attach the flag to the last appended event (copy to avoid mutating original)
                if current_step["events"]:
                    ev = dict(current_step["events"][-1])
                    ev["message_has_goal"] = int(msg_has_goal)
                    current_step["events"][-1] = ev
                # increment per-step counter
                if msg_has_goal:
                    current_step["message_goal_count"] = current_step.get("message_goal_count", 0) + 1
    
    # Add the final step if it has any content
    if current_step["tool_calls"] or current_step["message_content"].strip():
        current_step["context_window_position"] = estimate_context_window_position(
            current_step["relative_timestamp"], max_timestamp)
        steps.append(current_step)
    
    return steps

def has_goal_content(message_content, goal_info):
    """Check if message content contains goal-related references"""
    if not message_content:
        return 0
    
    message_lower = message_content.lower()
    
    # Check for primary goal content
    goal_match = False
    if goal_info["primary_goal"]:
        goal_lower = goal_info["primary_goal"].lower()
        # Check for partial matches (at least 3 words in common)
        goal_words = [w for w in goal_lower.split() if len(w) >= 3]
        matches = sum(1 for word in goal_words if word in message_lower)
        if matches >= min(2, len(goal_words)):  # At least 2 words or all words if less than 2
            goal_match = True
    
    # Check for goal keywords
    keyword_match = False
    for keyword in goal_info["goal_keywords"]:
        if keyword.lower() in message_lower:
            keyword_match = True
            break
    
    # Check for metrics/numbers that might relate to goals
    metric_match = False
    for metric in goal_info["goal_metrics"]:
        if metric.lower() in message_lower:
            metric_match = True
            break
    
    # Check for general goal-related terms
    general_goal_terms = [
        "profit", "money", "revenue", "income", "sales", "cost", "price", 
        "goal", "objective", "target", "success", "performance", "growth",
        "maximize", "optimize", "efficient", "strategy", "plan"
    ]
    
    general_match = any(term in message_lower for term in general_goal_terms)
    
    return 1 if (goal_match or keyword_match or metric_match or general_match) else 0

def process_json_file(json_file_path):
    """Process a single JSON file and extract goal-related data"""
    try:
        # Find the corresponding .eval file
        eval_file_path = find_corresponding_eval_file(json_file_path)
        if not eval_file_path:
            print(f"Warning: No corresponding .eval file found for {json_file_path}")
            return []
        
        # Extract goal information from .eval file
        goal_info = extract_goals_from_system_message(eval_file_path)
        
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
        conversation_data = extract_conversation_data(data, epoch, goal_info)
        
        # Create goal records for each step
        all_goal_data = []
        for step_data in conversation_data:
            goal_record = {
                "file": str(json_file_path),
                "run_id": run_id,
                "model": model_folder,
                "task": "vending_machine",
                "epoch": step_data["epoch"],
                "step": step_data["step"],
                "primary_goal": goal_info["primary_goal"],
                "goal_keywords": "; ".join(goal_info["goal_keywords"]) if goal_info["goal_keywords"] else None,
                "goal_metrics": "; ".join(goal_info["goal_metrics"]) if goal_info["goal_metrics"] else None,
                "success_criteria": "; ".join(goal_info["success_criteria"]) if goal_info["success_criteria"] else None,
                "constraints": "; ".join(goal_info["constraints"]) if goal_info["constraints"] else None,
                "strategies": "; ".join(goal_info["strategies"]) if goal_info["strategies"] else None,
                "context_window_position": step_data["context_window_position"],
                "total_tokens_at_step": step_data["total_tokens"],
                "relative_timestamp": step_data["relative_timestamp"],
                "tool_calls_at_step": "; ".join(step_data["tool_calls"]) if step_data["tool_calls"] else None,
                    "message_content": step_data["message_content"].strip() if step_data["message_content"] else None,
                    "message_content_length": len(step_data["message_content"]),
                    "message_goal_count": step_data.get("message_goal_count", 0),
                    "has_goal_content": has_goal_content(step_data["message_content"], goal_info),
                    "goal_action_type": categorize_goal_action(step_data["tool_calls"], step_data["message_content"])
            }
            
            all_goal_data.append(goal_record)
        
        return all_goal_data
        
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

def main():
    parser = argparse.ArgumentParser(description="Extract model goal data from JSON files")
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
    
    # Collect all goal data
    all_goal_data = []
    
    for i, file_path in enumerate(json_files, 1):
        print(f"Processing {i}/{len(json_files)}: {file_path.name}")
        
        goal_data = process_json_file(file_path)
        all_goal_data.extend(goal_data)
    
    # Write to CSV
    output_file = output_dir / "model_goals.csv"
    print(f"Writing {len(all_goal_data)} rows to {output_file}")
    
    if all_goal_data:
        fieldnames = [
            "file", "run_id", "model", "task", "epoch", "step",
            "primary_goal", "goal_keywords", "goal_metrics", "success_criteria", "constraints", "strategies",
            "context_window_position", "total_tokens_at_step",
            "relative_timestamp", "tool_calls_at_step", "message_content", "message_content_length", "message_goal_count",
            "has_goal_content", "goal_action_type"
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_goal_data)
    
    print("Done!")
    
    # Print summary
    if all_goal_data:
        unique_runs = len(set(row["run_id"] for row in all_goal_data))
        unique_models = len(set(row["model"] for row in all_goal_data))
        unique_goals = len(set(row["primary_goal"] for row in all_goal_data if row["primary_goal"]))
        
        # Count by goal action type
        action_counts = defaultdict(int)
        for row in all_goal_data:
            action_counts[row["goal_action_type"]] += 1
        
        # Count steps with goal content
        goal_content_count = sum(1 for row in all_goal_data if row["has_goal_content"] == 1)
        
        print(f"\nSummary:")
        print(f"- Total goal records: {len(all_goal_data)}")
        print(f"- Unique runs: {unique_runs}")
        print(f"- Unique models: {unique_models}")
        print(f"- Unique primary goals: {unique_goals}")
        print(f"- Steps with goal content: {goal_content_count}")
        print(f"- Goal action types:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {action}: {count}")
        
        # Show example goals found
        print(f"\nExample goals found:")
        goals = set()
        for row in all_goal_data:
            if row["primary_goal"]:
                goals.add(row["primary_goal"])
                if len(goals) >= 5:  # Show first 5
                    break
        
        for goal in list(goals)[:5]:
            print(f"  - {goal}")

if __name__ == "__main__":
    main()
