"""Logging utilities for VendoMini."""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class Logger:
    """Handle logging of experiment data."""
    
    def __init__(self, run_id: str, logs_dir: str = "logs"):
        """
        Initialize logger.
        
        Args:
            run_id: Unique identifier for this run
            logs_dir: Directory to save logs
        """
        self.run_id = run_id
        self.logs_dir = Path(logs_dir)
        self.run_dir = self.logs_dir / run_id
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.steps_file = self.run_dir / "steps.jsonl"
        self.summary_file = self.run_dir / "summary.json"
        
        # Open steps file for writing
        self.steps_fp = open(self.steps_file, 'w')
        
        # Metadata
        self.metadata = {
            'run_id': run_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
    
    def log_step(self, step_data: Dict[str, Any]):
        """
        Log a single step.
        
        Args:
            step_data: Step information to log
        """
        self.steps_fp.write(json.dumps(step_data) + '\n')
        self.steps_fp.flush()
    
    def log_summary(self, summary_data: Dict[str, Any]):
        """
        Log run summary.
        
        Args:
            summary_data: Summary information
        """
        self.metadata['end_time'] = datetime.now().isoformat()
        
        summary = {
            **self.metadata,
            **summary_data
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def close(self):
        """Close log files."""
        if hasattr(self, 'steps_fp') and not self.steps_fp.closed:
            self.steps_fp.close()


class ResultsAggregator:
    """Aggregate results from multiple runs."""
    
    @staticmethod
    def aggregate_to_csv(summaries: List[Dict[str, Any]], output_path: str):
        """
        Aggregate multiple run summaries to CSV.
        
        Args:
            summaries: List of summary dictionaries
            output_path: Path to output CSV file
        """
        if not summaries:
            return
        
        # Get all keys
        all_keys = set()
        for summary in summaries:
            all_keys.update(ResultsAggregator._flatten_dict(summary).keys())
        
        fieldnames = sorted(all_keys)
        
        # Write CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for summary in summaries:
                flat = ResultsAggregator._flatten_dict(summary)
                writer.writerow(flat)
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ResultsAggregator._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
