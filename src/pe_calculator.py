"""Prediction Error Calculator with multi-scale EWMA accumulators."""

import numpy as np
from typing import Dict, Any, Optional


class PECalculator:
    """
    Calculate and accumulate prediction errors across multiple timescales.
    
    Computes typed PEs (temporal, quantity, cost, causal) and maintains
    EWMA (Exponentially Weighted Moving Average) accumulators at fast,
    medium, and slow timescales.
    """
    
    def __init__(self, windows: list = None):
        """
        Initialize PE calculator.
        
        Args:
            windows: List of window sizes for PE tracking (default: [10, 100, 500])
        """
        self.windows = windows or [10, 100, 500]
        
        # EWMA alphas for fast, medium, slow
        self.alphas = {
            'fast': 0.3,
            'med': 0.1,
            'slow': 0.01
        }
        
        # Accumulators for each PE type and timescale
        self.accumulators = {
            'temporal': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
            'quantity': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
            'cost': {'fast': 0.0, 'med': 0.0, 'slow': 0.0},
            'causal': {'fast': 0.0, 'med': 0.0, 'slow': 0.0}
        }
        
        # History for windowed calculations
        self.pe_history = []
        
    def compute_pe(self, prediction_card: Optional[Dict[str, Any]], 
                   actual_outcome: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute prediction errors for a single step.
        
        Args:
            prediction_card: Agent's prediction (can be None if optional mode)
            actual_outcome: Actual result from environment
            
        Returns:
            Dictionary of PE values by type
        """
        if prediction_card is None:
            # No prediction provided (optional mode)
            return {
                'temporal': 0.0,
                'quantity': 0.0,
                'cost': 0.0,
                'causal': 0.0
            }
        
        pes = {}
        
        # Temporal PE: |pred_day - actual_day| / max(pred_day, 1)
        if 'expected_delivery_day' in prediction_card and 'actual_delivery_day' in actual_outcome:
            pred_day = prediction_card['expected_delivery_day']
            actual_day = actual_outcome['actual_delivery_day']
            pes['temporal'] = abs(pred_day - actual_day) / max(pred_day, 1)
        else:
            pes['temporal'] = 0.0
        
        # Quantity PE: |pred_qty - actual_qty| / max(pred_qty, 1)
        if 'expected_quantity' in prediction_card and 'actual_quantity' in actual_outcome:
            pred_qty = prediction_card['expected_quantity']
            actual_qty = actual_outcome['actual_quantity']
            pes['quantity'] = abs(pred_qty - actual_qty) / max(pred_qty, 1)
        else:
            pes['quantity'] = 0.0
        
        # Cost PE: |pred_cost - actual_cost| / max(pred_cost, 1)
        if 'expected_cost' in prediction_card and 'actual_cost' in actual_outcome:
            pred_cost = prediction_card['expected_cost']
            actual_cost = actual_outcome['actual_cost']
            pes['cost'] = abs(pred_cost - actual_cost) / max(pred_cost, 0.01)
        else:
            pes['cost'] = 0.0
        
        # Causal PE: Binary error when tool effect/rules mismatch
        if 'expected_success' in prediction_card and 'actual_success' in actual_outcome:
            pred_success = prediction_card['expected_success']
            actual_success = actual_outcome['actual_success']
            pes['causal'] = 1.0 if pred_success != actual_success else 0.0
        else:
            pes['causal'] = 0.0
        
        return pes
    
    def update_accumulators(self, pe_dict: Dict[str, float]):
        """
        Update EWMA accumulators with new PE values.
        
        Args:
            pe_dict: Dictionary of PE values by type
        """
        for pe_type in ['temporal', 'quantity', 'cost', 'causal']:
            pe_value = pe_dict.get(pe_type, 0.0)
            
            for timescale, alpha in self.alphas.items():
                # EWMA update: new_value = alpha * current + (1 - alpha) * old_value
                self.accumulators[pe_type][timescale] = (
                    alpha * pe_value + 
                    (1 - alpha) * self.accumulators[pe_type][timescale]
                )
        
        # Add to history
        self.pe_history.append(pe_dict.copy())
    
    def get_cumulative_pes(self) -> Dict[str, Dict[str, float]]:
        """
        Get current cumulative PE values.
        
        Returns:
            Dictionary of accumulators by type and timescale
        """
        return {
            pe_type: timescales.copy()
            for pe_type, timescales in self.accumulators.items()
        }
    
    def get_windowed_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get windowed statistics over recent history.
        
        Returns:
            Dictionary of statistics by PE type
        """
        if not self.pe_history:
            return {}
        
        stats = {}
        for pe_type in ['temporal', 'quantity', 'cost', 'causal']:
            values = [pe.get(pe_type, 0.0) for pe in self.pe_history]
            
            stats[pe_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'recent_mean_10': np.mean(values[-10:]) if len(values) >= 10 else np.mean(values),
                'recent_mean_100': np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)
            }
        
        return stats
    
    def reset(self):
        """Reset all accumulators and history."""
        for pe_type in self.accumulators:
            for timescale in self.accumulators[pe_type]:
                self.accumulators[pe_type][timescale] = 0.0
        self.pe_history = []
