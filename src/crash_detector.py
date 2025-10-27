"""Crash detector with multiple crash types and severity levels."""

from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CrashType:
    """Crash type constants."""
    LOOPING = "looping"
    INVALID_BURST = "invalid_burst"
    BUDGET_DENIAL = "budget_denial"
    DECOUPLING = "decoupling"
    EXPLORATION_COLLAPSE = "exploration_collapse"
    SLOW_DIVERGENCE = "slow_divergence"
    NONE = None


class CrashSeverity:
    """Crash severity levels."""
    SOFT = "soft"  # Recovers within 20 steps
    HARD = "hard"  # No recovery
    MISSION_ABANDON = "mission_abandon"


class CrashDetector:
    """
    Detect crashes using windowed heuristics.
    
    Monitors for various crash patterns:
    - Looping: Repeated identical actions
    - Invalid burst: High rate of failed actions
    - Budget denial: Continued ordering while broke
    - Decoupling: Actions contradict predictions
    - Exploration collapse: Low action diversity
    - Slow divergence: Incoherent state summaries
    """
    
    def __init__(self, threshold: str = "moderate", window_size: int = 20, continue_after_crash: int = 0):
        """
        Initialize crash detector.
        
        Args:
            threshold: Detection threshold (strict, moderate, lenient)
            window_size: Number of recent steps to analyze
            continue_after_crash: Number of steps to continue after crash detected (0 = stop immediately)
        """
        self.threshold = threshold
        self.window_size = window_size
        self.continue_after_crash = continue_after_crash
        
        # Threshold parameters
        self.thresholds = self._get_thresholds(threshold)
        
        # State
        self.crash_detected = False
        self.crash_type = CrashType.NONE
        self.crash_severity = None
        self.crash_step = None
        self.recovery_attempts = 0
        
    def _get_thresholds(self, threshold: str) -> Dict[str, Any]:
        """Get threshold parameters based on sensitivity level."""
        if threshold == "strict":
            return {
                'loop_repeat_count': 3,
                'invalid_rate': 0.3,  # 30% failures
                'budget_denial_count': 3,
                'decoupling_rate': 0.4,
                'entropy_threshold': 0.5,
                'similarity_threshold': 0.7
            }
        elif threshold == "lenient":
            return {
                'loop_repeat_count': 6,
                'invalid_rate': 0.5,
                'budget_denial_count': 5,
                'decoupling_rate': 0.6,
                'entropy_threshold': 0.2,
                'similarity_threshold': 0.5
            }
        else:  # moderate
            return {
                'loop_repeat_count': 4,
                'invalid_rate': 0.4,
                'budget_denial_count': 4,
                'decoupling_rate': 0.5,
                'entropy_threshold': 0.3,
                'similarity_threshold': 0.6
            }
    
    def update(self, history: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Analyze recent history for crash patterns.
        
        Args:
            history: List of recent step records
            
        Returns:
            (is_crashed, crash_type)
        """
        if self.crash_detected:
            return True, self.crash_type
        
        # Need minimum history to detect meaningful patterns
        min_steps = 5
        if len(history) < min_steps:
            # Not enough history yet
            return False, None
        
        # Take recent window
        window = history[-self.window_size:]
        
        # Check each crash type
        if self._check_looping(window):
            self.crash_detected = True
            self.crash_type = CrashType.LOOPING
            self.crash_step = len(history)
            return True, CrashType.LOOPING
        
        if self._check_invalid_burst(window):
            self.crash_detected = True
            self.crash_type = CrashType.INVALID_BURST
            self.crash_step = len(history)
            return True, CrashType.INVALID_BURST
        
        if self._check_budget_denial(window):
            self.crash_detected = True
            self.crash_type = CrashType.BUDGET_DENIAL
            self.crash_step = len(history)
            return True, CrashType.BUDGET_DENIAL
        
        if self._check_decoupling(window):
            self.crash_detected = True
            self.crash_type = CrashType.DECOUPLING
            self.crash_step = len(history)
            return True, CrashType.DECOUPLING
        
        if self._check_exploration_collapse(window):
            self.crash_detected = True
            self.crash_type = CrashType.EXPLORATION_COLLAPSE
            self.crash_step = len(history)
            return True, CrashType.EXPLORATION_COLLAPSE
        
        return False, None
    
    def _check_looping(self, window: List[Dict[str, Any]]) -> bool:
        """Check for repeated identical tool calls with no state change."""
        if len(window) < self.thresholds['loop_repeat_count']:
            return False
        
        # Extract actions
        actions = []
        for step in window:
            action = step.get('action', {})
            # Create action signature
            action_sig = f"{action.get('tool')}_{str(action.get('args', {}))}"
            actions.append(action_sig)
        
        # Count consecutive repeats
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(actions)):
            if actions[i] == actions[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive >= self.thresholds['loop_repeat_count']
    
    def _check_invalid_burst(self, window: List[Dict[str, Any]]) -> bool:
        """Check for high rate of failed actions."""
        if len(window) < 8:
            return False
        
        # Count failures
        failures = 0
        for step in window:
            result = step.get('result', {})
            
            # Fix: Handle tuple return from env.execute_tool()
            if isinstance(result, tuple):
                result = result[0] if result else {}
            
            if not result.get('success', True):
                failures += 1
        
        failure_rate = failures / len(window)
        return failure_rate >= self.thresholds['invalid_rate']
    
    def _check_budget_denial(self, window: List[Dict[str, Any]]) -> bool:
        """Check for repeated ordering while bankrupt."""
        order_attempts_while_broke = 0
        
        for step in window:
            action = step.get('action', {})
            observation = step.get('observation', {})
            
            if action.get('tool') == 'tool_order' and observation.get('budget', 0) < 0:
                order_attempts_while_broke += 1
        
        return order_attempts_while_broke >= self.thresholds['budget_denial_count']
    
    def _check_decoupling(self, window: List[Dict[str, Any]]) -> bool:
        """Check if actions contradict recent predictions."""
        decoupling_count = 0
        
        for step in window:
            prediction = step.get('prediction')
            action = step.get('action', {})
            
            if prediction and action:
                # Check if predicted tool matches actual tool
                pred_tool = prediction.get('tool')
                actual_tool = action.get('tool')
                
                if pred_tool and actual_tool and pred_tool != actual_tool:
                    decoupling_count += 1
        
        if len(window) > 0:
            decoupling_rate = decoupling_count / len(window)
            return decoupling_rate >= self.thresholds['decoupling_rate']
        
        return False
    
    def _check_exploration_collapse(self, window: List[Dict[str, Any]]) -> bool:
        """Check for low tool diversity (entropy)."""
        if len(window) < 10:
            return False
        
        # Extract tools used
        tools = []
        for step in window:
            action = step.get('action', {})
            tool = action.get('tool')
            if tool:
                tools.append(tool)
        
        if not tools:
            return False
        
        # Calculate entropy
        tool_counts = Counter(tools)
        total = len(tools)
        entropy = 0.0
        
        for count in tool_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize by max possible entropy
        max_entropy = np.log2(len(tool_counts)) if len(tool_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy <= self.thresholds['entropy_threshold']
    
    def is_crashed(self) -> bool:
        """Check if crash has been detected."""
        return self.crash_detected
    
    def should_terminate(self, current_step: int) -> bool:
        """
        Check if simulation should terminate based on crash state.
        
        Args:
            current_step: Current step number
            
        Returns:
            True if should terminate, False if should continue
        """
        if not self.crash_detected:
            return False
        
        # If continue_after_crash is 0, terminate immediately
        if self.continue_after_crash == 0:
            return True
        
        # Otherwise, check if we've been crashed for long enough
        steps_since_crash = current_step - self.crash_step
        return steps_since_crash >= self.continue_after_crash
    
    def get_crash_info(self) -> Dict[str, Any]:
        """Get crash information."""
        return {
            'crashed': self.crash_detected,
            'type': self.crash_type,
            'severity': self.crash_severity,
            'step': self.crash_step
        }
    
    def reset(self):
        """Reset crash detector state."""
        self.crash_detected = False
        self.crash_type = CrashType.NONE
        self.crash_severity = None
        self.crash_step = None
        self.recovery_attempts = 0
