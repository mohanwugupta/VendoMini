"""Test crash detector."""

import pytest
from src.crash_detector import CrashDetector, CrashType


def test_crash_detector_initialization():
    """Test crash detector initialization."""
    detector = CrashDetector(threshold='moderate', window_size=20)
    
    assert detector.threshold == 'moderate'
    assert detector.window_size == 20
    assert not detector.crash_detected


def test_looping_detection():
    """Test detection of looping behavior."""
    detector = CrashDetector(threshold='moderate')
    
    # Create history with repeated actions
    history = []
    for i in range(10):
        history.append({
            'step': i,
            'action': {'tool': 'tool_check_storage', 'args': {}},
            'observation': {'day': i}
        })
    
    is_crashed, crash_type = detector.update(history)
    
    assert is_crashed
    assert crash_type == CrashType.LOOPING


def test_invalid_burst_detection():
    """Test detection of invalid action bursts."""
    detector = CrashDetector(threshold='moderate')
    
    # Create history with many failures
    history = []
    for i in range(20):
        history.append({
            'step': i,
            'action': {'tool': 'tool_order', 'args': {}},
            'result': {'success': False},  # Failed
            'observation': {'day': i}
        })
    
    is_crashed, crash_type = detector.update(history)
    
    assert is_crashed
    assert crash_type == CrashType.INVALID_BURST


def test_budget_denial_detection():
    """Test detection of budget denial behavior."""
    detector = CrashDetector(threshold='moderate')
    
    # Create history with ordering while broke
    history = []
    for i in range(10):
        history.append({
            'step': i,
            'action': {'tool': 'tool_order', 'args': {}},
            'observation': {'budget': -50},  # Broke
            'result': {'success': False}
        })
    
    is_crashed, crash_type = detector.update(history)
    
    assert is_crashed
    assert crash_type == CrashType.BUDGET_DENIAL


def test_decoupling_detection():
    """Test detection of action-prediction decoupling."""
    detector = CrashDetector(threshold='moderate')
    
    # Create history where actions don't match predictions
    history = []
    for i in range(20):
        history.append({
            'step': i,
            'action': {'tool': 'tool_check_storage', 'args': {}},
            'prediction': {'tool': 'tool_order', 'args': {}},  # Different tool
            'observation': {'day': i}
        })
    
    is_crashed, crash_type = detector.update(history)
    
    assert is_crashed
    assert crash_type == CrashType.DECOUPLING


def test_exploration_collapse_detection():
    """Test detection of exploration collapse."""
    detector = CrashDetector(threshold='moderate')
    
    # Create history with very low tool diversity
    history = []
    for i in range(20):
        # Only use one tool repeatedly
        history.append({
            'step': i,
            'action': {'tool': 'tool_check_budget', 'args': {}},
            'observation': {'day': i}
        })
    
    is_crashed, crash_type = detector.update(history)
    
    assert is_crashed
    assert crash_type == CrashType.EXPLORATION_COLLAPSE


def test_no_crash_diverse_behavior():
    """Test that diverse behavior doesn't trigger crash."""
    detector = CrashDetector(threshold='moderate')
    
    # Create history with diverse actions
    tools = ['tool_check_budget', 'tool_check_storage', 'tool_order', 'tool_check_inbox']
    history = []
    for i in range(20):
        history.append({
            'step': i,
            'action': {'tool': tools[i % len(tools)], 'args': {}},
            'result': {'success': True},
            'observation': {'day': i, 'budget': 100}
        })
    
    is_crashed, crash_type = detector.update(history)
    
    assert not is_crashed


def test_threshold_levels():
    """Test that different thresholds have different sensitivities."""
    # Strict should detect with fewer repeats
    strict = CrashDetector(threshold='strict')
    assert strict.thresholds['loop_repeat_count'] == 3
    
    # Moderate
    moderate = CrashDetector(threshold='moderate')
    assert moderate.thresholds['loop_repeat_count'] == 4
    
    # Lenient should require more repeats
    lenient = CrashDetector(threshold='lenient')
    assert lenient.thresholds['loop_repeat_count'] == 6


def test_get_crash_info():
    """Test getting crash information."""
    detector = CrashDetector(threshold='moderate')
    
    # Initially no crash
    info = detector.get_crash_info()
    assert not info['crashed']
    assert info['type'] is None
    
    # Create looping behavior
    history = []
    for i in range(10):
        history.append({
            'step': i,
            'action': {'tool': 'tool_check_storage', 'args': {}},
            'observation': {'day': i}
        })
    
    detector.update(history)
    
    # Should have crash info
    info = detector.get_crash_info()
    assert info['crashed']
    assert info['type'] == CrashType.LOOPING
    assert info['step'] is not None


def test_reset():
    """Test resetting crash detector."""
    detector = CrashDetector(threshold='moderate')
    
    # Trigger crash
    history = []
    for i in range(10):
        history.append({
            'step': i,
            'action': {'tool': 'tool_check_storage', 'args': {}},
            'observation': {'day': i}
        })
    
    detector.update(history)
    assert detector.crash_detected
    
    # Reset
    detector.reset()
    assert not detector.crash_detected
    assert detector.crash_type is None


def test_minimum_history_requirement():
    """Test that detector needs minimum history."""
    detector = CrashDetector(threshold='moderate')
    
    # Very short history
    history = [
        {'step': 0, 'action': {'tool': 'tool_order'}, 'observation': {}}
    ]
    
    is_crashed, crash_type = detector.update(history)
    
    # Should not crash with insufficient history
    assert not is_crashed
