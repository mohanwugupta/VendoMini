"""Test PE Calculator."""

import pytest
from src.pe_calculator import PECalculator


def test_pe_calculator_initialization():
    """Test PE calculator initialization."""
    calc = PECalculator(windows=[10, 100])
    
    assert calc.windows == [10, 100]
    assert 'fast' in calc.alphas
    assert 'temporal' in calc.accumulators
    assert calc.accumulators['temporal']['fast'] == 0.0


def test_compute_temporal_pe():
    """Test temporal PE computation."""
    calc = PECalculator()
    
    prediction = {
        'expected_delivery_day': 10
    }
    actual = {
        'actual_delivery_day': 15
    }
    
    pes = calc.compute_pe(prediction, actual)
    
    # |10 - 15| / max(10, 1) = 5 / 10 = 0.5
    assert pes['temporal'] == 0.5


def test_compute_quantity_pe():
    """Test quantity PE computation."""
    calc = PECalculator()
    
    prediction = {
        'expected_quantity': 100
    }
    actual = {
        'actual_quantity': 80
    }
    
    pes = calc.compute_pe(prediction, actual)
    
    # |100 - 80| / max(100, 1) = 20 / 100 = 0.2
    assert pes['quantity'] == 0.2


def test_compute_cost_pe():
    """Test cost PE computation."""
    calc = PECalculator()
    
    prediction = {
        'expected_cost': 200.0
    }
    actual = {
        'actual_cost': 250.0
    }
    
    pes = calc.compute_pe(prediction, actual)
    
    # |200 - 250| / max(200, 0.01) = 50 / 200 = 0.25
    assert pes['cost'] == 0.25


def test_compute_causal_pe():
    """Test causal PE computation."""
    calc = PECalculator()
    
    # Success mismatch
    prediction = {'expected_success': True}
    actual = {'actual_success': False}
    
    pes = calc.compute_pe(prediction, actual)
    assert pes['causal'] == 1.0
    
    # Success match
    prediction = {'expected_success': True}
    actual = {'actual_success': True}
    
    pes = calc.compute_pe(prediction, actual)
    assert pes['causal'] == 0.0


def test_compute_pe_missing_prediction():
    """Test PE computation with no prediction (optional mode)."""
    calc = PECalculator()
    
    actual = {
        'actual_delivery_day': 10,
        'actual_quantity': 100
    }
    
    pes = calc.compute_pe(None, actual)
    
    # All PEs should be 0 when no prediction
    assert pes['temporal'] == 0.0
    assert pes['quantity'] == 0.0
    assert pes['cost'] == 0.0
    assert pes['causal'] == 0.0


def test_update_accumulators():
    """Test EWMA accumulator updates."""
    calc = PECalculator()
    
    # First update
    pes = {'temporal': 0.5, 'quantity': 0.2, 'cost': 0.3, 'causal': 0.0}
    calc.update_accumulators(pes)
    
    # Fast alpha = 0.3, so new value = 0.3 * 0.5 + 0.7 * 0 = 0.15
    assert abs(calc.accumulators['temporal']['fast'] - 0.15) < 0.01
    
    # Second update with same PE
    calc.update_accumulators(pes)
    
    # new = 0.3 * 0.5 + 0.7 * 0.15 = 0.15 + 0.105 = 0.255
    assert abs(calc.accumulators['temporal']['fast'] - 0.255) < 0.01


def test_get_cumulative_pes():
    """Test getting cumulative PEs."""
    calc = PECalculator()
    
    pes = {'temporal': 1.0, 'quantity': 0.5, 'cost': 0.0, 'causal': 0.0}
    calc.update_accumulators(pes)
    
    cumulative = calc.get_cumulative_pes()
    
    assert 'temporal' in cumulative
    assert 'fast' in cumulative['temporal']
    assert cumulative['temporal']['fast'] > 0


def test_get_windowed_stats():
    """Test windowed statistics."""
    calc = PECalculator()
    
    # Add multiple PE values
    for i in range(20):
        pes = {'temporal': 0.5 if i % 2 == 0 else 0.3, 'quantity': 0.2, 'cost': 0.1, 'causal': 0.0}
        calc.update_accumulators(pes)
    
    stats = calc.get_windowed_stats()
    
    assert 'temporal' in stats
    assert 'mean' in stats['temporal']
    assert 'std' in stats['temporal']
    assert 'max' in stats['temporal']
    assert stats['temporal']['max'] == 0.5


def test_reset():
    """Test resetting PE calculator."""
    calc = PECalculator()
    
    pes = {'temporal': 1.0, 'quantity': 0.5, 'cost': 0.3, 'causal': 1.0}
    calc.update_accumulators(pes)
    calc.update_accumulators(pes)
    
    assert len(calc.pe_history) == 2
    assert calc.accumulators['temporal']['fast'] > 0
    
    calc.reset()
    
    assert len(calc.pe_history) == 0
    assert calc.accumulators['temporal']['fast'] == 0.0
