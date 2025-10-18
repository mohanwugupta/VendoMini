"""Test VendoMini environment."""

import pytest
from src.env import VendoMiniEnv, ShockType


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        'experiment': {'seed': 42},
        'simulation': {
            'max_steps': 100,
            'complexity_level': 1,
            'initial_budget': 200,
            'pressure_level': 'medium'
        },
        'pe_induction': {
            'p_shock': 0.1,
            'pe_mag': 'medium',
            'pe_type_mix': 'realistic',
            'observability': 'full'
        }
    }


def test_env_initialization(basic_config):
    """Test environment initialization."""
    env = VendoMiniEnv(basic_config, seed=42)
    
    assert env.current_day == 0
    assert env.budget == 200
    assert len(env.skus) == 10  # complexity 1 = 10 SKUs
    assert len(env.suppliers) == 3  # complexity 1 = 3 suppliers
    assert len(env.orders) == 0


def test_env_reset(basic_config):
    """Test environment reset."""
    env = VendoMiniEnv(basic_config, seed=42)
    
    # Make some changes
    env.budget = 100
    env.current_day = 10
    
    # Reset
    obs = env.reset()
    
    assert env.current_day == 0
    assert env.budget == 200
    assert obs['day'] == 0


def test_tool_order(basic_config):
    """Test order placement tool."""
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    supplier_id = env.suppliers[0].id
    sku_id = env.skus[0].id
    
    result = env._tool_order(supplier_id, sku_id, 10)
    
    assert result['success'] is True
    assert 'order_id' in result
    assert 'eta_day' in result
    assert result['order_id'] in env.orders


def test_tool_check_storage(basic_config):
    """Test storage check tool."""
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    # Set some storage
    env.storage[env.skus[0].id] = 50
    
    result = env._tool_check_storage()
    
    assert result['success'] is True
    assert result['storage'][env.skus[0].id] == 50


def test_tool_check_budget(basic_config):
    """Test budget check tool."""
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    result = env._tool_check_budget()
    
    assert result['success'] is True
    assert result['budget'] == 200


def test_delivery_processing(basic_config):
    """Test order delivery processing."""
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    supplier_id = env.suppliers[0].id
    sku_id = env.skus[0].id
    
    # Place order
    result = env._tool_order(supplier_id, sku_id, 10)
    order_id = result['order_id']
    eta_day = result['eta_day']
    
    # Advance to delivery day
    env.current_day = eta_day
    initial_storage = env.storage[sku_id]
    
    env._process_deliveries()
    
    # Check that order was delivered
    assert env.storage[sku_id] == initial_storage + 10
    assert order_id not in env.orders
    assert len(env.inbox) > 0  # Should have delivery notification


def test_shock_injection_temporal(basic_config):
    """Test temporal shock injection."""
    basic_config['pe_induction']['p_shock'] = 1.0  # Always inject
    basic_config['pe_induction']['pe_type_mix'] = 'temporal_only'
    
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    # Place order
    supplier_id = env.suppliers[0].id
    sku_id = env.skus[0].id
    env._tool_order(supplier_id, sku_id, 10)
    
    original_eta = list(env.orders.values())[0].eta_day
    
    # Inject shock
    shock = env._inject_shock()
    
    if shock and shock.shock_type == ShockType.TEMPORAL:
        # Check that ETA was modified
        new_eta = list(env.orders.values())[0].eta_day
        assert new_eta > original_eta


def test_step_execution(basic_config):
    """Test full step execution."""
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    action = {
        'tool': 'tool_check_budget',
        'args': {}
    }
    
    obs, done = env.step(action, None)
    
    assert obs['day'] == 1  # Day advanced
    assert not done  # Shouldn't be done
    assert len(env.action_history) == 1


def test_complexity_levels():
    """Test that different complexity levels create different numbers of SKUs/suppliers."""
    config = {
        'experiment': {'seed': 42},
        'simulation': {
            'max_steps': 100,
            'complexity_level': 0,
            'initial_budget': 200,
            'pressure_level': 'medium'
        },
        'pe_induction': {
            'p_shock': 0.0,
            'pe_mag': 'low',
            'pe_type_mix': 'realistic',
            'observability': 'full'
        }
    }
    
    # Complexity 0
    env0 = VendoMiniEnv(config, seed=42)
    assert len(env0.skus) == 5
    assert len(env0.suppliers) == 2
    
    # Complexity 2
    config['simulation']['complexity_level'] = 2
    env2 = VendoMiniEnv(config, seed=42)
    assert len(env2.skus) == 15
    assert len(env2.suppliers) == 4


def test_budget_depletion():
    """Test that simulation ends when budget is too negative."""
    config = {
        'experiment': {'seed': 42},
        'simulation': {
            'max_steps': 100,
            'complexity_level': 0,
            'initial_budget': 10,  # Very low budget
            'pressure_level': 'medium'
        },
        'pe_induction': {
            'p_shock': 0.0,
            'pe_mag': 'low',
            'pe_type_mix': 'realistic',
            'observability': 'full'
        }
    }
    
    env = VendoMiniEnv(config, seed=42)
    env.reset()
    env.budget = -150  # Set very negative
    
    action = {'tool': 'tool_check_budget', 'args': {}}
    obs, done = env.step(action, None)
    
    assert done  # Should be done due to budget


def test_scratchpad_tools(basic_config):
    """Test scratchpad memory tools."""
    env = VendoMiniEnv(basic_config, seed=42)
    env.reset()
    
    # Write
    result = env._tool_write_scratchpad('key1', 'value1')
    assert result['success'] is True
    
    # Read
    result = env._tool_read_scratchpad('key1')
    assert result['success'] is True
    assert result['value'] == 'value1'
    
    # Delete
    result = env._tool_delete_scratchpad('key1')
    assert result['success'] is True
    
    # Read deleted key
    result = env._tool_read_scratchpad('key1')
    assert result['value'] is None
