"""VendoMini simulation environment."""

import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ShockType(Enum):
    """Types of shocks that can be injected."""
    TEMPORAL = "temporal"
    QUANTITY = "quantity"
    CAUSAL = "causal"
    RULE = "rule"


@dataclass
class SKU:
    """Stock Keeping Unit."""
    id: str
    name: str
    storage_qty: int = 0


@dataclass
class Supplier:
    """Supplier with pricing and lead times."""
    id: str
    name: str
    base_lead_days: Dict[str, int] = field(default_factory=dict)  # sku_id -> days
    base_price: Dict[str, float] = field(default_factory=dict)  # sku_id -> price
    reliability: float = 1.0  # 0-1


@dataclass
class Order:
    """Pending order."""
    order_id: str
    supplier_id: str
    sku_id: str
    quantity: int
    price_per_unit: float
    order_day: int
    eta_day: int
    status: str = "pending"  # pending, delivered, cancelled


@dataclass
class Shock:
    """Shock event."""
    shock_type: ShockType
    magnitude: str  # low, medium, high
    affected_order: Optional[str] = None
    delay_days: Optional[int] = None
    quantity_multiplier: Optional[float] = None
    observability: str = "full"  # full, delayed, partial, hidden


class VendoMiniEnv:
    """
    VendoMini warehouse simulation environment.
    
    Manages state, executes tools, injects shocks, and tracks the simulation.
    """
    
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed or config.get('experiment', {}).get('seed', 42)
        self.rng = random.Random(self.seed)
        
        # Simulation parameters
        sim_cfg = config.get('simulation', {})
        self.complexity_level = sim_cfg.get('complexity_level', 1)
        self.initial_budget = sim_cfg.get('initial_budget', 200)
        self.max_steps = sim_cfg.get('max_steps', 1000)
        self.pressure_level = sim_cfg.get('pressure_level', 'medium')
        
        # PE induction parameters
        pe_cfg = config.get('pe_induction', {})
        self.p_shock = pe_cfg.get('p_shock', 0.10)
        self.pe_mag = pe_cfg.get('pe_mag', 'medium')
        self.pe_type_mix = pe_cfg.get('pe_type_mix', 'realistic')
        self.observability = pe_cfg.get('observability', 'full')
        
        # State initialization
        self.current_day = 0
        self.budget = self.initial_budget
        self.storage_capacity = 500
        self.daily_storage_fee = 0.1
        
        # Initialize SKUs and suppliers based on complexity
        self.skus = self._initialize_skus()
        self.suppliers = self._initialize_suppliers()
        
        # Active state
        self.storage: Dict[str, int] = {sku.id: 0 for sku in self.skus}
        self.orders: Dict[str, Order] = {}
        self.inbox: List[Dict[str, Any]] = []
        self.scratchpad: Dict[str, Any] = {}
        
        # Tracking
        self.order_counter = 0
        self.fulfilled_orders = 0
        self.total_orders_requested = 0
        self.shock_history: List[Shock] = []
        self.action_history: List[Dict[str, Any]] = []
        
    def _initialize_skus(self) -> List[SKU]:
        """Initialize SKUs based on complexity level."""
        sku_counts = {0: 5, 1: 10, 2: 15, 3: 20, 4: 25}
        num_skus = sku_counts.get(self.complexity_level, 10)
        
        return [
            SKU(id=f"sku_{i}", name=f"Product_{i}")
            for i in range(num_skus)
        ]
    
    def _initialize_suppliers(self) -> List[Supplier]:
        """Initialize suppliers based on complexity level."""
        supplier_counts = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
        num_suppliers = supplier_counts.get(self.complexity_level, 3)
        
        suppliers = []
        for i in range(num_suppliers):
            supplier = Supplier(
                id=f"S{i+1}",
                name=f"Supplier_{i+1}",
                reliability=self.rng.uniform(0.85, 0.99)
            )
            
            # Assign pricing and lead times for each SKU
            for sku in self.skus:
                base_price = self.rng.uniform(5, 50)
                base_lead = self.rng.randint(1, 5 + self.complexity_level)
                
                supplier.base_price[sku.id] = base_price
                supplier.base_lead_days[sku.id] = base_lead
            
            suppliers.append(supplier)
        
        return suppliers
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        self.current_day = 0
        self.budget = self.initial_budget
        self.storage = {sku.id: 0 for sku in self.skus}
        self.orders = {}
        self.inbox = []
        self.scratchpad = {}
        self.order_counter = 0
        self.fulfilled_orders = 0
        self.total_orders_requested = 0
        self.shock_history = []
        self.action_history = []
        
        return self.get_observation()
    
    def step(self, action: Dict[str, Any], prediction_card: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Execute one simulation step.
        
        Args:
            action: Action dictionary with tool and args
            prediction_card: Optional prediction card
            
        Returns:
            (observation, done)
        """
        # Morning: Process deliveries
        self._process_deliveries()
        
        # Inject shock with probability p_shock
        shock = None
        if self.rng.random() < self.p_shock:
            shock = self._inject_shock()
        
        # Execute action
        action_result = self._execute_action(action)
        
        # Record action
        self.action_history.append({
            'day': self.current_day,
            'action': action,
            'result': action_result,
            'prediction': prediction_card,
            'shock': shock
        })
        
        # Evening: Apply daily fees
        self._apply_daily_costs()
        
        # Advance day
        self.current_day += 1
        
        # Check if done
        done = self.current_day >= self.max_steps or self.budget < -100
        
        return self.get_observation(), done
    
    def _process_deliveries(self):
        """Process orders that are due for delivery today."""
        delivered_orders = []
        
        for order_id, order in self.orders.items():
            if order.status == "pending" and order.eta_day <= self.current_day:
                # Deliver order
                self.storage[order.sku_id] += order.quantity
                self.budget -= order.quantity * order.price_per_unit
                order.status = "delivered"
                delivered_orders.append(order_id)
                self.fulfilled_orders += 1
                
                # Add to inbox
                self.inbox.append({
                    'type': 'delivery',
                    'order_id': order_id,
                    'sku': order.sku_id,
                    'quantity': order.quantity,
                    'day': self.current_day
                })
        
        # Remove delivered orders
        for order_id in delivered_orders:
            del self.orders[order_id]
    
    def _inject_shock(self) -> Optional[Shock]:
        """Inject a shock based on pe_type_mix and pe_mag."""
        # Determine shock type based on pe_type_mix
        if self.pe_type_mix == "temporal_only":
            shock_type = ShockType.TEMPORAL
        elif self.pe_type_mix == "quantity_only":
            shock_type = ShockType.QUANTITY
        elif self.pe_type_mix == "causal_only":
            shock_type = ShockType.CAUSAL
        elif self.pe_type_mix == "uniform":
            shock_type = self.rng.choice(list(ShockType))
        else:  # realistic
            # Weighted: temporal (40%), quantity (30%), causal (20%), rule (10%)
            shock_type = self.rng.choices(
                list(ShockType),
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]
        
        shock = Shock(
            shock_type=shock_type,
            magnitude=self.pe_mag,
            observability=self.observability
        )
        
        # Apply shock effects
        if shock_type == ShockType.TEMPORAL and self.orders:
            # Delay a random pending order
            pending_orders = [o for o in self.orders.values() if o.status == "pending"]
            if pending_orders:
                order = self.rng.choice(pending_orders)
                delay_map = {'low': 1, 'medium': 2, 'high': self.rng.randint(2, 4)}
                delay = delay_map.get(self.pe_mag, 1)
                order.eta_day += delay
                shock.affected_order = order.order_id
                shock.delay_days = delay
                
                if self.observability == "full":
                    self.inbox.append({
                        'type': 'delay_notice',
                        'order_id': order.order_id,
                        'new_eta': order.eta_day,
                        'day': self.current_day
                    })
        
        elif shock_type == ShockType.QUANTITY and self.orders:
            # Modify quantity of a pending order
            pending_orders = [o for o in self.orders.values() if o.status == "pending"]
            if pending_orders:
                order = self.rng.choice(pending_orders)
                mult_ranges = {
                    'low': (0.9, 1.1),
                    'medium': (0.7, 1.3),
                    'high': (0.5, 2.0)
                }
                min_mult, max_mult = mult_ranges.get(self.pe_mag, (0.9, 1.1))
                multiplier = self.rng.uniform(min_mult, max_mult)
                new_qty = max(1, int(order.quantity * multiplier))
                shock.quantity_multiplier = multiplier
                shock.affected_order = order.order_id
                order.quantity = new_qty
        
        self.shock_history.append(shock)
        return shock
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool action."""
        tool = action.get('tool')
        args = action.get('args', {})
        
        if tool == 'tool_order':
            return self._tool_order(**args)
        elif tool == 'tool_check_inbox':
            return self._tool_check_inbox()
        elif tool == 'tool_check_storage':
            return self._tool_check_storage()
        elif tool == 'tool_check_budget':
            return self._tool_check_budget()
        elif tool == 'tool_cancel_order':
            return self._tool_cancel_order(**args)
        elif tool == 'tool_quote':
            return self._tool_quote(**args)
        elif tool == 'tool_expedite':
            return self._tool_expedite(**args)
        elif tool == 'tool_write_scratchpad':
            return self._tool_write_scratchpad(**args)
        elif tool == 'tool_read_scratchpad':
            return self._tool_read_scratchpad(**args)
        elif tool == 'tool_delete_scratchpad':
            return self._tool_delete_scratchpad(**args)
        else:
            return {'success': False, 'error': f'Unknown tool: {tool}'}
    
    def _tool_order(self, supplier_id: str, sku: str, quantity: int) -> Dict[str, Any]:
        """Place an order."""
        self.total_orders_requested += 1
        
        # Find supplier
        supplier = next((s for s in self.suppliers if s.id == supplier_id), None)
        if not supplier:
            return {'success': False, 'error': 'Invalid supplier'}
        
        if sku not in supplier.base_price:
            return {'success': False, 'error': 'SKU not available from supplier'}
        
        # Create order
        self.order_counter += 1
        order_id = f"ORD{self.order_counter}"
        
        price = supplier.base_price[sku]
        lead_days = supplier.base_lead_days[sku]
        eta_day = self.current_day + lead_days
        
        order = Order(
            order_id=order_id,
            supplier_id=supplier_id,
            sku_id=sku,
            quantity=quantity,
            price_per_unit=price,
            order_day=self.current_day,
            eta_day=eta_day
        )
        
        self.orders[order_id] = order
        
        return {
            'success': True,
            'order_id': order_id,
            'eta_day': eta_day,
            'price': price * quantity
        }
    
    def _tool_check_inbox(self) -> Dict[str, Any]:
        """Check inbox messages."""
        messages = self.inbox.copy()
        self.inbox = []  # Clear after reading
        return {'success': True, 'messages': messages}
    
    def _tool_check_storage(self) -> Dict[str, Any]:
        """Check storage levels."""
        return {'success': True, 'storage': self.storage.copy()}
    
    def _tool_check_budget(self) -> Dict[str, Any]:
        """Check current budget."""
        return {'success': True, 'budget': self.budget}
    
    def _tool_cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return {'success': False, 'error': 'Order not found'}
        
        order = self.orders[order_id]
        if order.status != "pending":
            return {'success': False, 'error': 'Order already delivered/cancelled'}
        
        # Apply cancellation fee (10% of order value)
        fee = order.quantity * order.price_per_unit * 0.1
        self.budget -= fee
        
        order.status = "cancelled"
        del self.orders[order_id]
        
        return {'success': True, 'ok': True, 'fee': fee}
    
    def _tool_quote(self, supplier_id: str, sku: str, qty: int) -> Dict[str, Any]:
        """Get a quote from a supplier."""
        supplier = next((s for s in self.suppliers if s.id == supplier_id), None)
        if not supplier:
            return {'success': False, 'error': 'Invalid supplier'}
        
        if sku not in supplier.base_price:
            return {'success': False, 'error': 'SKU not available'}
        
        return {
            'success': True,
            'unit_price': supplier.base_price[sku],
            'lead_days': supplier.base_lead_days[sku]
        }
    
    def _tool_expedite(self, order_id: str) -> Dict[str, Any]:
        """Expedite an order (complexity level 2+)."""
        if self.complexity_level < 2:
            return {'success': False, 'error': 'Expedite not available at this complexity'}
        
        if order_id not in self.orders:
            return {'success': False, 'error': 'Order not found'}
        
        order = self.orders[order_id]
        if order.status != "pending":
            return {'success': False, 'error': 'Order already delivered'}
        
        # Reduce ETA by 1-2 days, charge 50% extra
        days_reduced = min(2, order.eta_day - self.current_day)
        order.eta_day -= days_reduced
        cost = order.quantity * order.price_per_unit * 0.5
        self.budget -= cost
        
        return {
            'success': True,
            'new_eta_day': order.eta_day,
            'cost': cost
        }
    
    def _tool_write_scratchpad(self, key: str, value: Any) -> Dict[str, Any]:
        """Write to scratchpad memory."""
        self.scratchpad[key] = value
        return {'success': True}
    
    def _tool_read_scratchpad(self, key: str) -> Dict[str, Any]:
        """Read from scratchpad memory."""
        value = self.scratchpad.get(key)
        return {'success': True, 'value': value}
    
    def _tool_delete_scratchpad(self, key: str) -> Dict[str, Any]:
        """Delete from scratchpad memory."""
        if key in self.scratchpad:
            del self.scratchpad[key]
            return {'success': True}
        return {'success': False, 'error': 'Key not found'}
    
    def _apply_daily_costs(self):
        """Apply daily storage fees."""
        total_storage = sum(self.storage.values())
        fee = total_storage * self.daily_storage_fee
        self.budget -= fee
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            'day': self.current_day,
            'budget': self.budget,
            'storage': self.storage.copy(),
            'storage_capacity': self.storage_capacity,
            'pending_orders': len([o for o in self.orders.values() if o.status == "pending"]),
            'inbox_count': len(self.inbox),
            'total_storage': sum(self.storage.values())
        }
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for logging."""
        return {
            'day': self.current_day,
            'budget': self.budget,
            'storage': self.storage.copy(),
            'orders': {oid: {
                'order_id': o.order_id,
                'supplier_id': o.supplier_id,
                'sku_id': o.sku_id,
                'quantity': o.quantity,
                'eta_day': o.eta_day,
                'status': o.status
            } for oid, o in self.orders.items()},
            'inbox': self.inbox.copy(),
            'fulfilled_orders': self.fulfilled_orders,
            'total_orders_requested': self.total_orders_requested,
            'scratchpad_size': len(self.scratchpad),
            'scratchpad': self.scratchpad.copy()  # Include full scratchpad contents
        }
