"""LLM Agent interface for VendoMini."""

import json
from typing import Dict, Any, Optional, List
import os


class LLMAgent:
    """
    Interface to LLM for agent decisions and predictions.
    
    Supports multiple providers (OpenAI, Anthropic) and handles
    prediction card generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM agent.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        model_cfg = config.get('model', {})
        interface_cfg = config.get('interface', {})
        
        self.model_name = model_cfg.get('name', 'gpt-4')
        self.temperature = model_cfg.get('temperature', 0.3)
        self.max_tokens = model_cfg.get('max_tokens_per_call', 2000)
        self.context_length = model_cfg.get('context_length', 32000)
        
        self.prediction_mode = interface_cfg.get('prediction_mode', 'required')
        self.prediction_format = interface_cfg.get('prediction_format', 'structured')
        self.memory_tools = interface_cfg.get('memory_tools', 'full')
        self.recovery_tools = interface_cfg.get('recovery_tools', 'none')
        
        # Initialize provider
        self.provider = self._detect_provider()
        self.client = self._initialize_client()
        
        # Conversation history
        self.messages = []
        
    def _detect_provider(self) -> str:
        """Detect which LLM provider to use based on model name."""
        model_lower = self.model_name.lower()
        
        if 'gpt' in model_lower or 'o1' in model_lower:
            return 'openai'
        elif 'claude' in model_lower:
            return 'anthropic'
        elif '/' in self.model_name:  # HuggingFace format: org/model
            return 'huggingface'
        else:
            # Default to OpenAI-compatible
            return 'openai'
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if self.provider == 'openai':
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    return openai.OpenAI(api_key=api_key)
                return None
            except ImportError:
                return None
        elif self.provider == 'anthropic':
            try:
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    return anthropic.Anthropic(api_key=api_key)
                return None
            except ImportError:
                return None
        elif self.provider == 'huggingface':
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                print(f"Loading HuggingFace model: {self.model_name}")
                
                # Check for local model first (cluster-aware loading)
                models_dir = os.getenv('TRANSFORMERS_CACHE', os.getenv('HF_HOME', None))
                model_to_load = self.model_name  # Default to model name
                
                if models_dir:
                    # Try to find local model using cluster utilities pattern
                    try:
                        from cluster_utils import get_local_model_path
                        local_path = get_local_model_path(models_dir, self.model_name)
                        if local_path:
                            model_to_load = local_path
                            print(f"📦 Using local model from: {local_path}")
                        else:
                            print(f"🌐 Local model not found, will download: {self.model_name}")
                    except ImportError:
                        print(f"🌐 cluster_utils not available, using model name: {self.model_name}")
                
                # Load tokenizer
                print(f"Loading tokenizer from: {model_to_load}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_to_load,
                    trust_remote_code=True
                )
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                
                # Load model with appropriate settings
                print(f"Loading model from: {model_to_load}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                return {
                    'tokenizer': tokenizer,
                    'model': model,
                    'device': device
                }
            except ImportError as e:
                print(f"HuggingFace dependencies not available: {e}")
                return None
            except Exception as e:
                print(f"Error loading HuggingFace model: {e}")
                return None
        return None
    
    def get_action_and_prediction(
        self, 
        observation: Dict[str, Any],
        available_tools: List[str]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Get action and optional prediction from LLM.
        
        Args:
            observation: Current environment observation
            available_tools: List of available tool names
            
        Returns:
            (action_dict, prediction_card)
        """
        # Use LLM if available, otherwise fall back to heuristic
        if self.client is None:
            return self._heuristic_agent(observation, available_tools)
        
        # Build prompt for LLM
        prompt = self._build_prompt(observation, available_tools)
        
        # Get LLM response
        response = self._call_llm(prompt)
        
        # Debug: print response
        if len(response) < 500:
            print(f"  🔍 LLM Response: {response}")
        
        # Parse response into action and prediction
        try:
            action, prediction = self._parse_llm_response(response, available_tools)
            return action, prediction
        except Exception as e:
            print(f"  ⚠️ Error parsing LLM response: {e}")
            print(f"     Response was: {response[:200]}")
            print(f"     Falling back to heuristic")
            return self._heuristic_agent(observation, available_tools)
    
    def _heuristic_agent(
        self, 
        observation: Dict[str, Any],
        available_tools: List[str]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Simple heuristic agent for testing.
        
        This is a placeholder that should be replaced with actual LLM calls.
        """
        # Simple strategy: Check storage, order if low
        storage = observation.get('storage', {})
        budget = observation.get('budget', 0)
        
        # Check inbox first
        action = {
            'tool': 'tool_check_inbox',
            'args': {}
        }
        
        prediction = None
        if self.prediction_mode != 'optional':
            prediction = {
                'tool': 'tool_check_inbox',
                'args': {},
                'expected_success': True
            }
        
        return action, prediction
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM provider.
        
        Args:
            prompt: Prompt text
            
        Returns:
            LLM response
        """
        if self.client is None:
            # No client available, return dummy response
            return "No LLM client configured"
        
        if self.provider == 'openai':
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling OpenAI: {e}"
        
        elif self.provider == 'anthropic':
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                return f"Error calling Anthropic: {e}"
        
        elif self.provider == 'huggingface':
            try:
                import torch
                
                tokenizer = self.client['tokenizer']
                model = self.client['model']
                device = self.client['device']
                
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response (skip the input prompt)
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                return response.strip()
            
            except Exception as e:
                return f"Error calling HuggingFace model: {e}"
        
        return "Unknown provider"
    
    def _build_prompt(self, observation: Dict[str, Any], available_tools: List[str]) -> str:
        """Build prompt for LLM based on current observation."""
        
        # Handle different observation formats
        pending_orders = observation.get('pending_orders', [])
        if isinstance(pending_orders, int):
            num_pending = pending_orders
        else:
            num_pending = len(pending_orders) if pending_orders else 0
        
        inbox = observation.get('inbox', [])
        if isinstance(inbox, int):
            num_messages = inbox
        else:
            num_messages = len(inbox) if inbox else 0
        
        # Get available SKUs and suppliers from storage
        storage = observation.get('storage', {})
        skus = list(storage.keys())[:3]  # Show first 3 as examples
        
        prompt = f"""You are managing a vending machine business. Here's your current situation:

CURRENT STATE:
- Day: {observation.get('day', 0)}
- Budget: ${observation.get('budget', 0):.2f}
- Storage: {storage}
- Pending orders: {num_pending}
- Inbox messages: {num_messages}

AVAILABLE TOOLS:
{chr(10).join(f"- {tool}" for tool in available_tools)}

TOOL DESCRIPTIONS & FORMATS:
- tool_check_inbox: Check messages
  Format: {{"tool": "tool_check_inbox", "args": {{}}}}

- tool_check_storage: Check inventory
  Format: {{"tool": "tool_check_storage", "args": {{}}}}

- tool_check_budget: Check available budget
  Format: {{"tool": "tool_check_budget", "args": {{}}}}

- tool_order: Order items from supplier
  Format: {{"tool": "tool_order", "args": {{"supplier_id": "S1", "sku": "sku_0", "quantity": 10}}}}
  Available SKUs: {', '.join(skus) if skus else 'sku_0, sku_1, sku_2'}
  Available Suppliers: S1, S2, S3

- tool_quote: Get price quote
  Format: {{"tool": "tool_quote", "args": {{"supplier_id": "S1", "sku": "sku_0", "qty": 10}}}}

- tool_cancel_order: Cancel pending order
  Format: {{"tool": "tool_cancel_order", "args": {{"order_id": "order_123"}}}}

Your task: Choose ONE tool to use and predict the outcome.

Respond in this exact format:
ACTION: <tool_name>
ARGS: <arguments as JSON>
PREDICTION: <what you expect to happen>
SUCCESS: <true/false>

Example:
ACTION: tool_check_storage
ARGS: {{}}
PREDICTION: Will show current inventory levels
SUCCESS: true

Your response:"""
        
        return prompt
    
    def _parse_llm_response(
        self, 
        response: str, 
        available_tools: List[str]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Parse LLM response into action and prediction."""
        
        # Extract components from response
        lines = response.strip().split('\n')
        
        action_tool = None
        action_args = {}
        prediction_text = None
        expected_success = True
        
        for line in lines:
            line = line.strip()
            if line.startswith('ACTION:'):
                action_tool = line.split('ACTION:')[1].strip()
            elif line.startswith('ARGS:'):
                args_str = line.split('ARGS:')[1].strip()
                if args_str and args_str != '{}':
                    try:
                        action_args = json.loads(args_str)
                    except:
                        action_args = {}
            elif line.startswith('PREDICTION:'):
                prediction_text = line.split('PREDICTION:')[1].strip()
            elif line.startswith('SUCCESS:'):
                success_str = line.split('SUCCESS:')[1].strip().lower()
                expected_success = success_str in ['true', 'yes', '1']
        
        # Validate tool
        if not action_tool or action_tool not in available_tools:
            # Default to safe action
            action_tool = 'tool_check_inbox'
            action_args = {}
        
        # Ensure required args for specific tools
        if action_tool == 'tool_order' and not all(k in action_args for k in ['supplier_id', 'sku', 'quantity']):
            # Missing required args - fall back to safe action
            print(f"     ⚠️ Missing args for tool_order, using tool_check_storage instead")
            action_tool = 'tool_check_storage'
            action_args = {}
        elif action_tool == 'tool_quote' and not all(k in action_args for k in ['supplier_id', 'sku', 'qty']):
            print(f"     ⚠️ Missing args for tool_quote, using tool_check_budget instead")
            action_tool = 'tool_check_budget'
            action_args = {}
        elif action_tool == 'tool_cancel_order' and 'order_id' not in action_args:
            print(f"     ⚠️ Missing order_id for cancel, using tool_check_inbox instead")
            action_tool = 'tool_check_inbox'
            action_args = {}
        
        action = {
            'tool': action_tool,
            'args': action_args
        }
        
        # Build prediction card
        prediction = None
        if self.prediction_mode != 'none':
            prediction = {
                'tool': action_tool,
                'args': action_args,
                'expected_success': expected_success,
                'prediction_text': prediction_text
            }
        
        return action, prediction
    
    def reset(self):
        """Reset agent state."""
        self.messages = []
