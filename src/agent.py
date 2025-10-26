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
                   Can be full config with 'agent' section or just agent config
        """
        self.config = config
        
        # Handle both config structures:
        # 1. Full config with agent.model and agent.interface
        # 2. Agent config with model and interface at top level
        if 'agent' in config:
            # Full config structure - extract agent section
            agent_cfg = config['agent']
            model_cfg = agent_cfg.get('model', {})
            interface_cfg = agent_cfg.get('interface', {})
        else:
            # Direct agent config structure
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
        
        # Check for HuggingFace format FIRST (org/model)
        if '/' in self.model_name:
            return 'huggingface'
        # Then check for OpenAI models (no slash, contains gpt or o1)
        elif 'gpt' in model_lower or 'o1' in model_lower:
            return 'openai'
        elif 'claude' in model_lower:
            return 'anthropic'
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
                
                # FORCE OFFLINE MODE - Don't contact HuggingFace servers
                # This is critical for cluster compute nodes without internet access
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                
                print(f"[*] Loading HuggingFace model: {self.model_name}")
                print(f"[*] OFFLINE MODE: Models must be pre-cached locally")
                
                # Check cache directories
                hf_home = os.getenv('HF_HOME')
                transformers_cache = os.getenv('TRANSFORMERS_CACHE')
                if hf_home:
                    print(f"[*] HF_HOME: {hf_home}")
                if transformers_cache:
                    print(f"[*] TRANSFORMERS_CACHE: {transformers_cache}")
                
                # Try to find the model in the local directory structure
                # Models might be in: models/org--name/ format (from --local-dir downloads)
                model_to_load = self.model_name
                
                # Check if model exists in simple directory format (from --local-dir)
                if hf_home:
                    simple_model_dir = os.path.join(hf_home, self.model_name.replace('/', '--'))
                    if os.path.exists(simple_model_dir):
                        print(f"[*] Found model in simple directory: {simple_model_dir}")
                        model_to_load = simple_model_dir
                    else:
                        print(f"[*] Simple directory not found: {simple_model_dir}")
                        print(f"[*] Will try using model name: {self.model_name}")
                
                # Load tokenizer
                print(f"[*] Loading tokenizer from: {model_to_load}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_to_load,
                    trust_remote_code=True,
                    local_files_only=True  # Don't try to download
                )
                
                # Ensure tokenizer has pad token set
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token:
                        tokenizer.pad_token = tokenizer.eos_token
                        print(f"[*] Set pad_token to eos_token: {tokenizer.eos_token}")
                    else:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        print(f"[*] Added [PAD] as pad_token")
                
                print(f"[*] Tokenizer loaded successfully")
                print(f"[*] Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
                print(f"[*] EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
                
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[*] Device: {device}")
                
                # Set memory management environment variable
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Clear GPU cache before loading
                if device == "cuda":
                    torch.cuda.empty_cache()
                    print(f"[*] Cleared CUDA cache")
                
                # Determine dtype - use bfloat16 for better stability on large models
                if device == "cuda":
                    # Check if bfloat16 is supported
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                        print(f"[*] Using bfloat16 (supported by GPU)")
                    else:
                        dtype = torch.float16
                        print(f"[*] Using float16 (bfloat16 not supported)")
                else:
                    dtype = torch.float32
                    print(f"[*] Using float32 (CPU mode)")
                
                # Set max memory per GPU - be more conservative
                num_gpus = torch.cuda.device_count() if device == "cuda" else 0
                
                # Get available memory per GPU
                if num_gpus > 0:
                    gpu_memory = []
                    for i in range(num_gpus):
                        mem = torch.cuda.get_device_properties(i).total_memory
                        gpu_memory.append(mem / (1024**3))  # Convert to GB
                        print(f"[*] GPU {i}: {gpu_memory[-1]:.2f} GB total")
                    
                    # Use 90% of available memory to leave headroom, allow CPU offloading
                    max_memory = {i: f"{int(gpu_memory[i] * 0.9)}GB" for i in range(num_gpus)}
                    max_memory["cpu"] = "120GB"  # Allow CPU offloading for layers that don't fit on GPU
                    print(f"[*] Max memory allocation: {max_memory}")
                else:
                    max_memory = None
                
                # Load model with appropriate settings for large models
                print(f"[*] Loading model weights...")
                model_kwargs = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "local_files_only": True,  # Don't try to download
                }
                
                # For multi-GPU or large models, use auto device mapping with max_memory and offloading
                if max_memory:
                    model_kwargs["device_map"] = "auto"  # Optimally distribute across GPUs and CPU
                    model_kwargs["max_memory"] = max_memory
                    model_kwargs["offload_folder"] = "./offload"  # Disk offload for very large models
                    model_kwargs["offload_state_dict"] = True  # Reduces peak memory during loading
                    print(f"[*] Using auto device mapping with max_memory constraints and CPU offloading")
                else:
                    model_kwargs["device_map"] = "auto"
                    print(f"[*] Using auto device mapping")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    **model_kwargs
                )
                
                print(f"[*] Model loaded successfully!")
                
                # Set up generation config with proper pad token
                try:
                    from transformers import GenerationConfig
                    model.generation_config = GenerationConfig.from_pretrained(model_to_load, local_files_only=True)
                    if model.generation_config.pad_token_id is None:
                        model.generation_config.pad_token_id = model.generation_config.eos_token_id
                        print(f"[*] Set pad_token_id to eos_token_id: {model.generation_config.eos_token_id}")
                except Exception as e:
                    print(f"[*] Could not load generation config: {e}")
                    # Set basic generation config
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                
                return {
                    'tokenizer': tokenizer,
                    'model': model,
                    'device': device
                }
            except ImportError as e:
                print(f"[ERROR] HuggingFace dependencies not available: {e}")
                return None
            except Exception as e:
                print(f"[ERROR] Failed to load HuggingFace model: {e}")
                import traceback
                traceback.print_exc()
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
            
        Raises:
            RuntimeError: If LLM client is not initialized
        """
        # Check if LLM is available - NO FALLBACK
        if self.client is None:
            raise RuntimeError(
                f"LLM client not initialized for model '{self.model_name}'. "
                f"Provider: {self.provider}. "
                f"This experiment cannot run without a working LLM."
            )
        
        # Build prompt for LLM
        prompt = self._build_prompt(observation, available_tools)
        
        # Get LLM response
        response = self._call_llm(prompt)
        
        # Debug: print response
        if len(response) < 500:
            print(f"  [DEBUG] LLM Response: {response}")
        
        # Parse response into action and prediction
        action, prediction = self._parse_llm_response(response, available_tools)
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
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate response with cache disabled to avoid DynamicCache issues
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True if self.temperature > 0 else False,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to avoid DynamicCache errors
                        num_beams=1,      # Use greedy or sampling, not beam search
                    )
                
                # Decode response (skip the input prompt)
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                return response.strip()
            
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"[ERROR] HuggingFace generation failed: {e}")
                print(f"[ERROR] Traceback: {error_details}")
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
            print(f"     [WARNING] Missing args for tool_order, using tool_check_storage instead")
            action_tool = 'tool_check_storage'
            action_args = {}
        elif action_tool == 'tool_quote' and not all(k in action_args for k in ['supplier_id', 'sku', 'qty']):
            print(f"     [WARNING] Missing args for tool_quote, using tool_check_budget instead")
            action_tool = 'tool_check_budget'
            action_args = {}
        elif action_tool == 'tool_cancel_order' and 'order_id' not in action_args:
            print(f"     [WARNING] Missing order_id for cancel, using tool_check_inbox instead")
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
