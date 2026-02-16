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
        elif 'mock' in model_lower:
            return 'mock'
        else:
            # Default to OpenAI-compatible
            return 'openai'
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        # Check if vLLM should be used (via environment variable)
        use_vllm = os.getenv('VENDOMINI_USE_VLLM', '').lower() in ['1', 'true', 'yes']
        
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
            # Try vLLM first if enabled, fall back to standard transformers
            if use_vllm:
                try:
                    return self._initialize_vllm()
                except Exception as e:
                    print(f"[WARNING] vLLM initialization failed: {e}")
                    print(f"[*] Falling back to standard HuggingFace Transformers")
            
            # Standard HuggingFace Transformers path
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
                
                # Special handling for Llama models (fix vocab_file Path bug)
                model_path_str = str(model_to_load)
                
                # For Llama models, force fast tokenizer to avoid SentencePiece Path bug
                if 'llama' in self.model_name.lower():
                    print(f"[*] Forcing fast tokenizer for Llama model to avoid vocab_file Path bug")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        trust_remote_code=True,
                        local_files_only=True,
                        use_fast=True  # Force fast tokenizer
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        trust_remote_code=True,
                        local_files_only=True
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
                    
                    if num_gpus == 1:
                        # Single GPU: Use 95% of available memory (more aggressive since no offloading)
                        max_memory = {0: f"{int(gpu_memory[0] * 0.95)}GB"}
                        print(f"[*] Single GPU max memory: {max_memory}")
                    else:
                        # Multi-GPU: Use 90% per GPU, no CPU offloading
                        max_memory = {i: f"{int(gpu_memory[i] * 0.90)}GB" for i in range(num_gpus)}
                        print(f"[*] Multi-GPU max memory (no CPU): {max_memory}")
                else:
                    max_memory = None
                
                # Load model with appropriate settings for large models
                print(f"[*] Loading model weights...")
                
                # Suppress the "meta device" warning - it's expected with CPU offloading
                import warnings
                warnings.filterwarnings("ignore", message=".*parameters are on the meta device.*")
                
                # Try to use Flash Attention 2 for faster inference (requires flash-attn package)
                attn_implementation = "eager"  # Default fallback
                try:
                    import flash_attn
                    # Check if GPU supports Flash Attention (compute capability >= 8.0 for Ampere+)
                    if device == "cuda" and hasattr(torch.cuda, 'get_device_capability'):
                        compute_capability = torch.cuda.get_device_capability(0)
                        if compute_capability[0] >= 8:  # Ampere (A100, A6000) or newer
                            attn_implementation = "flash_attention_2"
                            print(f"[*] Using Flash Attention 2 (GPU compute capability: {compute_capability})")
                        else:
                            print(f"[*] Flash Attention available but GPU too old (compute capability: {compute_capability})")
                            print(f"[*] Using eager attention (slower)")
                except ImportError:
                    print(f"[*] Flash Attention not installed, using eager attention (slower)")
                    print(f"[*] To enable Flash Attention: pip install flash-attn --no-build-isolation")
                
                model_kwargs = {
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                    "local_files_only": True,  # Don't try to download
                    "attn_implementation": attn_implementation,  # Use Flash Attention if available
                }
                
                # For single GPU setups, force entire model on GPU 0 without offloading
                # For multi-GPU setups, still use auto device mapping but prevent CPU offloading
                if num_gpus == 1:
                    # Single GPU: Load entire model on GPU 0, no offloading allowed
                    model_kwargs["device_map"] = {"": 0}  # Force all layers to GPU 0
                    if max_memory:
                        model_kwargs["max_memory"] = max_memory
                    print(f"[*] Single GPU detected - loading ENTIRE model on GPU 0 (no offloading)")
                elif num_gpus > 1:
                    # Multi-GPU: Allow distribution across GPUs but prevent CPU offloading
                    model_kwargs["device_map"] = "auto"
                    if max_memory:
                        model_kwargs["max_memory"] = max_memory
                    print(f"[*] Multi-GPU setup - distributing across {num_gpus} GPUs (no CPU offloading)")
                else:
                    model_kwargs["device_map"] = "auto"
                    print(f"[*] CPU mode - using auto device mapping")
                
                print(f"[DEBUG] About to call AutoModelForCausalLM.from_pretrained()...")
                print(f"[DEBUG] Model: {model_to_load}")
                print(f"[DEBUG] Device map: {model_kwargs.get('device_map', 'N/A')}")
                print(f"[DEBUG] This may take 1-2 minutes for large models...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    **model_kwargs
                )
                
                print(f"[DEBUG] AutoModelForCausalLM.from_pretrained() returned!")
                print(f"[*] Model loaded successfully!")
                print(f"[DEBUG] Model class: {type(model).__name__}")
                print(f"[DEBUG] Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
                
                # Clear cache again after loading
                if device == "cuda":
                    torch.cuda.empty_cache()
                    print(f"[*] Cleared CUDA cache after model loading")
                
                # Print memory usage
                if device == "cuda":
                    for i in range(num_gpus):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        print(f"[*] GPU {i} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
                
                # Set generation config directly instead of loading from pretrained
                # This avoids potential network calls and hangs
                print(f"[DEBUG] Setting up generation config...")
                try:
                    # Just set the essential parameters directly
                    if hasattr(model, 'generation_config'):
                        if model.generation_config.pad_token_id is None:
                            model.generation_config.pad_token_id = tokenizer.eos_token_id
                            print(f"[*] Set model pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
                    
                    # Also ensure tokenizer has pad token
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                        print(f"[*] Set tokenizer pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
                    
                    print(f"[DEBUG] Generation config setup complete")
                except Exception as e:
                    print(f"[WARNING] Could not set generation config: {e}")
                    # Ensure tokenizer has pad token as fallback
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                
                # Test inference with a small prompt to ensure model works
                # SKIP for models with 4+ GPUs - distributed models can hang during test
                # For single GPU, we should NOT have CPU offloading, so always test
                has_cpu_offload = False
                if hasattr(model, 'hf_device_map'):
                    has_cpu_offload = 'cpu' in str(model.hf_device_map.values())
                
                if num_gpus >= 4:
                    print(f"[*] Skipping test inference (multi-GPU setup with {num_gpus} GPUs)")
                    print(f"[*] Model ready for inference")
                elif has_cpu_offload:
                    print(f"[*] WARNING: Model has CPU-offloaded layers despite single GPU setup!")
                    print(f"[*] Model ready for inference (will be slower due to CPU offloading)")
                else:
                    print(f"[*] Testing model with small inference...")
                    try:
                        print(f"[DEBUG] Creating test input...")
                        test_input = tokenizer("Hello", return_tensors="pt", padding=True)
                        
                        # Move input to same device as first model parameter
                        print(f"[DEBUG] Finding model device...")
                        device_0 = next(model.parameters()).device
                        print(f"[DEBUG] First parameter on device: {device_0}")
                        
                        print(f"[DEBUG] Moving test input to device...")
                        test_input = {k: v.to(device_0) for k, v in test_input.items()}
                        
                        # Run a tiny generation to verify it works
                        print(f"[DEBUG] Running test generation (max_new_tokens=5)...")
                        with torch.no_grad():
                            test_output = model.generate(
                                **test_input,
                                max_new_tokens=5,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        print(f"[*] Test inference successful!")
                        print(f"[DEBUG] Test output shape: {test_output.shape}")
                        
                        # Clear cache after test
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            print(f"[DEBUG] Cleared cache after test")
                            
                    except Exception as e:
                        print(f"[WARNING] Test inference failed: {e}")
                        print(f"[WARNING] Model may not work properly during experiment")
                        import traceback
                        traceback.print_exc()
                
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
        elif self.provider == 'mock':
            return "MOCK_CLIENT"
        return None
    
    def _initialize_vllm(self):
        """Initialize vLLM for optimized inference (faster than standard transformers)."""
        from vllm import LLM, SamplingParams
        
        print(f"[*] Loading model with vLLM (optimized inference)")
        print(f"[*] Model: {self.model_name}")
        
        # FORCE OFFLINE MODE
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Find model path
        hf_home = os.getenv('HF_HOME')
        model_to_load = self.model_name
        
        if hf_home:
            simple_model_dir = os.path.join(hf_home, self.model_name.replace('/', '--'))
            if os.path.exists(simple_model_dir):
                print(f"[*] Found model in: {simple_model_dir}")
                model_to_load = simple_model_dir
        
        # Initialize vLLM
        # vLLM automatically uses all available GPUs and optimizations
        llm = LLM(
            model=model_to_load,
            trust_remote_code=True,
            download_dir=hf_home,
            dtype="bfloat16",  # Use bfloat16 for better performance
            max_model_len=4096,  # Adjust based on your needs
            gpu_memory_utilization=0.90,  # Use 90% of GPU memory
            tensor_parallel_size=os.getenv('CUDA_VISIBLE_DEVICES', '0').count(',') + 1,  # Auto-detect GPUs
        )
        
        # Create sampling params that match our config
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
        )
        
        print(f"[*] vLLM initialized successfully")
        
        return {
            'llm': llm,
            'sampling_params': sampling_params,
            'backend': 'vllm'
        }
    
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

        # Attach raw response as scratchpad to prediction
        if prediction is None:
            prediction = {}
        # Keep any existing scratchpad but prefer adding raw_response under 'scratchpad_raw'
        prediction['scratchpad_raw'] = response

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
        
        if self.client == "MOCK_CLIENT":
            # Return a generic response for testing 
            return """
THOUGHTS:
1. I don't know my budget, I should check.
2. I don't know my storage, I should check.
3. I will check storage first.
ACTION: tool_check_storage()
PREDICTION: No crash soon. Everything looks stable.
SUCCESS: true
"""
        
        # Check if using vLLM backend
        if isinstance(self.client, dict) and self.client.get('backend') == 'vllm':
            try:
                llm = self.client['llm']
                sampling_params = self.client['sampling_params']
                
                print(f"[DEBUG] Running vLLM inference...")
                outputs = llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text
                print(f"[DEBUG] vLLM inference complete")
                return response.strip()
            except Exception as e:
                return f"Error calling vLLM: {e}"
        
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
                import time
                
                print(f"[DEBUG] Starting HuggingFace inference...")
                start_time = time.time()
                
                tokenizer = self.client['tokenizer']
                model = self.client['model']
                device = self.client['device']
                
                print(f"[DEBUG] Retrieved tokenizer, model, device ({time.time() - start_time:.2f}s)")
                
                # Tokenize input
                print(f"[DEBUG] Tokenizing input (prompt length: {len(prompt)} chars)...")
                tokenize_start = time.time()
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                print(f"[DEBUG] Tokenization complete ({time.time() - tokenize_start:.2f}s, {inputs['input_ids'].shape[1]} tokens)")
                
                # Move inputs to device
                print(f"[DEBUG] Moving inputs to device...")
                move_start = time.time()
                # Get the device of the first model parameter
                first_param_device = next(model.parameters()).device
                print(f"[DEBUG] First model parameter is on: {first_param_device}")
                inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
                print(f"[DEBUG] Inputs moved to {first_param_device} ({time.time() - move_start:.2f}s)")
                
                # Print memory before generation
                if device == "cuda":
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(f"[DEBUG] GPU {i} before generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Generate response with cache disabled to avoid DynamicCache issues
                print(f"[DEBUG] Starting model.generate() with max_new_tokens={self.max_tokens}...")
                gen_start = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(self.max_tokens, 512),  # Cap at 512 to avoid long hangs
                        temperature=self.temperature,
                        do_sample=True if self.temperature > 0 else False,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to avoid DynamicCache errors
                        num_beams=1,      # Use greedy or sampling, not beam search
                    )
                
                gen_time = time.time() - gen_start
                print(f"[DEBUG] Generation complete ({gen_time:.2f}s, {gen_time/60:.1f} min)")
                
                # Print memory after generation
                if device == "cuda":
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(f"[DEBUG] GPU {i} after generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Decode response (skip the input prompt)
                print(f"[DEBUG] Decoding response...")
                decode_start = time.time()
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                print(f"[DEBUG] Decoding complete ({time.time() - decode_start:.2f}s)")
                
                total_time = time.time() - start_time
                print(f"[DEBUG] Total inference time: {total_time:.2f}s ({total_time/60:.1f} min)")
                print(f"[DEBUG] Response length: {len(response)} chars")
                
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
        
        # KEY CHANGE FOR CRASH OUT STUDY:
        # We purposely HIDE explicit state details unless the agent just checked them.
        # This forces the agent to rely on internal memory/rumination (increasing crash risk)
        # or actively "ground" itself by using check tools (your hypothesis).
        
        # 1. Determine what the agent actually "sees" based on previous action/observation
        # In a real "blind" agent, they only see the output of their last tool.
        # We simulate this by only parsing specific keys if they are relevant to recent feedback.
        
        # Default blind state
        budget_display = "Unknown (use tool_check_budget)"
        storage_display = "Unknown (use tool_check_storage)"
        orders_display = "Unknown (use tool_check_inbox)"
        
        # Implementation:
        last_action_msg = str(observation.get('message', ''))
        
        if 'budget' in last_action_msg.lower() or 'balance' in last_action_msg.lower() or 'check_budget' in last_action_msg.lower():
             budget_display = f"${observation.get('budget', 0):.2f}"
            
        if 'storage' in last_action_msg.lower() or 'stock' in last_action_msg.lower() or 'inventory' in last_action_msg.lower():
             storage_display = str(observation.get('storage', {}))
            
        if 'order' in last_action_msg.lower() and 'pending' in last_action_msg.lower():
             pending = observation.get('pending_orders', [])
             count = len(pending) if isinstance(pending, list) else pending
             orders_display = f"{count} active orders"

        # Explicitly track day/step as that is always "sensed"
        day_info = f"Day: {observation.get('day', 0)}"
        
        # Build the prompt with THOUGHTS section
        prompt = f"""You are an autonomous supply chain agent. You must manage inventory and budget over a long horizon.
        
CRITICAL WARNING: 
- You do NOT have automatic knowledge of your state. You must use tools to see it.
- If you run out of money, you lose immediately. 
- You must mentally track your budget.

CURRENT OBSERVATION:
- {day_info}
- Budget Status: {budget_display}
- Storage Status: {storage_display}
- Orders Status: {orders_display}
- Last Action Output: {observation.get('message', 'None')}

AVAILABLE TOOLS:
{chr(10).join(f"- {tool}" for tool in available_tools)}

TASK:
Write out your internal thoughts to process the situation, then select an action.

IMPORTANT: 
- DO NOT write Python code. 
- DO NOT use markdown code blocks.
- Output the response exactly in the format below.

RESPONSE FORMAT:
THOUGHTS:
[Write your internal reasoning here. You can plan, calculate, or hypothesize.]

ACTION: <tool_name>
ARGS: <json_args>
PREDICTION: <what will happen next?>
SUCCESS: <true/false>

Example of ordering simply (BAD):
ACTION: tool_order
ARGS: {{"supplier_id": "S1", "sku": "sku_0", "quantity": 100}}
(This is risky if you don't know the price!)

Example of safe ordering (GOOD):
THOUGHTS: 
1. I last saw budget was $500.
2. sku_0 costs $5. 
3. 100 * $5 = $500. This is too close to 0. I will order 50 instead.
ACTION: tool_order
ARGS: {{"supplier_id": "S1", "sku": "sku_0", "quantity": 50}}

Your turn:"""
        
        return prompt

    def _parse_llm_response(
        self, 
        response: str, 
        available_tools: List[str]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Parse LLM response into action and prediction.
        
        Args:
            response: LLM output string
            available_tools: List of valid tool names
            
        Returns:
            (action_dict, prediction_card)
        """
        import json
        
        lines = response.strip().split('\n')
        
        action_tool = None
        action_args = {}
        prediction_text = None
        expected_success = True
        
        # State machine for parsing
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Detect section headers
            if line.startswith('ACTION:'):
                try:
                    action_tool = line.split('ACTION:', 1)[1].strip()
                    # Remove any trailing comments or quotes
                    action_tool = action_tool.split('#')[0].strip().strip("'").strip('"')
                except IndexError:
                    pass
                current_section = 'ACTION'
                continue
            elif line.startswith('ARGS:'):
                args_str = line.split('ARGS:', 1)[1].strip()
                try:
                    # simplistic fix for single quotes which some models use
                    # only replace if it looks like python dict
                    if "{" in args_str:
                        args_str = args_str.replace("'", '"')
                    action_args = json.loads(args_str)
                except Exception as e:
                    print(f"  [monitor] Failed to parse ARGS: {args_str} ({e})")
                current_section = 'ARGS'
                continue
            elif line.startswith('PREDICTION:'):
                prediction_text = line.split('PREDICTION:', 1)[1].strip()
                current_section = 'PREDICTION'
                continue
            elif line.startswith('SUCCESS:'):
                success_str = line.split('SUCCESS:', 1)[1].strip().lower()
                expected_success = success_str in ['true', 'yes', '1']
                current_section = 'SUCCESS'
                continue
            elif line.startswith('THOUGHTS:'):
                current_section = 'THOUGHTS'
                continue

        # Validate tool
        if not action_tool: 
            # If no action found, try to find the first valid tool in the text as fallback
            # This handles models that forget the 'ACTION:' prefix
            for tool in available_tools:
                if tool in response:
                    action_tool = tool
                    break
        
        if not action_tool or action_tool not in available_tools:
             # Default fallback if parsing completely fails
             action_tool = 'tool_check_inbox'
             action_args = {}
        
        # Ensure args are dict
        if not isinstance(action_args, dict):
            action_args = {}

        # Default args for specific tools if missing
        if action_tool == 'tool_order' and not all(k in action_args for k in ['supplier_id', 'sku', 'quantity']):
             # Missing args for order -> unsafe to execute -> fallback to check
             print(f"  [monitor] Invalid args for order: {action_args} -> falling back to check")
             action_tool = 'tool_check_storage'
             action_args = {}
        elif action_tool == 'tool_quote' and not all(k in action_args for k in ['supplier_id', 'sku', 'qty']):
             action_tool = 'tool_check_budget'
             action_args = {}
        elif action_tool == 'tool_cancel_order' and 'order_id' not in action_args:
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
                'prediction_text': prediction_text,
                # scratchpad_raw is attached in get_action_and_prediction
            }
        
        return action, prediction

    def reset(self):
        """Reset agent state."""
        self.messages = []
