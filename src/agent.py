"""LLM Agent interface for VendoMini."""

import json
import os
from typing import Any, Dict, List, Optional

# JSON schema for the structured decision call (Phase 2 of two-phase inference).
# The 'tool' enum is filled in dynamically per call so the model cannot hallucinate
# a tool name.  Constrained decoding (vLLM guided_json) enforces this at token level.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "prediction_text": {"type": "string"},
        "expected_success": {"type": "boolean"},
    },
    "required": ["tool", "args", "prediction_text", "expected_success"],
    "additionalProperties": False,
}


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
        if "agent" in config:
            # Full config structure - extract agent section
            agent_cfg = config["agent"]
            model_cfg = agent_cfg.get("model", {})
            interface_cfg = agent_cfg.get("interface", {})
        else:
            # Direct agent config structure
            model_cfg = config.get("model", {})
            interface_cfg = config.get("interface", {})

        self.model_name = model_cfg.get("name", "gpt-4")
        self.temperature = model_cfg.get("temperature", 0.3)
        self.max_tokens = model_cfg.get("max_tokens_per_call", 2000)
        self.context_length = model_cfg.get("context_length", 32000)

        self.prediction_mode = interface_cfg.get("prediction_mode", "required")
        self.prediction_format = interface_cfg.get("prediction_format", "structured")
        self.memory_tools = interface_cfg.get("memory_tools", "full")
        self.recovery_tools = interface_cfg.get("recovery_tools", "none")

        # Initialize provider
        self.provider = self._detect_provider()
        self.client = self._initialize_client()

        # Conversation history
        self.messages = []

        # Mock agent step counter (used only when provider == 'mock')
        self._mock_step = 0

    def _detect_provider(self) -> str:
        """Detect which LLM provider to use based on model name."""
        model_lower = self.model_name.lower()

        # Check for HuggingFace format FIRST (org/model)
        if "/" in self.model_name:
            return "huggingface"
        # Then check for OpenAI models (no slash, contains gpt or o1)
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "mock" in model_lower:
            return "mock"
        else:
            # Default to OpenAI-compatible
            return "openai"

    def _initialize_client(self):
        """Initialize the LLM client."""
        # Check if vLLM should be used (via environment variable)
        use_vllm = os.getenv("VENDOMINI_USE_VLLM", "").lower() in ["1", "true", "yes"]

        if self.provider == "openai":
            try:
                import openai

                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    return openai.OpenAI(api_key=api_key)
                return None
            except ImportError:
                return None
        elif self.provider == "anthropic":
            try:
                import anthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    return anthropic.Anthropic(api_key=api_key)
                return None
            except ImportError:
                return None
        elif self.provider == "huggingface":
            # Try vLLM first if enabled, fall back to standard transformers
            if use_vllm:
                try:
                    return self._initialize_vllm()
                except Exception as e:
                    print(f"[WARNING] vLLM initialization failed: {e}")
                    print("[*] Falling back to standard HuggingFace Transformers")

            # Standard HuggingFace Transformers path
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # FORCE OFFLINE MODE - Don't contact HuggingFace servers
                # This is critical for cluster compute nodes without internet access
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

                print(f"[*] Loading HuggingFace model: {self.model_name}")
                print("[*] OFFLINE MODE: Models must be pre-cached locally")

                # Check cache directories
                hf_home = os.getenv("HF_HOME")
                transformers_cache = os.getenv("TRANSFORMERS_CACHE")
                if hf_home:
                    print(f"[*] HF_HOME: {hf_home}")
                if transformers_cache:
                    print(f"[*] TRANSFORMERS_CACHE: {transformers_cache}")

                # Try to find the model in the local directory structure
                # Pattern 1: flat --local-dir download  → {HF_HOME}/{org}--{model}/
                # Pattern 2: standard HF hub cache      → {HF_HOME}/models--{org}--{model}/snapshots/{hash}/
                model_to_load = self.model_name

                if hf_home:
                    flat_dir = os.path.join(hf_home, self.model_name.replace("/", "--"))
                    hub_dir = os.path.join(
                        hf_home, "models--" + self.model_name.replace("/", "--")
                    )

                    if os.path.isdir(flat_dir):
                        print(f"[*] Found model (flat --local-dir): {flat_dir}")
                        model_to_load = flat_dir
                    elif os.path.isdir(hub_dir):
                        # Resolve the most recent snapshot inside the HF hub cache dir
                        snapshots_dir = os.path.join(hub_dir, "snapshots")
                        if os.path.isdir(snapshots_dir):
                            snapshots = sorted(os.listdir(snapshots_dir))
                            if snapshots:
                                snapshot_path = os.path.join(
                                    snapshots_dir, snapshots[-1]
                                )
                                print(
                                    f"[*] Found model (HF hub cache snapshot): {snapshot_path}"
                                )
                                model_to_load = snapshot_path
                            else:
                                print(
                                    f"[*] Hub cache dir exists but has no snapshots: {hub_dir}"
                                )
                        else:
                            print(
                                f"[*] Hub cache dir exists but no 'snapshots' subdir: {hub_dir}"
                            )
                    else:
                        print("[*] Model not found locally. Checked:")
                        print(f"    flat:  {flat_dir}")
                        print(f"    hub:   {hub_dir}")
                        print(
                            f"[*] Will try model name with local_files_only=True: {self.model_name}"
                        )

                # Load tokenizer
                print(f"[*] Loading tokenizer from: {model_to_load}")

                # Special handling for Llama models (fix vocab_file Path bug)
                model_path_str = str(model_to_load)

                # For Llama models, force fast tokenizer to avoid SentencePiece Path bug
                if "llama" in self.model_name.lower():
                    print(
                        "[*] Forcing fast tokenizer for Llama model to avoid vocab_file Path bug"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        trust_remote_code=True,
                        local_files_only=True,
                        use_fast=True,  # Force fast tokenizer
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str, trust_remote_code=True, local_files_only=True
                    )

                # Ensure tokenizer has pad token set
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token:
                        tokenizer.pad_token = tokenizer.eos_token
                        print(f"[*] Set pad_token to eos_token: {tokenizer.eos_token}")
                    else:
                        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                        print("[*] Added [PAD] as pad_token")

                print("[*] Tokenizer loaded successfully")
                print(
                    f"[*] Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
                )
                print(
                    f"[*] EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
                )

                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[*] Device: {device}")

                # Set memory management environment variable
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

                # Clear GPU cache before loading
                if device == "cuda":
                    torch.cuda.empty_cache()
                    print("[*] Cleared CUDA cache")

                # Determine dtype - use bfloat16 for better stability on large models
                if device == "cuda":
                    # Check if bfloat16 is supported
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                        print("[*] Using bfloat16 (supported by GPU)")
                    else:
                        dtype = torch.float16
                        print("[*] Using float16 (bfloat16 not supported)")
                else:
                    dtype = torch.float32
                    print("[*] Using float32 (CPU mode)")

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
                        max_memory = {
                            i: f"{int(gpu_memory[i] * 0.90)}GB" for i in range(num_gpus)
                        }
                        print(f"[*] Multi-GPU max memory (no CPU): {max_memory}")
                else:
                    max_memory = None

                # Load model with appropriate settings for large models
                print("[*] Loading model weights...")

                # Suppress the "meta device" warning - it's expected with CPU offloading
                import warnings

                warnings.filterwarnings(
                    "ignore", message=".*parameters are on the meta device.*"
                )

                # Try to use Flash Attention 2 for faster inference (requires flash-attn package)
                attn_implementation = "eager"  # Default fallback
                try:
                    import flash_attn

                    # Check if GPU supports Flash Attention (compute capability >= 8.0 for Ampere+)
                    if device == "cuda" and hasattr(
                        torch.cuda, "get_device_capability"
                    ):
                        compute_capability = torch.cuda.get_device_capability(0)
                        if compute_capability[0] >= 8:  # Ampere (A100, A6000) or newer
                            attn_implementation = "flash_attention_2"
                            print(
                                f"[*] Using Flash Attention 2 (GPU compute capability: {compute_capability})"
                            )
                        else:
                            print(
                                f"[*] Flash Attention available but GPU too old (compute capability: {compute_capability})"
                            )
                            print("[*] Using eager attention (slower)")
                except ImportError:
                    print(
                        "[*] Flash Attention not installed, using eager attention (slower)"
                    )
                    print(
                        "[*] To enable Flash Attention: pip install flash-attn --no-build-isolation"
                    )

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
                    print(
                        "[*] Single GPU detected - loading ENTIRE model on GPU 0 (no offloading)"
                    )
                elif num_gpus > 1:
                    # Multi-GPU: Allow distribution across GPUs but prevent CPU offloading
                    model_kwargs["device_map"] = "auto"
                    if max_memory:
                        model_kwargs["max_memory"] = max_memory
                    print(
                        f"[*] Multi-GPU setup - distributing across {num_gpus} GPUs (no CPU offloading)"
                    )
                else:
                    model_kwargs["device_map"] = "auto"
                    print("[*] CPU mode - using auto device mapping")

                print(
                    "[DEBUG] About to call AutoModelForCausalLM.from_pretrained()..."
                )
                print(f"[DEBUG] Model: {model_to_load}")
                print(f"[DEBUG] Device map: {model_kwargs.get('device_map', 'N/A')}")
                print("[DEBUG] This may take 1-2 minutes for large models...")

                model = AutoModelForCausalLM.from_pretrained(
                    model_to_load, **model_kwargs
                )

                print("[DEBUG] AutoModelForCausalLM.from_pretrained() returned!")
                print("[*] Model loaded successfully!")
                print(f"[DEBUG] Model class: {type(model).__name__}")
                print(
                    f"[DEBUG] Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}"
                )

                # Clear cache again after loading
                if device == "cuda":
                    torch.cuda.empty_cache()
                    print("[*] Cleared CUDA cache after model loading")

                # Print memory usage
                if device == "cuda":
                    for i in range(num_gpus):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        total = torch.cuda.get_device_properties(i).total_memory / (
                            1024**3
                        )
                        print(
                            f"[*] GPU {i} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total"
                        )

                # Set generation config directly instead of loading from pretrained
                # This avoids potential network calls and hangs
                print("[DEBUG] Setting up generation config...")
                try:
                    # Just set the essential parameters directly
                    if hasattr(model, "generation_config"):
                        if model.generation_config.pad_token_id is None:
                            model.generation_config.pad_token_id = (
                                tokenizer.eos_token_id
                            )
                            print(
                                f"[*] Set model pad_token_id to eos_token_id: {tokenizer.eos_token_id}"
                            )

                    # Also ensure tokenizer has pad token
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                        print(
                            f"[*] Set tokenizer pad_token_id to eos_token_id: {tokenizer.eos_token_id}"
                        )

                    print("[DEBUG] Generation config setup complete")
                except Exception as e:
                    print(f"[WARNING] Could not set generation config: {e}")
                    # Ensure tokenizer has pad token as fallback
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id

                # Test inference with a small prompt to ensure model works
                # SKIP for models with 4+ GPUs - distributed models can hang during test
                # For single GPU, we should NOT have CPU offloading, so always test
                has_cpu_offload = False
                if hasattr(model, "hf_device_map"):
                    has_cpu_offload = "cpu" in str(model.hf_device_map.values())

                if num_gpus >= 4:
                    print(
                        f"[*] Skipping test inference (multi-GPU setup with {num_gpus} GPUs)"
                    )
                    print("[*] Model ready for inference")
                elif has_cpu_offload:
                    print(
                        "[*] WARNING: Model has CPU-offloaded layers despite single GPU setup!"
                    )
                    print(
                        "[*] Model ready for inference (will be slower due to CPU offloading)"
                    )
                else:
                    print("[*] Testing model with small inference...")
                    try:
                        print("[DEBUG] Creating test input...")
                        test_input = tokenizer(
                            "Hello", return_tensors="pt", padding=True
                        )

                        # Move input to same device as first model parameter
                        print("[DEBUG] Finding model device...")
                        device_0 = next(model.parameters()).device
                        print(f"[DEBUG] First parameter on device: {device_0}")

                        print("[DEBUG] Moving test input to device...")
                        test_input = {k: v.to(device_0) for k, v in test_input.items()}

                        # Run a tiny generation to verify it works
                        print("[DEBUG] Running test generation (max_new_tokens=5)...")
                        with torch.no_grad():
                            test_output = model.generate(
                                **test_input,
                                max_new_tokens=5,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        print("[*] Test inference successful!")
                        print(f"[DEBUG] Test output shape: {test_output.shape}")

                        # Clear cache after test
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            print("[DEBUG] Cleared cache after test")

                    except Exception as e:
                        print(f"[WARNING] Test inference failed: {e}")
                        print(
                            "[WARNING] Model may not work properly during experiment"
                        )
                        import traceback

                        traceback.print_exc()

                return {"tokenizer": tokenizer, "model": model, "device": device}
            except ImportError as e:
                print(f"[ERROR] HuggingFace dependencies not available: {e}")
                return None
            except Exception as e:
                print(f"[ERROR] Failed to load HuggingFace model: {e}")
                import traceback

                traceback.print_exc()
                return None
        elif self.provider == "mock":
            return "MOCK_CLIENT"
        return None

    def _initialize_vllm(self):
        """Initialize vLLM for optimized inference (faster than standard transformers)."""
        from vllm import LLM, SamplingParams

        print("[*] Loading model with vLLM (optimized inference)")
        print(f"[*] Model: {self.model_name}")

        # FORCE OFFLINE MODE
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Find model path
        hf_home = os.getenv("HF_HOME")
        model_to_load = self.model_name

        if hf_home:
            flat_dir = os.path.join(hf_home, self.model_name.replace("/", "--"))
            hub_dir = os.path.join(
                hf_home, "models--" + self.model_name.replace("/", "--")
            )

            if os.path.isdir(flat_dir):
                print(f"[*] Found model (flat --local-dir): {flat_dir}")
                model_to_load = flat_dir
            elif os.path.isdir(hub_dir):
                snapshots_dir = os.path.join(hub_dir, "snapshots")
                if os.path.isdir(snapshots_dir):
                    snapshots = sorted(os.listdir(snapshots_dir))
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshots[-1])
                        print(
                            f"[*] Found model (HF hub cache snapshot): {snapshot_path}"
                        )
                        model_to_load = snapshot_path
                    else:
                        print(
                            f"[*] Hub cache dir exists but has no snapshots: {hub_dir}"
                        )
                else:
                    print(
                        f"[*] Hub cache dir exists but no 'snapshots' subdir: {hub_dir}"
                    )
            else:
                print("[*] Model not found locally. Checked:")
                print(f"    flat:  {flat_dir}")
                print(f"    hub:   {hub_dir}")

        # Initialize vLLM
        # vLLM automatically uses all available GPUs and optimizations
        llm = LLM(
            model=model_to_load,
            trust_remote_code=True,
            download_dir=hf_home,
            dtype="bfloat16",  # Use bfloat16 for better performance
            max_model_len=4096,  # Adjust based on your needs
            gpu_memory_utilization=0.90,  # Use 90% of GPU memory
            tensor_parallel_size=os.getenv("CUDA_VISIBLE_DEVICES", "0").count(",")
            + 1,  # Auto-detect GPUs
        )

        # Scratchpad params — free reasoning, full token budget
        scratchpad_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95,
        )

        # Decision params base — greedy, short output, schema applied per-call
        decision_params_base = SamplingParams(
            temperature=0,
            max_tokens=2048,
        )

        print("[*] vLLM initialized successfully")

        return {
            "llm": llm,
            "sampling_params": scratchpad_params,  # kept for legacy _call_llm path
            "scratchpad_params": scratchpad_params,
            "decision_params_base": decision_params_base,
            "backend": "vllm",
        }

    def get_action_and_prediction(
        self, observation: Dict[str, Any], available_tools: List[str]
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

        # ── MOCK: use existing single-call path unchanged ─────────────────────
        if self.client == "MOCK_CLIENT":
            prompt = self._build_prompt(observation, available_tools)
            response = self._call_llm(prompt)
            action, prediction = self._parse_llm_response(response, available_tools)
            if prediction is None:
                prediction = {}
            prediction["scratchpad_raw"] = response
            return action, prediction

        # ── Inject system message on the very first step of each episode ──────
        # The system message carries the stable role/rules so they don't need to
        # be repeated in every user turn, saving tokens and giving a fixed anchor.
        if not self.messages:
            self.messages.append(
                {"role": "system", "content": self._build_system_message()}
            )

        # ── vLLM: two-phase structured path ───────────────────────────────────
        if isinstance(self.client, dict) and self.client.get("backend") == "vllm":
            return self._get_action_vllm(observation, available_tools)

        # ── Everything else (HF transformers, OpenAI, Anthropic) ─────────────
        prompt = self._build_prompt(observation, available_tools)

        # Append user turn, call LLM with full history, append assistant reply
        self.messages.append({"role": "user", "content": prompt})
        response = self._call_llm_with_history()
        self.messages.append({"role": "assistant", "content": response})

        # Trim history to stay within context window.
        # Always keep index 0 (system message) + last N user/assistant pairs.
        # For HF transformers, keep 5 turns (10 messages) — the 4096-token input cap
        # means longer histories get truncated from the front anyway, so keeping
        # fewer turns avoids paying the tokenization cost for tokens that get dropped.
        max_turns = 5  # keep last 5 user/assistant pairs = 10 messages
        if len(self.messages) > 1 + max_turns * 2:
            self.messages = [self.messages[0]] + self.messages[-(max_turns * 2) :]

        # Debug: print short responses in full
        if len(response) < 500:
            print(f"  [DEBUG] LLM Response: {response}")

        action, prediction = self._parse_llm_response(response, available_tools)
        if prediction is None:
            prediction = {}
        prediction["scratchpad_raw"] = response
        return action, prediction

    def _get_action_vllm(
        self, observation: Dict[str, Any], available_tools: List[str]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Two-phase vLLM inference with full conversation history:
          Phase 1 — free scratchpad reasoning sent as a multi-turn chat (with history)
          Phase 2 — constrained JSON decision (temperature=0, guided_json schema)

        History is maintained in self.messages (system msg at index 0, then
        alternating user/assistant turns). The system message is injected by
        get_action_and_prediction before this method is called.
        """
        from vllm import SamplingParams

        # ── Phase 1: scratchpad reasoning, history-aware ──────────────────────
        scratchpad_prompt = self._build_prompt(observation, available_tools)

        # Append this step's user turn to history, then call with full history
        self.messages.append({"role": "user", "content": scratchpad_prompt})
        scratchpad_raw = self._call_llm_vllm_chat(
            scratchpad_prompt,
            self.client["scratchpad_params"],
            messages=self.messages,  # pass full history
        )

        # Strip markdown fences before using scratchpad as Phase 2 context
        scratchpad = self._strip_markdown(scratchpad_raw)
        print(f"  [vllm] Scratchpad complete ({len(scratchpad)} chars)")

        # Append assistant reply to history (trim after to stay in context window)
        self.messages.append({"role": "assistant", "content": scratchpad_raw})
        max_turns = 10
        if len(self.messages) > 1 + max_turns * 2:
            self.messages = [self.messages[0]] + self.messages[-(max_turns * 2) :]

        # ── Phase 2: constrained JSON decision ────────────────────────────────
        schema = json.loads(json.dumps(DECISION_SCHEMA))
        schema["properties"]["tool"]["enum"] = available_tools

        decision_params = SamplingParams(
            temperature=0,
            max_tokens=300,
            guided_json=json.dumps(schema),
            guided_backend="outlines",
        )

        # Include real SKU/supplier IDs so the constrained decoder fills valid values
        sku_ids = observation.get(
            "sku_ids", list(observation.get("storage", {}).keys())
        )
        supplier_ids = observation.get("supplier_ids", [])

        decision_prompt = (
            "Based on your reasoning, output your single next action as JSON.\n\n"
            f"<reasoning>\n{scratchpad}\n</reasoning>\n\n"
            "Available tools:\n"
            + "\n".join(f"  - {t}" for t in available_tools)
            + f"\n\nValid SKU IDs: {sku_ids}"
            f"\nValid supplier IDs: {supplier_ids}"
            "\n\nJSON field guidance:"
            '\n  tool_order              → args: {"supplier_id": "<one of above>", "sku": "<one of above>", "quantity": <int>}'
            '\n  tool_ship_customer_order → args: {"customer_order_id": "CO<N>"}'
            "\n  check tools             → args: {}"
        )

        # Phase 2 is a single-turn call (no history — just the decision context)
        raw = self._call_llm_vllm_chat(decision_prompt, decision_params)
        print(f"  [vllm] Decision JSON: {raw}")

        decision = json.loads(raw)

        fake_response = (
            f"THOUGHTS:\n{scratchpad}\n"
            f"ACTION: {decision['tool']}\n"
            f"ARGS: {json.dumps(decision['args'])}\n"
            f"PREDICTION: {decision['prediction_text']}\n"
            f"SUCCESS: {'true' if decision['expected_success'] else 'false'}"
        )
        action, prediction = self._parse_llm_response(fake_response, available_tools)

        if prediction is None:
            prediction = {}
        prediction["scratchpad_raw"] = scratchpad_raw
        prediction["decision_raw"] = raw

        return action, prediction

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown code fences that some models emit despite being told not to."""
        import re

        lines = text.split("\n")
        cleaned = []
        in_fence = False
        for line in lines:
            # Match ``` with optional language tag: ```python, ```json, ``` etc.
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue  # drop the fence line itself
            if not in_fence:
                cleaned.append(line)
        # If we ended mid-fence (odd number of fences), toggle got out of sync —
        # fall back to regex strip of all fence markers instead.
        if in_fence:
            text = re.sub(r"```[\w]*\n?", "", text)
            text = re.sub(r"```", "", text)
            return text.strip()
        result = "\n".join(cleaned).strip()
        # If stripping removed everything meaningful, return original unchanged
        return result if len(result) > 20 else text

    def _call_llm_vllm_chat(
        self,
        prompt: str,
        sampling_params,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Call vLLM using the model's chat template.

        When `messages` is provided (Phase 1 / history-aware calls), the full
        conversation history is formatted and sent.  When omitted (Phase 2 /
        single-shot constrained calls), only the prompt is sent as a single user turn.

        apply_chat_template wraps the conversation as the model expects, preventing
        instruction-tuned models from echoing the THOUGHTS:/ACTION:/ARGS: format.
        """
        llm = self.client["llm"]
        tokenizer = llm.get_tokenizer()

        # Use provided history or fall back to a single-turn conversation
        chat_messages = (
            messages if messages is not None else [{"role": "user", "content": prompt}]
        )

        try:
            formatted = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,  # appends the assistant turn opener
            )
        except Exception:
            # Tokenizer has no chat template — fall back to raw prompt
            formatted = prompt

        outputs = llm.generate([formatted], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _call_llm_with_history(self) -> str:
        """
        Call the LLM using the full conversation history in self.messages.

        Gives the model memory of its prior actions and tool results so it does
        not restart reasoning from scratch on every step.
        """
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling OpenAI: {e}"

        elif self.provider == "anthropic":
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=self.messages,
                )
                return response.content[0].text
            except Exception as e:
                return f"Error calling Anthropic: {e}"

        elif self.provider == "huggingface":
            # Use apply_chat_template for proper chat formatting — the same approach
            # as _call_llm_vllm_chat.  Raw string concatenation ("User: ... \n\n
            # Assistant: ...") causes the model to continue the previous truncated
            # turn rather than starting a fresh assistant reply.
            tokenizer = self.client["tokenizer"]
            try:
                formatted = tokenizer.apply_chat_template(
                    self.messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Tokenizer has no chat template — fall back to plain last prompt
                formatted = self.messages[-1]["content"] if self.messages else ""
            return self._call_llm(formatted)

        # Fallback — should not be reached for supported providers
        return self._call_llm(self.messages[-1]["content"] if self.messages else "")

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
            # Cycle through a realistic customer-order fulfillment workflow so
            # the demo shows all phases: discover → stock check → ship → reorder.
            # Pattern (repeats every 5 steps):
            #   0 → check_inbox   (discover customer orders)
            #   1 → check_storage (verify stock levels)
            #   2 → ship          (fulfil oldest open customer order)
            #   3 → order         (replenish sku_0 from S1)
            #   4 → check_budget  (monitor finances)
            phase = self._mock_step % 5
            self._mock_step += 1

            # Derive the customer order ID to ship on this cycle
            # CO index increments by 1 each full cycle (every 5 steps)
            cycle = (self._mock_step - 1) // 5
            co_id = f"CO{cycle * 5 + 1}"  # CO1, CO6, CO11, ...

            mock_responses = [
                # Phase 0 — check inbox
                (
                    "THOUGHTS:\n"
                    "I need to check my inbox for new customer orders so I know what to fulfil.\n"
                    "ACTION: tool_check_inbox\n"
                    "ARGS: {}\n"
                    "PREDICTION: I will see any pending customer orders.\n"
                    "SUCCESS: true"
                ),
                # Phase 1 — check storage
                (
                    "THOUGHTS:\n"
                    "I need to verify my stock levels before committing to ship anything.\n"
                    "ACTION: tool_check_storage\n"
                    "ARGS: {}\n"
                    "PREDICTION: I will see current inventory per SKU.\n"
                    "SUCCESS: true"
                ),
                # Phase 2 — ship customer order
                (
                    f"THOUGHTS:\n"
                    f"I have customer order {co_id} to fulfil. I'll ship it now to earn revenue.\n"
                    f"ACTION: tool_ship_customer_order\n"
                    f'ARGS: {{"customer_order_id": "{co_id}"}}\n'
                    f"PREDICTION: The order will be shipped and revenue added to my budget.\n"
                    f"SUCCESS: true"
                ),
                # Phase 3 — replenish stock
                (
                    "THOUGHTS:\n"
                    "I should keep sku_0 stocked so I can fulfil future customer orders. "
                    "Ordering 10 units from S1.\n"
                    "ACTION: tool_order\n"
                    'ARGS: {"supplier_id": "S1", "sku": "sku_0", "quantity": 10}\n'
                    "PREDICTION: Stock will arrive within a few days.\n"
                    "SUCCESS: true"
                ),
                # Phase 4 — check budget
                (
                    "THOUGHTS:\n"
                    "I want to track my financial position after shipping and ordering.\n"
                    "ACTION: tool_check_budget\n"
                    "ARGS: {}\n"
                    "PREDICTION: I will see my current budget balance.\n"
                    "SUCCESS: true"
                ),
            ]
            return mock_responses[phase]

        # Check if using vLLM backend
        if isinstance(self.client, dict) and self.client.get("backend") == "vllm":
            try:
                llm = self.client["llm"]
                sampling_params = self.client["sampling_params"]

                print("[DEBUG] Running vLLM inference...")
                outputs = llm.generate([prompt], sampling_params)
                response = outputs[0].outputs[0].text
                print("[DEBUG] vLLM inference complete")
                return response.strip()
            except Exception as e:
                return f"Error calling vLLM: {e}"

        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling OpenAI: {e}"

        elif self.provider == "anthropic":
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                return f"Error calling Anthropic: {e}"

        elif self.provider == "huggingface":
            try:
                import time

                import torch

                print("[DEBUG] Starting HuggingFace inference...")
                start_time = time.time()

                tokenizer = self.client["tokenizer"]
                model = self.client["model"]
                device = self.client["device"]

                print(
                    f"[DEBUG] Retrieved tokenizer, model, device ({time.time() - start_time:.2f}s)"
                )

                # Tokenize input — cap at 4096 tokens.  History is included via
                # apply_chat_template so 4096 comfortably fits ~8 turns + system msg.
                # We do NOT use context_length (often 128k) as the cap because
                # tokenizing + attending over 8k+ tokens with HF is very slow.
                print(
                    f"[DEBUG] Tokenizing input (prompt length: {len(prompt)} chars)..."
                )
                tokenize_start = time.time()
                max_input_len = 4096
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_len,
                )
                print(
                    f"[DEBUG] Tokenization complete ({time.time() - tokenize_start:.2f}s, {inputs['input_ids'].shape[1]} tokens)"
                )

                # Move inputs to device
                print("[DEBUG] Moving inputs to device...")
                move_start = time.time()
                # Get the device of the first model parameter
                first_param_device = next(model.parameters()).device
                print(f"[DEBUG] First model parameter is on: {first_param_device}")
                inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
                print(
                    f"[DEBUG] Inputs moved to {first_param_device} ({time.time() - move_start:.2f}s)"
                )

                # Print memory before generation
                if device == "cuda":
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(
                            f"[DEBUG] GPU {i} before generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                        )

                # Generate response.
                # use_cache=True is critical for performance — without it the model
                # re-processes all input tokens for every generated token (O(n²)).
                # If the DynamicCache error resurfaces (transformers >= 4.36 issue),
                # we catch it and retry once with use_cache=False as a fallback.
                print(
                    f"[DEBUG] Starting model.generate() with max_new_tokens={min(self.max_tokens, 512)}..."
                )
                gen_start = time.time()

                def _generate(use_cache_flag: bool):
                    with torch.no_grad():
                        return model.generate(
                            **inputs,
                            max_new_tokens=min(self.max_tokens, 512),
                            temperature=self.temperature,
                            do_sample=True if self.temperature > 0 else False,
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id
                                else tokenizer.eos_token_id
                            ),
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=use_cache_flag,
                            num_beams=1,
                        )

                try:
                    outputs = _generate(use_cache_flag=True)
                except Exception as cache_err:
                    if (
                        "DynamicCache" in str(cache_err)
                        or "cache" in str(cache_err).lower()
                    ):
                        print(
                            f"[DEBUG] KV cache error ({cache_err}), retrying with use_cache=False"
                        )
                        outputs = _generate(use_cache_flag=False)
                    else:
                        raise

                gen_time = time.time() - gen_start
                print(
                    f"[DEBUG] Generation complete ({gen_time:.2f}s, {gen_time/60:.1f} min)"
                )

                # Print memory after generation
                if device == "cuda":
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        print(
                            f"[DEBUG] GPU {i} after generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                        )

                # Decode response (skip the input prompt)
                print("[DEBUG] Decoding response...")
                decode_start = time.time()
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                print(f"[DEBUG] Decoding complete ({time.time() - decode_start:.2f}s)")

                total_time = time.time() - start_time
                print(
                    f"[DEBUG] Total inference time: {total_time:.2f}s ({total_time/60:.1f} min)"
                )
                print(f"[DEBUG] Response length: {len(response)} chars")

                return response.strip()

            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                print(f"[ERROR] HuggingFace generation failed: {e}")
                print(f"[ERROR] Traceback: {error_details}")
                return f"Error calling HuggingFace model: {e}"

        return "Unknown provider"

    def _build_system_message(self) -> str:
        """
        Build the one-time system message injected at the start of every episode.

        Contains the stable role, objective, and rules that never change between
        steps — kept out of the per-step user prompt to save tokens and to give
        the model a stable anchor in its context window.
        """
        max_failures = self.config.get("demand", {}).get("max_failures", 25)
        expire_days = self.config.get("demand", {}).get("expire_after_days", 10)
        # max_steps may live under 'simulation.*' (phase overrides) or 'env.*' (base)
        sim_cfg = self.config.get("env", self.config.get("simulation", {}))
        sim_over = self.config.get("simulation", {})
        max_steps = sim_over.get("max_steps", sim_cfg.get("max_steps", 1000))

        return f"""You are an autonomous supply chain agent managing a warehouse.

OBJECTIVE: Earn revenue by fulfilling customer orders before they expire.

HOW REVENUE WORKS:
- Customers send orders to your inbox (type: customer_order). Each specifies a SKU, quantity, unit sale price, and due_day.
- To fill an order: (1) place a supplier order with tool_order to get stock, (2) once delivered, ship to the customer with tool_ship_customer_order.
- Revenue = unit_sale_price × quantity.
- Budget is deducted when supplier deliveries arrive (not when you place the order).

HOW TIME WORKS:
- You can take as many actions as you want within a single day.
- The calendar only advances when you call tool_end_day.
- You MUST call tool_end_day to trigger: deliveries arriving, new customer orders, and storage fees.
- If you never call tool_end_day, time stands still — no deliveries will arrive and no new demand will appear.

FAILURE CONDITIONS — the simulation ends if any of:
  1. Budget drops below -$100
  2. {max_failures} customer orders expire unfulfilled
  3. {max_steps} days pass

RULES:
- Customer orders EXPIRE after {expire_days} days from when they arrive — act promptly.
- Supplier orders have variable lead times (1-6 days) — order well before customer due dates.
- Storage fees accrue daily ($0.10/unit) — avoid excess stockpiling.
- You do NOT automatically see current state — use check tools to observe quantities and open orders.
  (SKU IDs and supplier IDs are always listed so you can form valid tool calls.)

TOOL SIGNATURES:
  tool_check_inbox    → {{}}   ← returns inbox messages + open_customer_orders list with IDs and urgency
  tool_check_storage  → {{}}
  tool_check_budget   → {{}}
  tool_quote          → {{"supplier_id": "S<N>", "sku": "sku_<N>"}}
  tool_order          → {{"supplier_id": "S<N>", "sku": "sku_<N>", "quantity": <int>}}
  tool_cancel_order   → {{"order_id": "ORD<N>"}}
  tool_ship_customer_order → {{"customer_order_id": "CO<N>"}}   ← get IDs from tool_check_inbox
  tool_end_day        → {{}}   ← REQUIRED to advance the calendar; call once when done for the day

Think through what you know and what the best next action is."""

    def _build_prompt(
        self, observation: Dict[str, Any], available_tools: List[str]
    ) -> str:
        """Build the per-step observation card sent as the user turn each step.

        Role, rules, and tool signatures live in _build_system_message() (sent once).
        This prompt contains only what changes step-to-step: the current observation,
        the blind-state display logic, and the required response format.

        BLIND-STATE DESIGN (intentional for the crash study):
        Quantities and order details are hidden unless the agent just checked them.
        SKU IDs and supplier IDs are ALWAYS shown — they are vocabulary, not state.
        """
        last_action_msg = str(observation.get("message", ""))

        # ── Blind state: reveal quantities only when the agent just checked them ──
        budget_display = "Unknown (use tool_check_budget)"
        storage_display = "Unknown (use tool_check_storage)"
        orders_display = "Unknown (use tool_check_inbox)"
        customer_orders_display = "Unknown (use tool_check_inbox)"

        msg_lower = last_action_msg.lower()
        if (
            "budget" in msg_lower
            or "balance" in msg_lower
            or "check_budget" in msg_lower
        ):
            budget_display = f"${observation.get('budget', 0):.2f}"

        if "storage" in msg_lower or "stock" in msg_lower or "inventory" in msg_lower:
            storage_display = str(observation.get("storage", {}))

        if "order" in msg_lower and "pending" in msg_lower:
            pending = observation.get("pending_orders", 0)
            count = len(pending) if isinstance(pending, list) else pending
            orders_display = f"{count} active supplier orders"

        if "customer_order" in msg_lower:
            customer_orders_display = (
                f"{observation.get('open_customer_orders', '?')} open customer orders"
            )

        # ── Always-visible metadata (needed to form valid tool calls) ────────────
        sku_ids = observation.get(
            "sku_ids", list(observation.get("storage", {}).keys())
        )
        supplier_ids = observation.get("supplier_ids", [])

        return f"""--- DAY {observation.get('day', '?')} ---

CURRENT STATE (what you can see right now):
  Day:                      {observation.get('day', 0)}
  Actions taken today:      {observation.get('action_count', '?')} total this episode
  Budget:                   {budget_display}
  Revenue earned so far:    ${observation.get('revenue', 0):.2f}
  Storage levels:           {storage_display}
  Pending supplier orders:  {orders_display}
  Open customer orders:     {customer_orders_display}
  Messages in inbox:        {observation.get('inbox_count', '?')}
  Customer orders shipped:  {observation.get('customer_orders_shipped', '?')}
  Customer orders failed:   {observation.get('customer_orders_failed', '?')}
  Last tool output: {observation.get('message', 'None (first step)')}

AVAILABLE SKUs (use these exact IDs when ordering):
  {sku_ids}

AVAILABLE SUPPLIERS (use these exact IDs when ordering):
  {supplier_ids}

AVAILABLE TOOLS:
{chr(10).join(f'  - {tool}' for tool in available_tools)}

You may call as many tools as you like before calling tool_end_day.
Call tool_end_day when you are done for this day — that advances the clock,
triggers deliveries, generates new demand, and applies storage fees.
Do NOT write Python code or use markdown code blocks.

THOUGHTS:
[your reasoning]

ACTION: <tool_name>
ARGS: <json args, e.g. {{}} or {{"supplier_id": "S1", "sku": "sku_0", "quantity": 10}}>
PREDICTION: <what you expect to happen>
SUCCESS: <true/false>
"""

    def _parse_llm_response(
        self, response: str, available_tools: List[str]
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

        lines = response.strip().split("\n")

        action_tool = None
        action_args = {}
        prediction_text = None
        expected_success = True

        # State machine for parsing
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if line.startswith("ACTION:"):
                try:
                    action_tool = line.split("ACTION:", 1)[1].strip()
                    # Remove any trailing comments or quotes
                    action_tool = (
                        action_tool.split("#")[0].strip().strip("'").strip('"')
                    )
                except IndexError:
                    pass
                current_section = "ACTION"
                continue
            elif line.startswith("ARGS:"):
                args_str = line.split("ARGS:", 1)[1].strip()
                try:
                    # simplistic fix for single quotes which some models use
                    # only replace if it looks like python dict
                    if "{" in args_str:
                        args_str = args_str.replace("'", '"')
                    action_args = json.loads(args_str)
                except Exception as e:
                    print(f"  [monitor] Failed to parse ARGS: {args_str} ({e})")
                current_section = "ARGS"
                continue
            elif line.startswith("PREDICTION:"):
                prediction_text = line.split("PREDICTION:", 1)[1].strip()
                current_section = "PREDICTION"
                continue
            elif line.startswith("SUCCESS:"):
                success_str = line.split("SUCCESS:", 1)[1].strip().lower()
                expected_success = success_str in ["true", "yes", "1"]
                current_section = "SUCCESS"
                continue
            elif line.startswith("THOUGHTS:"):
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
            action_tool = "tool_check_inbox"
            action_args = {}

        # Ensure args are dict
        if not isinstance(action_args, dict):
            action_args = {}

        # Validate args for tools that REQUIRE specific fields.
        # IMPORTANT: Do NOT silently redirect to a different tool — let the env
        # execute the action so it returns a {'success': False, 'error': ...} that
        # gets fed back into the agent's context via last_message.  Silent fallbacks
        # hide the failure from the model and prevent it from learning.
        if action_tool == "tool_order":
            missing = [
                k for k in ["supplier_id", "sku", "quantity"] if k not in action_args
            ]
            if missing:
                print(
                    f"  [monitor] tool_order missing args {missing}: {action_args} — sending to env anyway so agent sees the error"
                )
        elif action_tool == "tool_quote":
            missing = [k for k in ["supplier_id", "sku", "qty"] if k not in action_args]
            if missing:
                print(
                    f"  [monitor] tool_quote missing args {missing}: {action_args} — sending to env anyway so agent sees the error"
                )
        elif action_tool == "tool_cancel_order" and "order_id" not in action_args:
            print(
                f"  [monitor] tool_cancel_order missing order_id: {action_args} — sending to env anyway"
            )

        action = {"tool": action_tool, "args": action_args}

        # Build prediction card
        prediction = None
        if self.prediction_mode != "none":
            prediction = {
                "tool": action_tool,
                "args": action_args,
                "expected_success": expected_success,
                "prediction_text": prediction_text,
                # scratchpad_raw is attached in get_action_and_prediction
            }

        return action, prediction

    def reset(self):
        """Reset agent state."""
        self.messages = []
