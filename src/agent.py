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
        # For now, use a simple heuristic agent
        # TODO: Replace with actual LLM calls
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
                return f"Error: {str(e)}"
        
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
                return f"Error: {str(e)}"
        
        return "Unknown provider"
    
    def reset(self):
        """Reset agent state."""
        self.messages = []
