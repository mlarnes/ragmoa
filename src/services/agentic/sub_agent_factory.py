"""
Sub-Agent Factory

Factory for creating sub-agent instances (OpenAI, Gemini, etc.)
"""

import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from src.services.llm import get_llm
from config.settings import settings

logger = logging.getLogger(__name__)


class SubAgentFactory:
    """
    Factory for creating sub-agent instances.
    
    Each sub-agent is essentially an LLM with optional tools bound to it.
    """
    
    @staticmethod
    def create_sub_agent(
        provider: str,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLanguageModel:
        """
        Create a sub-agent instance.
        
        Args:
            provider: The LLM provider (openai, google, groq, etc.)
            temperature: Optional temperature override
            model_name: Optional model name override
            tools: Optional list of tools to bind to the sub-agent
            
        Returns:
            A configured LLM instance (sub-agent) with optional tools bound
        """
        # Get the base LLM
        sub_agent_llm = get_llm(
            temperature=temperature or 0.3,
            model_provider_override=provider,
            model_name_override=model_name,
        )
        
        # Bind tools if provided
        if tools:
            sub_agent_llm = sub_agent_llm.bind_tools(tools)
            logger.info(f"Created sub-agent '{provider}' with {len(tools)} tool(s)")
        else:
            logger.info(f"Created sub-agent '{provider}' without tools")
        
        return sub_agent_llm
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Get list of available sub-agent providers.
        
        Returns:
            List of provider names
        """
        return ["openai", "google", "groq", "ollama", "huggingface_api"]


def get_sub_agent(
    provider: str,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None,
) -> BaseLanguageModel:
    """
    Convenience function to create a sub-agent.
    
    Args:
        provider: The LLM provider (openai, google, groq, etc.)
        temperature: Optional temperature override
        model_name: Optional model name override
        tools: Optional list of tools to bind to the sub-agent
        
    Returns:
        A configured LLM instance (sub-agent) with optional tools bound
    """
    return SubAgentFactory.create_sub_agent(
        provider=provider,
        temperature=temperature,
        model_name=model_name,
        tools=tools,
    )

