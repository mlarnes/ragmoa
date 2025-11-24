"""
OpenAI Sub-Agent Tool

Specialized tool for invoking OpenAI sub-agents.
"""

import logging
from typing import Optional, Dict, Any

from langchain_core.tools import tool
from src.services.agentic.sub_agent_service import SubAgentService
from config.settings import settings

logger = logging.getLogger(__name__)


@tool
def invoke_openai_sub_agent(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    use_mcp_tools: bool = False,
    mcp_server_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invoke an OpenAI sub-agent (GPT-4, GPT-3.5, etc.) to process a query or task.
    
    OpenAI sub-agents support function calling natively, and can also use MCP tools
    if provided.
    
    Args:
        prompt: The main prompt/query to send to the OpenAI sub-agent
        system_prompt: Optional system prompt to guide the sub-agent's behavior
        temperature: Optional temperature override (default: 0.3)
        model_name: Optional model name (default: from settings, typically 'gpt-4')
        use_mcp_tools: Whether to enable MCP tools for this sub-agent
        mcp_server_name: Optional name of the MCP server to use (if use_mcp_tools is True)
    
    Returns:
        A dictionary containing:
        - 'content': The response content from the OpenAI sub-agent
        - 'tool_calls': Any function calls made by the sub-agent (if applicable)
        - 'provider': 'openai'
        - 'error': Error message if invocation failed
    """
    logger.info(f"Invoking OpenAI sub-agent: prompt_length={len(prompt)}, model={model_name or settings.DEFAULT_OPENAI_GENERATIVE_MODEL}")
    
    try:
        sub_agent_service = SubAgentService(
            provider="openai",
            temperature=temperature,
            model_name=model_name or settings.DEFAULT_OPENAI_GENERATIVE_MODEL,
            use_mcp_tools=use_mcp_tools,
            mcp_server_name=mcp_server_name,
        )
        
        response = sub_agent_service.invoke_sync(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = [
                {
                    "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                    "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                    "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                }
                for tc in response.tool_calls
            ]
        
        result = {
            "content": response.content if hasattr(response, 'content') else str(response),
            "tool_calls": tool_calls,
            "provider": "openai",
        }
        
        logger.info(f"OpenAI sub-agent responded successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error invoking OpenAI sub-agent: {e}", exc_info=True)
        return {
            "content": "",
            "tool_calls": [],
            "provider": "openai",
            "error": str(e),
        }

