"""
Sub-Agent Tool

Generic tool for invoking sub-agents (OpenAI, Gemini, etc.)
"""

import logging
from typing import Optional, Dict, Any
import json

from langchain_core.tools import tool
from src.services.agentic.sub_agent_service import SubAgentService

logger = logging.getLogger(__name__)


@tool
def invoke_sub_agent_tool(
    provider: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    use_mcp_tools: bool = False,
    mcp_server_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invoke a sub-agent (OpenAI, Gemini, Groq, etc.) to process a query or task.
    
    This tool allows the master agent to delegate tasks to specialized sub-agents.
    Each sub-agent can have its own tools (via MCP or default provider tools).
    
    Args:
        provider: The sub-agent provider to use. Options: 'openai', 'google', 'groq', 'ollama', 'huggingface_api'
        prompt: The main prompt/query to send to the sub-agent
        system_prompt: Optional system prompt to guide the sub-agent's behavior
        temperature: Optional temperature override for the sub-agent (default: 0.3)
        model_name: Optional model name override (e.g., 'gpt-4', 'gemini-pro')
        use_mcp_tools: Whether to enable MCP tools for this sub-agent
        mcp_server_name: Optional name of the MCP server to use (if use_mcp_tools is True)
    
    Returns:
        A dictionary containing:
        - 'content': The response content from the sub-agent
        - 'tool_calls': Any tool calls made by the sub-agent (if applicable)
        - 'provider': The provider used
        - 'error': Error message if invocation failed
    """
    logger.info(f"Invoking sub-agent tool: provider={provider}, prompt_length={len(prompt)}")
    
    try:
        # Create sub-agent service
        sub_agent_service = SubAgentService(
            provider=provider,
            temperature=temperature,
            model_name=model_name,
            use_mcp_tools=use_mcp_tools,
            mcp_server_name=mcp_server_name,
        )
        
        # Invoke the sub-agent (synchronous for tool execution)
        response = sub_agent_service.invoke_sync(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        
        # Extract tool calls if present
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
            "provider": provider,
        }
        
        logger.info(f"Sub-agent '{provider}' responded successfully (content_length={len(result['content'])})")
        return result
        
    except Exception as e:
        logger.error(f"Error invoking sub-agent '{provider}': {e}", exc_info=True)
        return {
            "content": "",
            "tool_calls": [],
            "provider": provider,
            "error": str(e),
        }

