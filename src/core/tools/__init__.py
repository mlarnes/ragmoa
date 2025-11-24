"""
Tools Module

Exports all available tools for the master agent.
"""

from src.core.tools.sub_agent_tool import invoke_sub_agent_tool
from src.core.tools.openai_tool import invoke_openai_sub_agent
from src.core.tools.google_gemini_tool import invoke_google_gemini_sub_agent
from src.core.tools.registry import ToolRegistry

# Auto-register all tools in the registry
ToolRegistry.register("invoke_sub_agent_tool", invoke_sub_agent_tool)
ToolRegistry.register("invoke_openai_sub_agent", invoke_openai_sub_agent)
ToolRegistry.register("invoke_google_gemini_sub_agent", invoke_google_gemini_sub_agent)

__all__ = [
    "invoke_sub_agent_tool",
    "invoke_openai_sub_agent",
    "invoke_google_gemini_sub_agent",
    "get_all_tools",
    "ToolRegistry",
]


def get_all_tools():
    """Returns a list of all available tools."""
    return ToolRegistry.get_all()

