"""
Sub-Agent Service

Service for invoking sub-agents with prompts and tools.
"""

import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from src.services.agentic.sub_agent_factory import get_sub_agent
from src.services.agentic.mcp_integration import get_mcp_tools

logger = logging.getLogger(__name__)


class SubAgentService:
    """
    Service for invoking sub-agents.
    
    Handles:
    - Creating sub-agent instances
    - Invoking sub-agents with prompts
    - Managing tools (MCP or default)
    """
    
    def __init__(
        self,
        provider: str,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        use_mcp_tools: bool = False,
        mcp_server_name: Optional[str] = None,
        custom_tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize sub-agent service.
        
        Args:
            provider: The LLM provider (openai, google, groq, etc.)
            temperature: Optional temperature override
            model_name: Optional model name override
            use_mcp_tools: Whether to use MCP tools
            mcp_server_name: Name of MCP server to use (if use_mcp_tools is True)
            custom_tools: Custom tools to bind to the sub-agent
        """
        self.provider = provider
        self.temperature = temperature
        self.model_name = model_name
        self.use_mcp_tools = use_mcp_tools
        self.mcp_server_name = mcp_server_name
        self.custom_tools = custom_tools or []
        
        # Tools to bind to the sub-agent
        tools_to_bind = self._prepare_tools()
        
        # Create the sub-agent
        self.sub_agent = get_sub_agent(
            provider=provider,
            temperature=temperature,
            model_name=model_name,
            tools=tools_to_bind,
        )
    
    def _prepare_tools(self) -> List[BaseTool]:
        """
        Prepare tools for the sub-agent.
        
        Priority:
        1. Custom tools (if provided)
        2. MCP tools (if use_mcp_tools is True)
        3. Default tools from provider (if supported)
        
        Returns:
            List of tools to bind to the sub-agent
        """
        tools = []
        
        # Add custom tools first
        if self.custom_tools:
            tools.extend(self.custom_tools)
            logger.info(f"Added {len(self.custom_tools)} custom tool(s) to sub-agent '{self.provider}'")
        
        # Add MCP tools if requested
        if self.use_mcp_tools:
            mcp_tools = get_mcp_tools(self.mcp_server_name)
            if mcp_tools:
                tools.extend(mcp_tools)
                logger.info(f"Added {len(mcp_tools)} MCP tool(s) to sub-agent '{self.provider}'")
        
        # Note: Default provider tools (like function calling) are handled
        # by the LLM itself, not through explicit tool binding
        
        return tools
    
    async def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context_messages: Optional[List[BaseMessage]] = None,
    ) -> AIMessage:
        """
        Invoke the sub-agent with a prompt.
        
        Args:
            prompt: The main prompt/query for the sub-agent
            system_prompt: Optional system prompt
            context_messages: Optional context messages for conversation history
            
        Returns:
            AIMessage from the sub-agent
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add context messages if provided
        if context_messages:
            messages.extend(context_messages)
        
        # Add the main prompt
        messages.append(HumanMessage(content=prompt))
        
        logger.info(f"Invoking sub-agent '{self.provider}' with prompt: {prompt[:100]}...")
        
        try:
            response = await self.sub_agent.ainvoke(messages)
            
            # Ensure we return an AIMessage
            if isinstance(response, AIMessage):
                return response
            elif isinstance(response, dict):
                # Extract AIMessage from dict
                if "messages" in response:
                    for msg in reversed(response["messages"]):
                        if isinstance(msg, AIMessage):
                            return msg
                return AIMessage(content=str(response.get("content", response)))
            elif isinstance(response, list):
                # Extract AIMessage from list
                for msg in reversed(response):
                    if isinstance(msg, AIMessage):
                        return msg
                return AIMessage(content=str(response))
            else:
                return AIMessage(content=str(response))
                
        except Exception as e:
            logger.error(f"Error invoking sub-agent '{self.provider}': {e}", exc_info=True)
            return AIMessage(content=f"Error: {str(e)}")
    
    def invoke_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context_messages: Optional[List[BaseMessage]] = None,
    ) -> AIMessage:
        """
        Synchronous version of invoke.
        
        Args:
            prompt: The main prompt/query for the sub-agent
            system_prompt: Optional system prompt
            context_messages: Optional context messages for conversation history
            
        Returns:
            AIMessage from the sub-agent
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        if context_messages:
            messages.extend(context_messages)
        
        messages.append(HumanMessage(content=prompt))
        
        logger.info(f"Invoking sub-agent '{self.provider}' (sync) with prompt: {prompt[:100]}...")
        
        try:
            response = self.sub_agent.invoke(messages)
            
            if isinstance(response, AIMessage):
                return response
            elif isinstance(response, dict):
                if "messages" in response:
                    for msg in reversed(response["messages"]):
                        if isinstance(msg, AIMessage):
                            return msg
                return AIMessage(content=str(response.get("content", response)))
            elif isinstance(response, list):
                for msg in reversed(response):
                    if isinstance(msg, AIMessage):
                        return msg
                return AIMessage(content=str(response))
            else:
                return AIMessage(content=str(response))
                
        except Exception as e:
            logger.error(f"Error invoking sub-agent '{self.provider}': {e}", exc_info=True)
            return AIMessage(content=f"Error: {str(e)}")

