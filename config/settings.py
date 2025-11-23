"""
Configuration Module

This module defines the central configuration for the React-Gated MOA system using Pydantic settings.
It handles:
- Environment variable loading
- Default configuration values
- Configuration validation
- Provider-specific settings (LLM, Embeddings, etc.)

Configuration is loaded from:
1. Environment variables
2. .env file
3. Default values (as fallback)
"""

from typing import List, Optional
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """
    Centralized application settings.
    
    Settings are loaded from environment variables and/or a .env file.
    All settings can be overridden by environment variables using the same name.
    """
    
    # --- General Project Settings ---
    PROJECT_NAME: str = "ReAct-Gated Mixture-of-Agents"
    DEBUG: bool = False
    PYTHON_ENV: str = "development"  # development, staging, production

    # --- API Keys & Authentication ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

    # --- SQLite Configuration (for LangGraph checkpoints) ---
    SQLITE_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite"

    # --- LLM Provider Configuration ---
    DEFAULT_LLM_MODEL_PROVIDER: str = "groq"  # Groq for unlimited/free tier usage
    DEFAULT_OPENAI_GENERATIVE_MODEL: str = "gpt-4"
    HUGGINGFACE_REPO_ID: Optional[str] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"
    OLLAMA_GENERATIVE_MODEL_NAME: Optional[str] = "mistral"
    GROQ_MODEL_NAME: Optional[str] = "llama-3.3-70b-versatile"  # Default Groq model
    GOOGLE_GEMINI_MODEL_NAME: Optional[str] = "gemini-pro"  # Default Gemini model

    # --- Agentic Workflow Configuration ---
    # Summary and Memory Management
    SUMMARY_THRESHOLD: int = 15  # Number of messages before triggering summarization (more frequent = better memory management)
    MESSAGES_TO_KEEP_AFTER_SUMMARY: int = 3  # Number of recent messages to keep after summarization
    
    # Workflow Safety
    MAX_ITERATIONS: int = 100  # Maximum iterations to prevent infinite loops (higher for complex workflows)
    
    # LLM Temperature Settings
    AGENT_TEMPERATURE: float = 0.3  # Master agent: balanced creativity and accuracy
    SUMMARY_LLM_TEMPERATURE: float = 0.1  # Summary agent: low temperature for factual consistency
    SUB_AGENT_TEMPERATURE: float = 0.3  # Default temperature for sub-agents (can be overridden per invocation)
    
    # --- Sub-Agent Configuration ---
    # Available sub-agents: openai, google, groq, ollama, huggingface_api
    ENABLED_SUB_AGENTS: List[str] = ["openai", "google", "groq"]  # List of enabled sub-agent providers
    DEFAULT_SUB_AGENT_PROVIDER: str = "openai"  # Default sub-agent to use when not specified
    
    # MCP (Model Context Protocol) Configuration
    MCP_ENABLED: bool = False  # Enable MCP tool integration for sub-agents
    MCP_SERVER_NAME: Optional[str] = None  # Default MCP server name (if None, uses all available)
    
    # --- Data Directory ---
    DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"

    # --- Evaluation Configuration ---
    EVALUATION_DATASET_PATH: Optional[str] = str(DATA_DIR / "evaluation/synthesis_eval_dataset.json")

    # --- API Configuration ---
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]  # Override in production with specific origins

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create global settings instance
settings = Settings()

if __name__ == "__main__":
    """Print current configuration when run directly."""
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")

    print("\n--- Generative LLM Configuration ---")
    print(f"Default Generative LLM Provider: {settings.DEFAULT_LLM_MODEL_PROVIDER}")
    print(f"  OpenAI Model: {settings.DEFAULT_OPENAI_GENERATIVE_MODEL}")
    print(f"  HuggingFace Repo ID: {settings.HUGGINGFACE_REPO_ID}")
    print(f"  Ollama Model: {settings.OLLAMA_GENERATIVE_MODEL_NAME}")
    print(f"  Ollama Base URL: {settings.OLLAMA_BASE_URL}")
    print(f"  Groq Model: {settings.GROQ_MODEL_NAME}")
    print(f"  Google Gemini Model: {settings.GOOGLE_GEMINI_MODEL_NAME}")

    print("\n--- API Keys (Presence) ---")
    print(f"OpenAI API Key: {'✓' if settings.OPENAI_API_KEY else '✗'}")
    print(f"HuggingFace API Key: {'✓' if settings.HUGGINGFACE_API_KEY else '✗'}")
    print(f"Anthropic API Key: {'✓' if settings.ANTHROPIC_API_KEY else '✗'}")
    print(f"Groq API Key: {'✓' if settings.GROQ_API_KEY else '✗'}")
    print(f"Google API Key: {'✓' if settings.GOOGLE_API_KEY else '✗'}")
    print(f"Tavily API Key: {'✓' if settings.TAVILY_API_KEY else '✗'}")
    print(f"Weights & Biases API Key: {'✓' if settings.WANDB_API_KEY else '✗'}")

    print("\n--- SQLite Configuration (Checkpoints) ---")
    print(f"SQLite DB Path: {settings.SQLITE_DB_PATH}")

    print("\n--- Data & Paths ---")
    print(f"Data Directory: {settings.DATA_DIR}")
    
    print("\n--- Agentic Workflow Configuration ---")
    print(f"Summary Threshold: {settings.SUMMARY_THRESHOLD} messages")
    print(f"Messages to Keep After Summary: {settings.MESSAGES_TO_KEEP_AFTER_SUMMARY}")
    print(f"Max Iterations: {settings.MAX_ITERATIONS}")
    print(f"Master Agent Temperature: {settings.AGENT_TEMPERATURE}")
    print(f"Summary LLM Temperature: {settings.SUMMARY_LLM_TEMPERATURE}")
    print(f"Sub-Agent Default Temperature: {settings.SUB_AGENT_TEMPERATURE}")
    
    print("\n--- Sub-Agent Configuration ---")
    print(f"Enabled Sub-Agents: {', '.join(settings.ENABLED_SUB_AGENTS)}")
    print(f"Default Sub-Agent Provider: {settings.DEFAULT_SUB_AGENT_PROVIDER}")
    print(f"MCP Enabled: {settings.MCP_ENABLED}")
    if settings.MCP_SERVER_NAME:
        print(f"MCP Server Name: {settings.MCP_SERVER_NAME}")