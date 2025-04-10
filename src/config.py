"""
Configuration Management for the MCP Analytical System.
This module provides configuration management using Pydantic models.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    default_provider: str = Field(default="groq", description="Default LLM provider to use")
    default_model: str = Field(default="llama4-scout-17b-16e", description="Default model to use")
    temperature: float = Field(default=0.7, description="Default temperature for LLM calls")
    max_tokens: int = Field(default=4096, description="Default max tokens for LLM calls")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for LLM calls")
    timeout_seconds: int = Field(default=30, description="Timeout in seconds for LLM calls")

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

class ResearchConfig(BaseModel):
    """Configuration for research services."""
    perplexity_api_key: Optional[str] = Field(default=None, description="Perplexity API key")
    brave_api_key: Optional[str] = Field(default=None, description="Brave Search API key")
    academic_api_key: Optional[str] = Field(default=None, description="Academic API key")
    default_depth: str = Field(default="standard", description="Default research depth")
    cache_results: bool = Field(default=True, description="Whether to cache research results")
    max_sources: int = Field(default=10, description="Maximum number of sources to return")

    @validator('default_depth')
    def validate_depth(cls, v):
        if v not in ["quick", "standard", "deep"]:
            raise ValueError("Research depth must be one of: quick, standard, deep")
        return v

class WorkflowConfig(BaseModel):
    """Configuration for workflow orchestration."""
    max_retries: int = Field(default=3, description="Maximum number of retries for failed techniques")
    adaptive_workflow: bool = Field(default=True, description="Whether to enable adaptive workflows")
    default_strategy: str = Field(default="auto", description="Default strategy type (auto, predictive, causal, evaluative)")
    enable_fallbacks: bool = Field(default=True, description="Whether to enable fallback techniques")
    execution_timeout: int = Field(default=300, description="Timeout in seconds for workflow execution")

class StorageConfig(BaseModel):
    """Configuration for storage services."""
    redis_url: Optional[str] = Field(default=None, description="Redis URL for context storage")
    enable_persistence: bool = Field(default=False, description="Whether to enable persistence")
    ttl_seconds: int = Field(default=86400, description="Time-to-live in seconds for stored data")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    log_to_file: bool = Field(default=False, description="Whether to log to file")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    @validator('level')
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v

class SystemConfig(BaseModel):
    """Main configuration for the MCP Analytical System."""
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    research: ResearchConfig = Field(default_factory=ResearchConfig, description="Research configuration")
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Workflow configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    debug_mode: bool = Field(default=False, description="Whether to enable debug mode")
    environment: str = Field(default="development", description="Environment (development, testing, production)")

    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ["development", "testing", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {', '.join(valid_envs)}")
        return v

def load_config_from_env() -> SystemConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        SystemConfig instance with values from environment variables
    """
    # Load LLM config
    llm_config = LLMConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        default_provider=os.environ.get("DEFAULT_LLM_PROVIDER", "groq"),
        default_model=os.environ.get("DEFAULT_LLM_MODEL", "llama4-scout-17b-16e"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
        retry_attempts=int(os.environ.get("LLM_RETRY_ATTEMPTS", "3")),
        timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
    )
    
    # Load Research config
    research_config = ResearchConfig(
        perplexity_api_key=os.environ.get("PERPLEXITY_API_KEY"),
        brave_api_key=os.environ.get("BRAVE_API_KEY"),
        academic_api_key=os.environ.get("ACADEMIC_API_KEY"),
        default_depth=os.environ.get("DEFAULT_RESEARCH_DEPTH", "standard"),
        cache_results=os.environ.get("CACHE_RESEARCH_RESULTS", "True").lower() == "true",
        max_sources=int(os.environ.get("MAX_RESEARCH_SOURCES", "10"))
    )
    
    # Load Workflow config
    workflow_config = WorkflowConfig(
        max_retries=int(os.environ.get("MAX_TECHNIQUE_RETRIES", "3")),
        adaptive_workflow=os.environ.get("ENABLE_ADAPTIVE_WORKFLOW", "True").lower() == "true",
        default_strategy=os.environ.get("DEFAULT_STRATEGY", "auto"),
        enable_fallbacks=os.environ.get("ENABLE_FALLBACKS", "True").lower() == "true",
        execution_timeout=int(os.environ.get("WORKFLOW_EXECUTION_TIMEOUT", "300"))
    )
    
    # Load Storage config
    storage_config = StorageConfig(
        redis_url=os.environ.get("REDIS_URL"),
        enable_persistence=os.environ.get("ENABLE_PERSISTENCE", "False").lower() == "true",
        ttl_seconds=int(os.environ.get("STORAGE_TTL_SECONDS", "86400"))
    )
    
    # Load Logging config
    logging_config = LoggingConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format=os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        log_to_file=os.environ.get("LOG_TO_FILE", "False").lower() == "true",
        log_file=os.environ.get("LOG_FILE")
    )
    
    # Create and return SystemConfig
    return SystemConfig(
        llm=llm_config,
        research=research_config,
        workflow=workflow_config,
        storage=storage_config,
        logging=logging_config,
        debug_mode=os.environ.get("DEBUG_MODE", "False").lower() == "true",
        environment=os.environ.get("ENVIRONMENT", "development")
    )

# Global config instance
_config = None

def get_config() -> SystemConfig:
    """
    Get the system configuration.
    
    Returns:
        SystemConfig instance
    """
    global _config
    if _config is None:
        try:
            _config = load_config_from_env()
            logger.info(f"Loaded configuration for environment: {_config.environment}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            _config = SystemConfig()
    
    return _config

def configure_logging():
    """Configure logging based on the system configuration."""
    config = get_config()
    
    # Set log level
    log_level = getattr(logging, config.logging.level)
    logging.getLogger().setLevel(log_level)
    
    # Set log format
    formatter = logging.Formatter(config.logging.format)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    
    # Add file handler if enabled
    if config.logging.log_to_file and config.logging.log_file:
        file_handler = logging.FileHandler(config.logging.log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging configured with level: {config.logging.level}")
