"""
Base MCP class for all Modular Cognitive Processors.
This module provides the BaseMCP abstract class that all MCPs must inherit from.
"""

import logging
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseMCP(ABC):
    """
    Base class for all Modular Cognitive Processors (MCPs).
    
    This abstract class defines the interface that all MCPs must implement.
    It provides common functionality for logging, error handling, and validation.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize the MCP.
        
        Args:
            name: Name of the MCP
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"mcp.{name}")
        self.logger.info(f"Initialized {name} MCP")
    
    @abstractmethod
    def process(self, context: Dict) -> Dict:
        """
        Process a request with the given context.
        
        Args:
            context: Dictionary containing request parameters
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def validate_output(self, output: Dict) -> bool:
        """
        Validate the output of the MCP.
        
        Args:
            output: Dictionary containing MCP output
            
        Returns:
            True if output is valid, False otherwise
        """
        # Basic validation - ensure output is a dictionary
        if not isinstance(output, dict):
            self.logger.error(f"Invalid output type: {type(output)}, expected dict")
            return False
        
        # Check for error key
        if "error" in output:
            self.logger.warning(f"Output contains error: {output['error']}")
            return False
        
        return True
    
    def handle_error(self, error: Exception, context: Dict) -> Dict:
        """
        Handle an error that occurred during processing.
        
        Args:
            error: The exception that occurred
            context: The context that was being processed
            
        Returns:
            Dictionary containing error information
        """
        self.logger.error(f"Error in {self.name}: {error}")
        
        return {
            "error": str(error),
            "error_type": type(error).__name__,
            "mcp": self.name,
            "timestamp": time.time()
        }
    
    def log_operation(self, operation: str, details: Dict) -> None:
        """
        Log an operation performed by the MCP.
        
        Args:
            operation: Name of the operation
            details: Dictionary containing operation details
        """
        self.logger.info(f"Operation '{operation}' in {self.name}: {details}")
