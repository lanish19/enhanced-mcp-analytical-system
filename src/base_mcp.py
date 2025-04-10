"""
Base MCP class for all MCPs in the system.
This module provides the BaseMCP abstract class that all MCPs must inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseMCP(ABC):
    """
    Abstract base class for all MCPs in the system.
    
    This class defines the common interface and functionality that all MCPs must implement.
    """
    
    def __init__(self, name: str, description: str, version: str):
        """
        Initialize the BaseMCP.
        
        Args:
            name: Name of the MCP
            description: Description of the MCP's functionality
            version: Version of the MCP
        """
        self.name = name
        self.description = description
        self.version = version
        logger.info(f"Initialized {name} (v{version})")
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the MCP.
        
        Returns:
            Dictionary with MCP information
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version
        }
    
    def __str__(self) -> str:
        """
        Get string representation of the MCP.
        
        Returns:
            String representation
        """
        return f"{self.name} (v{self.version}): {self.description}"
