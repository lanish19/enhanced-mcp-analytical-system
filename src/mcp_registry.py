"""
MCP Registry for managing and accessing MCPs.
This module provides the MCPRegistry class for registering and retrieving MCPs.
"""

import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPRegistry:
    """
    Registry for managing and accessing MCPs.
    
    This class provides functionality for:
    1. Registering MCPs by name
    2. Retrieving MCPs by name
    3. Listing all available MCPs
    4. Checking if an MCP exists
    """
    
    def __init__(self):
        """Initialize the MCP registry."""
        self.mcps = {}
        logger.info("Initialized MCP Registry")
    
    def register_mcp(self, name: str, mcp: Any) -> None:
        """
        Register an MCP with the registry.
        
        Args:
            name: Name of the MCP
            mcp: MCP instance
        """
        if name in self.mcps:
            logger.warning(f"Overwriting existing MCP: {name}")
        
        self.mcps[name] = mcp
        logger.info(f"Registered MCP: {name}")
    
    def get_mcp(self, name: str) -> Optional[Any]:
        """
        Get an MCP by name.
        
        Args:
            name: Name of the MCP to retrieve
            
        Returns:
            MCP instance if found, None otherwise
        """
        if name not in self.mcps:
            logger.warning(f"MCP not found: {name}")
            return None
        
        return self.mcps[name]
    
    def list_mcps(self) -> Dict[str, Any]:
        """
        List all registered MCPs.
        
        Returns:
            Dictionary of MCP names to MCP instances
        """
        return self.mcps
    
    def has_mcp(self, name: str) -> bool:
        """
        Check if an MCP exists in the registry.
        
        Args:
            name: Name of the MCP to check
            
        Returns:
            True if the MCP exists, False otherwise
        """
        return name in self.mcps
