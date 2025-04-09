"""
MCP Registry for managing all MCPs and Analytical Techniques.
This module provides the MCPRegistry singleton class for registering and retrieving MCPs and techniques.
"""

import logging
from typing import Dict, Any, Optional, List
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPRegistry:
    """
    Registry for all MCPs and Analytical Techniques.
    
    This singleton class provides a central registry for all MCPs and techniques,
    allowing them to be registered and retrieved by name.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MCPRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.mcps = {}
        self.techniques = {}
        self._initialized = True
        logger.info("Initialized MCP Registry")
    
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of the registry.
        
        Returns:
            MCPRegistry: The singleton instance
        """
        if cls._instance is None:
            return cls()
        return cls._instance
    
    def register_mcp(self, name: str, mcp) -> None:
        """
        Register an MCP with the registry.
        
        Args:
            name: Name of the MCP
            mcp: The MCP instance
        """
        if name in self.mcps:
            logger.warning(f"Overwriting existing MCP: {name}")
        
        self.mcps[name] = mcp
        logger.info(f"Registered MCP: {name}")
    
    def register_technique(self, name: str, technique) -> None:
        """
        Register an Analytical Technique with the registry.
        
        Args:
            name: Name of the technique
            technique: The technique instance
        """
        if name in self.techniques:
            logger.warning(f"Overwriting existing technique: {name}")
        
        self.techniques[name] = technique
        logger.info(f"Registered technique: {name}")
    
    def get_mcp(self, name: str):
        """
        Get an MCP by name.
        
        Args:
            name: Name of the MCP
            
        Returns:
            The MCP instance, or None if not found
        """
        if name not in self.mcps:
            logger.warning(f"MCP not found: {name}")
            return None
        
        return self.mcps[name]
    
    def get_technique(self, name: str):
        """
        Get an Analytical Technique by name.
        
        Args:
            name: Name of the technique
            
        Returns:
            The technique instance, or None if not found
        """
        if name not in self.techniques:
            logger.warning(f"Technique not found: {name}")
            return None
        
        return self.techniques[name]
    
    def get_all_mcps(self) -> Dict[str, Any]:
        """
        Get all registered MCPs.
        
        Returns:
            Dictionary of all MCPs, keyed by name
        """
        return self.mcps.copy()
    
    def get_all_techniques(self) -> Dict[str, Any]:
        """
        Get all registered Analytical Techniques.
        
        Returns:
            Dictionary of all techniques, keyed by name
        """
        return self.techniques.copy()
    
    def get_mcps_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get all MCPs in a specific category.
        
        Args:
            category: Category of MCPs to retrieve
            
        Returns:
            Dictionary of MCPs in the category, keyed by name
        """
        result = {}
        for name, mcp in self.mcps.items():
            if hasattr(mcp, "category") and mcp.category == category:
                result[name] = mcp
        return result
    
    def get_techniques_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get all Analytical Techniques in a specific category.
        
        Args:
            category: Category of techniques to retrieve
            
        Returns:
            Dictionary of techniques in the category, keyed by name
        """
        result = {}
        for name, technique in self.techniques.items():
            if hasattr(technique, "category") and technique.category == category:
                result[name] = technique
        return result
    
    def clear(self) -> None:
        """
        Clear all registered MCPs and techniques.
        """
        self.mcps = {}
        self.techniques = {}
        logger.info("Cleared MCP Registry")
