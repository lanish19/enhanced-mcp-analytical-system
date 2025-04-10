"""
Analytical Technique base class for all analytical techniques.
This module provides the AnalyticalTechnique abstract class that all techniques must inherit from.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from src.analysis_context import AnalysisContext
from src.mcp_registry import MCPRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalyticalTechnique(ABC):
    """
    Base class for all Analytical Techniques.
    
    This abstract class defines the interface that all techniques must implement.
    It provides common functionality for validation, error handling, and metadata.
    """
    
    def __init__(self, name: str, description: str, required_mcps: List[str] = None,
                 compatible_techniques: List[str] = None, incompatible_techniques: List[str] = None):
        """
        Initialize the technique.
        
        Args:
            name: Name of the technique
            description: Description of the technique
            required_mcps: List of MCPs required by this technique
            compatible_techniques: List of techniques that work well with this one
            incompatible_techniques: List of techniques that conflict with this one
        """
        self.name = name
        self.description = description
        self.required_mcps = required_mcps or []
        self.compatible_techniques = compatible_techniques or []
        self.incompatible_techniques = incompatible_techniques or []
        self.logger = logging.getLogger(f"technique.{name}")
        self.logger.info(f"Initialized {name} technique")
    
    @abstractmethod
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        pass
    
    def validate_parameters(self, parameters: Dict) -> bool:
        """
        Validate the parameters for the technique.
        
        Args:
            parameters: Technique parameters
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Basic validation - ensure parameters is a dictionary
        if parameters is None:
            parameters = {}
        
        if not isinstance(parameters, dict):
            self.logger.error(f"Invalid parameters type: {type(parameters)}, expected dict")
            return False
        
        return True
    
    def validate_output(self, output: Dict) -> bool:
        """
        Validate the output of the technique.
        
        Args:
            output: Technique output
            
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
    
    def handle_error(self, error: Exception, context: AnalysisContext) -> Dict:
        """
        Handle an error that occurred during execution.
        
        Args:
            error: The exception that occurred
            context: The analysis context
            
        Returns:
            Dictionary containing error information
        """
        self.logger.error(f"Error in {self.name}: {error}")
        
        return {
            "error": str(error),
            "error_type": type(error).__name__,
            "technique": self.name,
            "timestamp": time.time()
        }
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about the technique.
        
        Returns:
            Dictionary containing technique metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "required_mcps": self.required_mcps,
            "compatible_techniques": self.compatible_techniques,
            "incompatible_techniques": self.incompatible_techniques
        }
    
    def check_required_mcps(self) -> bool:
        """
        Check if all required MCPs are available.
        
        Returns:
            True if all required MCPs are available, False otherwise
        """
        registry = MCPRegistry.get_instance()
        
        for mcp_name in self.required_mcps:
            if not registry.get_mcp(mcp_name):
                self.logger.error(f"Required MCP not found: {mcp_name}")
                return False
        
        return True
    
    def get_mcp(self, name: str):
        """
        Get an MCP by name.
        
        Args:
            name: Name of the MCP
            
        Returns:
            The MCP instance, or None if not found
        """
        registry = MCPRegistry.get_instance()
        return registry.get_mcp(name)
    
    def ground_llm_with_context(self, prompt: str, context: AnalysisContext) -> str:
        """
        Ground an LLM prompt with context from research results.
        
        Args:
            prompt: The original prompt
            context: The analysis context
            
        Returns:
            Grounded prompt with relevant context
        """
        # Get research results
        research_results = context.get("research_results")
        if not research_results:
            return prompt
        
        # Extract key findings and evidence
        key_findings = ""
        evidence = ""
        
        if isinstance(research_results, dict):
            if "research_results" in research_results and isinstance(research_results["research_results"], dict):
                if "key_findings" in research_results["research_results"]:
                    key_findings = research_results["research_results"]["key_findings"]
                if "evidence" in research_results["research_results"]:
                    evidence = research_results["research_results"]["evidence"]
        
        # Add context to prompt
        grounded_prompt = f"""
        {prompt}
        
        Consider the following research findings and evidence when formulating your response:
        
        Key Findings:
        {key_findings}
        
        Evidence:
        {evidence}
        """
        
        return grounded_prompt
