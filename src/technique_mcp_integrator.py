"""
Technique MCP Integrator for the MCP Analytical System.
This module provides the TechniqueMCPIntegrator class for integrating analytical techniques with MCPs.
"""

import logging
import os
import importlib.util
import sys
from typing import Dict, List, Any, Optional

from .mcp_registry import MCPRegistry
from .analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechniqueMCPIntegrator:
    """
    Integrator for analytical techniques and MCPs.
    
    This class provides functionality for:
    1. Loading technique modules dynamically
    2. Registering techniques with the MCP registry
    3. Mapping techniques to appropriate MCPs
    4. Managing technique dependencies
    5. Executing techniques with proper context management
    """
    
    def __init__(self, mcp_registry: MCPRegistry, techniques_dir: str = "src/techniques"):
        """
        Initialize the TechniqueMCPIntegrator.
        
        Args:
            mcp_registry: Registry of available MCPs
            techniques_dir: Directory containing technique modules
        """
        self.mcp_registry = mcp_registry
        self.techniques_dir = techniques_dir
        self.techniques = {}
        
        # Load techniques if directory exists
        if os.path.exists(techniques_dir):
            self._load_techniques()
        else:
            logger.warning(f"Techniques directory not found: {techniques_dir}")
    
    def _load_techniques(self):
        """Load technique modules from the techniques directory."""
        logger.info(f"Loading techniques from {self.techniques_dir}")
        
        # Get all Python files in the techniques directory
        technique_files = [f for f in os.listdir(self.techniques_dir) if f.endswith('.py')]
        
        # Load each technique module
        for file_name in technique_files:
            module_name = file_name[:-3]  # Remove .py extension
            module_path = os.path.join(self.techniques_dir, file_name)
            
            try:
                # Load module dynamically
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # Check if module has a technique class
                    if hasattr(module, 'AnalyticalTechnique'):
                        technique_class = getattr(module, 'AnalyticalTechnique')
                        technique = technique_class()
                        self.techniques[technique.name] = technique
                        logger.info(f"Loaded technique: {technique.name}")
                    else:
                        logger.warning(f"No AnalyticalTechnique class found in {file_name}")
                else:
                    logger.warning(f"Could not load module spec for {file_name}")
            
            except Exception as e:
                logger.error(f"Error loading technique {module_name}: {str(e)}")
    
    def get_technique(self, name: str) -> Optional[Any]:
        """
        Get a technique by name.
        
        Args:
            name: Name of the technique
            
        Returns:
            Technique instance if found, None otherwise
        """
        return self.techniques.get(name)
    
    def get_all_techniques(self) -> Dict[str, Any]:
        """
        Get all available techniques.
        
        Returns:
            Dictionary of technique names to technique instances
        """
        return self.techniques
    
    def get_techniques_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get techniques by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            Dictionary of technique names to technique instances in the category
        """
        return {name: technique for name, technique in self.techniques.items() 
                if hasattr(technique, 'category') and technique.category == category}
    
    def get_techniques_by_question_type(self, question_type: str) -> Dict[str, Any]:
        """
        Get techniques suitable for a question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            Dictionary of technique names to technique instances suitable for the question type
        """
        return {name: technique for name, technique in self.techniques.items() 
                if hasattr(technique, 'suitable_for_question_types') and 
                question_type in technique.suitable_for_question_types}
    
    def execute_step(self, technique_name: str, context: AnalysisContext, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a technique step with the given context and parameters.
        
        Args:
            technique_name: Name of the technique to execute
            context: Analysis context containing all relevant data
            parameters: Additional parameters for the technique
            
        Returns:
            Technique execution results
            
        Raises:
            ValueError: If the technique is not found
        """
        logger.info(f"Executing technique step: {technique_name}")
        
        try:
            # Get the technique from the registry
            technique = self.get_technique(technique_name)
            if not technique:
                error_msg = f"Technique not found: {technique_name}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Prepare input data for the technique
            input_data = {
                "question": context.question,
                "context": context,
                **(parameters or {})
            }
            
            # Execute the technique
            logger.info(f"Calling execute method on technique: {technique_name}")
            result = technique.execute(input_data)
            
            # Validate result format
            if not isinstance(result, dict):
                logger.warning(f"Technique {technique_name} returned non-dictionary result: {result}")
                result = {"result": result, "status": "completed"}
            
            # Add execution metadata
            result["technique"] = technique_name
            if "status" not in result:
                result["status"] = "completed"
            result["timestamp"] = import_time() if 'time' in sys.modules else None
            
            logger.info(f"Technique {technique_name} executed successfully")
            return result
            
        except ValueError as ve:
            # Re-raise ValueError for expected errors
            logger.warning(f"ValueError executing technique {technique_name}: {ve}")
            raise
        
        except Exception as e:
            # Catch and log unexpected errors
            error_msg = f"Error executing technique {technique_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return error result
            return {
                "technique": technique_name,
                "status": "failed",
                "error": str(e),
                "error_type": "unexpected",
                "timestamp": import_time() if 'time' in sys.modules else None
            }

    def execute_technique(self, technique_name: str, input_data: Dict) -> Dict:
        """
        Legacy method for executing a technique with the given input data.
        
        Args:
            technique_name: Name of the technique to execute
            input_data: Input data for the technique
            
        Returns:
            Technique execution results
            
        Raises:
            ValueError: If the technique is not found
        """
        logger.warning("Using deprecated execute_technique method. Use execute_step instead.")
        
        technique = self.get_technique(technique_name)
        if not technique:
            error_msg = f"Technique not found: {technique_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Executing technique: {technique_name}")
        
        try:
            result = technique.execute(input_data)
            return result
        except Exception as e:
            error_msg = f"Error executing technique {technique_name}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

# Helper function to import time module only when needed
def import_time():
    import time
    return time.time()
