"""
Analysis Context for storing and managing analysis state.
This module provides the AnalysisContext class for maintaining analysis state across techniques.
"""

import logging
import time
import copy
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisContext:
    """
    Context for storing and managing analysis state.
    
    This class provides a structured way to store and retrieve analysis data,
    including question information, research results, technique results,
    intermediate findings, assumptions, and uncertainties.
    """
    
    def __init__(self):
        """Initialize the analysis context."""
        self.data = {}
        self.question = ""
        self.question_analysis = {}
        self.research_results = {}
        self.technique_results = {}
        self.intermediate_findings = []
        self.assumptions = []
        self.uncertainties = []
        self.parameters = {}
        self.creation_time = time.time()
        logger.info("Initialized Analysis Context")
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a value to the context.
        
        Args:
            key: Key to store the value under
            value: Value to store
        """
        if key == "question":
            self.question = value
        elif key == "question_analysis":
            self.question_analysis = value
        elif key == "research_results":
            self.research_results = value
        elif key == "parameters":
            self.parameters = value
        else:
            self.data[key] = value
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the context.
        
        Args:
            key: Key to retrieve
            
        Returns:
            The value, or None if not found
        """
        if key == "question":
            return self.question
        elif key == "question_analysis":
            return self.question_analysis
        elif key == "research_results":
            return self.research_results
        elif key == "technique_results":
            return self.technique_results
        elif key == "intermediate_findings":
            return self.intermediate_findings
        elif key == "assumptions":
            return self.assumptions
        elif key == "uncertainties":
            return self.uncertainties
        elif key == "parameters":
            return self.parameters
        else:
            return self.data.get(key)
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a value in the context.
        
        Args:
            key: Key to update
            value: New value
        """
        if key == "question":
            self.question = value
        elif key == "question_analysis":
            self.question_analysis = value
        elif key == "research_results":
            self.research_results = value
        elif key == "parameters":
            self.parameters = value
        else:
            self.data[key] = value
    
    def add_technique_result(self, technique: str, result: Any) -> None:
        """
        Add a technique result to the context.
        
        Args:
            technique: Name of the technique
            result: Result of the technique
        """
        self.technique_results[technique] = result
        
        # Extract findings, assumptions, and uncertainties if available
        if isinstance(result, dict):
            # Extract findings
            if "findings" in result and isinstance(result["findings"], list):
                for finding in result["findings"]:
                    if isinstance(finding, dict):
                        self.add_finding(finding)
            
            # Extract assumptions
            if "assumptions" in result and isinstance(result["assumptions"], list):
                for assumption in result["assumptions"]:
                    if isinstance(assumption, dict):
                        self.add_assumption(assumption)
            
            # Extract uncertainties
            if "uncertainties" in result and isinstance(result["uncertainties"], list):
                for uncertainty in result["uncertainties"]:
                    if isinstance(uncertainty, dict):
                        self.add_uncertainty(uncertainty)
    
    def add_finding(self, finding: Dict) -> None:
        """
        Add an intermediate finding to the context.
        
        Args:
            finding: Dictionary containing finding information
        """
        if not isinstance(finding, dict):
            logger.warning(f"Invalid finding type: {type(finding)}, expected dict")
            return
        
        # Add timestamp if not present
        if "timestamp" not in finding:
            finding["timestamp"] = time.time()
        
        self.intermediate_findings.append(finding)
    
    def add_assumption(self, assumption: Dict) -> None:
        """
        Add an assumption to the context.
        
        Args:
            assumption: Dictionary containing assumption information
        """
        if not isinstance(assumption, dict):
            logger.warning(f"Invalid assumption type: {type(assumption)}, expected dict")
            return
        
        # Add timestamp if not present
        if "timestamp" not in assumption:
            assumption["timestamp"] = time.time()
        
        self.assumptions.append(assumption)
    
    def add_uncertainty(self, uncertainty: Dict) -> None:
        """
        Add an uncertainty to the context.
        
        Args:
            uncertainty: Dictionary containing uncertainty information
        """
        if not isinstance(uncertainty, dict):
            logger.warning(f"Invalid uncertainty type: {type(uncertainty)}, expected dict")
            return
        
        # Add timestamp if not present
        if "timestamp" not in uncertainty:
            uncertainty["timestamp"] = time.time()
        
        self.uncertainties.append(uncertainty)
    
    def get_findings_by_source(self, source: str) -> List[Dict]:
        """
        Get findings from a specific source.
        
        Args:
            source: Source of the findings
            
        Returns:
            List of findings from the source
        """
        return [f for f in self.intermediate_findings if f.get("source") == source]
    
    def get_assumptions_by_source(self, source: str) -> List[Dict]:
        """
        Get assumptions from a specific source.
        
        Args:
            source: Source of the assumptions
            
        Returns:
            List of assumptions from the source
        """
        return [a for a in self.assumptions if a.get("source") == source]
    
    def get_uncertainties_by_source(self, source: str) -> List[Dict]:
        """
        Get uncertainties from a specific source.
        
        Args:
            source: Source of the uncertainties
            
        Returns:
            List of uncertainties from the source
        """
        return [u for u in self.uncertainties if u.get("source") == source]
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Args:
            name: Name of the parameter
            default: Default value if parameter not found
            
        Returns:
            Parameter value, or default if not found
        """
        return self.parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set a parameter value.
        
        Args:
            name: Name of the parameter
            value: Value to set
        """
        self.parameters[name] = value
    
    def serialize(self) -> Dict:
        """
        Serialize the context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "question": self.question,
            "question_analysis": self.question_analysis,
            "research_results": self.research_results,
            "technique_results": self.technique_results,
            "intermediate_findings": self.intermediate_findings,
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
            "parameters": self.parameters,
            "data": self.data,
            "creation_time": self.creation_time,
            "last_updated": time.time()
        }
    
    def deserialize(self, data: Dict) -> None:
        """
        Deserialize a dictionary into the context.
        
        Args:
            data: Dictionary representation of the context
        """
        self.question = data.get("question", "")
        self.question_analysis = data.get("question_analysis", {})
        self.research_results = data.get("research_results", {})
        self.technique_results = data.get("technique_results", {})
        self.intermediate_findings = data.get("intermediate_findings", [])
        self.assumptions = data.get("assumptions", [])
        self.uncertainties = data.get("uncertainties", [])
        self.parameters = data.get("parameters", {})
        self.data = data.get("data", {})
        self.creation_time = data.get("creation_time", time.time())
    
    def clone(self):
        """
        Create a deep copy of the context.
        
        Returns:
            New AnalysisContext instance with the same data
        """
        new_context = AnalysisContext()
        new_context.deserialize(self.serialize())
        return new_context
