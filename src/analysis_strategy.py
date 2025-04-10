"""
Analysis Strategy for the MCP Analytical System.
This module provides the AnalysisStrategy class for representing and managing analysis workflows.
"""

import logging
import json
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisStrategy:
    """
    Strategy for representing and managing analysis workflows.
    
    This class provides functionality for:
    1. Defining a sequence of analytical techniques to execute
    2. Specifying parameters for each technique
    3. Defining adaptation criteria for dynamic workflow adjustment
    4. Converting strategy to/from dictionary for serialization
    """
    
    def __init__(self, strategy_data: Dict[str, Any]):
        """
        Initialize the analysis strategy.
        
        Args:
            strategy_data: Dictionary containing strategy data
        """
        self.name = strategy_data.get("name", "Unnamed Strategy")
        self.description = strategy_data.get("description", "No description provided")
        self.adaptive = strategy_data.get("adaptive", False)
        self.steps = strategy_data.get("steps", [])
        
        # Validate steps
        for i, step in enumerate(self.steps):
            if "technique" not in step:
                logger.warning(f"Step {i} missing required 'technique' field, adding placeholder")
                step["technique"] = "placeholder"
            if "purpose" not in step:
                step["purpose"] = f"Execute {step['technique']}"
            if "parameters" not in step:
                step["parameters"] = {}
            if "adaptive_criteria" not in step:
                step["adaptive_criteria"] = []
        
        logger.info(f"Initialized AnalysisStrategy '{self.name}' with {len(self.steps)} steps")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the strategy to a dictionary.
        
        Returns:
            Dictionary representation of the strategy
        """
        return {
            "name": self.name,
            "description": self.description,
            "adaptive": self.adaptive,
            "steps": self.steps
        }
    
    @classmethod
    def from_dict(cls, strategy_dict: Dict[str, Any]) -> 'AnalysisStrategy':
        """
        Create a strategy from a dictionary.
        
        Args:
            strategy_dict: Dictionary containing strategy data
            
        Returns:
            AnalysisStrategy instance
        """
        return cls(strategy_dict)
    
    @classmethod
    def from_context(cls, context) -> 'AnalysisStrategy':
        """
        Create a strategy based on analysis context.
        
        Args:
            context: Analysis context
            
        Returns:
            AnalysisStrategy instance
        """
        # Extract question analysis from context
        question_analysis = context.get("question_analysis", {})
        question_type = question_analysis.get("type", "unknown")
        domains = question_analysis.get("domains", [])
        
        # Create strategy based on question type
        if question_type == "predictive":
            return cls.create_predictive_strategy(domains)
        elif question_type == "causal":
            return cls.create_causal_strategy(domains)
        elif question_type == "evaluative":
            return cls.create_evaluative_strategy(domains)
        else:
            return cls.create_default_strategy()
    
    @classmethod
    def create_predictive_strategy(cls, domains: List[str]) -> 'AnalysisStrategy':
        """
        Create a strategy for predictive questions.
        
        Args:
            domains: List of relevant domains
            
        Returns:
            AnalysisStrategy instance
        """
        strategy_data = {
            "name": "Predictive Analysis Strategy",
            "description": f"Strategy for predictive analysis in {', '.join(domains)} domains",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research and generate hypotheses",
                    "parameters": {},
                    "adaptive_criteria": ["conflicting_evidence_found"]
                },
                {
                    "technique": "scenario_triangulation",
                    "purpose": "Generate multiple plausible futures",
                    "parameters": {"num_scenarios": 3},
                    "adaptive_criteria": []
                },
                {
                    "technique": "uncertainty_mapping",
                    "purpose": "Map and quantify areas of uncertainty",
                    "parameters": {},
                    "adaptive_criteria": ["overall_uncertainty > 0.7"]
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate final synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                }
            ]
        }
        
        return cls(strategy_data)
    
    @classmethod
    def create_causal_strategy(cls, domains: List[str]) -> 'AnalysisStrategy':
        """
        Create a strategy for causal questions.
        
        Args:
            domains: List of relevant domains
            
        Returns:
            AnalysisStrategy instance
        """
        strategy_data = {
            "name": "Causal Analysis Strategy",
            "description": f"Strategy for causal analysis in {', '.join(domains)} domains",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research and generate hypotheses",
                    "parameters": {},
                    "adaptive_criteria": ["conflicting_evidence_found"]
                },
                {
                    "technique": "causal_network_analysis",
                    "purpose": "Identify causal relationships",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "key_assumptions_check",
                    "purpose": "Identify and challenge key assumptions",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate final synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                }
            ]
        }
        
        return cls(strategy_data)
    
    @classmethod
    def create_evaluative_strategy(cls, domains: List[str]) -> 'AnalysisStrategy':
        """
        Create a strategy for evaluative questions.
        
        Args:
            domains: List of relevant domains
            
        Returns:
            AnalysisStrategy instance
        """
        strategy_data = {
            "name": "Evaluative Analysis Strategy",
            "description": f"Strategy for evaluative analysis in {', '.join(domains)} domains",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research and generate hypotheses",
                    "parameters": {},
                    "adaptive_criteria": ["conflicting_evidence_found"]
                },
                {
                    "technique": "analysis_of_competing_hypotheses",
                    "purpose": "Systematically evaluate competing hypotheses",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "argument_mapping",
                    "purpose": "Map arguments for and against",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate final synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                }
            ]
        }
        
        return cls(strategy_data)
    
    @classmethod
    def create_default_strategy(cls) -> 'AnalysisStrategy':
        """
        Create a default strategy.
        
        Returns:
            AnalysisStrategy instance
        """
        strategy_data = {
            "name": "Default Analysis Strategy",
            "description": "Basic analytical strategy with essential techniques",
            "adaptive": False,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research and generate hypotheses",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate final synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                }
            ]
        }
        
        return cls(strategy_data)
    
    def add_step(self, technique: str, purpose: str, parameters: Dict[str, Any] = None, 
                adaptive_criteria: List = None, position: int = None) -> None:
        """
        Add a step to the strategy.
        
        Args:
            technique: Name of the technique to execute
            purpose: Purpose of the technique
            parameters: Parameters for the technique
            adaptive_criteria: Criteria for adapting the workflow
            position: Position to insert the step (None for append)
        """
        step = {
            "technique": technique,
            "purpose": purpose,
            "parameters": parameters or {},
            "adaptive_criteria": adaptive_criteria or []
        }
        
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        logger.info(f"Added step '{technique}' to strategy '{self.name}'")
    
    def remove_step(self, index: int) -> None:
        """
        Remove a step from the strategy.
        
        Args:
            index: Index of the step to remove
        """
        if 0 <= index < len(self.steps):
            removed_step = self.steps.pop(index)
            logger.info(f"Removed step '{removed_step['technique']}' from strategy '{self.name}'")
        else:
            logger.warning(f"Cannot remove step at index {index}, out of range")
    
    def get_step(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get a step from the strategy.
        
        Args:
            index: Index of the step to get
            
        Returns:
            Step dictionary if found, None otherwise
        """
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
    
    def update_step(self, index: int, **kwargs) -> None:
        """
        Update a step in the strategy.
        
        Args:
            index: Index of the step to update
            **kwargs: Fields to update
        """
        if 0 <= index < len(self.steps):
            for key, value in kwargs.items():
                if key in self.steps[index]:
                    self.steps[index][key] = value
            logger.info(f"Updated step {index} in strategy '{self.name}'")
        else:
            logger.warning(f"Cannot update step at index {index}, out of range")
    
    def __str__(self) -> str:
        """
        Get string representation of the strategy.
        
        Returns:
            String representation
        """
        return f"AnalysisStrategy(name='{self.name}', steps={len(self.steps)})"
