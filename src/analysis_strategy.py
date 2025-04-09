"""
Analysis strategy module for the MCP architecture.
This module provides the AnalysisStrategy class for representing and managing analytical workflows.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import json
import copy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisStep:
    """
    Represents a single step in an analysis strategy.
    
    Attributes:
        technique: Name of the analytical technique to execute
        parameters: Parameters for the technique execution
        dependencies: List of step indices that must be completed before this step
        optional: Whether this step is optional
    """
    
    def __init__(self, technique: str, parameters: Dict = None, dependencies: List[int] = None, optional: bool = False):
        """
        Initialize an analysis step.
        
        Args:
            technique: Name of the analytical technique to execute
            parameters: Parameters for the technique execution
            dependencies: List of step indices that must be completed before this step
            optional: Whether this step is optional
        """
        self.technique = technique
        self.parameters = parameters or {}
        self.dependencies = dependencies or []
        self.optional = optional
    
    def to_dict(self) -> Dict:
        """
        Convert the step to a dictionary.
        
        Returns:
            Dictionary representation of the step
        """
        return {
            "technique": self.technique,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "optional": self.optional
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisStep':
        """
        Create a step from a dictionary.
        
        Args:
            data: Dictionary representation of the step
            
        Returns:
            AnalysisStep instance
        """
        return cls(
            technique=data.get("technique", ""),
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            optional=data.get("optional", False)
        )

class AnalysisStrategy:
    """
    Represents an analysis strategy consisting of a sequence of steps.
    
    Attributes:
        steps: List of analysis steps
        metadata: Strategy metadata
    """
    
    def __init__(self):
        """Initialize an analysis strategy."""
        self.steps = []
        self.metadata = {}
    
    def add_step(self, technique: str, parameters: Dict = None, dependencies: List[int] = None, optional: bool = False) -> int:
        """
        Add a step to the strategy.
        
        Args:
            technique: Name of the analytical technique to execute
            parameters: Parameters for the technique execution
            dependencies: List of step indices that must be completed before this step
            optional: Whether this step is optional
            
        Returns:
            Index of the added step
        """
        step = AnalysisStep(technique, parameters, dependencies, optional)
        self.steps.append(step)
        return len(self.steps) - 1
    
    def insert_step(self, index: int, technique: str, parameters: Dict = None, dependencies: List[int] = None, optional: bool = False) -> int:
        """
        Insert a step at the specified index.
        
        Args:
            index: Index at which to insert the step
            technique: Name of the analytical technique to execute
            parameters: Parameters for the technique execution
            dependencies: List of step indices that must be completed before this step
            optional: Whether this step is optional
            
        Returns:
            Index of the inserted step
        """
        step = AnalysisStep(technique, parameters, dependencies, optional)
        self.steps.insert(index, step)
        
        # Update dependencies for steps after the inserted step
        for i in range(index + 1, len(self.steps)):
            for j, dep in enumerate(self.steps[i].dependencies):
                if dep >= index:
                    self.steps[i].dependencies[j] = dep + 1
        
        return index
    
    def remove_step(self, index: int) -> Optional[AnalysisStep]:
        """
        Remove a step from the strategy.
        
        Args:
            index: Index of the step to remove
            
        Returns:
            Removed step or None if index is invalid
        """
        if index < 0 or index >= len(self.steps):
            return None
        
        removed_step = self.steps.pop(index)
        
        # Update dependencies for remaining steps
        for step in self.steps:
            # Remove dependencies on the removed step
            step.dependencies = [dep for dep in step.dependencies if dep != index]
            
            # Adjust dependencies for steps after the removed step
            step.dependencies = [dep - 1 if dep > index else dep for dep in step.dependencies]
        
        return removed_step
    
    def get_step(self, index: int) -> Optional[AnalysisStep]:
        """
        Get a step by index.
        
        Args:
            index: Index of the step
            
        Returns:
            Step at the specified index or None if index is invalid
        """
        if index < 0 or index >= len(self.steps):
            return None
        
        return self.steps[index]
    
    def update_step(self, index: int, technique: str = None, parameters: Dict = None, dependencies: List[int] = None, optional: bool = None) -> bool:
        """
        Update a step in the strategy.
        
        Args:
            index: Index of the step to update
            technique: New technique name (if None, keep existing)
            parameters: New parameters (if None, keep existing)
            dependencies: New dependencies (if None, keep existing)
            optional: New optional flag (if None, keep existing)
            
        Returns:
            True if the step was updated, False otherwise
        """
        step = self.get_step(index)
        if not step:
            return False
        
        if technique is not None:
            step.technique = technique
        
        if parameters is not None:
            step.parameters = parameters
        
        if dependencies is not None:
            step.dependencies = dependencies
        
        if optional is not None:
            step.optional = optional
        
        return True
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the strategy.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from the strategy.
        
        Args:
            key: Metadata key
            default: Default value if key is not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict:
        """
        Convert the strategy to a dictionary.
        
        Returns:
            Dictionary representation of the strategy
        """
        return {
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisStrategy':
        """
        Create a strategy from a dictionary.
        
        Args:
            data: Dictionary representation of the strategy
            
        Returns:
            AnalysisStrategy instance
        """
        strategy = cls()
        
        # Add steps
        for step_data in data.get("steps", []):
            step = AnalysisStep.from_dict(step_data)
            strategy.steps.append(step)
        
        # Add metadata
        strategy.metadata = data.get("metadata", {})
        
        return strategy
    
    def copy(self) -> 'AnalysisStrategy':
        """
        Create a deep copy of the strategy.
        
        Returns:
            Copy of the strategy
        """
        strategy = AnalysisStrategy()
        strategy.steps = [copy.deepcopy(step) for step in self.steps]
        strategy.metadata = copy.deepcopy(self.metadata)
        return strategy
    
    def get_next_executable_steps(self, completed_steps: List[int]) -> List[int]:
        """
        Get the indices of steps that can be executed next.
        
        Args:
            completed_steps: Indices of completed steps
            
        Returns:
            List of step indices that can be executed next
        """
        executable_steps = []
        
        for i, step in enumerate(self.steps):
            # Skip if already completed
            if i in completed_steps:
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_steps for dep in step.dependencies):
                executable_steps.append(i)
        
        return executable_steps
    
    def is_complete(self, completed_steps: List[int]) -> bool:
        """
        Check if the strategy is complete.
        
        Args:
            completed_steps: Indices of completed steps
            
        Returns:
            True if all required steps are completed, False otherwise
        """
        for i, step in enumerate(self.steps):
            if i not in completed_steps and not step.optional:
                return False
        
        return True
    
    def get_remaining_steps(self, completed_steps: List[int]) -> List[int]:
        """
        Get the indices of steps that remain to be executed.
        
        Args:
            completed_steps: Indices of completed steps
            
        Returns:
            List of step indices that remain to be executed
        """
        return [i for i in range(len(self.steps)) if i not in completed_steps]
    
    def get_execution_plan(self) -> List[List[int]]:
        """
        Get a plan for executing the steps in the strategy.
        
        Returns:
            List of lists, where each inner list contains step indices that can be executed in parallel
        """
        execution_plan = []
        completed_steps = []
        
        while not self.is_complete(completed_steps):
            executable_steps = self.get_next_executable_steps(completed_steps)
            
            if not executable_steps:
                # No more steps can be executed, but strategy is not complete
                # This indicates a dependency cycle or missing steps
                break
            
            execution_plan.append(executable_steps)
            completed_steps.extend(executable_steps)
        
        return execution_plan
