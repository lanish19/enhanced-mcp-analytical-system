"""
Analysis Context for the MCP Analytical System.
This module provides the AnalysisContext class for storing and retrieving analysis state.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for structured data
class PreliminaryResearchOutput(BaseModel):
    """Structured output from preliminary research."""
    key_facts: List[str] = Field(default_factory=list, description="Key facts discovered during research")
    relevant_domains: List[str] = Field(default_factory=list, description="Domains relevant to the question")
    information_sources: List[Dict[str, str]] = Field(default_factory=list, description="Sources of information")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified gaps in available information")
    research_confidence: str = Field(default="medium", description="Confidence level in the research (low, medium, high)")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="Suggested follow-up research areas")

class QuestionAnalysisOutput(BaseModel):
    """Structured output from question analysis."""
    question_type: str = Field(description="Type of question (e.g., factual, analytical, predictive)")
    complexity_level: str = Field(description="Complexity level of the question (low, medium, high)")
    temporal_focus: str = Field(description="Temporal focus of the question (past, present, future)")
    scope: str = Field(description="Scope of the question (narrow, moderate, broad)")
    relevant_domains: List[str] = Field(default_factory=list, description="Domains relevant to the question")
    key_entities: List[str] = Field(default_factory=list, description="Key entities mentioned in the question")
    required_expertise: List[str] = Field(default_factory=list, description="Types of expertise required")
    uncertainty_level: str = Field(default="medium", description="Level of uncertainty involved (low, medium, high)")
    recommended_techniques: List[str] = Field(default_factory=list, description="Recommended analytical techniques")
    data_requirements: List[str] = Field(default_factory=list, description="Data required for analysis")

class AnalysisContext:
    """
    Context for storing and retrieving analysis state.
    
    This class provides functionality for:
    1. Storing and retrieving key-value pairs
    2. Tracking events during analysis
    3. Storing and retrieving MCP results
    4. Converting context to dictionary for serialization
    """
    
    def __init__(self):
        """Initialize the analysis context."""
        self._data = {}
        self._events = []
        self._mcp_results = {}
        
        # Add timestamp
        self._data["created_at"] = time.time()
        logger.info("Initialized AnalysisContext")
    
    def add(self, key: str, value: Any) -> None:
        """
        Add a key-value pair to the context.
        
        Args:
            key: Key to store the value under
            value: Value to store
        """
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context.
        
        Args:
            key: Key to retrieve the value for
            default: Default value to return if key not found
            
        Returns:
            The value for the key, or the default if not found
        """
        return self._data.get(key, default)
    
    def add_event(self, event_type: str, event_name: str, event_data: Dict = None) -> None:
        """
        Add an event to the context.
        
        Args:
            event_type: Type of event
            event_name: Name of the event
            event_data: Additional event data
        """
        event = {
            "type": event_type,
            "name": event_name,
            "timestamp": time.time(),
            "data": event_data or {}
        }
        self._events.append(event)
    
    def get_events(self, event_type: str = None) -> List[Dict]:
        """
        Get events from the context.
        
        Args:
            event_type: Optional type of events to filter by
            
        Returns:
            List of events
        """
        if event_type:
            return [event for event in self._events if event["type"] == event_type]
        return self._events
    
    def add_mcp_result(self, mcp_name: str, result: Any) -> None:
        """
        Add an MCP result to the context.
        
        Args:
            mcp_name: Name of the MCP
            result: Result from the MCP
        """
        self._mcp_results[mcp_name] = result
        
        # Also add an event for the MCP result
        self.add_event("mcp_result", f"{mcp_name} result added", {"mcp_name": mcp_name})
    
    def get_mcp_result(self, mcp_name: str) -> Optional[Any]:
        """
        Get an MCP result from the context.
        
        Args:
            mcp_name: Name of the MCP
            
        Returns:
            The MCP result, or None if not found
        """
        return self._mcp_results.get(mcp_name)
    
    def get_mcp_results(self) -> Dict[str, Any]:
        """
        Get all MCP results from the context.
        
        Returns:
            Dictionary of MCP names to results
        """
        return self._mcp_results
    
    def to_dict(self) -> Dict:
        """
        Convert the context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "data": self._data,
            "events": self._events,
            "mcp_results": self._mcp_results
        }
    
    def __str__(self) -> str:
        """
        Get string representation of the context.
        
        Returns:
            String representation
        """
        return f"AnalysisContext(keys={list(self._data.keys())}, events={len(self._events)}, mcp_results={list(self._mcp_results.keys())})"
    
    # Properties for convenient access to common data
    @property
    def question(self) -> str:
        """Get the question being analyzed."""
        return self.get("question", "")
    
    @property
    def question_analysis(self) -> Dict:
        """Get the question analysis."""
        return self.get("question_analysis", {})
    
    @property
    def preliminary_research(self) -> Dict:
        """Get the preliminary research."""
        return self.get("preliminary_research", {})
    
    @property
    def strategy(self) -> Dict:
        """Get the analysis strategy."""
        return self.get("strategy", {})
