"""
Technique-MCP Integration Module for connecting analytical techniques with MCPs.

This module provides the TechniqueMCPIntegrator class that serves as a bridge between
analytical techniques and the MCP infrastructure, enabling dynamic workflow adaptation
and technique selection based on interim findings.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import random

from src.base_mcp import BaseMCP
from src.analytical_technique import AnalyticalTechnique
from src.mcp_registry import MCPRegistry
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechniqueMCPIntegrator:
    """
    Integrator class that connects analytical techniques with MCPs.
    
    This class serves as a bridge between the analytical techniques and the MCP infrastructure,
    enabling techniques to leverage MCP capabilities and facilitating dynamic workflow adaptation
    based on interim findings.
    """
    
    def __init__(self, mcp_registry: MCPRegistry):
        """
        Initialize the TechniqueMCPIntegrator.
        
        Args:
            mcp_registry: Registry of available MCPs
        """
        self.mcp_registry = mcp_registry
        self.technique_mcp_mappings = self._initialize_technique_mcp_mappings()
        self.technique_dependencies = self._initialize_technique_dependencies()
        self.technique_complementary_pairs = self._initialize_technique_complementary_pairs()
        self.execution_history = []
        
        logger.info("Initialized TechniqueMCPIntegrator")
    
    def _initialize_technique_mcp_mappings(self) -> Dict[str, List[str]]:
        """
        Initialize mappings between techniques and the MCPs they require.
        
        Returns:
            Dictionary mapping technique names to lists of required MCP names
        """
        # Define which MCPs each technique requires
        mappings = {
            
             # Economic techniques
            "economic_forecasting" : ["economics"],
            # Research-intensive techniques
            "research_to_hypothesis": ["research", "llama4_scout"],
            "scenario_triangulation": ["research", "llama4_scout", "redis_context_store"],
            "historical_analogies": ["research", "llama4_scout"],
            
            # Domain-specific techniques
            "causal_network_analysis": ["llama4_scout", "economics", "geopolitics"],
            "cross_impact_analysis": ["llama4_scout", "economics", "geopolitics"],
            "system_dynamics_modeling": ["llama4_scout", "economics", "geopolitics"],
            
            # Reasoning-intensive techniques
            "analysis_of_competing_hypotheses": ["llama4_scout", "redis_context_store"],
            "key_assumptions_check": ["llama4_scout"],
            "backward_reasoning": ["llama4_scout"],
            
            # Perspective-based techniques
            "multi_persona": ["llama4_scout"],
            "red_teaming": ["llama4_scout"],
            "consensus_challenge": ["llama4_scout", "redis_context_store"],
            
            # Uncertainty-focused techniques
            "uncertainty_mapping": ["llama4_scout", "redis_context_store"],
            "premortem_analysis": ["llama4_scout"],
            "indicators_development": ["llama4_scout", "research"],
            
            # Structured techniques
            "argument_mapping": ["llama4_scout"],
            "bias_detection": ["llama4_scout"],
            "decision_tree_analysis": ["llama4_scout"],
            
            # Forecasting techniques
            "delphistic_forecasting": ["llama4_scout", "research"],
            
            # Synthesis techniques
            "synthesis_generation": ["llama4_scout", "redis_context_store"]
        }
        
        return mappings
    
    def _initialize_technique_dependencies(self) -> Dict[str, List[str]]:
        """
        Initialize dependencies between techniques.
        
        Returns:
            Dictionary mapping technique names to lists of prerequisite technique names
        """
        # Define which techniques depend on other techniques
        dependencies = {
            "economic_forecasting" : [],
            "scenario_triangulation": [],
            "consensus_challenge": [],
            "multi_persona": [],
            "backward_reasoning": [],
            "research_to_hypothesis": [],
            "causal_network_analysis": ["research_to_hypothesis"],
            "key_assumptions_check": ["research_to_hypothesis"],
            "analysis_of_competing_hypotheses": ["research_to_hypothesis"],
            "uncertainty_mapping": ["research_to_hypothesis"],
            "red_teaming": ["research_to_hypothesis"],
            "premortem_analysis": ["scenario_triangulation"],
            "synthesis_generation": ["analysis_of_competing_hypotheses", "uncertainty_mapping"],
            "cross_impact_analysis": ["causal_network_analysis"],
            "system_dynamics_modeling": ["causal_network_analysis"],
            "indicators_development": ["uncertainty_mapping"],
            "argument_mapping": ["research_to_hypothesis"],
            "bias_detection": [],
            "decision_tree_analysis": ["causal_network_analysis"],
            "delphistic_forecasting": ["scenario_triangulation"],
            "historical_analogies": []
        }
        
        return dependencies
    
    def _initialize_technique_complementary_pairs(self) -> List[Tuple[str, str]]:
        """
        Initialize complementary pairs of techniques that work well together.
        
        Returns:
            List of tuples containing pairs of complementary technique names
        """
        # Define pairs of techniques that complement each other
        complementary_pairs = [
            ("research_to_hypothesis", "key_assumptions_check"),
            ("scenario_triangulation", "premortem_analysis"),
            ("causal_network_analysis", "system_dynamics_modeling"),
            ("analysis_of_competing_hypotheses", "uncertainty_mapping"),
            ("multi_persona", "red_teaming"),
            ("consensus_challenge", "bias_detection"),
            ("backward_reasoning", "decision_tree_analysis"),
            ("historical_analogies", "scenario_triangulation"),
            ("uncertainty_mapping", "indicators_development"),
            ("causal_network_analysis", "cross_impact_analysis")
        ]
        
        return complementary_pairs
    
    def execute_technique(self, technique: AnalyticalTechnique, context: AnalysisContext, 
                         parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an analytical technique with MCP support.
        
        Args:
            technique: The analytical technique to execute
            context: The analysis context
            parameters: Optional parameters for technique execution
            
        Returns:
            Execution results
        """
        technique_name = technique.name
        logger.info(f"Executing technique: {technique_name}")
        
        # Get required MCPs for this technique
        required_mcps = self.technique_mcp_mappings.get(technique_name, [])
        
        # Check if all required MCPs are available
        missing_mcps = [mcp_name for mcp_name in required_mcps if not self.mcp_registry.has_mcp(mcp_name)]
        if missing_mcps:
            error_msg = f"Missing required MCPs for technique {technique_name}: {', '.join(missing_mcps)}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "failed"}
        
        # Get MCP instances
        mcp_instances = {mcp_name: self.mcp_registry.get_mcp(mcp_name) for mcp_name in required_mcps}
        
        # Prepare execution context with MCPs
        execution_context = {
            "mcps": mcp_instances,
            "context": context,
            "parameters": parameters or {}
        }
        
        # Execute technique with MCP support
        start_time = time.time()
        try:
            result = technique.execute(execution_context)
            
            # Record execution in history
            execution_record = {
                "technique": technique_name,
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "required_mcps": required_mcps,
                "parameters": parameters,
                "status": "success",
                "result_summary": self._summarize_result(result)
            }
            self.execution_history.append(execution_record)
            
            return result
        except Exception as e:
            logger.error(f"Error executing technique {technique_name}: {str(e)}")
            
            # Record execution failure in history
            execution_record = {
                "technique": technique_name,
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "required_mcps": required_mcps,
                "parameters": parameters,
                "status": "failed",
                "error": str(e)
            }
            self.execution_history.append(execution_record)
            
            return {"error": str(e), "status": "failed"}
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of technique execution results for history tracking.
        
        Args:
            result: The full execution result
            
        Returns:
            Summarized result
        """
        # Create a simplified summary without large data structures
        summary = {}
        
        # Copy basic metadata
        if "status" in result:
            summary["status"] = result["status"]
        
        if "confidence" in result:
            summary["confidence"] = result["confidence"]
        
        if "uncertainty" in result:
            summary["uncertainty"] = result["uncertainty"]
        
        # Summarize findings
        if "findings" in result:
            if isinstance(result["findings"], list):
                summary["findings_count"] = len(result["findings"])
                if result["findings"]:
                    summary["findings_sample"] = result["findings"][0]
            elif isinstance(result["findings"], dict):
                summary["findings_keys"] = list(result["findings"].keys())
        
        # Summarize hypotheses
        if "hypotheses" in result:
            if isinstance(result["hypotheses"], list):
                summary["hypotheses_count"] = len(result["hypotheses"])
        
        # Summarize scenarios
        if "scenarios" in result:
            if isinstance(result["scenarios"], list):
                summary["scenarios_count"] = len(result["scenarios"])
        
        return summary
    
    def get_technique_dependencies(self, technique_name: str) -> List[str]:
        """
        Get the dependencies for a technique.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            List of prerequisite technique names
        """
        return self.technique_dependencies.get(technique_name, [])
    
    def get_complementary_techniques(self, technique_name: str) -> List[str]:
        """
        Get complementary techniques for a given technique.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            List of complementary technique names
        """
        complementary = []
        for pair in self.technique_complementary_pairs:
            if technique_name == pair[0]:
                complementary.append(pair[1])
            elif technique_name == pair[1]:
                complementary.append(pair[0])
        
        return complementary
    
    def get_required_mcps(self, technique_name: str) -> List[str]:
        """
        Get the required MCPs for a technique.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            List of required MCP names
        """
        return self.technique_mcp_mappings.get(technique_name, [])
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history.
        
        Returns:
            List of execution records
        """
        return self.execution_history
    
    def get_last_execution(self, technique_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the last execution record, optionally filtered by technique name.
        
        Args:
            technique_name: Optional name of the technique to filter by
            
        Returns:
            Last execution record or None if no matching record
        """
        if not self.execution_history:
            return None
        
        if technique_name:
            # Filter by technique name
            filtered_history = [record for record in self.execution_history if record["technique"] == technique_name]
            if not filtered_history:
                return None
            return filtered_history[-1]
        else:
            # Return the most recent execution
            return self.execution_history[-1]
    
    def suggest_next_techniques(self, context: AnalysisContext, current_technique: str = None) -> List[Dict[str, Any]]:
        """
        Suggest next techniques to execute based on the current context and execution history.
        
        Args:
            context: The analysis context
            current_technique: Optional name of the current technique
            
        Returns:
            List of suggested technique names with rationale
        """
        suggestions = []
        
        # Get all available techniques
        all_techniques = list(self.technique_dependencies.keys())
        
        # Get executed techniques
        executed_techniques = [record["technique"] for record in self.execution_history]
        
        # If no techniques have been executed yet, suggest initial techniques
        if not executed_techniques:
            # Always start with research_to_hypothesis for research-based questions
            if "research_to_hypothesis" in all_techniques:
                suggestions.append({
                    "technique": "research_to_hypothesis",
                    "rationale": "Initial research and hypothesis formation",
                    "priority": "high"
                })
            
            # For predictive questions, suggest scenario_triangulation
            if context.question_type == "predictive" and "scenario_triangulation" in all_techniques:
                suggestions.append({
                    "technique": "scenario_triangulation",
                    "rationale": "Generate scenarios for predictive question",
                    "priority": "high"
                })
            
            # For causal questions, suggest causal_network_analysis
            if context.question_type == "causal" and "causal_network_analysis" in all_techniques:
                suggestions.append({
                    "technique": "causal_network_analysis",
                    "rationale": "Analyze causal relationships for causal question",
                    "priority": "high"
                })
            
            # For evaluative questions, suggest multi_persona
            if context.question_type == "evaluative" and "multi_persona" in all_techniques:
                suggestions.append({
                    "technique": "multi_persona",
                    "rationale": "Evaluate from multiple perspectives for evaluative question",
                    "priority": "high"
                })
            
            return suggestions
        
        # If current_technique is provided, suggest complementary techniques
        if current_technique:
            complementary = self.get_complementary_techniques(current_technique)
            for technique in complementary:
                if technique not in executed_techniques:
                    suggestions.append({
                        "technique": technique,
                        "rationale": f"Complementary to {current_technique}",
                        "priority": "medium"
                    })
        
        # Check for techniques whose dependencies have been satisfied
        for technique in all_techniques:
            if technique not in executed_techniques:
                dependencies = self.get_technique_dependencies(technique)
                if all(dep in executed_techniques for dep in dependencies):
                    suggestions.append({
                        "technique": technique,
                        "rationale": "Dependencies satisfied",
                        "priority": "medium"
                    })
        
        # Suggest techniques based on question type and context
        if context.question_type == "predictive" and "uncertainty_mapping" not in executed_techniques:
            suggestions.append({
                "technique": "uncertainty_mapping",
                "rationale": "Map uncertainties for predictive question",
                "priority": "medium"
            })
        
        if context.question_type == "causal" and "cross_impact_analysis" not in executed_techniques:
            suggestions.append({
                "technique": "cross_impact_analysis",
                "rationale": "Analyze cross-impacts for causal question",
                "priority": "medium"
            })
        
        if context.question_type == "evaluative" and "key_assumptions_check" not in executed_techniques:
            suggestions.append({
                "technique": "key_assumptions_check",
                "rationale": "Check key assumptions for evaluative question",
                "priority": "medium"
            })
        
        # Check for high uncertainty and suggest uncertainty-focused techniques
        if context.uncertainty_level > 0.7:
            uncertainty_techniques = ["uncertainty_mapping", "premortem_analysis", "indicators_development"]
            for technique in uncertainty_techniques:
                if technique not in executed_techniques:
                    suggestions.append({
                        "technique": technique,
                        "rationale": "Address high uncertainty",
                        "priority": "high"
                    })
        
        # Check for conflicting hypotheses and suggest ACH
        if context.has_conflicting_hypotheses() and "analysis_of_competing_hypotheses" not in executed_techniques:
            suggestions.append({
                "technique": "analysis_of_competing_hypotheses",
                "rationale": "Resolve conflicting hypotheses",
                "priority": "high"
            })
        
        # Always suggest synthesis_generation as a final technique
        if len(executed_techniques) >= 3 and "synthesis_generation" not in executed_techniques:
            suggestions.append({
                "technique": "synthesis_generation",
                "rationale": "Synthesize findings from multiple techniques",
                "priority": "high" if len(executed_techniques) >= 5 else "medium"
            })
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen_techniques = set()
        for suggestion in suggestions:
            if suggestion["technique"] not in seen_techniques:
                unique_suggestions.append(suggestion)
                seen_techniques.add(suggestion["technique"])
        
        return unique_suggestions
    
    def adapt_workflow(self, context: AnalysisContext, interim_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt the workflow based on interim findings.
        
        Args:
            context: The analysis context
            interim_findings: Interim findings from executed techniques
            
        Returns:
            Adaptation recommendations
        """
        logger.info("Adapting workflow based on interim findings")
        
        # Initialize recommendations
        recommendations = {
            "add_techniques": [],
            "skip_techniques": [],
            "modify_parameters": {},
            "update_context": {},
            "rationale": {}
        }
        
        # Get executed techniques
        executed_techniques = [record["technique"] for record in self.execution_history]
        
        # Check for high uncertainty
        if "uncertainty_level" in interim_findings:
            uncertainty_level = interim_findings["uncertainty_level"]
            context.uncertainty_level = uncertainty_level
            
            if uncertainty_level > 0.7:
                # High uncertainty - add uncertainty-focused techniques
                uncertainty_techniques = ["uncertainty_mapping", "premortem_analysis", "indicators_development"]
                for technique in uncertainty_techniques:
                    if technique not in executed_techniques and technique not in recommendations["add_techniques"]:
                        recommendations["add_techniques"].append(technique)
                        recommendations["rationale"][f"add_{technique}"] = f"High uncertainty level ({uncertainty_level:.2f})"
            elif uncertainty_level < 0.3:
                # Low uncertainty - skip uncertainty-focused techniques
                uncertainty_techniques = ["uncertainty_mapping", "premortem_analysis"]
                for technique in uncertainty_techniques:
                    if technique not in executed_techniques and technique not in recommendations["skip_techniques"]:
                        recommendations["skip_techniques"].append(technique)
                        recommendations["rationale"][f"skip_{technique}"] = f"Low uncertainty level ({uncertainty_level:.2f})"
        
        # Check for conflicting hypotheses
        if "hypotheses" in interim_findings:
            hypotheses = interim_findings["hypotheses"]
            context.update_hypotheses(hypotheses)
            
            if context.has_conflicting_hypotheses():
                # Add ACH for conflicting hypotheses
                if "analysis_of_competing_hypotheses" not in executed_techniques and "analysis_of_competing_hypotheses" not in recommendations["add_techniques"]:
                    recommendations["add_techniques"].append("analysis_of_competing_hypotheses")
                    recommendations["rationale"]["add_analysis_of_competing_hypotheses"] = "Conflicting hypotheses detected"
        
        # Check for strong causal relationships
        if "causal_relationships" in interim_findings:
            causal_relationships = interim_findings["causal_relationships"]
            context.update_causal_relationships(causal_relationships)
            
            if context.has_strong_causal_relationships():
                # Add system dynamics modeling for strong causal relationships
                if "system_dynamics_modeling" not in executed_techniques and "system_dynamics_modeling" not in recommendations["add_techniques"]:
                    recommendations["add_techniques"].append("system_dynamics_modeling")
                    recommendations["rationale"]["add_system_dynamics_modeling"] = "Strong causal relationships detected"
        
        # Check for domain-specific findings
        if "domain_findings" in interim_findings:
            domain_findings = interim_findings["domain_findings"]
            
            # Check for economic findings
            if "economic" in domain_findings:
                # Modify parameters for economic-focused techniques
                if "causal_network_analysis" in recommendations["add_techniques"] or "causal_network_analysis" not in executed_techniques:
                    recommendations["modify_parameters"]["causal_network_analysis"] = {"focus_domain": "economic"}
                    recommendations["rationale"]["modify_causal_network_analysis"] = "Economic domain findings detected"
            
            # Check for geopolitical findings
            if "geopolitical" in domain_findings:
                # Modify parameters for geopolitical-focused techniques
                if "scenario_triangulation" in recommendations["add_techniques"] or "scenario_triangulation" not in executed_techniques:
                    recommendations["modify_parameters"]["scenario_triangulation"] = {"focus_domain": "geopolitical"}
                    recommendations["rationale"]["modify_scenario_triangulation"] = "Geopolitical domain findings detected"
        
        # Check for time horizon
        if "time_horizon" in interim_findings:
            time_horizon = interim_findings["time_horizon"]
            context.time_horizon = time_horizon
            
            if time_horizon == "long_term":
                # Add scenario-based techniques for long-term questions
                long_term_techniques = ["scenario_triangulation", "delphistic_forecasting"]
                for technique in long_term_techniques:
                    if technique not in executed_techniques and technique not in recommendations["add_techniques"]:
                        recommendations["add_techniques"].append(technique)
                        recommendations["rationale"][f"add_{technique}"] = "Long-term time horizon detected"
            elif time_horizon == "short_term":
                # Skip scenario-based techniques for short-term questions
                if "delphistic_forecasting" not in executed_techniques and "delphistic_forecasting" not in recommendations["skip_techniques"]:
                    recommendations["skip_techniques"].append("delphistic_forecasting")
                    recommendations["rationale"]["skip_delphistic_forecasting"] = "Short-term time horizon detected"
        
        # Update context with any new information
        if "key_entities" in interim_findings:
            context.key_entities = interim_findings["key_entities"]
            recommendations["update_context"]["key_entities"] = interim_findings["key_entities"]
        
        if "key_relationships" in interim_findings:
            context.key_relationships = interim_findings["key_relationships"]
            recommendations["update_context"]["key_relationships"] = interim_findings["key_relationships"]
        
        # Remove empty entries
        if not recommendations["add_techniques"]:
            del recommendations["add_techniques"]
        
        if not recommendations["skip_techniques"]:
            del recommendations["skip_techniques"]
        
        if not recommendations["modify_parameters"]:
            del recommendations["modify_parameters"]
        
        if not recommendations["update_context"]:
            del recommendations["update_context"]
        
        return recommendations
    
    def evaluate_technique_effectiveness(self, technique_name: str) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a technique based on execution history.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            Effectiveness evaluation
        """
        # Filter execution history for this technique
        technique_executions = [record for record in self.execution_history if record["technique"] == technique_name]
        
        if not technique_executions:
            return {
                "technique": technique_name,
                "executions": 0,
                "effectiveness": "unknown",
                "average_duration": 0,
                "success_rate": 0
            }
        
        # Calculate metrics
        num_executions = len(technique_executions)
        num_successes = sum(1 for record in technique_executions if record["status"] == "success")
        success_rate = num_successes / num_executions if num_executions > 0 else 0
        
        # Calculate average duration
        durations = [record["duration"] for record in technique_executions if "duration" in record]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Determine effectiveness
        effectiveness = "unknown"
        if success_rate > 0.8:
            effectiveness = "high"
        elif success_rate > 0.5:
            effectiveness = "medium"
        else:
            effectiveness = "low"
        
        return {
            "technique": technique_name,
            "executions": num_executions,
            "effectiveness": effectiveness,
            "average_duration": avg_duration,
            "success_rate": success_rate
        }
    
    def get_technique_mcp_usage(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics on MCP usage by techniques.
        
        Returns:
            Dictionary mapping technique names to dictionaries of MCP usage counts
        """
        usage = {}
        
        for record in self.execution_history:
            technique = record["technique"]
            if technique not in usage:
                usage[technique] = {}
            
            for mcp in record["required_mcps"]:
                if mcp not in usage[technique]:
                    usage[technique][mcp] = 0
                usage[technique][mcp] += 1
        
        return usage
