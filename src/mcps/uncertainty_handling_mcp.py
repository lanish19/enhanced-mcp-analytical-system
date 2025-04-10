"""
Uncertainty Handling MCP for managing and communicating uncertainty in analysis.
This module provides the UncertaintyHandlingMCP class for structured uncertainty analysis.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional

from src.base_mcp import BaseMCP
from src.utils.llm_integration import call_llm, parse_json_response, LLMCallError, LLMParsingError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UncertaintyHandlingMCP(BaseMCP):
    """
    Uncertainty Handling MCP for managing and communicating uncertainty in analysis.
    
    This MCP provides capabilities for:
    1. Identifying sources of uncertainty in analysis
    2. Quantifying uncertainty levels and confidence intervals
    3. Mapping uncertainty across different aspects of analysis
    4. Communicating uncertainty clearly in conclusions
    5. Developing strategies for reducing critical uncertainties
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the UncertaintyHandlingMCP.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__(
            name="UncertaintyHandlingMCP",
            description="Manages and communicates uncertainty in analysis",
            version="1.0.0"
        )
        
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "gpt-4o")
        
        # Define uncertainty types
        self.uncertainty_types = {
            "data_uncertainty": "Uncertainty arising from limitations in available data",
            "model_uncertainty": "Uncertainty in the models or frameworks used for analysis",
            "parameter_uncertainty": "Uncertainty in specific parameters or variables",
            "structural_uncertainty": "Uncertainty about the fundamental structure of the system",
            "linguistic_uncertainty": "Uncertainty arising from ambiguity or vagueness in language",
            "temporal_uncertainty": "Uncertainty related to future developments or changes over time",
            "source_reliability": "Uncertainty about the reliability of information sources",
            "measurement_uncertainty": "Uncertainty in how variables are measured or quantified",
            "sampling_uncertainty": "Uncertainty arising from sampling methods or sample size",
            "expert_disagreement": "Uncertainty due to disagreement among experts",
            "unknown_unknowns": "Uncertainty about factors we don't know we don't know"
        }
        
        # Operation handlers
        self.operation_handlers = {
            "uncertainty_identification": self._uncertainty_identification,
            "uncertainty_quantification": self._uncertainty_quantification,
            "uncertainty_mapping": self._uncertainty_mapping,
            "uncertainty_communication": self._uncertainty_communication,
            "uncertainty_reduction": self._uncertainty_reduction
        }
        
        logger.info("Initialized UncertaintyHandlingMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in UncertaintyHandlingMCP")
        
        # Validate input
        if not isinstance(input_data, dict):
            error_msg = "Input must be a dictionary"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get operation type
        operation = input_data.get("operation")
        if not operation:
            error_msg = "No operation specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Check if operation is supported
        if operation not in self.operation_handlers:
            error_msg = f"Unsupported operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Handle operation
        try:
            result = self.operation_handlers[operation](input_data)
            return result
        except Exception as e:
            error_msg = f"Error processing operation {operation}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _uncertainty_identification(self, input_data: Dict) -> Dict:
        """
        Identify sources of uncertainty in analysis.
        
        Args:
            input_data: Input data dictionary with analysis content
            
        Returns:
            Identified sources of uncertainty
        """
        logger.info("Identifying sources of uncertainty")
        
        # Get analysis to evaluate
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for uncertainty identification"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get specific uncertainty types to focus on if provided
        focus_types = input_data.get("focus_types", [])
        
        # Construct prompt for uncertainty identification
        system_prompt = """You are an expert in uncertainty analysis and epistemology. 
        Your task is to carefully analyze the provided content to identify sources of uncertainty, 
        assessing their nature, scope, and potential impact on conclusions."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        prompt += "TYPES OF UNCERTAINTY TO IDENTIFY:\n"
        if focus_types:
            # Only include specified uncertainty types
            for utype in focus_types:
                if utype in self.uncertainty_types:
                    prompt += f"- {utype}: {self.uncertainty_types[utype]}\n"
        else:
            # Include all uncertainty types
            for utype, description in self.uncertainty_types.items():
                prompt += f"- {utype}: {description}\n"
        
        prompt += """\nPlease carefully analyze the content to identify sources of uncertainty. 
        Structure your response as JSON with the following fields:
        - overall_uncertainty_assessment: Overall assessment of uncertainty in the analysis
        - identified_uncertainties: Array of identified uncertainties, each with:
          - uncertainty_type: Type of uncertainty
          - description: Specific description of this uncertainty
          - evidence: Evidence from the analysis that indicates this uncertainty
          - impact: How this uncertainty might affect conclusions
          - severity: Severity of the uncertainty (low, medium, high)
        - key_knowledge_gaps: Critical gaps in knowledge or information
        - compounding_factors: How different uncertainties might interact or compound
        - recommendations: Specific recommendations for addressing key uncertainties"""
        
        # Call LLM for uncertainty identification
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            if focus_types:
                parsed_result["focus_types"] = focus_types
            
            return {
                "operation": "uncertainty_identification",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in uncertainty identification: {str(e)}")
            return {
                "error": str(e),
                "operation": "uncertainty_identification",
                "input": analysis
            }
    
    def _uncertainty_quantification(self, input_data: Dict) -> Dict:
        """
        Quantify uncertainty levels and confidence intervals.
        
        Args:
            input_data: Input data dictionary with analysis and uncertainties
            
        Returns:
            Quantified uncertainty levels
        """
        logger.info("Quantifying uncertainty levels")
        
        # Get analysis to evaluate
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for uncertainty quantification"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get pre-identified uncertainties if available
        identified_uncertainties = input_data.get("identified_uncertainties", [])
        
        # Construct prompt for uncertainty quantification
        system_prompt = """You are an expert in uncertainty quantification and probabilistic reasoning. 
        Your task is to quantify the levels of uncertainty in the provided analysis, estimating confidence 
        intervals, probability ranges, or qualitative certainty levels as appropriate."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        if identified_uncertainties:
            prompt += "PRE-IDENTIFIED UNCERTAINTIES:\n"
            for i, uncertainty in enumerate(identified_uncertainties):
                prompt += f"{i+1}. {uncertainty}\n"
            prompt += "\n"
        
        prompt += """Please quantify the levels of uncertainty in this analysis. 
        Structure your response as JSON with the following fields:
        - overall_confidence: Overall confidence level in the analysis conclusions (0-100%)
        - key_claims: Array of key claims or conclusions, each with:
          - claim: The specific claim or conclusion
          - confidence: Estimated confidence level (0-100%)
          - confidence_interval: Confidence interval if applicable (e.g., "±10%")
          - probability_range: Probability range if applicable (e.g., "60-80%")
          - qualitative_certainty: Qualitative certainty level (very low, low, medium, high, very high)
          - supporting_factors: Factors that increase confidence
          - limiting_factors: Factors that decrease confidence
        - uncertainty_distribution: How uncertainty is distributed across different aspects
        - confidence_calibration: Assessment of whether confidence levels are well-calibrated
        - recommendations: Recommendations for communicating uncertainty"""
        
        # Call LLM for uncertainty quantification
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "uncertainty_quantification",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in uncertainty quantification: {str(e)}")
            return {
                "error": str(e),
                "operation": "uncertainty_quantification",
                "input": analysis
            }
    
    def _uncertainty_mapping(self, input_data: Dict) -> Dict:
        """
        Map uncertainty across different aspects of analysis.
        
        Args:
            input_data: Input data dictionary with analysis content
            
        Returns:
            Uncertainty map across analysis components
        """
        logger.info("Mapping uncertainty across analysis")
        
        # Get analysis to evaluate
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for uncertainty mapping"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get analysis components if provided
        components = input_data.get("components", [])
        
        # Construct prompt for uncertainty mapping
        system_prompt = """You are an expert in uncertainty analysis and systems thinking. 
        Your task is to map uncertainty across different components or aspects of the provided analysis, 
        identifying how uncertainty propagates through the analytical system."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        if components:
            prompt += "ANALYSIS COMPONENTS TO MAP:\n"
            for component in components:
                prompt += f"- {component}\n"
            prompt += "\n"
        else:
            prompt += "Please identify the key components of this analysis and map uncertainty across them.\n\n"
        
        prompt += """Please create a comprehensive uncertainty map for this analysis. 
        Structure your response as JSON with the following fields:
        - identified_components: Array of key analysis components or aspects
        - uncertainty_map: Array of component uncertainty assessments, each with:
          - component: The analysis component or aspect
          - uncertainty_level: Level of uncertainty (low, medium, high)
          - uncertainty_types: Types of uncertainty affecting this component
          - key_uncertainties: Specific uncertainties affecting this component
          - upstream_dependencies: Components that feed into this one
          - downstream_impacts: Components affected by uncertainty in this one
        - critical_pathways: Pathways through which uncertainty propagates most significantly
        - uncertainty_hotspots: Components with highest uncertainty concentration
        - systemic_uncertainties: Uncertainties that affect multiple components
        - recommendations: Recommendations for managing uncertainty across components"""
        
        # Call LLM for uncertainty mapping
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            if components:
                parsed_result["provided_components"] = components
            
            return {
                "operation": "uncertainty_mapping",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in uncertainty mapping: {str(e)}")
            return {
                "error": str(e),
                "operation": "uncertainty_mapping",
                "input": analysis
            }
    
    def _uncertainty_communication(self, input_data: Dict) -> Dict:
        """
        Generate clear uncertainty communication for conclusions.
        
        Args:
            input_data: Input data dictionary with analysis and conclusions
            
        Returns:
            Uncertainty-aware communication of conclusions
        """
        logger.info("Generating uncertainty communication")
        
        # Get analysis and conclusions
        analysis = input_data.get("analysis", "")
        conclusions = input_data.get("conclusions", "")
        
        if not analysis and not conclusions:
            error_msg = "Neither analysis nor conclusions provided for uncertainty communication"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get uncertainty data if available
        uncertainty_data = input_data.get("uncertainty_data", {})
        
        # Construct prompt for uncertainty communication
        system_prompt = """You are an expert in communicating uncertainty and probabilistic information. 
        Your task is to reformulate conclusions to clearly and accurately communicate uncertainty, 
        avoiding both overconfidence and excessive hedging."""
        
        prompt = ""
        if analysis:
            prompt += f"ANALYSIS:\n{analysis}\n\n"
        
        if conclusions:
            prompt += f"CONCLUSIONS TO REFORMULATE:\n{conclusions}\n\n"
        
        if uncertainty_data:
            prompt += f"UNCERTAINTY DATA:\n{json.dumps(uncertainty_data, indent=2)}\n\n"
        
        prompt += """Please reformulate the conclusions to clearly communicate uncertainty. 
        Structure your response as JSON with the following fields:
        - uncertainty_aware_conclusions: Reformulated conclusions with appropriate uncertainty language
        - confidence_statements: Explicit statements about confidence levels for key points
        - uncertainty_qualifiers: Specific qualifiers used to express uncertainty
        - alternative_possibilities: Clear articulation of alternative possibilities
        - knowledge_limitations: Explicit acknowledgment of knowledge limitations
        - verbal_probability_expressions: Verbal expressions of probability used
        - numerical_expressions: Numerical expressions of uncertainty used
        - visualization_recommendations: Recommendations for visualizing uncertainty
        - communication_principles: Principles followed in the uncertainty communication"""
        
        # Call LLM for uncertainty communication
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "uncertainty_communication",
                "input": conclusions or analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in uncertainty communication: {str(e)}")
            return {
                "error": str(e),
                "operation": "uncertainty_communication",
                "input": conclusions or analysis
            }
    
    def _uncertainty_reduction(self, input_data: Dict) -> Dict:
        """
        Develop strategies for reducing critical uncertainties.
        
        Args:
            input_data: Input data dictionary with analysis and uncertainties
            
        Returns:
            Strategies for uncertainty reduction
        """
        logger.info("Developing uncertainty reduction strategies")
        
        # Get analysis to evaluate
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for uncertainty reduction"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get identified uncertainties if available
        identified_uncertainties = input_data.get("identified_uncertainties", [])
        
        # Construct prompt for uncertainty reduction
        system_prompt = """You are an expert in uncertainty reduction and research design. 
        Your task is to develop strategies for reducing critical uncertainties in the analysis, 
        identifying specific approaches to gather information, test assumptions, or otherwise 
        increase confidence in conclusions."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        if identified_uncertainties:
            prompt += "IDENTIFIED UNCERTAINTIES:\n"
            for i, uncertainty in enumerate(identified_uncertainties):
                prompt += f"{i+1}. {uncertainty}\n"
            prompt += "\n"
        
        prompt += """Please develop strategies for reducing critical uncertainties in this analysis. 
        Structure your response as JSON with the following fields:
        - prioritized_uncertainties: Array of uncertainties prioritized by importance and reducibility
        - reduction_strategies: Array of uncertainty reduction strategies, each with:
          - target_uncertainty: The uncertainty this strategy addresses
          - approach: Specific approach for reducing this uncertainty
          - information_needs: Additional information needed
          - research_methods: Recommended research methods or data collection
          - feasibility: Feasibility of this reduction strategy (low, medium, high)
          - potential_impact: Potential impact on overall confidence (low, medium, high)
          - timeframe: Estimated timeframe for implementation
        - quick_wins: Strategies that could be implemented quickly with significant impact
        - long_term_approaches: Strategies requiring more time but with substantial benefits
        - irreducible_uncertainties: Uncertainties that likely cannot be significantly reduced
        - recommendations: Overall recommendations for uncertainty reduction"""
        
        # Call LLM for uncertainty reduction
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.2
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "uncertainty_reduction",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in uncertainty reduction: {str(e)}")
            return {
                "error": str(e),
                "operation": "uncertainty_reduction",
                "input": analysis
            }
