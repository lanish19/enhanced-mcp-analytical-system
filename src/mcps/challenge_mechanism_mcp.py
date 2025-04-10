"""
Challenge Mechanism MCP for critical evaluation and bias detection.
This module provides the ChallengeMechanismMCP class for implementing structured challenges to analysis.
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

class ChallengeMechanismMCP(BaseMCP):
    """
    Challenge Mechanism MCP for critical evaluation and bias detection.
    
    This MCP provides capabilities for:
    1. Detecting cognitive biases in analysis
    2. Challenging assumptions through structured techniques
    3. Identifying logical fallacies and reasoning errors
    4. Generating alternative hypotheses and explanations
    5. Stress-testing conclusions through adversarial analysis
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ChallengeMechanismMCP.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__(
            name="ChallengeMechanismMCP",
            description="Implements structured challenges to analysis",
            version="1.0.0"
        )
        
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "gpt-4o")
        
        # Define cognitive biases to detect
        self.cognitive_biases = {
            "confirmation_bias": "Tendency to search for, interpret, and recall information that confirms pre-existing beliefs",
            "availability_bias": "Overestimating the likelihood of events based on their availability in memory",
            "anchoring_bias": "Over-reliance on the first piece of information encountered",
            "hindsight_bias": "Tendency to perceive past events as having been predictable",
            "recency_bias": "Overweighting recent events or information",
            "groupthink": "Tendency for groups to make irrational decisions due to conformity pressure",
            "authority_bias": "Tendency to attribute greater accuracy to opinions of authority figures",
            "sunk_cost_fallacy": "Continuing a behavior based on previously invested resources",
            "framing_effect": "Drawing different conclusions from the same information presented differently",
            "overconfidence_bias": "Excessive confidence in one's own answers or abilities",
            "status_quo_bias": "Preference for the current state of affairs",
            "bandwagon_effect": "Adopting beliefs or behaviors because others have done so",
            "optimism_bias": "Tendency to overestimate positive outcomes and underestimate negative ones",
            "pessimism_bias": "Tendency to overestimate negative outcomes and underestimate positive ones",
            "fundamental_attribution_error": "Overemphasizing personality-based explanations for behaviors observed in others"
        }
        
        # Define logical fallacies to detect
        self.logical_fallacies = {
            "ad_hominem": "Attacking the person instead of addressing their argument",
            "straw_man": "Misrepresenting an argument to make it easier to attack",
            "false_dichotomy": "Presenting only two options when others exist",
            "slippery_slope": "Arguing that a small step will lead to a chain of events ending in disaster",
            "circular_reasoning": "Using the conclusion as a premise",
            "appeal_to_nature": "Arguing that something is good because it's natural",
            "appeal_to_authority": "Using an authority's statement as evidence without addressing the argument",
            "hasty_generalization": "Drawing a conclusion based on insufficient evidence",
            "post_hoc_ergo_propter_hoc": "Assuming that because B followed A, A caused B",
            "appeal_to_emotion": "Manipulating emotions instead of using valid reasoning",
            "tu_quoque": "Avoiding criticism by pointing out similar behavior in others",
            "no_true_scotsman": "Redefining terms to exclude counterexamples",
            "burden_of_proof": "Claiming something is true without providing evidence",
            "anecdotal_evidence": "Using personal experience or isolated examples instead of sound reasoning",
            "appeal_to_popularity": "Claiming something is true because many people believe it"
        }
        
        # Operation handlers
        self.operation_handlers = {
            "bias_detection": self._bias_detection,
            "assumption_challenge": self._assumption_challenge,
            "logical_fallacy_check": self._logical_fallacy_check,
            "alternative_hypothesis": self._alternative_hypothesis,
            "red_team_analysis": self._red_team_analysis,
            "premortem_analysis": self._premortem_analysis
        }
        
        logger.info("Initialized ChallengeMechanismMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in ChallengeMechanismMCP")
        
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
    
    def _bias_detection(self, input_data: Dict) -> Dict:
        """
        Detect cognitive biases in analysis.
        
        Args:
            input_data: Input data dictionary with analysis content
            
        Returns:
            Detected biases and their impact
        """
        logger.info("Detecting cognitive biases")
        
        # Get analysis to check
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for bias detection"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get specific biases to check if provided
        target_biases = input_data.get("target_biases", [])
        
        # Construct prompt for bias detection
        system_prompt = """You are an expert in cognitive biases and critical thinking. 
        Your task is to carefully analyze the provided content for evidence of cognitive biases, 
        identifying specific instances where biases may be influencing the analysis, and assessing 
        their potential impact on conclusions."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        prompt += "COGNITIVE BIASES TO DETECT:\n"
        if target_biases:
            # Only include specified biases
            for bias in target_biases:
                if bias in self.cognitive_biases:
                    prompt += f"- {bias}: {self.cognitive_biases[bias]}\n"
        else:
            # Include all biases
            for bias, description in self.cognitive_biases.items():
                prompt += f"- {bias}: {description}\n"
        
        prompt += """\nPlease carefully analyze the content for evidence of these cognitive biases. 
        Structure your response as JSON with the following fields:
        - overall_assessment: Overall assessment of bias presence in the analysis
        - detected_biases: Array of detected biases, each with:
          - bias_type: Type of cognitive bias detected
          - evidence: Specific evidence or examples from the text
          - impact: How this bias might impact the analysis or conclusions
          - severity: Severity of the bias (low, medium, high)
        - potential_biases: Biases that might be present but with less clear evidence
        - bias_interactions: How multiple biases might interact or compound
        - debiasing_recommendations: Specific recommendations to mitigate detected biases"""
        
        # Call LLM for bias detection
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
            if target_biases:
                parsed_result["target_biases"] = target_biases
            
            return {
                "operation": "bias_detection",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in bias detection: {str(e)}")
            return {
                "error": str(e),
                "operation": "bias_detection",
                "input": analysis
            }
    
    def _assumption_challenge(self, input_data: Dict) -> Dict:
        """
        Challenge key assumptions in analysis.
        
        Args:
            input_data: Input data dictionary with analysis content
            
        Returns:
            Identified assumptions and their evaluation
        """
        logger.info("Challenging assumptions")
        
        # Get analysis to check
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for assumption challenge"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get explicit assumptions if provided
        explicit_assumptions = input_data.get("explicit_assumptions", [])
        
        # Construct prompt for assumption challenge
        system_prompt = """You are an expert in critical thinking and assumption analysis. 
        Your task is to identify and challenge key assumptions in the provided analysis, 
        evaluating their validity, impact, and alternatives."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        if explicit_assumptions:
            prompt += "EXPLICIT ASSUMPTIONS TO EVALUATE:\n"
            for assumption in explicit_assumptions:
                prompt += f"- {assumption}\n"
            prompt += "\nIn addition to these explicit assumptions, identify and challenge any implicit assumptions.\n\n"
        
        prompt += """Please identify and challenge key assumptions in this analysis. 
        Structure your response as JSON with the following fields:
        - overall_assessment: Overall assessment of the role of assumptions in the analysis
        - identified_assumptions: Array of identified assumptions, each with:
          - assumption: The assumption statement
          - type: Whether the assumption is explicit or implicit
          - evidence: Evidence from the text that indicates this assumption
          - validity: Assessment of the assumption's validity (low, medium, high)
          - impact: How critical this assumption is to the conclusions
          - alternatives: Alternative assumptions that could be made
        - key_dependencies: How assumptions depend on or relate to each other
        - sensitivity_analysis: How conclusions might change if key assumptions were different
        - recommendations: Recommendations for addressing problematic assumptions"""
        
        # Call LLM for assumption challenge
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
            if explicit_assumptions:
                parsed_result["explicit_assumptions_provided"] = explicit_assumptions
            
            return {
                "operation": "assumption_challenge",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in assumption challenge: {str(e)}")
            return {
                "error": str(e),
                "operation": "assumption_challenge",
                "input": analysis
            }
    
    def _logical_fallacy_check(self, input_data: Dict) -> Dict:
        """
        Check for logical fallacies in reasoning.
        
        Args:
            input_data: Input data dictionary with analysis content
            
        Returns:
            Detected logical fallacies and their impact
        """
        logger.info("Checking for logical fallacies")
        
        # Get analysis to check
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for logical fallacy check"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get specific fallacies to check if provided
        target_fallacies = input_data.get("target_fallacies", [])
        
        # Construct prompt for fallacy detection
        system_prompt = """You are an expert in logic and critical reasoning. 
        Your task is to carefully analyze the provided content for logical fallacies and reasoning errors, 
        identifying specific instances where fallacious reasoning may be undermining the analysis."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        prompt += "LOGICAL FALLACIES TO DETECT:\n"
        if target_fallacies:
            # Only include specified fallacies
            for fallacy in target_fallacies:
                if fallacy in self.logical_fallacies:
                    prompt += f"- {fallacy}: {self.logical_fallacies[fallacy]}\n"
        else:
            # Include all fallacies
            for fallacy, description in self.logical_fallacies.items():
                prompt += f"- {fallacy}: {description}\n"
        
        prompt += """\nPlease carefully analyze the content for evidence of these logical fallacies. 
        Structure your response as JSON with the following fields:
        - overall_assessment: Overall assessment of logical reasoning in the analysis
        - detected_fallacies: Array of detected fallacies, each with:
          - fallacy_type: Type of logical fallacy detected
          - evidence: Specific evidence or examples from the text
          - impact: How this fallacy might impact the analysis or conclusions
          - severity: Severity of the fallacy (low, medium, high)
        - reasoning_strengths: Areas where the reasoning is particularly strong
        - reasoning_weaknesses: General weaknesses in reasoning beyond specific fallacies
        - improvement_recommendations: Specific recommendations to improve logical reasoning"""
        
        # Call LLM for fallacy detection
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
            if target_fallacies:
                parsed_result["target_fallacies"] = target_fallacies
            
            return {
                "operation": "logical_fallacy_check",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in logical fallacy check: {str(e)}")
            return {
                "error": str(e),
                "operation": "logical_fallacy_check",
                "input": analysis
            }
    
    def _alternative_hypothesis(self, input_data: Dict) -> Dict:
        """
        Generate and evaluate alternative hypotheses.
        
        Args:
            input_data: Input data dictionary with analysis and current hypothesis
            
        Returns:
            Alternative hypotheses and their evaluation
        """
        logger.info("Generating alternative hypotheses")
        
        # Get analysis and current hypothesis
        analysis = input_data.get("analysis", "")
        current_hypothesis = input_data.get("current_hypothesis", "")
        
        if not analysis and not current_hypothesis:
            error_msg = "Neither analysis nor current hypothesis provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get number of alternatives to generate
        num_alternatives = input_data.get("num_alternatives", 3)
        
        # Construct prompt for alternative hypothesis generation
        system_prompt = """You are an expert in hypothesis generation and evaluation. 
        Your task is to generate alternative hypotheses that could explain the same evidence or address 
        the same question as the current analysis, evaluating their relative strengths and weaknesses."""
        
        prompt = ""
        if analysis:
            prompt += f"CURRENT ANALYSIS:\n{analysis}\n\n"
        
        if current_hypothesis:
            prompt += f"CURRENT HYPOTHESIS:\n{current_hypothesis}\n\n"
        
        prompt += f"Please generate {num_alternatives} alternative hypotheses that could explain the same evidence or address the same question. "
        prompt += """Structure your response as JSON with the following fields:
        - current_hypothesis_summary: Summary of the current hypothesis or main conclusion
        - alternative_hypotheses: Array of alternative hypotheses, each with:
          - hypothesis: Clear statement of the alternative hypothesis
          - supporting_evidence: Evidence that supports this alternative
          - contradicting_evidence: Evidence that contradicts this alternative
          - explanatory_power: How well this alternative explains the available evidence
          - parsimony: How simple or complex this alternative is
          - testability: How easily this alternative could be tested
          - plausibility: Overall plausibility assessment (low, medium, high)
        - comparative_analysis: How the alternatives compare to the current hypothesis
        - key_discriminating_evidence: Evidence that would help distinguish between hypotheses
        - recommendations: Recommendations for hypothesis evaluation and testing"""
        
        # Call LLM for alternative hypothesis generation
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.4  # Higher temperature for creative alternatives
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            parsed_result["num_alternatives_requested"] = num_alternatives
            
            return {
                "operation": "alternative_hypothesis",
                "input": current_hypothesis or analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in alternative hypothesis generation: {str(e)}")
            return {
                "error": str(e),
                "operation": "alternative_hypothesis",
                "input": current_hypothesis or analysis
            }
    
    def _red_team_analysis(self, input_data: Dict) -> Dict:
        """
        Perform adversarial red team analysis.
        
        Args:
            input_data: Input data dictionary with analysis to challenge
            
        Returns:
            Red team critique and recommendations
        """
        logger.info("Performing red team analysis")
        
        # Get analysis to challenge
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for red team challenge"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get red team focus areas if provided
        focus_areas = input_data.get("focus_areas", [])
        
        # Construct prompt for red team analysis
        system_prompt = """You are an expert red team analyst specializing in adversarial analysis. 
        Your task is to critically challenge the provided analysis from multiple adversarial perspectives, 
        identifying weaknesses, blind spots, and vulnerabilities in the reasoning and conclusions."""
        
        prompt = f"ANALYSIS TO CHALLENGE:\n{analysis}\n\n"
        
        if focus_areas:
            prompt += "FOCUS AREAS FOR RED TEAM CHALLENGE:\n"
            for area in focus_areas:
                prompt += f"- {area}\n"
            prompt += "\n"
        
        prompt += """As a red team, your job is to challenge this analysis as strongly as possible. 
        Structure your response as JSON with the following fields:
        - red_team_assessment: Overall assessment of vulnerabilities in the analysis
        - key_vulnerabilities: Array of key vulnerabilities, each with:
          - vulnerability: Description of the vulnerability
          - evidence: Evidence from the analysis that indicates this vulnerability
          - impact: How this vulnerability could undermine conclusions
          - severity: Severity of the vulnerability (low, medium, high)
        - blind_spots: Important considerations or perspectives missing from the analysis
        - adversarial_scenarios: Scenarios where the analysis could lead to incorrect conclusions
        - stress_test_results: How the analysis holds up under various stress tests
        - strengthening_recommendations: Specific recommendations to strengthen the analysis"""
        
        # Call LLM for red team analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.3
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            if focus_areas:
                parsed_result["focus_areas"] = focus_areas
            
            return {
                "operation": "red_team_analysis",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in red team analysis: {str(e)}")
            return {
                "error": str(e),
                "operation": "red_team_analysis",
                "input": analysis
            }
    
    def _premortem_analysis(self, input_data: Dict) -> Dict:
        """
        Perform premortem analysis to identify potential failure modes.
        
        Args:
            input_data: Input data dictionary with analysis and conclusion
            
        Returns:
            Premortem analysis identifying potential failure modes
        """
        logger.info("Performing premortem analysis")
        
        # Get analysis and conclusion
        analysis = input_data.get("analysis", "")
        conclusion = input_data.get("conclusion", "")
        
        if not analysis and not conclusion:
            error_msg = "Neither analysis nor conclusion provided for premortem"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Construct prompt for premortem analysis
        system_prompt = """You are an expert in premortem analysis, a technique where you imagine a future 
        where a conclusion or decision has failed, and then work backward to determine what could have led to the failure. 
        Your task is to identify potential failure modes, warning signs, and preventive measures."""
        
        prompt = ""
        if analysis:
            prompt += f"CURRENT ANALYSIS:\n{analysis}\n\n"
        
        if conclusion:
            prompt += f"CONCLUSION/DECISION:\n{conclusion}\n\n"
        
        prompt += """Imagine that this analysis or conclusion has failed spectacularly. 
        It's one year in the future, and the conclusion has been proven completely wrong or the decision has led to disaster. 
        
        Structure your response as JSON with the following fields:
        - premortem_scenario: Description of the failure scenario
        - failure_modes: Array of potential failure modes, each with:
          - failure_mode: Description of how and why the analysis/conclusion failed
          - early_warning_signs: Indicators that might have predicted this failure
          - contributing_factors: Factors that contributed to this failure mode
          - likelihood: Likelihood of this failure mode (low, medium, high)
          - impact: Severity of impact if this failure occurs (low, medium, high)
        - common_vulnerabilities: Vulnerabilities that appear across multiple failure modes
        - preventive_measures: Specific measures that could prevent these failures
        - monitoring_recommendations: What should be monitored to detect early signs of failure
        - contingency_recommendations: How to respond if failure begins to occur"""
        
        # Call LLM for premortem analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.3
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "premortem_analysis",
                "input": conclusion or analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in premortem analysis: {str(e)}")
            return {
                "error": str(e),
                "operation": "premortem_analysis",
                "input": conclusion or analysis
            }
