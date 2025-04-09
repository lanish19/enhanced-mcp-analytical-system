"""
Bias Detection Technique implementation.
This module provides the BiasDetectionTechnique class for identifying cognitive and analytical biases.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional

from .analytical_technique import AnalyticalTechnique
from utils.llm_integration import call_llm, extract_content, parse_json_response, MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiasDetectionTechnique(AnalyticalTechnique):
    """
    Identifies cognitive and analytical biases in reasoning and analysis.
    
    This technique systematically examines analysis for common cognitive biases,
    blind spots, and flawed reasoning patterns that might distort conclusions.
    """
    
    def execute(self, context, parameters):
        """
        Execute the bias detection technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing bias detection results
        """
        logger.info(f"Executing BiasDetectionTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        target_techniques = parameters.get("target_techniques", [])
        bias_categories = parameters.get("bias_categories", ["confirmation", "anchoring", "availability", "groupthink"])
        
        # Step 1: Collect analysis to examine
        analysis_content = self._collect_analysis(context, target_techniques)
        
        # Step 2: Detect potential biases
        potential_biases = self._detect_biases(context.question, analysis_content, bias_categories)
        
        # Step 3: Assess bias impact
        bias_impact = self._assess_bias_impact(context.question, potential_biases)
        
        # Step 4: Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(potential_biases, bias_impact)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, potential_biases, bias_impact, mitigation_strategies)
        
        return {
            "technique": "Bias Detection",
            "status": "Completed",
            "analysis_examined": list(analysis_content.keys()),
            "potential_biases": potential_biases,
            "bias_impact": bias_impact,
            "mitigation_strategies": mitigation_strategies,
            "final_judgment": synthesis.get("final_judgment", "No judgment provided"),
            "judgment_rationale": synthesis.get("judgment_rationale", "No rationale provided"),
            "confidence_level": synthesis.get("confidence_level", "Medium"),
            "potential_biases": synthesis.get("potential_biases", [])
        }
    
    def get_required_mcps(self):
        """
        Return list of MCPs that enhance this technique.
        
        Returns:
            List of MCP names that enhance this technique
        """
        return ["cognitive_bias_mcp", "meta_analysis_mcp"]
    
    def _collect_analysis(self, context, target_techniques):
        """
        Collect analysis content to examine for biases.
        
        Args:
            context: The analysis context
            target_techniques: List of technique names to target
            
        Returns:
            Dictionary mapping technique names to their content
        """
        logger.info(f"Collecting analysis content from {len(target_techniques) if target_techniques else 'all'} techniques...")
        
        analysis_content = {}
        
        # If no specific techniques are specified, include all available results
        if not target_techniques:
            for technique, result in context.results.items():
                analysis_content[technique] = result
        else:
            # Include only specified techniques
            for technique in target_techniques:
                if technique in context.results:
                    analysis_content[technique] = context.results[technique]
                else:
                    logger.warning(f"Technique {technique} not found in results")
        
        logger.info(f"Collected analysis content from {len(analysis_content)} techniques")
        return analysis_content
    
    def _detect_biases(self, question, analysis_content, bias_categories):
        """
        Detect potential biases in the analysis content.
        
        Args:
            question: The analytical question
            analysis_content: Dictionary mapping technique names to their content
            bias_categories: List of bias categories to check for
            
        Returns:
            Dictionary mapping technique names to detected biases
        """
        logger.info(f"Detecting potential biases in {len(analysis_content)} techniques...")
        
        # Use cognitive bias MCP if available
        bias_mcp = self.mcp_registry.get_mcp("cognitive_bias_mcp")
        
        if bias_mcp:
            try:
                logger.info("Using cognitive bias MCP")
                potential_biases = bias_mcp.detect_biases(question, analysis_content, bias_categories)
                return potential_biases
            except Exception as e:
                logger.error(f"Error using cognitive bias MCP: {e}")
                # Fall through to LLM-based detection
        
        potential_biases = {}
        
        for technique, content in analysis_content.items():
            logger.info(f"Detecting biases in technique: {technique}...")
            
            # Extract key fields that are likely to contain biases
            bias_fields = [
                "final_judgment",
                "judgment_rationale",
                "key_findings",
                "key_insights",
                "key_drivers",
                "confidence_level"
            ]
            
            # Prepare a summary of the technique results
            result_summary = {}
            for field in bias_fields:
                if field in content:
                    result_summary[field] = content[field]
            
            # If no key fields found, use the entire content
            if not result_summary:
                result_summary = content
            
            # Use LLM to detect biases
            prompt = f"""
            Detect potential cognitive and analytical biases in the following analysis:
            
            Question: "{question}"
            
            Technique: {technique}
            
            Analysis Content: {json.dumps(result_summary, indent=2)}
            
            Check for the following bias categories:
            {", ".join(bias_categories)}
            
            For each potential bias:
            1. Identify the specific type of bias
            2. Provide evidence from the analysis that suggests this bias
            3. Explain how this bias might be distorting the analysis
            4. Rate the confidence in this bias detection (High/Medium/Low)
            
            Common cognitive biases include:
            - Confirmation bias: Seeking or interpreting evidence in ways that confirm existing beliefs
            - Anchoring bias: Over-relying on the first piece of information encountered
            - Availability bias: Overestimating the likelihood of events based on how easily examples come to mind
            - Groupthink: Tendency for groups to reach consensus without critical evaluation
            - Status quo bias: Preference for the current state of affairs
            - Hindsight bias: Tendency to see past events as having been predictable
            - Overconfidence bias: Excessive confidence in one's abilities or judgments
            - Framing effect: Drawing different conclusions based on how information is presented
            
            Return your analysis as a JSON object with the following structure:
            {{
                "detected_biases": [
                    {{
                        "bias_type": "Name of the bias",
                        "bias_category": "Category of the bias",
                        "evidence": "Evidence from the analysis that suggests this bias",
                        "potential_distortion": "How this bias might be distorting the analysis",
                        "detection_confidence": "High/Medium/Low"
                    }},
                    ...
                ]
            }}
            """
            
            model_config = MODEL_CONFIG["sonar"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error detecting biases: {parsed_response.get('error')}")
                    potential_biases[technique] = {"detected_biases": []}
                else:
                    potential_biases[technique] = parsed_response
            
            except Exception as e:
                logger.error(f"Error parsing bias detection: {e}")
                potential_biases[technique] = {"detected_biases": []}
        
        return potential_biases
    
    def _assess_bias_impact(self, question, potential_biases):
        """
        Assess the impact of detected biases on the analysis.
        
        Args:
            question: The analytical question
            potential_biases: Dictionary mapping technique names to detected biases
            
        Returns:
            Dictionary containing bias impact assessment
        """
        logger.info("Assessing bias impact...")
        
        # Use meta analysis MCP if available
        meta_mcp = self.mcp_registry.get_mcp("meta_analysis_mcp")
        
        if meta_mcp:
            try:
                logger.info("Using meta analysis MCP")
                bias_impact = meta_mcp.assess_bias_impact(question, potential_biases)
                return bias_impact
            except Exception as e:
                logger.error(f"Error using meta analysis MCP: {e}")
                # Fall through to LLM-based assessment
        
        # Flatten all biases into a single list
        all_biases = []
        for technique, biases in potential_biases.items():
            for bias in biases.get("detected_biases", []):
                all_biases.append({
                    "technique": technique,
                    "bias_type": bias.get("bias_type", ""),
                    "bias_category": bias.get("bias_category", ""),
                    "evidence": bias.get("evidence", ""),
                    "potential_distortion": bias.get("potential_distortion", ""),
                    "detection_confidence": bias.get("detection_confidence", "Medium")
                })
        
        # Use LLM to assess bias impact
        prompt = f"""
        Assess the impact of the following detected biases on the analysis of this question:
        
        Question: "{question}"
        
        Detected Biases:
        {json.dumps(all_biases, indent=2)}
        
        For this assessment:
        1. Identify the most significant biases that are likely affecting the analysis
        2. Assess how these biases might be influencing the overall conclusions
        3. Evaluate whether certain techniques are more affected by biases than others
        4. Determine the overall level of bias in the analysis
        
        Return your assessment as a JSON object with the following structure:
        {{
            "most_significant_biases": [
                {{
                    "bias_type": "Name of the bias",
                    "techniques_affected": ["Technique 1", "Technique 2", ...],
                    "significance": "High/Medium/Low",
                    "rationale": "Explanation of why this bias is significant"
                }},
                ...
            ],
            "conclusion_impact": "Description of how biases might be influencing overall conclusions",
            "technique_vulnerability": [
                {{
                    "technique": "Name of the technique",
                    "vulnerability_level": "High/Medium/Low",
                    "explanation": "Explanation of why this technique is vulnerable to bias"
                }},
                ...
            ],
            "overall_bias_level": "High/Medium/Low",
            "overall_assessment": "Overall assessment of how biases are affecting the analysis"
        }}
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error assessing bias impact: {parsed_response.get('error')}")
                return self._generate_fallback_bias_impact(all_biases)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing bias impact assessment: {e}")
            return self._generate_fallback_bias_impact(all_biases)
    
    def _generate_fallback_bias_impact(self, all_biases):
        """
        Generate fallback bias impact assessment when normal assessment fails.
        
        Args:
            all_biases: List of all bias dictionaries
            
        Returns:
            Dictionary containing fallback bias impact assessment
        """
        # Extract unique bias types and techniques
        bias_types = list(set([bias.get("bias_type", f"Bias {i+1}") for i, bias in enumerate(all_biases)]))
        techniques = list(set([bias.get("technique", f"Technique {i+1}") for i, bias in enumerate(all_biases)]))
        
        # Select a few biases to highlight as most significant
        most_significant = bias_types[:3] if len(bias_types) >= 3 else bias_types
        
        return {
            "most_significant_biases": [
                {
                    "bias_type": bias_type,
                    "techniques_affected": techniques[:2],
                    "significance": "Medium",
                    "rationale": f"This bias appears consistently across multiple techniques and could significantly distort conclusions."
                } for bias_type in most_significant
            ],
            "conclusion_impact": "The detected biases may be leading to overconfidence in certain conclusions and insufficient consideration of alternative perspectives. The analysis might be skewed toward confirming initial hypotheses rather than objectively evaluating all evidence.",
            "technique_vulnerability": [
                {
                    "technique": technique,
                    "vulnerability_level": "Medium",
                    "explanation": "This technique relies heavily on subjective judgment and interpretation, making it particularly vulnerable to cognitive biases."
                } for technique in techniques[:3]
            ],
            "overall_bias_level": "Medium",
            "overall_assessment": "While the analysis contains valuable insights, there are several cognitive biases that may be distorting the conclusions. These biases should be addressed through targeted mitigation strategies to improve the objectivity and reliability of the analysis."
        }
    
    def _generate_mitigation_strategies(self, potential_biases, bias_impact):
        """
        Generate strategies to mitigate detected biases.
        
        Args:
            potential_biases: Dictionary mapping technique names to detected biases
            bias_impact: Dictionary containing bias impact assessment
            
        Returns:
            Dictionary containing mitigation strategies
        """
        logger.info("Generating bias mitigation strategies...")
        
        # Use cognitive bias MCP if available
        bias_mcp = self.mcp_registry.get_mcp("cognitive_bias_mcp")
        
        if bias_mcp:
            try:
                logger.info("Using cognitive bias MCP")
                mitigation_strategies = bias_mcp.generate_mitigation_strategies(potential_biases, bias_impact)
                return mitigation_strategies
            except Exception as e:
                logger.error(f"Error using cognitive bias MCP: {e}")
                # Fall through to LLM-based generation
        
        # Extract most significant biases from impact assessment
        most_significant = bias_impact.get("most_significant_biases", [])
        
        # Use LLM to generate mitigation strategies
        prompt = f"""
        Generate strategies to mitigate the following biases in the analysis:
        
        Most Significant Biases:
        {json.dumps(most_significant, indent=2)}
        
        Overall Bias Level: {bias_impact.get("overall_bias_level", "Medium")}
        
        Overall Assessment: {bias_impact.get("overall_assessment", "")}
        
        For each significant bias:
        1. Develop 2-3 specific strategies to mitigate its impact
        2. Explain how each strategy addresses the bias
        3. Assess the potential effectiveness of each strategy
        
        Also provide:
        1. General strategies to improve overall analytical objectivity
        2. Recommendations for analytical process improvements
        
        Return your strategies as a JSON object with the following structure:
        {{
            "bias_specific_strategies": [
                {{
                    "bias_type": "Name of the bias",
                    "strategies": [
                        {{
                            "strategy": "Description of the strategy",
                            "implementation": "How to implement this strategy",
                            "effectiveness": "High/Medium/Low",
                            "rationale": "Why this strategy would be effective"
                        }},
                        ...
                    ]
                }},
                ...
            ],
            "general_strategies": [
                {{
                    "strategy": "Description of the general strategy",
                    "target_area": "Area of analysis this strategy targets",
                    "implementation": "How to implement this strategy"
                }},
                ...
            ],
            "process_improvements": [
                {{
                    "improvement": "Description of the process improvement",
                    "implementation": "How to implement this improvement",
                    "expected_benefit": "Expected benefit of this improvement"
                }},
                ...
            ]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating mitigation strategies: {parsed_response.get('error')}")
                return self._generate_fallback_mitigation_strategies(most_significant)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing mitigation strategies: {e}")
            return self._generate_fallback_mitigation_strategies(most_significant)
    
    def _generate_fallback_mitigation_strategies(self, most_significant):
        """
        Generate fallback mitigation strategies when normal generation fails.
        
        Args:
            most_significant: List of most significant bias dictionaries
            
        Returns:
            Dictionary containing fallback mitigation strategies
        """
        # Extract bias types
        bias_types = [bias.get("bias_type", f"Bias {i+1}") for i, bias in enumerate(most_significant)]
        
        # Ensure we have at least one bias type
        if not bias_types:
            bias_types = ["Confirmation bias", "Anchoring bias", "Availability bias"]
        
        return {
            "bias_specific_strategies": [
                {
                    "bias_type": bias_type,
                    "strategies": [
                        {
                            "strategy": "Structured devil's advocate approach",
                            "implementation": "Assign a team member or process step specifically to challenge the prevailing view",
                            "effectiveness": "High",
                            "rationale": "Directly counters the tendency to favor confirming evidence by institutionalizing dissent"
                        },
                        {
                            "strategy": "Pre-commitment to evaluation criteria",
                            "implementation": "Define evaluation criteria before seeing the evidence or results",
                            "effectiveness": "Medium",
                            "rationale": "Reduces the ability to selectively interpret evidence by establishing objective standards in advance"
                        }
                    ]
                } for bias_type in bias_types
            ],
            "general_strategies": [
                {
                    "strategy": "Blind analysis techniques",
                    "target_area": "Data interpretation",
                    "implementation": "Analyze data without knowing which hypothesis it relates to until after analysis is complete"
                },
                {
                    "strategy": "Diverse analytical team",
                    "target_area": "Team composition",
                    "implementation": "Ensure analytical teams include members with diverse backgrounds, expertise, and viewpoints"
                },
                {
                    "strategy": "Explicit uncertainty quantification",
                    "target_area": "Confidence assessment",
                    "implementation": "Require explicit reasoning about uncertainty ranges and confidence levels for all key judgments"
                }
            ],
            "process_improvements": [
                {
                    "improvement": "Structured analytical technique rotation",
                    "implementation": "Systematically apply multiple analytical techniques to the same question",
                    "expected_benefit": "Reduces the impact of technique-specific biases and provides multiple perspectives"
                },
                {
                    "improvement": "Mandatory reflection periods",
                    "implementation": "Build in mandatory pauses for reflection before finalizing conclusions",
                    "expected_benefit": "Allows time to recognize and correct for biases before they become embedded in final judgments"
                },
                {
                    "improvement": "Regular bias awareness training",
                    "implementation": "Conduct regular training on cognitive bias recognition and mitigation",
                    "expected_benefit": "Increases analysts' ability to self-monitor for biases and apply appropriate countermeasures"
                }
            ]
        }
    
    def _generate_synthesis(self, question, potential_biases, bias_impact, mitigation_strategies):
        """
        Generate a synthesis of the bias detection.
        
        Args:
            question: The analytical question
            potential_biases: Dictionary mapping technique names to detected biases
            bias_impact: Dictionary containing bias impact assessment
            mitigation_strategies: Dictionary containing mitigation strategies
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of bias detection...")
        
        prompt = f"""
        Synthesize the following bias detection analysis for the question:
        
        "{question}"
        
        Overall Bias Level: {bias_impact.get("overall_bias_level", "Medium")}
        
        Most Significant Biases:
        {json.dumps([b.get("bias_type", "") for b in bias_impact.get("most_significant_biases", [])], indent=2)}
        
        Key Mitigation Strategies:
        {json.dumps([{
            "bias_type": s.get("bias_type", ""),
            "strategies": [strategy.get("strategy", "") for strategy in s.get("strategies", [])]
        } for s in mitigation_strategies.get("bias_specific_strategies", [])], indent=2)}
        
        Based on this bias detection:
        1. How reliable is the current analysis given the detected biases?
        2. Which conclusions are most likely to be distorted by biases?
        3. How effectively can the biases be mitigated?
        4. What is the appropriate level of confidence in the analysis?
        
        Provide:
        1. A final judgment that addresses the original question in light of bias detection
        2. A rationale for this judgment
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this meta-analysis itself
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question in light of bias detection",
            "judgment_rationale": "Explanation for your judgment",
            "reliability_assessment": "Assessment of the reliability of the current analysis",
            "most_distorted_conclusions": ["Conclusion 1", "Conclusion 2", ...],
            "mitigation_effectiveness": "Assessment of how effectively biases can be mitigated",
            "confidence_level": "High/Medium/Low",
            "potential_biases": ["Bias 1", "Bias 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating synthesis: {parsed_response.get('error')}")
                return {
                    "final_judgment": "Error generating synthesis",
                    "judgment_rationale": parsed_response.get('error', "Unknown error"),
                    "reliability_assessment": "Unable to assess reliability due to synthesis error",
                    "most_distorted_conclusions": ["Error in synthesis generation"],
                    "mitigation_effectiveness": "Unable to assess mitigation effectiveness due to synthesis error",
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "reliability_assessment": "Unable to assess reliability due to synthesis error",
                "most_distorted_conclusions": ["Error in synthesis generation"],
                "mitigation_effectiveness": "Unable to assess mitigation effectiveness due to synthesis error",
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
