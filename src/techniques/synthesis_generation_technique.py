"""
Synthesis Generation Technique implementation.
This module provides the SynthesisGenerationTechnique class for integrating multiple analyses.
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

class SynthesisGenerationTechnique(AnalyticalTechnique):
    """
    Integrates multiple analyses into a coherent synthesis.
    
    This technique combines insights from multiple analytical techniques,
    resolves contradictions, identifies patterns, and generates a comprehensive
    synthesis that addresses the original question.
    """
    
    def execute(self, context, parameters):
        """
        Execute the synthesis generation technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing synthesis generation results
        """
        logger.info(f"Executing SynthesisGenerationTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        techniques_to_include = parameters.get("techniques_to_include", [])
        synthesis_focus = parameters.get("synthesis_focus", "comprehensive")  # comprehensive, contradictions, patterns
        
        # Step 1: Collect results from specified techniques
        technique_results = self._collect_technique_results(context, techniques_to_include)
        
        # Step 2: Identify key insights from each technique
        key_insights = self._identify_key_insights(technique_results)
        
        # Step 3: Identify patterns and contradictions
        patterns_contradictions = self._identify_patterns_contradictions(key_insights)
        
        # Step 4: Generate integrated synthesis
        integrated_synthesis = self._generate_integrated_synthesis(
            context.question, 
            key_insights, 
            patterns_contradictions, 
            synthesis_focus
        )
        
        # Step 5: Generate final assessment
        final_assessment = self._generate_final_assessment(
            context.question, 
            integrated_synthesis
        )
        
        return {
            "technique": "Synthesis Generation",
            "status": "Completed",
            "techniques_included": list(technique_results.keys()),
            "key_insights": key_insights,
            "patterns_contradictions": patterns_contradictions,
            "integrated_synthesis": integrated_synthesis,
            "final_judgment": final_assessment.get("final_judgment", "No judgment provided"),
            "judgment_rationale": final_assessment.get("judgment_rationale", "No rationale provided"),
            "confidence_level": final_assessment.get("confidence_level", "Medium"),
            "potential_biases": final_assessment.get("potential_biases", [])
        }
    
    def get_required_mcps(self):
        """
        Return list of MCPs that enhance this technique.
        
        Returns:
            List of MCP names that enhance this technique
        """
        return ["synthesis_mcp"]
    
    def _collect_technique_results(self, context, techniques_to_include):
        """
        Collect results from specified techniques.
        
        Args:
            context: The analysis context
            techniques_to_include: List of technique names to include
            
        Returns:
            Dictionary mapping technique names to their results
        """
        logger.info(f"Collecting results from techniques: {techniques_to_include if techniques_to_include else 'all'}")
        
        technique_results = {}
        
        # If no specific techniques are specified, include all available results
        if not techniques_to_include:
            technique_results = context.results
        else:
            # Include only specified techniques
            for technique in techniques_to_include:
                if technique in context.results:
                    technique_results[technique] = context.results[technique]
                else:
                    logger.warning(f"Technique {technique} not found in results")
        
        logger.info(f"Collected results from {len(technique_results)} techniques")
        return technique_results
    
    def _identify_key_insights(self, technique_results):
        """
        Identify key insights from each technique.
        
        Args:
            technique_results: Dictionary mapping technique names to their results
            
        Returns:
            Dictionary mapping technique names to their key insights
        """
        logger.info(f"Identifying key insights from {len(technique_results)} techniques")
        
        # Use synthesis MCP if available
        synthesis_mcp = self.mcp_registry.get_mcp("synthesis_mcp")
        
        if synthesis_mcp:
            try:
                logger.info("Using synthesis MCP")
                key_insights = synthesis_mcp.extract_key_insights(technique_results)
                return key_insights
            except Exception as e:
                logger.error(f"Error using synthesis MCP: {e}")
                # Fall through to LLM-based extraction
        
        key_insights = {}
        
        for technique, results in technique_results.items():
            logger.info(f"Extracting key insights from technique: {technique}")
            
            # Extract key fields that are likely to contain insights
            insight_fields = [
                "final_judgment",
                "judgment_rationale",
                "key_findings",
                "key_drivers",
                "key_pathways",
                "significant_vulnerabilities",
                "critical_risks",
                "critical_uncertainties"
            ]
            
            # Prepare a summary of the technique results
            result_summary = {}
            for field in insight_fields:
                if field in results:
                    result_summary[field] = results[field]
            
            # If no key fields found, use the entire results
            if not result_summary:
                result_summary = results
            
            prompt = f"""
            Extract the key insights from the following analytical technique results:
            
            Technique: {technique}
            
            Results: {json.dumps(result_summary, indent=2)}
            
            Identify 3-5 key insights that:
            1. Represent the most important findings or conclusions
            2. Capture unique contributions from this specific technique
            3. Would be valuable to include in an integrated synthesis
            
            For each insight:
            1. Provide a clear statement of the insight
            2. Indicate the confidence level associated with this insight (High/Medium/Low)
            3. Note any important caveats or limitations
            
            Return your response as a JSON object with the following structure:
            {{
                "insights": [
                    {{
                        "statement": "Clear statement of the insight",
                        "confidence": "High/Medium/Low",
                        "caveats": ["Caveat 1", "Caveat 2", ...]
                    }},
                    ...
                ]
            }}
            """
            
            model_config = MODEL_CONFIG["llama4"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error extracting insights: {parsed_response.get('error')}")
                    key_insights[technique] = self._generate_fallback_insights(technique)
                else:
                    key_insights[technique] = parsed_response.get("insights", [])
            
            except Exception as e:
                logger.error(f"Error parsing insights: {e}")
                key_insights[technique] = self._generate_fallback_insights(technique)
        
        return key_insights
    
    def _generate_fallback_insights(self, technique):
        """
        Generate fallback insights when extraction fails.
        
        Args:
            technique: Name of the technique
            
        Returns:
            List of fallback insight dictionaries
        """
        return [
            {
                "statement": f"Key insight from {technique} (fallback due to extraction error)",
                "confidence": "Low",
                "caveats": ["Generated as fallback due to extraction error"]
            }
        ]
    
    def _identify_patterns_contradictions(self, key_insights):
        """
        Identify patterns and contradictions across insights.
        
        Args:
            key_insights: Dictionary mapping technique names to their key insights
            
        Returns:
            Dictionary containing patterns and contradictions
        """
        logger.info("Identifying patterns and contradictions across insights")
        
        # Flatten all insights into a single list
        all_insights = []
        for technique, insights in key_insights.items():
            for insight in insights:
                insight_copy = insight.copy()
                insight_copy["technique"] = technique
                all_insights.append(insight_copy)
        
        # Use LLM to identify patterns and contradictions
        prompt = f"""
        Identify patterns and contradictions in the following insights from multiple analytical techniques:
        
        Insights:
        {json.dumps(all_insights, indent=2)}
        
        For this analysis:
        1. Identify common themes or patterns that appear across multiple techniques
        2. Identify contradictions or tensions between insights from different techniques
        3. Note insights that are unique to a single technique but particularly important
        
        Return your response as a JSON object with the following structure:
        {{
            "patterns": [
                {{
                    "theme": "Name of the pattern or theme",
                    "description": "Description of this pattern",
                    "supporting_insights": ["Insight 1", "Insight 2", ...],
                    "techniques_involved": ["Technique 1", "Technique 2", ...]
                }},
                ...
            ],
            "contradictions": [
                {{
                    "description": "Description of the contradiction",
                    "insight_1": "First contradictory insight",
                    "technique_1": "Technique for first insight",
                    "insight_2": "Second contradictory insight",
                    "technique_2": "Technique for second insight",
                    "potential_resolution": "Potential way to resolve this contradiction"
                }},
                ...
            ],
            "unique_insights": [
                {{
                    "insight": "The unique insight",
                    "technique": "Source technique",
                    "importance": "Explanation of why this unique insight is important"
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
                logger.error(f"Error identifying patterns and contradictions: {parsed_response.get('error')}")
                return {
                    "patterns": [],
                    "contradictions": [],
                    "unique_insights": []
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing patterns and contradictions: {e}")
            return {
                "patterns": [],
                "contradictions": [],
                "unique_insights": []
            }
    
    def _generate_integrated_synthesis(self, question, key_insights, patterns_contradictions, synthesis_focus):
        """
        Generate an integrated synthesis of all insights.
        
        Args:
            question: The analytical question
            key_insights: Dictionary mapping technique names to their key insights
            patterns_contradictions: Dictionary containing patterns and contradictions
            synthesis_focus: Focus of the synthesis (comprehensive, contradictions, patterns)
            
        Returns:
            Dictionary containing the integrated synthesis
        """
        logger.info(f"Generating integrated synthesis with focus: {synthesis_focus}")
        
        # Use synthesis MCP if available
        synthesis_mcp = self.mcp_registry.get_mcp("synthesis_mcp")
        
        if synthesis_mcp:
            try:
                logger.info("Using synthesis MCP")
                integrated_synthesis = synthesis_mcp.generate_synthesis(
                    question, key_insights, patterns_contradictions, synthesis_focus
                )
                return integrated_synthesis
            except Exception as e:
                logger.error(f"Error using synthesis MCP: {e}")
                # Fall through to LLM-based generation
        
        # Prepare focus-specific instructions
        focus_instructions = ""
        if synthesis_focus == "comprehensive":
            focus_instructions = """
            Generate a comprehensive synthesis that:
            1. Integrates insights from all techniques
            2. Addresses patterns, contradictions, and unique insights
            3. Provides a holistic answer to the original question
            """
        elif synthesis_focus == "contradictions":
            focus_instructions = """
            Generate a synthesis that focuses on resolving contradictions:
            1. Highlight the key contradictions between different techniques
            2. Analyze the potential reasons for these contradictions
            3. Propose resolutions or ways to reconcile these contradictions
            4. Explain how these resolutions affect the answer to the original question
            """
        elif synthesis_focus == "patterns":
            focus_instructions = """
            Generate a synthesis that focuses on patterns and themes:
            1. Highlight the key patterns and themes across different techniques
            2. Analyze what these patterns reveal about the question
            3. Explain how these patterns strengthen confidence in certain conclusions
            4. Use these patterns to provide a robust answer to the original question
            """
        
        # Use LLM to generate integrated synthesis
        prompt = f"""
        Generate an integrated synthesis for the following analytical question:
        
        "{question}"
        
        Key Insights by Technique:
        {json.dumps(key_insights, indent=2)}
        
        Patterns and Contradictions:
        {json.dumps(patterns_contradictions, indent=2)}
        
        {focus_instructions}
        
        Return your synthesis as a JSON object with the following structure:
        {{
            "integrated_answer": "Comprehensive answer to the original question",
            "key_themes": ["Theme 1", "Theme 2", ...],
            "resolved_contradictions": [
                {{
                    "contradiction": "Description of the contradiction",
                    "resolution": "How this contradiction was resolved"
                }},
                ...
            ],
            "confidence_assessment": {{
                "overall_confidence": "High/Medium/Low",
                "high_confidence_areas": ["Area 1", "Area 2", ...],
                "low_confidence_areas": ["Area 1", "Area 2", ...]
            }},
            "remaining_uncertainties": ["Uncertainty 1", "Uncertainty 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating integrated synthesis: {parsed_response.get('error')}")
                return {
                    "integrated_answer": f"Error generating synthesis: {parsed_response.get('error', 'Unknown error')}",
                    "key_themes": [],
                    "resolved_contradictions": [],
                    "confidence_assessment": {
                        "overall_confidence": "Low",
                        "high_confidence_areas": [],
                        "low_confidence_areas": ["All areas due to synthesis error"]
                    },
                    "remaining_uncertainties": ["Unable to identify uncertainties due to synthesis error"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing integrated synthesis: {e}")
            return {
                "integrated_answer": f"Error generating synthesis: {str(e)}",
                "key_themes": [],
                "resolved_contradictions": [],
                "confidence_assessment": {
                    "overall_confidence": "Low",
                    "high_confidence_areas": [],
                    "low_confidence_areas": ["All areas due to synthesis error"]
                },
                "remaining_uncertainties": ["Unable to identify uncertainties due to synthesis error"]
            }
    
    def _generate_final_assessment(self, question, integrated_synthesis):
        """
        Generate a final assessment based on the integrated synthesis.
        
        Args:
            question: The analytical question
            integrated_synthesis: Dictionary containing the integrated synthesis
            
        Returns:
            Dictionary containing the final assessment
        """
        logger.info("Generating final assessment")
        
        prompt = f"""
        Generate a final assessment for the following analytical question based on the integrated synthesis:
        
        Question: "{question}"
        
        Integrated Synthesis:
        {json.dumps(integrated_synthesis, indent=2)}
        
        Provide:
        1. A final judgment that directly answers the question
        2. A rationale for this judgment based on the integrated synthesis
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        5. Recommendations for further analysis or research
        
        Return your assessment as a JSON object with the following structure:
        {{
            "final_judgment": "Clear answer to the original question",
            "judgment_rationale": "Explanation for your judgment based on the synthesis",
            "confidence_level": "High/Medium/Low",
            "potential_biases": ["Bias 1", "Bias 2", ...],
            "further_research": ["Research recommendation 1", "Research recommendation 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating final assessment: {parsed_response.get('error')}")
                return {
                    "final_judgment": "Error generating final assessment",
                    "judgment_rationale": parsed_response.get('error', "Unknown error"),
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"],
                    "further_research": ["Review the integrated synthesis manually"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing final assessment: {e}")
            return {
                "final_judgment": f"Error generating final assessment: {str(e)}",
                "judgment_rationale": "Error in assessment generation",
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"],
                "further_research": ["Review the integrated synthesis manually"]
            }
