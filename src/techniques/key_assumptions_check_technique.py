"""
Key Assumptions Check Technique implementation.
This module provides the KeyAssumptionsCheckTechnique class for identifying and evaluating key assumptions.
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

class KeyAssumptionsCheckTechnique(AnalyticalTechnique):
    """
    Identifies and evaluates key assumptions underlying an analysis.
    
    This technique identifies implicit and explicit assumptions in the analysis,
    evaluates their validity, and assesses the impact if they are wrong.
    """
    
    def execute(self, context, parameters):
        """
        Execute the key assumptions check technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing key assumptions check results
        """
        logger.info(f"Executing KeyAssumptionsCheckTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        target_analysis = parameters.get("target_analysis", "question")
        target_technique = parameters.get("target_technique", None)
        
        # Step 1: Identify the target content to analyze
        target_content = self._identify_target_content(context, target_analysis, target_technique)
        
        # Step 2: Extract assumptions
        assumptions = self._extract_assumptions(context.question, target_content)
        
        # Step 3: Evaluate assumptions
        evaluated_assumptions = self._evaluate_assumptions(assumptions)
        
        # Step 4: Identify critical assumptions
        critical_assumptions = self._identify_critical_assumptions(evaluated_assumptions)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, evaluated_assumptions, critical_assumptions)
        
        return {
            "technique": "Key Assumptions Check",
            "status": "Completed",
            "target_analysis": target_analysis,
            "target_technique": target_technique,
            "assumptions": evaluated_assumptions,
            "critical_assumptions": critical_assumptions,
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
        return ["assumption_extraction_mcp"]
    
    def _identify_target_content(self, context, target_analysis, target_technique):
        """
        Identify the target content to analyze for assumptions.
        
        Args:
            context: The analysis context
            target_analysis: Type of analysis to target (question, technique, all)
            target_technique: Specific technique to target if target_analysis is 'technique'
            
        Returns:
            Dictionary containing target content
        """
        logger.info(f"Identifying target content for analysis: {target_analysis}")
        
        if target_analysis == "question":
            # Just analyze the question itself
            return {
                "type": "question",
                "content": context.question
            }
        
        elif target_analysis == "technique" and target_technique:
            # Analyze results from a specific technique
            if target_technique in context.results:
                return {
                    "type": "technique",
                    "technique": target_technique,
                    "content": context.results[target_technique]
                }
            else:
                logger.warning(f"Technique {target_technique} not found in results, falling back to question")
                return {
                    "type": "question",
                    "content": context.question
                }
        
        elif target_analysis == "all":
            # Analyze all available results
            all_content = {
                "type": "all",
                "question": context.question,
                "techniques": {}
            }
            
            for technique, result in context.results.items():
                all_content["techniques"][technique] = result
            
            return all_content
        
        else:
            # Default to analyzing the question
            logger.warning(f"Invalid target_analysis: {target_analysis}, falling back to question")
            return {
                "type": "question",
                "content": context.question
            }
    
    def _extract_assumptions(self, question, target_content):
        """
        Extract assumptions from the target content.
        
        Args:
            question: The analytical question
            target_content: Target content to analyze
            
        Returns:
            List of assumption dictionaries
        """
        logger.info("Extracting assumptions...")
        
        # Use assumption extraction MCP if available
        assumption_mcp = self.mcp_registry.get_mcp("assumption_extraction_mcp")
        
        if assumption_mcp:
            try:
                logger.info("Using assumption extraction MCP")
                assumptions = assumption_mcp.extract_assumptions(target_content)
                return assumptions
            except Exception as e:
                logger.error(f"Error using assumption extraction MCP: {e}")
                # Fall through to LLM-based extraction
        
        # Prepare content for LLM-based extraction
        content_description = ""
        
        if target_content["type"] == "question":
            content_description = f"Question: {target_content['content']}"
        
        elif target_content["type"] == "technique":
            content_description = f"""
            Question: {question}
            
            Technique: {target_content['technique']}
            
            Results: {json.dumps(target_content['content'], indent=2)}
            """
        
        elif target_content["type"] == "all":
            content_description = f"Question: {target_content['question']}\n\n"
            
            for technique, result in target_content.get("techniques", {}).items():
                content_description += f"Technique: {technique}\n\n"
                content_description += f"Results: {json.dumps(result, indent=2)}\n\n"
        
        # Use LLM to extract assumptions
        prompt = f"""
        Identify the key assumptions in the following analytical content:
        
        {content_description}
        
        An assumption is a statement that is taken as true without proof or demonstration. 
        Focus on identifying both explicit assumptions (clearly stated) and implicit assumptions 
        (unstated but necessary for the analysis to be valid).
        
        For each assumption:
        1. Clearly state the assumption
        2. Indicate whether it is explicit or implicit
        3. Explain why this is an assumption (what makes it unproven)
        4. Explain why this assumption is important to the analysis
        
        Return your response as a JSON object with the following structure:
        {{
            "assumptions": [
                {{
                    "statement": "Clear statement of the assumption",
                    "type": "Explicit/Implicit",
                    "explanation": "Explanation of why this is an assumption",
                    "importance": "Explanation of why this assumption is important"
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
                logger.error(f"Error extracting assumptions: {parsed_response.get('error')}")
                return self._generate_fallback_assumptions(question)
            
            assumptions = parsed_response.get("assumptions", [])
            
            if not assumptions:
                logger.warning("No assumptions extracted")
                return self._generate_fallback_assumptions(question)
            
            return assumptions
        
        except Exception as e:
            logger.error(f"Error parsing assumptions: {e}")
            return self._generate_fallback_assumptions(question)
    
    def _generate_fallback_assumptions(self, question):
        """
        Generate fallback assumptions when extraction fails.
        
        Args:
            question: The analytical question
            
        Returns:
            List of fallback assumption dictionaries
        """
        return [
            {
                "statement": f"The question '{question[:50]}...' is well-formed and answerable",
                "type": "Implicit",
                "explanation": "The question assumes that there is a clear answer that can be determined",
                "importance": "If this assumption is false, the entire analysis may be invalid"
            },
            {
                "statement": "Current information is sufficient to address the question",
                "type": "Implicit",
                "explanation": "The analysis assumes that available information is adequate",
                "importance": "If critical information is missing, conclusions may be incorrect"
            },
            {
                "statement": "Past patterns will continue into the future",
                "type": "Implicit",
                "explanation": "Many analyses assume continuity rather than disruption",
                "importance": "If major disruptions occur, predictions based on past patterns may fail"
            }
        ]
    
    def _evaluate_assumptions(self, assumptions):
        """
        Evaluate the validity and impact of assumptions.
        
        Args:
            assumptions: List of assumption dictionaries
            
        Returns:
            List of evaluated assumption dictionaries
        """
        logger.info(f"Evaluating {len(assumptions)} assumptions...")
        
        evaluated_assumptions = []
        
        for assumption in assumptions:
            logger.info(f"Evaluating assumption: {assumption.get('statement', '')[:50]}...")
            
            prompt = f"""
            Evaluate the following assumption:
            
            Assumption: {assumption.get('statement', '')}
            Type: {assumption.get('type', 'Implicit')}
            Explanation: {assumption.get('explanation', '')}
            Importance: {assumption.get('importance', '')}
            
            For this evaluation:
            1. Assess the validity of this assumption (High/Medium/Low)
            2. Provide a rationale for your validity assessment
            3. Assess the potential impact if this assumption is wrong (High/Medium/Low)
            4. Provide a rationale for your impact assessment
            5. Suggest how this assumption could be tested or verified
            
            Return your evaluation as a JSON object with the following structure:
            {{
                "validity": "High/Medium/Low",
                "validity_rationale": "Explanation for validity assessment",
                "impact": "High/Medium/Low",
                "impact_rationale": "Explanation for impact assessment",
                "testing_methods": ["Method 1", "Method 2", ...]
            }}
            """
            
            model_config = MODEL_CONFIG["sonar"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error evaluating assumption: {parsed_response.get('error')}")
                    evaluation = {
                        "validity": "Unknown",
                        "validity_rationale": parsed_response.get('error', "Unknown error"),
                        "impact": "Unknown",
                        "impact_rationale": "Error in evaluation",
                        "testing_methods": ["Unable to suggest testing methods due to evaluation error"]
                    }
                else:
                    evaluation = parsed_response
                
                # Combine assumption and evaluation
                evaluated_assumption = assumption.copy()
                evaluated_assumption.update(evaluation)
                evaluated_assumptions.append(evaluated_assumption)
            
            except Exception as e:
                logger.error(f"Error parsing assumption evaluation: {e}")
                evaluation = {
                    "validity": "Unknown",
                    "validity_rationale": f"Error parsing evaluation: {str(e)}",
                    "impact": "Unknown",
                    "impact_rationale": "Error in evaluation",
                    "testing_methods": []
                }
                
                # Combine assumption and evaluation
                evaluated_assumption = assumption.copy()
                evaluated_assumption.update(evaluation)
                evaluated_assumptions.append(evaluated_assumption)
        
        return evaluated_assumptions
    
    def _identify_critical_assumptions(self, evaluated_assumptions):
        """
        Identify critical assumptions based on validity and impact.
        
        Args:
            evaluated_assumptions: List of evaluated assumption dictionaries
            
        Returns:
            List of critical assumption dictionaries
        """
        logger.info("Identifying critical assumptions...")
        
        critical_assumptions = []
        
        for assumption in evaluated_assumptions:
            validity = assumption.get("validity", "Unknown")
            impact = assumption.get("impact", "Unknown")
            
            # Critical assumptions are those with low validity and high impact
            if validity == "Low" and impact == "High":
                critical_assumptions.append({
                    "statement": assumption.get("statement", ""),
                    "validity": validity,
                    "impact": impact,
                    "criticality": "High",
                    "rationale": "Low validity combined with high impact makes this assumption highly critical"
                })
            
            # Also include medium validity with high impact
            elif validity == "Medium" and impact == "High":
                critical_assumptions.append({
                    "statement": assumption.get("statement", ""),
                    "validity": validity,
                    "impact": impact,
                    "criticality": "Medium",
                    "rationale": "Medium validity combined with high impact makes this assumption moderately critical"
                })
        
        return critical_assumptions
    
    def _generate_synthesis(self, question, evaluated_assumptions, critical_assumptions):
        """
        Generate a synthesis of the key assumptions check.
        
        Args:
            question: The analytical question
            evaluated_assumptions: List of evaluated assumption dictionaries
            critical_assumptions: List of critical assumption dictionaries
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of key assumptions check...")
        
        prompt = f"""
        Synthesize the following key assumptions check for the question:
        
        "{question}"
        
        Evaluated Assumptions:
        {json.dumps(evaluated_assumptions, indent=2)}
        
        Critical Assumptions:
        {json.dumps(critical_assumptions, indent=2)}
        
        Based on this key assumptions check:
        1. How robust is the overall analysis given these assumptions?
        2. Which assumptions most threaten the validity of the analysis?
        3. What steps could be taken to strengthen the analysis?
        
        Provide:
        1. A final judgment about the robustness of the analysis
        2. A rationale for this judgment
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment of the analysis robustness",
            "judgment_rationale": "Explanation for your judgment",
            "key_vulnerabilities": ["Vulnerability 1", "Vulnerability 2", ...],
            "recommended_actions": ["Action 1", "Action 2", ...],
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
                    "key_vulnerabilities": ["Error in synthesis generation"],
                    "recommended_actions": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_vulnerabilities": ["Error in synthesis generation"],
                "recommended_actions": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
