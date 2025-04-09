"""
Argument Mapping Technique implementation.
This module provides the ArgumentMappingTechnique class for structured analysis of arguments.
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

class ArgumentMappingTechnique(AnalyticalTechnique):
    """
    Maps the structure of arguments and counter-arguments on a topic.
    
    This technique creates a structured representation of arguments,
    counter-arguments, and supporting evidence to clarify reasoning
    and identify logical strengths and weaknesses.
    """
    
    def execute(self, context, parameters):
        """
        Execute the argument mapping technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing argument mapping results
        """
        logger.info(f"Executing ArgumentMappingTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        source_technique = parameters.get("source_technique", None)
        max_depth = parameters.get("max_depth", 3)
        
        # Step 1: Identify key claims
        key_claims = self._identify_key_claims(context, source_technique)
        
        # Step 2: Map arguments for each claim
        argument_maps = self._map_arguments(context.question, key_claims, max_depth)
        
        # Step 3: Evaluate argument strength
        argument_evaluation = self._evaluate_arguments(argument_maps)
        
        # Step 4: Identify logical fallacies
        logical_fallacies = self._identify_logical_fallacies(argument_maps)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, key_claims, argument_maps, argument_evaluation, logical_fallacies)
        
        return {
            "technique": "Argument Mapping",
            "status": "Completed",
            "key_claims": key_claims,
            "argument_maps": argument_maps,
            "argument_evaluation": argument_evaluation,
            "logical_fallacies": logical_fallacies,
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
        return ["logical_reasoning_mcp", "evidence_evaluation_mcp"]
    
    def _identify_key_claims(self, context, source_technique):
        """
        Identify key claims related to the question.
        
        Args:
            context: The analysis context
            source_technique: Name of technique to source claims from
            
        Returns:
            List of key claim dictionaries
        """
        logger.info("Identifying key claims...")
        
        # Try to get claims from source technique if specified
        if source_technique and source_technique in context.results:
            source_results = context.results[source_technique]
            
            # Look for claims in common fields
            potential_claims = []
            
            # Check for final judgment
            if "final_judgment" in source_results:
                potential_claims.append({
                    "claim": source_results["final_judgment"],
                    "type": "Conclusion",
                    "source": f"{source_technique} final judgment"
                })
            
            # Check for key insights
            if "key_insights" in source_results:
                for i, insight in enumerate(source_results["key_insights"]):
                    potential_claims.append({
                        "claim": insight,
                        "type": "Insight",
                        "source": f"{source_technique} key insight {i+1}"
                    })
            
            # Check for competing hypotheses
            if "hypotheses" in source_results:
                for i, hypothesis in enumerate(source_results["hypotheses"]):
                    if isinstance(hypothesis, dict) and "statement" in hypothesis:
                        potential_claims.append({
                            "claim": hypothesis["statement"],
                            "type": "Hypothesis",
                            "source": f"{source_technique} hypothesis {i+1}"
                        })
                    else:
                        potential_claims.append({
                            "claim": str(hypothesis),
                            "type": "Hypothesis",
                            "source": f"{source_technique} hypothesis {i+1}"
                        })
            
            # If we found potential claims, process them
            if potential_claims:
                logger.info(f"Found {len(potential_claims)} potential claims from {source_technique}")
                return self._process_claims(potential_claims)
        
        # Use logical reasoning MCP if available
        logical_mcp = self.mcp_registry.get_mcp("logical_reasoning_mcp")
        
        if logical_mcp:
            try:
                logger.info("Using logical reasoning MCP")
                key_claims = logical_mcp.identify_key_claims(context.question)
                return key_claims
            except Exception as e:
                logger.error(f"Error using logical reasoning MCP: {e}")
                # Fall through to LLM-based identification
        
        # Use LLM to identify key claims
        prompt = f"""
        Identify key claims related to the following analytical question:
        
        "{context.question}"
        
        A claim is an assertion that can be argued for or against.
        
        For this analysis:
        1. Identify 4-6 key claims that are central to answering the question
        2. Include a mix of different types of claims (e.g., factual, causal, normative, predictive)
        3. Ensure the claims represent different perspectives on the question
        4. Provide a clear statement of each claim
        
        Return your response as a JSON object with the following structure:
        {{
            "key_claims": [
                {{
                    "claim": "Clear statement of the claim",
                    "type": "Type of claim (Factual/Causal/Normative/Predictive)",
                    "perspective": "Perspective this claim represents"
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
                logger.error(f"Error identifying key claims: {parsed_response.get('error')}")
                return self._generate_fallback_key_claims(context.question)
            
            key_claims = parsed_response.get("key_claims", [])
            
            if not key_claims:
                logger.warning("No key claims identified")
                return self._generate_fallback_key_claims(context.question)
            
            return self._process_claims(key_claims)
        
        except Exception as e:
            logger.error(f"Error parsing key claims: {e}")
            return self._generate_fallback_key_claims(context.question)
    
    def _process_claims(self, claims):
        """
        Process and normalize claims.
        
        Args:
            claims: List of raw claim dictionaries
            
        Returns:
            List of processed claim dictionaries
        """
        processed_claims = []
        seen_claims = set()
        
        for claim in claims:
            # Extract claim text and ensure it exists
            claim_text = claim.get("claim", "")
            if not claim_text:
                continue
            
            # Skip duplicates (based on normalized text)
            normalized_text = claim_text.lower().strip()
            if normalized_text in seen_claims:
                continue
            
            seen_claims.add(normalized_text)
            
            # Ensure all required fields exist
            processed_claim = {
                "claim": claim_text,
                "type": claim.get("type", "General"),
                "perspective": claim.get("perspective", "Neutral")
            }
            
            processed_claims.append(processed_claim)
        
        return processed_claims
    
    def _generate_fallback_key_claims(self, question):
        """
        Generate fallback key claims when identification fails.
        
        Args:
            question: The analytical question
            
        Returns:
            List of fallback key claim dictionaries
        """
        return [
            {
                "claim": "The current approach is likely to succeed with minor adjustments",
                "type": "Predictive",
                "perspective": "Optimistic"
            },
            {
                "claim": "Significant structural changes are necessary for success",
                "type": "Normative",
                "perspective": "Reformist"
            },
            {
                "claim": "External factors will have more influence than internal decisions",
                "type": "Causal",
                "perspective": "Externalist"
            },
            {
                "claim": "Historical patterns suggest a cyclical outcome rather than linear progress",
                "type": "Factual",
                "perspective": "Historical"
            }
        ]
    
    def _map_arguments(self, question, key_claims, max_depth):
        """
        Map arguments for each key claim.
        
        Args:
            question: The analytical question
            key_claims: List of key claim dictionaries
            max_depth: Maximum depth of argument mapping
            
        Returns:
            Dictionary mapping claim texts to their argument maps
        """
        logger.info(f"Mapping arguments for {len(key_claims)} key claims (max depth: {max_depth})...")
        
        # Use logical reasoning MCP if available
        logical_mcp = self.mcp_registry.get_mcp("logical_reasoning_mcp")
        
        if logical_mcp:
            try:
                logger.info("Using logical reasoning MCP")
                argument_maps = logical_mcp.map_arguments(question, key_claims, max_depth)
                return argument_maps
            except Exception as e:
                logger.error(f"Error using logical reasoning MCP: {e}")
                # Fall through to LLM-based mapping
        
        argument_maps = {}
        
        for claim_dict in key_claims:
            claim = claim_dict.get("claim", "")
            claim_type = claim_dict.get("type", "")
            
            logger.info(f"Mapping arguments for claim: {claim[:50]}...")
            
            # Use LLM to map arguments for this claim
            prompt = f"""
            Create an argument map for the following claim related to this question:
            
            Question: "{question}"
            
            Claim: "{claim}"
            Claim Type: {claim_type}
            
            For this argument map:
            1. Identify 2-3 supporting arguments for the claim
            2. Identify 2-3 opposing arguments against the claim
            3. For each supporting and opposing argument, identify:
               a. 1-2 pieces of evidence or sub-arguments that support it
               b. 1-2 potential objections or rebuttals to it
            4. Continue this process up to a maximum depth of {max_depth} levels
            
            Return your argument map as a JSON object with the following structure:
            {{
                "claim": "The main claim being mapped",
                "supporting_arguments": [
                    {{
                        "argument": "Statement of the supporting argument",
                        "evidence": ["Evidence 1", "Evidence 2", ...],
                        "objections": ["Objection 1", "Objection 2", ...],
                        "sub_arguments": [
                            {{
                                "argument": "Statement of the sub-argument",
                                "evidence": ["Evidence 1", "Evidence 2", ...],
                                "objections": ["Objection 1", "Objection 2", ...]
                            }},
                            ...
                        ]
                    }},
                    ...
                ],
                "opposing_arguments": [
                    {{
                        "argument": "Statement of the opposing argument",
                        "evidence": ["Evidence 1", "Evidence 2", ...],
                        "objections": ["Objection 1", "Objection 2", ...],
                        "sub_arguments": [
                            {{
                                "argument": "Statement of the sub-argument",
                                "evidence": ["Evidence 1", "Evidence 2", ...],
                                "objections": ["Objection 1", "Objection 2", ...]
                            }},
                            ...
                        ]
                    }},
                    ...
                ]
            }}
            """
            
            model_config = MODEL_CONFIG["sonar_deep"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error mapping arguments: {parsed_response.get('error')}")
                    argument_maps[claim] = self._generate_fallback_argument_map(claim)
                else:
                    argument_maps[claim] = parsed_response
            
            except Exception as e:
                logger.error(f"Error parsing argument map: {e}")
                argument_maps[claim] = self._generate_fallback_argument_map(claim)
        
        return argument_maps
    
    def _generate_fallback_argument_map(self, claim):
        """
        Generate fallback argument map when mapping fails.
        
        Args:
            claim: The claim text
            
        Returns:
            Dictionary containing fallback argument map
        """
        return {
            "claim": claim,
            "supporting_arguments": [
                {
                    "argument": f"Supporting argument 1 for: {claim[:30]}...",
                    "evidence": [
                        "Historical precedent suggests this is likely",
                        "Expert consensus aligns with this view"
                    ],
                    "objections": [
                        "The historical context has changed significantly",
                        "Expert consensus may be subject to groupthink"
                    ],
                    "sub_arguments": []
                },
                {
                    "argument": f"Supporting argument 2 for: {claim[:30]}...",
                    "evidence": [
                        "Quantitative data shows a consistent pattern",
                        "Theoretical models predict this outcome"
                    ],
                    "objections": [
                        "The data may be incomplete or biased",
                        "Alternative models suggest different outcomes"
                    ],
                    "sub_arguments": []
                }
            ],
            "opposing_arguments": [
                {
                    "argument": f"Opposing argument 1 against: {claim[:30]}...",
                    "evidence": [
                        "Recent developments contradict this claim",
                        "Case studies show exceptions to the pattern"
                    ],
                    "objections": [
                        "These developments may be temporary anomalies",
                        "Case studies may not be representative"
                    ],
                    "sub_arguments": []
                },
                {
                    "argument": f"Opposing argument 2 against: {claim[:30]}...",
                    "evidence": [
                        "Structural factors create significant barriers",
                        "Alternative explanations are more parsimonious"
                    ],
                    "objections": [
                        "These barriers can be overcome with sufficient resources",
                        "Parsimony may come at the cost of explanatory power"
                    ],
                    "sub_arguments": []
                }
            ]
        }
    
    def _evaluate_arguments(self, argument_maps):
        """
        Evaluate the strength of arguments in the maps.
        
        Args:
            argument_maps: Dictionary mapping claim texts to their argument maps
            
        Returns:
            Dictionary containing argument evaluation
        """
        logger.info("Evaluating argument strength...")
        
        # Use evidence evaluation MCP if available
        evidence_mcp = self.mcp_registry.get_mcp("evidence_evaluation_mcp")
        
        if evidence_mcp:
            try:
                logger.info("Using evidence evaluation MCP")
                argument_evaluation = evidence_mcp.evaluate_arguments(argument_maps)
                return argument_evaluation
            except Exception as e:
                logger.error(f"Error using evidence evaluation MCP: {e}")
                # Fall through to LLM-based evaluation
        
        evaluations = {}
        
        for claim, argument_map in argument_maps.items():
            logger.info(f"Evaluating arguments for claim: {claim[:50]}...")
            
            # Use LLM to evaluate arguments for this claim
            prompt = f"""
            Evaluate the strength of arguments in the following argument map:
            
            Claim: "{claim}"
            
            Supporting Arguments:
            {json.dumps([{
                "argument": arg.get("argument", ""),
                "evidence": arg.get("evidence", []),
                "objections": arg.get("objections", [])
            } for arg in argument_map.get("supporting_arguments", [])], indent=2)}
            
            Opposing Arguments:
            {json.dumps([{
                "argument": arg.get("argument", ""),
                "evidence": arg.get("evidence", []),
                "objections": arg.get("objections", [])
            } for arg in argument_map.get("opposing_arguments", [])], indent=2)}
            
            For this evaluation:
            1. Assess the overall strength of supporting vs. opposing arguments
            2. Identify the strongest individual arguments on each side
            3. Evaluate the quality of evidence used
            4. Identify key unresolved issues or questions
            
            Return your evaluation as a JSON object with the following structure:
            {{
                "overall_assessment": {{
                    "supporting_strength": "Strong/Moderate/Weak",
                    "opposing_strength": "Strong/Moderate/Weak",
                    "balance": "Supporting arguments stronger/Opposing arguments stronger/Roughly balanced",
                    "rationale": "Explanation for this assessment"
                }},
                "strongest_supporting": {{
                    "argument": "The strongest supporting argument",
                    "strength_rationale": "Why this argument is strong"
                }},
                "strongest_opposing": {{
                    "argument": "The strongest opposing argument",
                    "strength_rationale": "Why this argument is strong"
                }},
                "evidence_quality": {{
                    "supporting_evidence_quality": "High/Medium/Low",
                    "opposing_evidence_quality": "High/Medium/Low",
                    "evidence_gaps": ["Gap 1", "Gap 2", ...]
                }},
                "unresolved_issues": ["Issue 1", "Issue 2", ...]
            }}
            """
            
            model_config = MODEL_CONFIG["sonar"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error evaluating arguments: {parsed_response.get('error')}")
                    evaluations[claim] = self._generate_fallback_argument_evaluation()
                else:
                    evaluations[claim] = parsed_response
            
            except Exception as e:
                logger.error(f"Error parsing argument evaluation: {e}")
                evaluations[claim] = self._generate_fallback_argument_evaluation()
        
        return evaluations
    
    def _generate_fallback_argument_evaluation(self):
        """
        Generate fallback argument evaluation when evaluation fails.
        
        Returns:
            Dictionary containing fallback argument evaluation
        """
        return {
            "overall_assessment": {
                "supporting_strength": "Moderate",
                "opposing_strength": "Moderate",
                "balance": "Roughly balanced",
                "rationale": "Both supporting and opposing arguments present reasonable cases with some evidence, but also have significant limitations."
            },
            "strongest_supporting": {
                "argument": "Supporting argument 1",
                "strength_rationale": "This argument is backed by multiple lines of evidence and addresses potential objections."
            },
            "strongest_opposing": {
                "argument": "Opposing argument 1",
                "strength_rationale": "This argument identifies important limitations and provides alternative explanations."
            },
            "evidence_quality": {
                "supporting_evidence_quality": "Medium",
                "opposing_evidence_quality": "Medium",
                "evidence_gaps": [
                    "Limited empirical testing of key assumptions",
                    "Insufficient consideration of alternative contexts",
                    "Lack of long-term historical data"
                ]
            },
            "unresolved_issues": [
                "The relative importance of different causal factors",
                "How conditions might change in different scenarios",
                "The applicability of theoretical models to real-world complexity"
            ]
        }
    
    def _identify_logical_fallacies(self, argument_maps):
        """
        Identify logical fallacies in the arguments.
        
        Args:
            argument_maps: Dictionary mapping claim texts to their argument maps
            
        Returns:
            Dictionary mapping claim texts to their logical fallacies
        """
        logger.info("Identifying logical fallacies...")
        
        # Use logical reasoning MCP if available
        logical_mcp = self.mcp_registry.get_mcp("logical_reasoning_mcp")
        
        if logical_mcp:
            try:
                logger.info("Using logical reasoning MCP")
                logical_fallacies = logical_mcp.identify_fallacies(argument_maps)
                return logical_fallacies
            except Exception as e:
                logger.error(f"Error using logical reasoning MCP: {e}")
                # Fall through to LLM-based identification
        
        fallacies = {}
        
        for claim, argument_map in argument_maps.items():
            logger.info(f"Identifying fallacies for claim: {claim[:50]}...")
            
            # Extract all arguments for this claim
            all_arguments = []
            
            for arg in argument_map.get("supporting_arguments", []):
                all_arguments.append({
                    "argument": arg.get("argument", ""),
                    "type": "supporting",
                    "evidence": arg.get("evidence", []),
                    "objections": arg.get("objections", [])
                })
            
            for arg in argument_map.get("opposing_arguments", []):
                all_arguments.append({
                    "argument": arg.get("argument", ""),
                    "type": "opposing",
                    "evidence": arg.get("evidence", []),
                    "objections": arg.get("objections", [])
                })
            
            # Use LLM to identify fallacies
            prompt = f"""
            Identify potential logical fallacies in the following arguments related to this claim:
            
            Claim: "{claim}"
            
            Arguments:
            {json.dumps(all_arguments, indent=2)}
            
            For this analysis:
            1. Identify any logical fallacies present in the arguments
            2. Explain why each identified pattern constitutes a fallacy
            3. Suggest how the argument could be reformulated to avoid the fallacy
            
            Common logical fallacies include:
            - Ad hominem: Attacking the person instead of addressing their argument
            - Appeal to authority: Relying on an authority figure rather than evidence or reasoning
            - Appeal to popularity: Assuming something is true because many people believe it
            - False dichotomy: Presenting only two options when others exist
            - Hasty generalization: Drawing conclusions from insufficient evidence
            - Post hoc ergo propter hoc: Assuming that because B followed A, A caused B
            - Slippery slope: Claiming one small step will inevitably lead to extreme consequences
            - Straw man: Misrepresenting an opponent's argument to make it easier to attack
            
            Return your analysis as a JSON object with the following structure:
            {{
                "fallacies": [
                    {{
                        "argument": "The argument containing the fallacy",
                        "fallacy_type": "Name of the fallacy",
                        "explanation": "Explanation of why this is a fallacy",
                        "reformulation": "Suggested reformulation to avoid the fallacy"
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
                    logger.error(f"Error identifying fallacies: {parsed_response.get('error')}")
                    fallacies[claim] = {"fallacies": []}
                else:
                    fallacies[claim] = parsed_response
            
            except Exception as e:
                logger.error(f"Error parsing fallacies: {e}")
                fallacies[claim] = {"fallacies": []}
        
        return fallacies
    
    def _generate_synthesis(self, question, key_claims, argument_maps, argument_evaluation, logical_fallacies):
        """
        Generate a synthesis of the argument mapping.
        
        Args:
            question: The analytical question
            key_claims: List of key claim dictionaries
            argument_maps: Dictionary mapping claim texts to their argument maps
            argument_evaluation: Dictionary containing argument evaluation
            logical_fallacies: Dictionary mapping claim texts to their logical fallacies
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of argument mapping...")
        
        # Prepare summaries for the prompt
        claims_summary = []
        for claim_dict in key_claims:
            claim = claim_dict.get("claim", "")
            claim_type = claim_dict.get("type", "")
            
            # Get evaluation for this claim
            evaluation = argument_evaluation.get(claim, {})
            overall = evaluation.get("overall_assessment", {})
            balance = overall.get("balance", "Unknown")
            
            # Get fallacies for this claim
            claim_fallacies = logical_fallacies.get(claim, {}).get("fallacies", [])
            fallacy_count = len(claim_fallacies)
            
            claims_summary.append(f"{claim} ({claim_type}) - Balance: {balance}, Fallacies: {fallacy_count}")
        
        # Use LLM to generate synthesis
        prompt = f"""
        Synthesize the following argument mapping analysis for the question:
        
        "{question}"
        
        Key Claims:
        {json.dumps(claims_summary, indent=2)}
        
        Argument Evaluations:
        {json.dumps([{
            "claim": claim,
            "balance": evaluation.get("overall_assessment", {}).get("balance", "Unknown"),
            "supporting_strength": evaluation.get("overall_assessment", {}).get("supporting_strength", "Unknown"),
            "opposing_strength": evaluation.get("overall_assessment", {}).get("opposing_strength", "Unknown")
        } for claim, evaluation in argument_evaluation.items()], indent=2)}
        
        Logical Fallacies:
        {json.dumps([{
            "claim": claim,
            "fallacy_count": len(fallacies.get("fallacies", []))
        } for claim, fallacies in logical_fallacies.items()], indent=2)}
        
        Based on this argument mapping:
        1. What are the most well-supported claims?
        2. What are the most significant logical weaknesses?
        3. What key issues remain unresolved?
        4. How does this analysis inform the original question?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the argument mapping
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "well_supported_claims": ["Claim 1", "Claim 2", ...],
            "logical_weaknesses": ["Weakness 1", "Weakness 2", ...],
            "unresolved_issues": ["Issue 1", "Issue 2", ...],
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
                    "well_supported_claims": ["Error in synthesis generation"],
                    "logical_weaknesses": ["Error in synthesis generation"],
                    "unresolved_issues": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "well_supported_claims": ["Error in synthesis generation"],
                "logical_weaknesses": ["Error in synthesis generation"],
                "unresolved_issues": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
