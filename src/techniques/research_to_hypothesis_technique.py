"""
Research-to-Hypothesis Technique implementation.
This module provides the ResearchToHypothesisTechnique class for conducting research and testing hypotheses.
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

class ResearchToHypothesisTechnique(AnalyticalTechnique):
    """
    Conducts research to generate and test hypotheses based on evidence.
    
    This technique performs research on the question, generates multiple hypotheses
    based on that research, and then evaluates each hypothesis against the evidence.
    """
    
    def execute(self, context, parameters):
        """
        Execute the research-to-hypothesis technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing research-to-hypothesis results
        """
        logger.info(f"Executing ResearchToHypothesisTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        num_hypotheses = parameters.get("num_hypotheses", 3)
        research_depth = parameters.get("research_depth", "medium")
        
        # Step 1: Conduct research
        research_results = self._conduct_research(context.question, research_depth)
        
        # Step 2: Generate hypotheses
        hypotheses = self._generate_hypotheses(context.question, research_results, num_hypotheses)
        
        # Step 3: Evaluate hypotheses against evidence
        evaluated_hypotheses = self._evaluate_hypotheses(hypotheses, research_results)
        
        # Step 4: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, evaluated_hypotheses)
        
        return {
            "technique": "Research-to-Hypothesis",
            "status": "Completed",
            "research_results": research_results,
            "hypotheses": evaluated_hypotheses,
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
        return ["brave_search_mcp", "academic_search_mcp", "content_extraction_mcp"]
    
    def _conduct_research(self, question, research_depth):
        """
        Conduct research on the question.
        
        Args:
            question: The analytical question
            research_depth: Depth of research (low, medium, high)
            
        Returns:
            Dictionary containing research results
        """
        logger.info(f"Conducting research with depth: {research_depth}...")
        
        # Use search MCPs if available
        brave_search_mcp = self.mcp_registry.get_mcp("brave_search_mcp")
        academic_search_mcp = self.mcp_registry.get_mcp("academic_search_mcp")
        content_extraction_mcp = self.mcp_registry.get_mcp("content_extraction_mcp")
        
        # Define number of sources based on research depth
        num_sources = {
            "low": 3,
            "medium": 5,
            "high": 8
        }.get(research_depth, 5)
        
        # Initialize research results
        research_results = {
            "sources": [],
            "key_findings": [],
            "contradictions": [],
            "knowledge_gaps": []
        }
        
        # Use Brave Search MCP if available
        if brave_search_mcp:
            try:
                logger.info(f"Using Brave Search MCP to search for: {question}")
                search_results = brave_search_mcp.search(question, num_results=num_sources)
                
                # Extract content if content extraction MCP is available
                if content_extraction_mcp and search_results:
                    for result in search_results:
                        try:
                            extracted_content = content_extraction_mcp.extract_content(result.get("url"))
                            research_results["sources"].append({
                                "title": result.get("title"),
                                "url": result.get("url"),
                                "content": extracted_content.get("content"),
                                "summary": extracted_content.get("summary")
                            })
                        except Exception as e:
                            logger.warning(f"Error extracting content from {result.get('url')}: {e}")
                            # Add source with just the snippet
                            research_results["sources"].append({
                                "title": result.get("title"),
                                "url": result.get("url"),
                                "content": result.get("snippet"),
                                "summary": result.get("snippet")
                            })
                else:
                    # Add sources with just the snippets
                    for result in search_results:
                        research_results["sources"].append({
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "content": result.get("snippet"),
                            "summary": result.get("snippet")
                        })
            except Exception as e:
                logger.error(f"Error using Brave Search MCP: {e}")
        
        # Use Academic Search MCP for medium and high depth research
        if academic_search_mcp and research_depth in ["medium", "high"]:
            try:
                logger.info(f"Using Academic Search MCP to search for: {question}")
                academic_results = academic_search_mcp.search(question, num_results=num_sources // 2)
                
                # Extract content if content extraction MCP is available
                if content_extraction_mcp and academic_results:
                    for result in academic_results:
                        try:
                            extracted_content = content_extraction_mcp.extract_content(result.get("url"))
                            research_results["sources"].append({
                                "title": result.get("title"),
                                "url": result.get("url"),
                                "content": extracted_content.get("content"),
                                "summary": extracted_content.get("summary"),
                                "type": "academic"
                            })
                        except Exception as e:
                            logger.warning(f"Error extracting content from {result.get('url')}: {e}")
                            # Add source with just the abstract
                            research_results["sources"].append({
                                "title": result.get("title"),
                                "url": result.get("url"),
                                "content": result.get("abstract"),
                                "summary": result.get("abstract"),
                                "type": "academic"
                            })
                else:
                    # Add sources with just the abstracts
                    for result in academic_results:
                        research_results["sources"].append({
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "content": result.get("abstract"),
                            "summary": result.get("abstract"),
                            "type": "academic"
                        })
            except Exception as e:
                logger.error(f"Error using Academic Search MCP: {e}")
        
        # If no sources found or MCPs not available, use LLM to generate research
        if not research_results["sources"]:
            logger.warning("No research sources found, using LLM to generate research")
            research_results = self._generate_research_fallback(question, research_depth)
        else:
            # Analyze research to extract key findings, contradictions, and knowledge gaps
            research_results.update(self._analyze_research(question, research_results["sources"]))
        
        return research_results
    
    def _generate_research_fallback(self, question, research_depth):
        """
        Generate fallback research when search MCPs are not available or fail.
        
        Args:
            question: The analytical question
            research_depth: Depth of research (low, medium, high)
            
        Returns:
            Dictionary containing fallback research results
        """
        logger.info("Generating fallback research...")
        
        prompt = f"""
        Conduct a thorough research analysis on the following question:
        
        "{question}"
        
        Since you don't have direct access to search results, please:
        1. Draw on your knowledge to identify what would likely be the most relevant and credible sources on this topic
        2. Summarize what these sources would likely say about the question
        3. Identify likely areas of consensus and disagreement in the literature
        4. Note important knowledge gaps that would likely exist
        
        Provide your response as a JSON object with the following structure:
        {{
            "sources": [
                {{
                    "title": "Likely source title",
                    "author": "Likely author(s)",
                    "publication": "Likely publication venue",
                    "year": "Likely publication year",
                    "summary": "Summary of what this source would likely say"
                }},
                ...
            ],
            "key_findings": ["Key finding 1", "Key finding 2", ...],
            "contradictions": ["Contradiction 1", "Contradiction 2", ...],
            "knowledge_gaps": ["Knowledge gap 1", "Knowledge gap 2", ...]
        }}
        
        Note: This is a simulation of research based on your knowledge, not actual search results.
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating fallback research: {parsed_response.get('error')}")
                return {
                    "sources": [
                        {
                            "title": "Error generating research",
                            "summary": parsed_response.get('error', "Unknown error"),
                            "content": "Error in research generation"
                        }
                    ],
                    "key_findings": ["Error in research generation"],
                    "contradictions": [],
                    "knowledge_gaps": ["Unable to identify knowledge gaps due to research error"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing fallback research: {e}")
            return {
                "sources": [
                    {
                        "title": f"Error generating research: {str(e)}",
                        "summary": "Error in research generation",
                        "content": "Error in research generation"
                    }
                ],
                "key_findings": ["Error in research generation"],
                "contradictions": [],
                "knowledge_gaps": ["Unable to identify knowledge gaps due to research error"]
            }
    
    def _analyze_research(self, question, sources):
        """
        Analyze research sources to extract key findings, contradictions, and knowledge gaps.
        
        Args:
            question: The analytical question
            sources: List of research sources
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing {len(sources)} research sources...")
        
        # Prepare source summaries for the prompt
        source_summaries = []
        for i, source in enumerate(sources):
            summary = source.get("summary", "No summary available")
            title = source.get("title", f"Source {i+1}")
            source_summaries.append(f"Source {i+1} - {title}: {summary}")
        
        prompt = f"""
        Analyze the following research sources related to the question:
        
        "{question}"
        
        Sources:
        {json.dumps(source_summaries, indent=2)}
        
        Based on these sources:
        1. Identify key findings that are supported by multiple sources
        2. Identify contradictions or disagreements between sources
        3. Identify important knowledge gaps not addressed by the sources
        
        Return your analysis as a JSON object with the following structure:
        {{
            "key_findings": ["Finding 1", "Finding 2", ...],
            "contradictions": ["Contradiction 1", "Contradiction 2", ...],
            "knowledge_gaps": ["Gap 1", "Gap 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error analyzing research: {parsed_response.get('error')}")
                return {
                    "key_findings": ["Error analyzing research findings"],
                    "contradictions": ["Error analyzing research contradictions"],
                    "knowledge_gaps": ["Error analyzing research knowledge gaps"]
                }
            
            return {
                "key_findings": parsed_response.get("key_findings", []),
                "contradictions": parsed_response.get("contradictions", []),
                "knowledge_gaps": parsed_response.get("knowledge_gaps", [])
            }
        
        except Exception as e:
            logger.error(f"Error parsing research analysis: {e}")
            return {
                "key_findings": [f"Error analyzing research: {str(e)}"],
                "contradictions": [],
                "knowledge_gaps": []
            }
    
    def _generate_hypotheses(self, question, research_results, num_hypotheses):
        """
        Generate hypotheses based on research results.
        
        Args:
            question: The analytical question
            research_results: Dictionary containing research results
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of hypothesis dictionaries
        """
        logger.info(f"Generating {num_hypotheses} hypotheses...")
        
        prompt = f"""
        Based on the following research related to the question:
        
        "{question}"
        
        Key Findings:
        {json.dumps(research_results.get("key_findings", []), indent=2)}
        
        Contradictions:
        {json.dumps(research_results.get("contradictions", []), indent=2)}
        
        Knowledge Gaps:
        {json.dumps(research_results.get("knowledge_gaps", []), indent=2)}
        
        Generate {num_hypotheses} distinct hypotheses that could answer the question. For each hypothesis:
        1. Provide a clear statement of the hypothesis
        2. Explain the reasoning behind this hypothesis
        3. Identify key assumptions underlying this hypothesis
        
        Return your hypotheses as a JSON object with the following structure:
        {{
            "hypotheses": [
                {{
                    "statement": "Hypothesis statement",
                    "reasoning": "Reasoning behind this hypothesis",
                    "key_assumptions": ["Assumption 1", "Assumption 2", ...]
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
                logger.error(f"Error generating hypotheses: {parsed_response.get('error')}")
                return self._generate_fallback_hypotheses(question, num_hypotheses)
            
            hypotheses = parsed_response.get("hypotheses", [])
            
            if not hypotheses or len(hypotheses) < num_hypotheses:
                logger.warning(f"Insufficient hypotheses generated: {len(hypotheses)}")
                # Add fallback hypotheses if needed
                hypotheses.extend(self._generate_fallback_hypotheses(question, num_hypotheses - len(hypotheses)))
            
            return hypotheses[:num_hypotheses]
        
        except Exception as e:
            logger.error(f"Error parsing hypotheses: {e}")
            return self._generate_fallback_hypotheses(question, num_hypotheses)
    
    def _generate_fallback_hypotheses(self, question, num_hypotheses):
        """
        Generate fallback hypotheses when normal generation fails.
        
        Args:
            question: The analytical question
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of fallback hypothesis dictionaries
        """
        fallback_hypotheses = [
            {
                "statement": f"Primary hypothesis for '{question[:50]}...'",
                "reasoning": "This is the most straightforward explanation based on available information.",
                "key_assumptions": ["Assumption 1", "Assumption 2"]
            },
            {
                "statement": f"Alternative hypothesis for '{question[:50]}...'",
                "reasoning": "This explanation considers alternative factors not addressed in the primary hypothesis.",
                "key_assumptions": ["Assumption 1", "Assumption 2"]
            },
            {
                "statement": f"Contrarian hypothesis for '{question[:50]}...'",
                "reasoning": "This explanation challenges conventional thinking on the subject.",
                "key_assumptions": ["Assumption 1", "Assumption 2"]
            },
            {
                "statement": f"Synthesis hypothesis for '{question[:50]}...'",
                "reasoning": "This explanation combines elements from multiple perspectives.",
                "key_assumptions": ["Assumption 1", "Assumption 2"]
            },
            {
                "statement": f"Null hypothesis for '{question[:50]}...'",
                "reasoning": "This explanation suggests that the premise of the question may be incorrect.",
                "key_assumptions": ["Assumption 1", "Assumption 2"]
            }
        ]
        
        return fallback_hypotheses[:num_hypotheses]
    
    def _evaluate_hypotheses(self, hypotheses, research_results):
        """
        Evaluate hypotheses against research evidence.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            research_results: Dictionary containing research results
            
        Returns:
            List of evaluated hypothesis dictionaries
        """
        logger.info(f"Evaluating {len(hypotheses)} hypotheses...")
        
        evaluated_hypotheses = []
        
        for hypothesis in hypotheses:
            logger.info(f"Evaluating hypothesis: {hypothesis.get('statement', '')[:50]}...")
            
            prompt = f"""
            Evaluate the following hypothesis against the research evidence:
            
            Hypothesis: {hypothesis.get('statement', '')}
            Reasoning: {hypothesis.get('reasoning', '')}
            Key Assumptions: {json.dumps(hypothesis.get('key_assumptions', []))}
            
            Research Findings:
            {json.dumps(research_results.get("key_findings", []), indent=2)}
            
            Contradictions in Research:
            {json.dumps(research_results.get("contradictions", []), indent=2)}
            
            For this evaluation:
            1. Identify evidence that supports the hypothesis
            2. Identify evidence that contradicts the hypothesis
            3. Assess the validity of key assumptions
            4. Provide an overall assessment of the hypothesis (Strong/Moderate/Weak)
            5. Suggest how the hypothesis could be refined
            
            Return your evaluation as a JSON object with the following structure:
            {{
                "supporting_evidence": ["Evidence 1", "Evidence 2", ...],
                "contradicting_evidence": ["Evidence 1", "Evidence 2", ...],
                "assumption_assessment": [
                    {{"assumption": "Assumption 1", "validity": "High/Medium/Low", "rationale": "Explanation"}}
                ],
                "overall_assessment": "Strong/Moderate/Weak",
                "assessment_rationale": "Explanation for overall assessment",
                "refinement_suggestions": ["Suggestion 1", "Suggestion 2", ...]
            }}
            """
            
            model_config = MODEL_CONFIG["sonar"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error evaluating hypothesis: {parsed_response.get('error')}")
                    evaluation = {
                        "supporting_evidence": ["Error evaluating supporting evidence"],
                        "contradicting_evidence": ["Error evaluating contradicting evidence"],
                        "assumption_assessment": [
                            {"assumption": "Error in evaluation", "validity": "Unknown", "rationale": parsed_response.get('error', "Unknown error")}
                        ],
                        "overall_assessment": "Unknown",
                        "assessment_rationale": "Error in evaluation",
                        "refinement_suggestions": ["Unable to provide refinement suggestions due to evaluation error"]
                    }
                else:
                    evaluation = parsed_response
                
                # Combine hypothesis and evaluation
                evaluated_hypothesis = hypothesis.copy()
                evaluated_hypothesis.update(evaluation)
                evaluated_hypotheses.append(evaluated_hypothesis)
            
            except Exception as e:
                logger.error(f"Error parsing hypothesis evaluation: {e}")
                evaluation = {
                    "supporting_evidence": [f"Error evaluating hypothesis: {str(e)}"],
                    "contradicting_evidence": [],
                    "assumption_assessment": [
                        {"assumption": "Error in evaluation", "validity": "Unknown", "rationale": f"Exception: {str(e)}"}
                    ],
                    "overall_assessment": "Unknown",
                    "assessment_rationale": "Error in evaluation",
                    "refinement_suggestions": []
                }
                
                # Combine hypothesis and evaluation
                evaluated_hypothesis = hypothesis.copy()
                evaluated_hypothesis.update(evaluation)
                evaluated_hypotheses.append(evaluated_hypothesis)
        
        return evaluated_hypotheses
    
    def _generate_synthesis(self, question, evaluated_hypotheses):
        """
        Generate a synthesis of the research-to-hypothesis analysis.
        
        Args:
            question: The analytical question
            evaluated_hypotheses: List of evaluated hypothesis dictionaries
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of research-to-hypothesis analysis...")
        
        prompt = f"""
        Synthesize the following research-to-hypothesis analysis for the question:
        
        "{question}"
        
        Evaluated Hypotheses:
        {json.dumps(evaluated_hypotheses, indent=2)}
        
        Based on this analysis:
        1. Which hypothesis is best supported by the evidence and why?
        2. How confident can we be in this assessment?
        3. What are the key uncertainties that remain?
        4. What biases might be affecting this analysis?
        
        Provide:
        1. A final judgment that integrates the hypothesis evaluation
        2. A rationale for this judgment
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your integrated judgment",
            "judgment_rationale": "Explanation for your judgment",
            "confidence_level": "High/Medium/Low",
            "key_uncertainties": ["Uncertainty 1", "Uncertainty 2", ...],
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
                    "confidence_level": "Low",
                    "key_uncertainties": ["Unable to identify uncertainties due to synthesis error"],
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "confidence_level": "Low",
                "key_uncertainties": ["Unable to identify uncertainties due to synthesis error"],
                "potential_biases": ["Technical error bias"]
            }
