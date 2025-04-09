"""
Llama4 Scout MCP for LLM-based analysis using Llama 4 Scout (17Bx16E) via Groq.
This module provides the Llama4ScoutMCP class for advanced analytical reasoning.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
import requests

from src.base_mcp import BaseMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Llama4ScoutMCP(BaseMCP):
    """
    Llama4 Scout MCP for LLM-based analysis using Llama 4 Scout (17Bx16E) via Groq.
    
    This MCP provides capabilities for:
    1. Chain-of-thought reasoning
    2. Uncertainty quantification
    3. Bias detection
    4. Multi-step analysis
    5. Strategy generation and adaptation
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Llama4ScoutMCP.
        
        Args:
            api_key: Groq API key (if None, will try to get from environment variable)
        """
        super().__init__(
            name="llama4_scout",
            description="LLM-based analysis using Llama 4 Scout (17Bx16E) via Groq",
            version="1.0.0"
        )
        
        # Set API key
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No Groq API key provided, LLM functionality will be limited")
        
        # Set model parameters
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.temperature = 0.2
        self.max_tokens = 4096
        self.top_p = 0.9
        
        # Operation handlers
        self.operation_handlers = {
            "analyze_question": self._analyze_question,
            "generate_strategy": self._generate_strategy,
            "adapt_strategy": self._adapt_strategy,
            "synthesize_results": self._synthesize_results,
            "final_synthesis": self._final_synthesis,
            "analyze_evidence": self._analyze_evidence,
            "detect_biases": self._detect_biases,
            "evaluate_uncertainty": self._evaluate_uncertainty,
            "generate_hypotheses": self._generate_hypotheses
        }
        
        logger.info("Initialized Llama4ScoutMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in Llama4ScoutMCP")
        
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
    
    def _call_llm(self, prompt: str, system_prompt: str = None, temperature: float = None, max_tokens: int = None) -> Dict:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (if None, uses default)
            temperature: Temperature parameter (if None, uses default)
            max_tokens: Maximum tokens to generate (if None, uses default)
            
        Returns:
            LLM response
        """
        # Check if API key is available
        if not self.api_key:
            logger.warning("No API key available, using mock LLM response")
            return self._mock_llm_response(prompt)
        
        # Set default system prompt if not provided
        if system_prompt is None:
            system_prompt = "You are an expert analytical assistant with deep expertise in structured analytic techniques, critical thinking, and evidence-based reasoning. Provide detailed, well-reasoned analysis based on the information provided."
        
        # Set parameters
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": tokens,
            "top_p": self.top_p
        }
        
        # Call API
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "details": response.text}
            
            # Parse response
            result = response.json()
            
            # Extract content
            content = result["choices"][0]["message"]["content"]
            
            return {"content": content, "model": self.model}
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return {"error": f"Error calling LLM API: {str(e)}"}
    
    def _mock_llm_response(self, prompt: str) -> Dict:
        """
        Generate a mock LLM response for testing without API access.
        
        Args:
            prompt: User prompt
            
        Returns:
            Mock LLM response
        """
        logger.info("Generating mock LLM response")
        
        # Generate different responses based on prompt keywords
        if "analyze_question" in prompt.lower():
            return {
                "content": json.dumps({
                    "question_type": "predictive",
                    "domains": ["economic", "political"],
                    "complexity": "high",
                    "uncertainty": "medium",
                    "time_horizon": "medium-term",
                    "potential_biases": ["recency", "availability"],
                    "key_entities": ["global markets", "central banks", "governments"],
                    "key_concepts": ["inflation", "monetary policy", "fiscal stimulus"]
                }),
                "model": "mock-llama4-scout"
            }
        
        elif "generate_strategy" in prompt.lower():
            return {
                "content": json.dumps({
                    "steps": [
                        {"technique": "research_to_hypothesis", "parameters": {"num_hypotheses": 3}},
                        {"technique": "scenario_triangulation", "parameters": {"num_scenarios": 3}, "dependencies": [0]},
                        {"technique": "key_assumptions_check", "parameters": {}, "dependencies": [1]},
                        {"technique": "uncertainty_mapping", "parameters": {}, "dependencies": [2]},
                        {"technique": "synthesis_generation", "parameters": {}, "dependencies": [3]}
                    ]
                }),
                "model": "mock-llama4-scout"
            }
        
        elif "adapt_strategy" in prompt.lower():
            return {
                "content": json.dumps({
                    "adapted_steps": [
                        {"technique": "analysis_of_competing_hypotheses", "parameters": {}, "dependencies": [0, 1]},
                        {"technique": "red_teaming", "parameters": {}, "dependencies": [2]},
                        {"technique": "synthesis_generation", "parameters": {}, "dependencies": [3]}
                    ]
                }),
                "model": "mock-llama4-scout"
            }
        
        elif "synthesize_results" in prompt.lower() or "final_synthesis" in prompt.lower():
            return {
                "content": json.dumps({
                    "summary": "This is a mock synthesis of the analysis results.",
                    "key_findings": [
                        "Finding 1: Mock finding with high confidence",
                        "Finding 2: Mock finding with medium confidence",
                        "Finding 3: Mock finding with low confidence"
                    ],
                    "confidence_assessment": {
                        "overall_confidence": "medium",
                        "explanation": "This is a mock confidence assessment."
                    },
                    "uncertainties": [
                        {"uncertainty": "Mock uncertainty 1", "impact": "high"},
                        {"uncertainty": "Mock uncertainty 2", "impact": "medium"}
                    ]
                }),
                "model": "mock-llama4-scout"
            }
        
        else:
            return {
                "content": "This is a mock response from the LLM.",
                "model": "mock-llama4-scout"
            }
    
    def _analyze_question(self, input_data: Dict) -> Dict:
        """
        Analyze a question to determine its characteristics.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Question analysis
        """
        logger.info("Analyzing question")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get research data if available
        research_data = input_data.get("research_data", {})
        key_insights = input_data.get("key_insights", [])
        
        # Prepare prompt
        system_prompt = """You are an expert analytical assistant specializing in question analysis. Your task is to analyze the given question and determine its characteristics. Provide your analysis in JSON format with the following fields:
- question_type: The type of question (predictive, causal, evaluative, decision, descriptive)
- domains: List of domains relevant to the question (economic, political, technological, social, environmental, security)
- complexity: Complexity level of the question (low, medium, high)
- uncertainty: Level of inherent uncertainty in the question (low, medium, high)
- time_horizon: Time horizon of the question (short-term, medium-term, long-term)
- potential_biases: List of cognitive biases that might affect analysis of this question
- key_entities: List of key entities mentioned or implied in the question
- key_concepts: List of key concepts relevant to the question"""
        
        # Build prompt with research data if available
        prompt = f"Question: {question}\n\n"
        
        if key_insights:
            prompt += "Key Insights from Preliminary Research:\n"
            for i, insight in enumerate(key_insights):
                prompt += f"{i+1}. {insight}\n"
            prompt += "\n"
        
        prompt += "Please analyze this question and provide a detailed characterization in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        analysis = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            required_fields = ["question_type", "domains", "complexity"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = "unknown"
            
            return analysis
            
        except Exception as e:
            error_msg = f"Error parsing question analysis: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _generate_strategy(self, input_data: Dict) -> Dict:
        """
        Generate an analysis strategy based on question characteristics.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Analysis strategy
        """
        logger.info("Generating analysis strategy")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get question analysis
        question_analysis = input_data.get("question_analysis", {})
        if not question_analysis:
            error_msg = "No question analysis provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get available techniques
        available_techniques = input_data.get("available_techniques", [])
        if not available_techniques:
            error_msg = "No available techniques provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get research data if available
        research_data = input_data.get("research_data", {})
        key_insights = input_data.get("key_insights", [])
        initial_hypotheses = input_data.get("initial_hypotheses", [])
        recommended_workflow = input_data.get("recommended_workflow", {})
        
        # Prepare system prompt
        system_prompt = """You are an expert analytical strategist specializing in designing analytical workflows. Your task is to generate an optimal sequence of analytical techniques to answer the given question based on its characteristics. Provide your strategy in JSON format with a 'steps' array, where each step has:
- technique: Name of the analytical technique
- parameters: Parameters for the technique
- dependencies: List of step indices that must be completed before this step (optional)
- optional: Whether this step is optional (default: false)

Choose techniques that complement each other and address the specific characteristics of the question. Consider the question type, domains, complexity, and uncertainty in your strategy design."""
        
        # Build prompt
        prompt = f"Question: {question}\n\n"
        prompt += f"Question Analysis:\n{json.dumps(question_analysis, indent=2)}\n\n"
        
        if key_insights:
            prompt += "Key Insights from Preliminary Research:\n"
            for i, insight in enumerate(key_insights):
                prompt += f"{i+1}. {insight}\n"
            prompt += "\n"
        
        if initial_hypotheses:
            prompt += "Initial Hypotheses from Preliminary Research:\n"
            for i, hypothesis in enumerate(initial_hypotheses):
                prompt += f"{i+1}. {hypothesis}\n"
            prompt += "\n"
        
        if recommended_workflow:
            prompt += f"Recommended Workflow from Preliminary Research:\n{json.dumps(recommended_workflow, indent=2)}\n\n"
        
        prompt += "Available Analytical Techniques:\n"
        for technique in available_techniques:
            prompt += f"- {technique}\n"
        prompt += "\n"
        
        prompt += "Please generate an optimal analysis strategy with a sequence of techniques to answer this question. Provide your strategy in JSON format with a 'steps' array."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                strategy = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    strategy = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        strategy = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            if "steps" not in strategy:
                strategy["steps"] = []
            
            # Validate each step
            for i, step in enumerate(strategy["steps"]):
                if "technique" not in step:
                    step["technique"] = "unknown"
                if "parameters" not in step:
                    step["parameters"] = {}
                if "dependencies" not in step:
                    step["dependencies"] = []
                if "optional" not in step:
                    step["optional"] = False
                
                # Validate technique is available
                if step["technique"] not in available_techniques:
                    logger.warning(f"Technique {step['technique']} is not available, replacing with default")
                    step["technique"] = available_techniques[0] if available_techniques else "unknown"
            
            return strategy
            
        except Exception as e:
            error_msg = f"Error parsing strategy: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _adapt_strategy(self, input_data: Dict) -> Dict:
        """
        Adapt an analysis strategy based on interim results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Adapted strategy
        """
        logger.info("Adapting analysis strategy")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get adaptation triggers
        adaptation_triggers = input_data.get("adaptation_triggers", {})
        if not adaptation_triggers:
            error_msg = "No adaptation triggers provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get completed steps
        completed_steps = input_data.get("completed_steps", [])
        
        # Get completed results
        completed_results = input_data.get("completed_results", {})
        
        # Get remaining steps
        remaining_steps = input_data.get("remaining_steps", [])
        
        # Get available techniques
        available_techniques = input_data.get("available_techniques", [])
        if not available_techniques:
            error_msg = "No available techniques provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Prepare system prompt
        system_prompt = """You are an expert analytical strategist specializing in adapting analytical workflows. Your task is to adapt the remaining steps of an analysis strategy based on interim results and adaptation triggers. Provide your adapted strategy in JSON format with an 'adapted_steps' array, where each step has:
- technique: Name of the analytical technique
- parameters: Parameters for the technique
- dependencies: List of step indices that must be completed before this step (optional)
- optional: Whether this step is optional (default: false)

Choose techniques that address the specific adaptation triggers and complement the already completed steps. Consider the question type, domains, complexity, and uncertainty in your adaptation."""
        
        # Build prompt
        prompt = f"Question: {question}\n\n"
        
        prompt += "Adaptation Triggers:\n"
        for trigger, value in adaptation_triggers.items():
            if value:
                prompt += f"- {trigger}: {value}\n"
        prompt += "\n"
        
        prompt += "Completed Steps:\n"
        for i, step in enumerate(completed_steps):
            prompt += f"{i+1}. {step['technique']}\n"
            if step['technique'] in completed_results:
                result = completed_results[step['technique']]
                if "findings" in result:
                    prompt += "   Key Findings:\n"
                    for finding in result["findings"][:3]:  # Limit to top 3 findings
                        prompt += f"   - {finding.get('finding', '')}\n"
        prompt += "\n"
        
        prompt += "Remaining Steps:\n"
        for i, step in enumerate(remaining_steps):
            prompt += f"{i+1}. {step['technique']}\n"
        prompt += "\n"
        
        prompt += "Available Analytical Techniques:\n"
        for technique in available_techniques:
            prompt += f"- {technique}\n"
        prompt += "\n"
        
        prompt += "Please adapt the remaining steps of the analysis strategy to address the adaptation triggers. Provide your adapted strategy in JSON format with an 'adapted_steps' array."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                adapted_strategy = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    adapted_strategy = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        adapted_strategy = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            if "adapted_steps" not in adapted_strategy:
                adapted_strategy["adapted_steps"] = []
            
            # Validate each step
            for i, step in enumerate(adapted_strategy["adapted_steps"]):
                if "technique" not in step:
                    step["technique"] = "unknown"
                if "parameters" not in step:
                    step["parameters"] = {}
                if "dependencies" not in step:
                    step["dependencies"] = []
                if "optional" not in step:
                    step["optional"] = False
                
                # Validate technique is available
                if step["technique"] not in available_techniques:
                    logger.warning(f"Technique {step['technique']} is not available, replacing with default")
                    step["technique"] = available_techniques[0] if available_techniques else "unknown"
            
            return adapted_strategy
            
        except Exception as e:
            error_msg = f"Error parsing adapted strategy: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _synthesize_results(self, input_data: Dict) -> Dict:
        """
        Synthesize analysis results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Synthesis of results
        """
        logger.info("Synthesizing analysis results")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get step results
        step_results = input_data.get("step_results", {})
        if not step_results:
            error_msg = "No step results provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Prepare system prompt
        system_prompt = """You are an expert analytical synthesist specializing in integrating findings from multiple analytical techniques. Your task is to synthesize the results of an analysis workflow into a coherent response to the original question. Provide your synthesis in JSON format with the following fields:
- summary: A concise summary of the analysis findings
- key_findings: List of key findings from the analysis
- confidence_assessment: Assessment of confidence in the findings (overall_confidence and explanation)
- uncertainties: List of key uncertainties in the analysis (each with uncertainty and impact)
- recommendations: List of recommendations based on the analysis (optional)

Ensure your synthesis is balanced, acknowledges uncertainties, and directly addresses the original question."""
        
        # Build prompt
        prompt = f"Question: {question}\n\n"
        
        prompt += "Analysis Results:\n"
        for technique, result in step_results.items():
            prompt += f"\n## {technique} Results:\n"
            
            if "findings" in result:
                prompt += "Findings:\n"
                for finding in result["findings"]:
                    prompt += f"- {finding.get('finding', '')}\n"
            
            if "hypotheses" in result:
                prompt += "Hypotheses:\n"
                for hypothesis in result["hypotheses"]:
                    prompt += f"- {hypothesis.get('hypothesis', '')} (Confidence: {hypothesis.get('confidence', 'unknown')})\n"
            
            if "confidence_assessment" in result:
                prompt += f"Confidence: {result['confidence_assessment'].get('overall_confidence', 'unknown')}\n"
            
            if "uncertainties" in result:
                prompt += "Uncertainties:\n"
                for uncertainty in result["uncertainties"]:
                    prompt += f"- {uncertainty.get('uncertainty', '')}\n"
        
        prompt += "\nPlease synthesize these analysis results into a coherent response to the original question. Provide your synthesis in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                synthesis = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    synthesis = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        synthesis = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            required_fields = ["summary", "key_findings", "confidence_assessment", "uncertainties"]
            for field in required_fields:
                if field not in synthesis:
                    if field == "confidence_assessment":
                        synthesis[field] = {"overall_confidence": "medium", "explanation": "Default confidence assessment"}
                    else:
                        synthesis[field] = []
            
            return synthesis
            
        except Exception as e:
            error_msg = f"Error parsing synthesis: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _final_synthesis(self, input_data: Dict) -> Dict:
        """
        Generate final synthesis based on synthesis result and all step results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Final synthesis
        """
        logger.info("Generating final synthesis")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get synthesis result
        synthesis_result = input_data.get("synthesis_result", {})
        if not synthesis_result:
            error_msg = "No synthesis result provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get step results
        step_results = input_data.get("step_results", {})
        
        # Prepare system prompt
        system_prompt = """You are an expert analytical synthesist specializing in creating final analytical products. Your task is to refine and enhance the synthesis of an analysis workflow into a polished final response to the original question. Provide your final synthesis in JSON format with the following fields:
- summary: A concise summary of the analysis findings
- key_findings: List of key findings from the analysis
- confidence_assessment: Assessment of confidence in the findings (overall_confidence and explanation)
- uncertainties: List of key uncertainties in the analysis (each with uncertainty and impact)
- recommendations: List of recommendations based on the analysis
- limitations: List of limitations of the analysis
- future_research: Suggestions for future research

Ensure your final synthesis is comprehensive, balanced, acknowledges uncertainties, and directly addresses the original question."""
        
        # Build prompt
        prompt = f"Question: {question}\n\n"
        
        prompt += "Initial Synthesis:\n"
        prompt += json.dumps(synthesis_result, indent=2)
        prompt += "\n\n"
        
        if step_results:
            prompt += "Additional Analysis Results:\n"
            for technique, result in step_results.items():
                if technique != "synthesis_generation":
                    prompt += f"\n## {technique} Results:\n"
                    
                    if "findings" in result:
                        prompt += "Findings:\n"
                        for finding in result["findings"]:
                            prompt += f"- {finding.get('finding', '')}\n"
                    
                    if "hypotheses" in result:
                        prompt += "Hypotheses:\n"
                        for hypothesis in result["hypotheses"]:
                            prompt += f"- {hypothesis.get('hypothesis', '')} (Confidence: {hypothesis.get('confidence', 'unknown')})\n"
                    
                    if "uncertainties" in result:
                        prompt += "Uncertainties:\n"
                        for uncertainty in result["uncertainties"]:
                            prompt += f"- {uncertainty.get('uncertainty', '')}\n"
        
        prompt += "\nPlease refine and enhance this synthesis into a polished final response to the original question. Provide your final synthesis in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                final_synthesis = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    final_synthesis = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        final_synthesis = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            required_fields = ["summary", "key_findings", "confidence_assessment", "uncertainties", "recommendations"]
            for field in required_fields:
                if field not in final_synthesis:
                    if field == "confidence_assessment":
                        final_synthesis[field] = {"overall_confidence": "medium", "explanation": "Default confidence assessment"}
                    else:
                        final_synthesis[field] = []
            
            # Add question to final synthesis
            final_synthesis["question"] = question
            
            return final_synthesis
            
        except Exception as e:
            error_msg = f"Error parsing final synthesis: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _analyze_evidence(self, input_data: Dict) -> Dict:
        """
        Analyze evidence to assess its relevance and credibility.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Evidence analysis
        """
        logger.info("Analyzing evidence")
        
        # Get evidence
        evidence = input_data.get("evidence", [])
        if not evidence:
            error_msg = "No evidence provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get hypotheses if available
        hypotheses = input_data.get("hypotheses", [])
        
        # Prepare system prompt
        system_prompt = """You are an expert evidence analyst specializing in assessing the relevance, credibility, and implications of evidence. Your task is to analyze the given evidence and provide an assessment. If hypotheses are provided, assess how the evidence relates to each hypothesis. Provide your analysis in JSON format with the following fields:
- evidence_assessments: List of assessments for each evidence item (relevance, credibility, implications)
- hypothesis_impacts: List of impacts on each hypothesis (if hypotheses are provided)
- overall_assessment: Overall assessment of the evidence

Ensure your analysis is objective, balanced, and considers alternative interpretations of the evidence."""
        
        # Build prompt
        prompt = "Evidence to Analyze:\n"
        for i, item in enumerate(evidence):
            prompt += f"{i+1}. {item}\n"
        prompt += "\n"
        
        if hypotheses:
            prompt += "Hypotheses to Consider:\n"
            for i, hypothesis in enumerate(hypotheses):
                prompt += f"{i+1}. {hypothesis}\n"
            prompt += "\n"
        
        prompt += "Please analyze this evidence and provide a detailed assessment in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        analysis = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            if "evidence_assessments" not in analysis:
                analysis["evidence_assessments"] = []
            
            if "overall_assessment" not in analysis:
                analysis["overall_assessment"] = "No overall assessment provided."
            
            return analysis
            
        except Exception as e:
            error_msg = f"Error parsing evidence analysis: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _detect_biases(self, input_data: Dict) -> Dict:
        """
        Detect potential biases in analysis.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Bias detection results
        """
        logger.info("Detecting biases")
        
        # Get analysis content
        analysis_content = input_data.get("analysis_content", "")
        if not analysis_content:
            error_msg = "No analysis content provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get findings if available
        findings = input_data.get("findings", [])
        
        # Get hypotheses if available
        hypotheses = input_data.get("hypotheses", [])
        
        # Prepare system prompt
        system_prompt = """You are an expert bias detector specializing in identifying cognitive biases in analytical content. Your task is to detect potential biases in the given analysis content, findings, and hypotheses. Provide your detection results in JSON format with the following fields:
- detected_biases: List of detected biases (each with bias_type, description, severity, and mitigation)
- overall_bias_assessment: Overall assessment of bias in the analysis
- structural_debiasing_recommendations: Recommendations for structural debiasing

Focus on common analytical biases such as confirmation bias, availability bias, anchoring, groupthink, and hindsight bias."""
        
        # Build prompt
        prompt = "Analysis Content to Examine:\n"
        prompt += analysis_content
        prompt += "\n\n"
        
        if findings:
            prompt += "Findings to Examine:\n"
            for i, finding in enumerate(findings):
                prompt += f"{i+1}. {finding}\n"
            prompt += "\n"
        
        if hypotheses:
            prompt += "Hypotheses to Examine:\n"
            for i, hypothesis in enumerate(hypotheses):
                prompt += f"{i+1}. {hypothesis}\n"
            prompt += "\n"
        
        prompt += "Please detect potential biases in this analytical content and provide a detailed assessment in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                detection = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    detection = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        detection = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            if "detected_biases" not in detection:
                detection["detected_biases"] = []
            
            if "overall_bias_assessment" not in detection:
                detection["overall_bias_assessment"] = "No overall bias assessment provided."
            
            if "structural_debiasing_recommendations" not in detection:
                detection["structural_debiasing_recommendations"] = []
            
            return detection
            
        except Exception as e:
            error_msg = f"Error parsing bias detection: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _evaluate_uncertainty(self, input_data: Dict) -> Dict:
        """
        Evaluate uncertainty in analysis.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Uncertainty evaluation
        """
        logger.info("Evaluating uncertainty")
        
        # Get analysis content
        analysis_content = input_data.get("analysis_content", "")
        if not analysis_content:
            error_msg = "No analysis content provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get findings if available
        findings = input_data.get("findings", [])
        
        # Get hypotheses if available
        hypotheses = input_data.get("hypotheses", [])
        
        # Prepare system prompt
        system_prompt = """You are an expert uncertainty analyst specializing in identifying and characterizing uncertainties in analytical content. Your task is to evaluate uncertainties in the given analysis content, findings, and hypotheses. Provide your evaluation in JSON format with the following fields:
- identified_uncertainties: List of identified uncertainties (each with description, type, impact, and reducibility)
- overall_uncertainty_assessment: Overall assessment of uncertainty in the analysis
- uncertainty_visualization_recommendations: Recommendations for visualizing the uncertainties

Focus on different types of uncertainties such as data uncertainty, model uncertainty, linguistic uncertainty, and decision uncertainty."""
        
        # Build prompt
        prompt = "Analysis Content to Examine:\n"
        prompt += analysis_content
        prompt += "\n\n"
        
        if findings:
            prompt += "Findings to Examine:\n"
            for i, finding in enumerate(findings):
                prompt += f"{i+1}. {finding}\n"
            prompt += "\n"
        
        if hypotheses:
            prompt += "Hypotheses to Examine:\n"
            for i, hypothesis in enumerate(hypotheses):
                prompt += f"{i+1}. {hypothesis}\n"
            prompt += "\n"
        
        prompt += "Please evaluate uncertainties in this analytical content and provide a detailed assessment in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                evaluation = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        evaluation = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            if "identified_uncertainties" not in evaluation:
                evaluation["identified_uncertainties"] = []
            
            if "overall_uncertainty_assessment" not in evaluation:
                evaluation["overall_uncertainty_assessment"] = "No overall uncertainty assessment provided."
            
            if "uncertainty_visualization_recommendations" not in evaluation:
                evaluation["uncertainty_visualization_recommendations"] = []
            
            return evaluation
            
        except Exception as e:
            error_msg = f"Error parsing uncertainty evaluation: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def _generate_hypotheses(self, input_data: Dict) -> Dict:
        """
        Generate hypotheses based on research data.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Generated hypotheses
        """
        logger.info("Generating hypotheses")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get research data
        research_data = input_data.get("research_data", {})
        
        # Get key insights if available
        key_insights = input_data.get("key_insights", [])
        
        # Prepare system prompt
        system_prompt = """You are an expert hypothesis generator specializing in creating well-formed, testable hypotheses based on research data. Your task is to generate a set of competing hypotheses that could answer the given question. Provide your hypotheses in JSON format with the following fields:
- hypotheses: List of hypotheses (each with id, hypothesis, confidence, evidence, contradicts)
- hypothesis_relationships: Relationships between hypotheses (complementary, contradictory, independent)
- key_uncertainties: Key uncertainties that affect hypothesis evaluation

Ensure your hypotheses are diverse, covering different possible explanations, and are specific enough to be testable."""
        
        # Build prompt
        prompt = f"Question: {question}\n\n"
        
        if key_insights:
            prompt += "Key Insights from Research:\n"
            for i, insight in enumerate(key_insights):
                prompt += f"{i+1}. {insight}\n"
            prompt += "\n"
        
        if research_data:
            prompt += "Research Data Summary:\n"
            prompt += json.dumps(research_data, indent=2)[:2000]  # Limit to 2000 chars to avoid token limits
            prompt += "\n\n"
        
        prompt += "Please generate a set of competing hypotheses that could answer this question. Provide your hypotheses in JSON format."
        
        # Call LLM
        llm_response = self._call_llm(prompt, system_prompt)
        
        # Check for errors
        if "error" in llm_response:
            return llm_response
        
        # Parse response
        try:
            content = llm_response["content"]
            
            # Try to extract JSON from the response
            try:
                # Check if the response is already valid JSON
                hypotheses = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    hypotheses = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_like:
                        hypotheses = json.loads(json_like.group(1))
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
            
            # Validate required fields
            if "hypotheses" not in hypotheses:
                hypotheses["hypotheses"] = []
            
            # Validate each hypothesis
            for i, hypothesis in enumerate(hypotheses["hypotheses"]):
                if "id" not in hypothesis:
                    hypothesis["id"] = f"H{i+1}"
                if "hypothesis" not in hypothesis:
                    hypothesis["hypothesis"] = f"Hypothesis {i+1}"
                if "confidence" not in hypothesis:
                    hypothesis["confidence"] = 0.5
                if "evidence" not in hypothesis:
                    hypothesis["evidence"] = []
                if "contradicts" not in hypothesis:
                    hypothesis["contradicts"] = []
            
            return hypotheses
            
        except Exception as e:
            error_msg = f"Error parsing hypotheses: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "llm_response": llm_response["content"]}
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this MCP.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "model": self.model,
            "operations": list(self.operation_handlers.keys()),
            "api_available": bool(self.api_key)
        }
