"""
LLM Integration utilities for the MCP Analytical System.
This module provides functions for calling LLMs and parsing responses.
"""

import logging
import json
import time
import random
from typing import Dict, Any, Optional, List, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import tenacity for retry logic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Define custom exceptions
class LLMCallError(Exception):
    """Exception raised when there's an error calling the LLM."""
    pass

class LLMParsingError(Exception):
    """Exception raised when there's an error parsing the LLM response."""
    pass

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    api_type: str = "openai",
    response_format: Optional[Dict] = None
) -> str:
    """
    Call an LLM with the given prompt and parameters.
    
    Args:
        prompt: The prompt to send to the LLM
        system_prompt: Optional system prompt for the LLM
        model: The model to use
        api_key: API key for the LLM provider
        temperature: Temperature parameter for the LLM
        max_tokens: Maximum number of tokens to generate
        api_type: Type of API to use (openai, anthropic, groq)
        response_format: Optional response format specification
        
    Returns:
        The LLM response as a string
    
    Raises:
        LLMCallError: If there's an error calling the LLM
    """
    logger.info(f"Calling LLM with API type: {api_type}")
    
    # Check if API key is provided
    if not api_key:
        logger.warning("No API key provided, using mock LLM response")
        return generate_mock_llm_response(prompt, system_prompt, api_type)
    
    try:
        # Call the appropriate API based on api_type
        if api_type.lower() == "openai":
            return call_openai_api(prompt, system_prompt, model, api_key, temperature, max_tokens, response_format)
        elif api_type.lower() == "anthropic":
            return call_anthropic_api(prompt, system_prompt, model, api_key, temperature, max_tokens)
        elif api_type.lower() == "groq":
            return call_groq_api(prompt, system_prompt, model, api_key, temperature, max_tokens)
        else:
            error_msg = f"Unsupported API type: {api_type}"
            logger.error(error_msg)
            raise LLMCallError(error_msg)
    
    except Exception as e:
        error_msg = f"Error calling LLM: {str(e)}"
        logger.error(error_msg)
        raise LLMCallError(error_msg)

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_openai_api(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[Dict]
) -> str:
    """
    Call the OpenAI API with the given parameters.
    
    Args:
        prompt: The prompt to send to the API
        system_prompt: Optional system prompt
        model: The model to use
        api_key: OpenAI API key
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens to generate
        response_format: Optional response format specification
        
    Returns:
        The API response as a string
    
    Raises:
        Exception: If there's an error calling the API
    """
    try:
        import openai
        
        # Configure the client
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if response_format:
            params["response_format"] = response_format
        
        # Call the API
        response = client.chat.completions.create(**params)
        
        # Extract and return the response content
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_anthropic_api(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: Optional[int]
) -> str:
    """
    Call the Anthropic API with the given parameters.
    
    Args:
        prompt: The prompt to send to the API
        system_prompt: Optional system prompt
        model: The model to use
        api_key: Anthropic API key
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The API response as a string
    
    Raises:
        Exception: If there's an error calling the API
    """
    try:
        import anthropic
        
        # Configure the client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        if system_prompt:
            params["system"] = system_prompt
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Call the API
        response = client.messages.create(**params)
        
        # Extract and return the response content
        return response.content[0].text
    
    except Exception as e:
        logger.error(f"Error calling Anthropic API: {str(e)}")
        raise

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_groq_api(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: Optional[int]
) -> str:
    """
    Call the Groq API with the given parameters.
    
    Args:
        prompt: The prompt to send to the API
        system_prompt: Optional system prompt
        model: The model to use
        api_key: Groq API key
        temperature: Temperature parameter
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The API response as a string
    
    Raises:
        Exception: If there's an error calling the API
    """
    try:
        import groq
        
        # Configure the client
        client = groq.Groq(api_key=api_key)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Call the API
        response = client.chat.completions.create(**params)
        
        # Extract and return the response content
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise

def generate_mock_llm_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_type: str = "openai"
) -> str:
    """
    Generate a mock LLM response for testing purposes.
    
    Args:
        prompt: The prompt that would be sent to the LLM
        system_prompt: Optional system prompt
        api_type: Type of API being mocked
        
    Returns:
        A mock LLM response as a string
    """
    logger.info(f"Generating mock LLM response for {api_type}")
    
    # Check if prompt is asking for JSON
    if "JSON" in prompt or "json" in prompt:
        # Generate mock JSON response based on prompt content
        if "physical sciences" in prompt or "physics" in prompt:
            return generate_mock_physical_sciences_json()
        elif "life sciences" in prompt or "biology" in prompt:
            return generate_mock_life_sciences_json()
        elif "economic" in prompt or "financial" in prompt:
            return generate_mock_economic_financial_json()
        elif "persona" in prompt or "thinking style" in prompt:
            return generate_mock_thinking_persona_json()
        elif "bias" in prompt or "cognitive bias" in prompt:
            return generate_mock_bias_detection_json()
        elif "assumption" in prompt:
            return generate_mock_assumption_challenge_json()
        elif "uncertainty" in prompt:
            return generate_mock_uncertainty_json()
        elif "question analysis" in prompt:
            return generate_mock_question_analysis_json()
        else:
            # Generic JSON response
            return generate_mock_generic_json()
    else:
        # Generate mock text response
        return "This is a mock LLM response for testing purposes. In a production environment, this would be replaced with an actual response from the LLM API."

def parse_json_response(response: str) -> Dict:
    """
    Parse a JSON response from an LLM.
    
    Args:
        response: The LLM response as a string
        
    Returns:
        The parsed JSON as a dictionary
        
    Raises:
        LLMParsingError: If there's an error parsing the JSON
    """
    try:
        # Try to parse the entire response as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the response
        try:
            # Look for JSON between triple backticks
            import re
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Look for JSON between curly braces
            json_match = re.search(r"(\{[\s\S]*\})", response)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If we can't find JSON, raise an error
            raise LLMParsingError("Could not extract JSON from LLM response")
        
        except Exception as e:
            error_msg = f"Error parsing JSON response: {str(e)}"
            logger.error(error_msg)
            raise LLMParsingError(error_msg)

# Mock JSON response generators for testing
def generate_mock_physical_sciences_json() -> str:
    """Generate a mock JSON response for physical sciences analysis."""
    return """
    {
        "domain_assessment": "The content discusses quantum computing applications in material science.",
        "key_insights": [
            "Quantum computing enables simulation of quantum systems",
            "Material science applications include new materials discovery",
            "Computational efficiency is significantly improved for certain problems"
        ],
        "sub_domain_analyses": {
            "physics": {
                "relevance": "high",
                "key_points": ["Quantum mechanics principles", "Superposition and entanglement"]
            },
            "chemistry": {
                "relevance": "high",
                "key_points": ["Molecular modeling", "Reaction pathway simulation"]
            },
            "earth_sciences": {
                "relevance": "low",
                "key_points": []
            },
            "climate_science": {
                "relevance": "low",
                "key_points": []
            }
        },
        "confidence": "medium",
        "data_needs": ["Specific quantum algorithms", "Benchmark comparisons"]
    }
    """

def generate_mock_life_sciences_json() -> str:
    """Generate a mock JSON response for life sciences analysis."""
    return """
    {
        "domain_assessment": "The content discusses CRISPR gene editing applications in medicine.",
        "key_insights": [
            "CRISPR enables precise genetic modifications",
            "Potential applications in treating genetic disorders",
            "Ethical considerations around germline editing"
        ],
        "sub_domain_analyses": {
            "genetics": {
                "relevance": "high",
                "key_points": ["DNA modification", "Gene knockout techniques"]
            },
            "ecology": {
                "relevance": "low",
                "key_points": []
            },
            "evolution": {
                "relevance": "medium",
                "key_points": ["Potential impact on natural selection"]
            },
            "biodiversity": {
                "relevance": "low",
                "key_points": []
            }
        },
        "confidence": "high",
        "data_needs": ["Clinical trial results", "Long-term safety studies"]
    }
    """

def generate_mock_economic_financial_json() -> str:
    """Generate a mock JSON response for economic/financial analysis."""
    return """
    {
        "domain_assessment": "The content discusses central bank digital currencies and monetary policy.",
        "key_insights": [
            "CBDCs could transform monetary policy implementation",
            "Potential impacts on commercial banking systems",
            "Privacy and surveillance considerations"
        ],
        "sub_domain_analyses": {
            "macroeconomics": {
                "relevance": "high",
                "key_points": ["Monetary policy transmission", "Financial stability"]
            },
            "markets": {
                "relevance": "high",
                "key_points": ["Banking sector disruption", "Payment systems evolution"]
            },
            "industries": {
                "relevance": "medium",
                "key_points": ["Fintech sector impacts"]
            },
            "trade": {
                "relevance": "medium",
                "key_points": ["Cross-border payment efficiency"]
            }
        },
        "confidence": "medium",
        "data_needs": ["CBDC pilot program results", "Consumer adoption metrics"]
    }
    """

def generate_mock_thinking_persona_json() -> str:
    """Generate a mock JSON response for thinking persona analysis."""
    return """
    {
        "persona_perspective": "From an analytical perspective, AI transformation across industries requires systematic evaluation of technical capabilities, adoption barriers, and sector-specific applications.",
        "key_insights": [
            "Industry transformation will likely occur at different rates based on data availability and problem complexity",
            "Ethical concerns require structured frameworks for evaluation and mitigation",
            "Technical limitations remain significant for certain applications"
        ],
        "blind_spots": [
            "May undervalue cultural and organizational resistance factors",
            "Could overemphasize quantifiable impacts while missing qualitative changes"
        ],
        "questions_raised": [
            "What metrics should be used to measure transformation progress?",
            "How can we systematically evaluate ethical implications across different contexts?"
        ],
        "recommendations": [
            "Develop sector-specific transformation roadmaps with clear metrics",
            "Establish cross-disciplinary ethics review processes"
        ],
        "confidence": "medium"
    }
    """

def generate_mock_bias_detection_json() -> str:
    """Generate a mock JSON response for bias detection."""
    return """
    {
        "overall_assessment": "The analysis exhibits several cognitive biases that may affect its conclusions.",
        "detected_biases": [
            {
                "bias_type": "overconfidence_bias",
                "evidence": "The definitive claim that AI will 'definitely transform all industries within 5 years'",
                "impact": "Leads to underestimation of uncertainty and variability in adoption timelines",
                "severity": "high"
            },
            {
                "bias_type": "technological_determinism",
                "evidence": "Assumption that technological advancement automatically leads to widespread adoption",
                "impact": "Neglects social, economic, and organizational factors affecting technology adoption",
                "severity": "medium"
            }
        ],
        "potential_biases": [
            "recency_bias",
            "availability_bias"
        ],
        "bias_interactions": "Overconfidence and technological determinism reinforce each other, creating an overly simplistic view of technological change",
        "debiasing_recommendations": [
            "Explicitly consider multiple timelines and adoption scenarios",
            "Incorporate historical examples of technology adoption challenges",
            "Seek diverse perspectives on potential barriers to transformation"
        ]
    }
    """

def generate_mock_assumption_challenge_json() -> str:
    """Generate a mock JSON response for assumption challenge."""
    return """
    {
        "overall_assessment": "The analysis relies on several key assumptions that warrant examination.",
        "identified_assumptions": [
            {
                "assumption": "Battery technology will continue to improve rapidly",
                "type": "explicit",
                "evidence": "Statement that 'battery technology is improving rapidly'",
                "validity": "medium",
                "impact": "high",
                "alternatives": "Battery improvement could plateau or face resource constraints"
            },
            {
                "assumption": "Consumer preferences will shift predominantly toward electric vehicles",
                "type": "implicit",
                "evidence": "Claim that EVs will 'dominate the market'",
                "validity": "medium",
                "impact": "high",
                "alternatives": "Hybrid technologies or alternative fuels could capture significant market share"
            }
        ],
        "key_dependencies": "The market dominance prediction depends heavily on both technological improvement and consumer preference assumptions",
        "sensitivity_analysis": "If battery technology improvement slows or consumer preferences remain divided, market penetration could be significantly lower than predicted",
        "recommendations": [
            "Develop multiple scenarios with different battery technology trajectories",
            "Analyze consumer preference data across different market segments",
            "Consider infrastructure development timelines as a potential constraint"
        ]
    }
    """

def generate_mock_uncertainty_json() -> str:
    """Generate a mock JSON response for uncertainty analysis."""
    return """
    {
        "overall_uncertainty_assessment": "The analysis contains significant uncertainties across multiple dimensions.",
        "identified_uncertainties": [
            {
                "uncertainty_type": "model_uncertainty",
                "description": "Uncertainty in climate model projections",
                "evidence": "Wide temperature range (1.5-4.5°C) in projections",
                "impact": "Affects adaptation planning and mitigation urgency",
                "severity": "high"
            },
            {
                "uncertainty_type": "temporal_uncertainty",
                "description": "Uncertainty about timing of temperature changes",
                "evidence": "Long timeframe (until 2100) with unspecified intermediate points",
                "impact": "Complicates near-term policy decisions",
                "severity": "medium"
            }
        ],
        "key_knowledge_gaps": [
            "Regional climate impacts",
            "Feedback loop mechanisms and tipping points"
        ],
        "compounding_factors": "Model and temporal uncertainties interact, creating greater uncertainty about near-term regional impacts",
        "recommendations": [
            "Use ensemble modeling approaches",
            "Develop robust decision-making frameworks that account for uncertainty",
            "Prioritize research on key knowledge gaps"
        ]
    }
    """

def generate_mock_question_analysis_json() -> str:
    """Generate a mock JSON response for question analysis."""
    return """
    {
        "question_type": "impact_assessment",
        "complexity_level": "high",
        "temporal_focus": "future",
        "scope": "industry_specific",
        "relevant_domains": ["economics", "technology", "business"],
        "key_entities": ["quantum computing", "economic impacts"],
        "required_expertise": ["quantum technology", "economic analysis", "industry forecasting"],
        "uncertainty_level": "high",
        "recommended_techniques": [
            "scenario_analysis",
            "expert_consultation",
            "trend_extrapolation",
            "comparative_analysis"
        ],
        "data_requirements": [
            "quantum computing development timeline",
            "industry adoption patterns",
            "economic impact metrics"
        ]
    }
    """

def generate_mock_generic_json() -> str:
    """Generate a generic mock JSON response."""
    return """
    {
        "analysis": "This is a mock analysis for testing purposes.",
        "key_points": [
            "First key point for testing",
            "Second key point for testing",
            "Third key point for testing"
        ],
        "confidence": "medium",
        "timestamp": "2025-04-10T17:00:00Z"
    }
    """
