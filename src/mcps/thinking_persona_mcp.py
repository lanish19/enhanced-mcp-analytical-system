"""
Cognitive Diversity Module for implementing diverse thinking personas.
This module provides the ThinkingPersonaMCP class for applying different cognitive perspectives to analysis.
"""

import logging
import time
import json
import random
from typing import Dict, List, Any, Optional

from src.base_mcp import BaseMCP
from src.utils.llm_integration import call_llm, parse_json_response, LLMCallError, LLMParsingError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThinkingPersonaMCP(BaseMCP):
    """
    Thinking Persona MCP for applying diverse cognitive perspectives to analysis.
    
    This MCP provides capabilities for:
    1. Applying different thinking styles to analysis
    2. Generating insights from diverse cognitive perspectives
    3. Challenging assumptions through cognitive diversity
    4. Enhancing analytical depth through multiple viewpoints
    5. Reducing cognitive biases through perspective-taking
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ThinkingPersonaMCP.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__(
            name="ThinkingPersonaMCP",
            description="Applies diverse cognitive perspectives to analysis",
            version="1.0.0"
        )
        
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.model = self.config.get("model", "gpt-4o")
        
        # Define thinking personas
        self.personas = {
            "analytical": {
                "description": "Logical, methodical, and detail-oriented thinker focused on breaking down problems into components",
                "strengths": ["logical reasoning", "systematic analysis", "attention to detail", "critical evaluation"],
                "weaknesses": ["may overlook intuitive insights", "can be overly cautious", "may miss creative alternatives"],
                "prompt_style": "Analyze this systematically, breaking it down into logical components. Consider the evidence methodically and evaluate each element critically."
            },
            "creative": {
                "description": "Imaginative, innovative thinker focused on generating novel possibilities and connections",
                "strengths": ["idea generation", "novel connections", "unconventional approaches", "future possibilities"],
                "weaknesses": ["may lack practical grounding", "can overlook constraints", "sometimes imprecise"],
                "prompt_style": "Think creatively about this, generating novel possibilities and unexpected connections. Consider unconventional approaches and imagine future scenarios."
            },
            "strategic": {
                "description": "Big-picture thinker focused on long-term implications, patterns, and strategic positioning",
                "strengths": ["long-term perspective", "pattern recognition", "systems thinking", "strategic positioning"],
                "weaknesses": ["may miss tactical details", "can overlook immediate concerns", "sometimes too abstract"],
                "prompt_style": "Take a strategic perspective, focusing on the big picture and long-term implications. Identify patterns and consider how elements interact as a system."
            },
            "skeptical": {
                "description": "Questioning, critical thinker focused on identifying flaws, contradictions, and alternative explanations",
                "strengths": ["identifying weaknesses", "challenging assumptions", "considering alternatives", "detecting inconsistencies"],
                "weaknesses": ["can be overly negative", "may undervalue consensus views", "sometimes dismissive"],
                "prompt_style": "Adopt a skeptical mindset, questioning assumptions and identifying potential flaws. Consider alternative explanations and look for contradictions or inconsistencies."
            },
            "pragmatic": {
                "description": "Practical, results-oriented thinker focused on feasibility, implementation, and real-world constraints",
                "strengths": ["practical application", "resource awareness", "implementation focus", "constraint recognition"],
                "weaknesses": ["may limit creative thinking", "can be too conservative", "sometimes short-sighted"],
                "prompt_style": "Think pragmatically about this, focusing on practical implementation and real-world constraints. Consider feasibility, resources required, and concrete next steps."
            },
            "empathetic": {
                "description": "Human-centered thinker focused on stakeholder perspectives, values, and social/emotional factors",
                "strengths": ["stakeholder consideration", "values awareness", "social impact understanding", "emotional intelligence"],
                "weaknesses": ["may overweight subjective factors", "can lack analytical rigor", "sometimes biased by empathy"],
                "prompt_style": "Consider this from an empathetic perspective, focusing on how different stakeholders would be affected. Think about values, social impacts, and emotional factors involved."
            }
        }
        
        # Operation handlers
        self.operation_handlers = {
            "apply_persona": self._apply_persona,
            "multi_persona_analysis": self._multi_persona_analysis,
            "get_personas": self._get_personas,
            "cognitive_diversity_check": self._cognitive_diversity_check,
            "perspective_shift": self._perspective_shift
        }
        
        logger.info("Initialized ThinkingPersonaMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in ThinkingPersonaMCP")
        
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
    
    def _apply_persona(self, input_data: Dict) -> Dict:
        """
        Apply a specific thinking persona to analyze content.
        
        Args:
            input_data: Input data dictionary with persona and content
            
        Returns:
            Analysis results from the persona's perspective
        """
        logger.info("Applying thinking persona")
        
        # Get persona
        persona_name = input_data.get("persona", "")
        if not persona_name:
            error_msg = "No persona specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Check if persona exists
        if persona_name not in self.personas:
            error_msg = f"Unknown persona: {persona_name}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get content to analyze
        content = input_data.get("content", "")
        if not content:
            error_msg = "No content provided for analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context if provided
        context = input_data.get("context", {})
        
        # Get persona details
        persona = self.personas[persona_name]
        
        # Construct prompt for persona-based analysis
        system_prompt = f"""You are an expert analyst with a {persona_name} thinking style. 
        {persona['description']}
        
        Your strengths include: {', '.join(persona['strengths'])}
        
        Analyze the provided content from your {persona_name} perspective, highlighting insights 
        that would be particularly visible through this cognitive lens. Maintain this perspective 
        consistently throughout your analysis."""
        
        prompt = f"CONTENT TO ANALYZE:\n{content}\n\n"
        
        if context:
            prompt += f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        
        prompt += f"{persona['prompt_style']}\n\n"
        
        prompt += """Structure your response as JSON with the following fields:
        - persona_perspective: Overall assessment from your specific thinking perspective
        - key_insights: List of insights particularly visible from this perspective
        - blind_spots: Potential blind spots or limitations of this perspective
        - questions_raised: Important questions this perspective would ask
        - recommendations: Recommendations based on this perspective's analysis
        - confidence: Confidence level in the analysis (low, medium, high)"""
        
        # Call LLM for persona-based analysis
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.4  # Slightly higher temperature for persona diversity
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["persona"] = persona_name
            parsed_result["persona_description"] = persona["description"]
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "apply_persona",
                "persona": persona_name,
                "input": content,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error applying {persona_name} persona: {str(e)}")
            return {
                "error": str(e),
                "operation": "apply_persona",
                "persona": persona_name,
                "input": content
            }
    
    def _multi_persona_analysis(self, input_data: Dict) -> Dict:
        """
        Apply multiple thinking personas to analyze content.
        
        Args:
            input_data: Input data dictionary with personas and content
            
        Returns:
            Analysis results from multiple perspectives with synthesis
        """
        logger.info("Performing multi-persona analysis")
        
        # Get content to analyze
        content = input_data.get("content", "")
        if not content:
            error_msg = "No content provided for analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get personas to apply
        personas = input_data.get("personas", [])
        if not personas:
            # Default to using all personas
            personas = list(self.personas.keys())
        
        # Validate personas
        valid_personas = [p for p in personas if p in self.personas]
        if not valid_personas:
            error_msg = "No valid personas specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Apply each persona
        persona_results = {}
        for persona_name in valid_personas:
            logger.info(f"Applying {persona_name} persona")
            
            # Create input for single persona analysis
            persona_input = {
                "persona": persona_name,
                "content": content,
                "context": input_data.get("context", {})
            }
            
            # Apply persona
            result = self._apply_persona(persona_input)
            
            # Check for errors
            if "error" in result:
                logger.warning(f"Error applying {persona_name} persona: {result['error']}")
                continue
            
            # Store result
            persona_results[persona_name] = result["output"]
        
        # Check if we have any valid results
        if not persona_results:
            error_msg = "No valid persona analyses generated"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Generate synthesis of multiple perspectives
        synthesis = self._synthesize_perspectives(content, persona_results)
        
        return {
            "operation": "multi_persona_analysis",
            "personas_applied": list(persona_results.keys()),
            "input": content,
            "persona_results": persona_results,
            "synthesis": synthesis
        }
    
    def _synthesize_perspectives(self, content: str, persona_results: Dict) -> Dict:
        """
        Synthesize insights from multiple persona perspectives.
        
        Args:
            content: Original content analyzed
            persona_results: Results from multiple personas
            
        Returns:
            Synthesized insights
        """
        logger.info("Synthesizing multiple perspectives")
        
        # Construct prompt for synthesis
        system_prompt = """You are an expert meta-analyst specializing in integrating diverse cognitive perspectives. 
        Your task is to synthesize insights from multiple thinking styles, identifying areas of consensus, 
        complementary insights, and productive tensions between different perspectives."""
        
        prompt = f"ORIGINAL CONTENT:\n{content}\n\n"
        prompt += "PERSPECTIVES FROM DIFFERENT THINKING STYLES:\n"
        
        for persona_name, result in persona_results.items():
            prompt += f"\n{persona_name.upper()} PERSPECTIVE:\n"
            prompt += f"Overall assessment: {result.get('persona_perspective', 'Not provided')}\n"
            
            key_insights = result.get('key_insights', [])
            if key_insights:
                prompt += "Key insights:\n"
                for i, insight in enumerate(key_insights):
                    prompt += f"- {insight}\n"
            
            questions = result.get('questions_raised', [])
            if questions:
                prompt += "Questions raised:\n"
                for i, question in enumerate(questions):
                    prompt += f"- {question}\n"
        
        prompt += """\nPlease synthesize these diverse perspectives into an integrated analysis. 
        Structure your response as JSON with the following fields:
        - integrated_assessment: Overall assessment that integrates multiple perspectives
        - areas_of_consensus: Points where different perspectives converge
        - complementary_insights: How different perspectives complement each other
        - productive_tensions: Valuable tensions or contradictions between perspectives
        - blind_spots_addressed: How cognitive diversity addresses potential blind spots
        - integrated_recommendations: Recommendations informed by multiple perspectives
        - meta_insights: Insights about the analysis process itself"""
        
        # Call LLM for synthesis
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
            parsed_result["perspectives_integrated"] = list(persona_results.keys())
            parsed_result["timestamp"] = time.time()
            
            return parsed_result
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error synthesizing perspectives: {str(e)}")
            
            # Return basic synthesis on error
            return {
                "error": str(e),
                "integrated_assessment": "Error generating integrated assessment",
                "perspectives_integrated": list(persona_results.keys()),
                "timestamp": time.time()
            }
    
    def _get_personas(self, input_data: Dict) -> Dict:
        """
        Get information about available thinking personas.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Information about available personas
        """
        logger.info("Getting persona information")
        
        # Get specific persona if requested
        persona_name = input_data.get("persona")
        if persona_name:
            if persona_name in self.personas:
                persona_info = self.personas[persona_name]
                return {
                    "operation": "get_personas",
                    "persona": persona_name,
                    "info": persona_info
                }
            else:
                return {
                    "error": f"Unknown persona: {persona_name}",
                    "operation": "get_personas",
                    "available_personas": list(self.personas.keys())
                }
        
        # Return all personas
        return {
            "operation": "get_personas",
            "available_personas": list(self.personas.keys()),
            "personas": self.personas
        }
    
    def _cognitive_diversity_check(self, input_data: Dict) -> Dict:
        """
        Check if an analysis would benefit from additional cognitive perspectives.
        
        Args:
            input_data: Input data dictionary with analysis content
            
        Returns:
            Assessment of cognitive diversity needs
        """
        logger.info("Performing cognitive diversity check")
        
        # Get analysis to check
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for cognitive diversity check"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get current perspectives if provided
        current_perspectives = input_data.get("current_perspectives", [])
        
        # Construct prompt for diversity check
        system_prompt = """You are an expert in cognitive diversity and analytical thinking. 
        Your task is to assess whether an analysis would benefit from additional cognitive 
        perspectives, identifying potential blind spots and recommending specific thinking 
        styles that would enhance the analysis."""
        
        prompt = f"ANALYSIS TO EVALUATE:\n{analysis}\n\n"
        
        if current_perspectives:
            prompt += f"PERSPECTIVES ALREADY APPLIED: {', '.join(current_perspectives)}\n\n"
        
        prompt += """Please assess the cognitive diversity of this analysis. 
        Structure your response as JSON with the following fields:
        - diversity_assessment: Overall assessment of cognitive diversity in the analysis
        - identified_perspectives: Thinking perspectives that appear to be represented
        - missing_perspectives: Thinking perspectives that would add valuable diversity
        - potential_blind_spots: Blind spots due to limited cognitive diversity
        - recommended_personas: Specific thinking personas to apply (from: analytical, creative, strategic, skeptical, pragmatic, empathetic)
        - diversity_score: Score from 1-10 representing the cognitive diversity of the analysis"""
        
        # Call LLM for diversity check
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
            parsed_result["current_perspectives"] = current_perspectives
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "cognitive_diversity_check",
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in cognitive diversity check: {str(e)}")
            return {
                "error": str(e),
                "operation": "cognitive_diversity_check",
                "input": analysis
            }
    
    def _perspective_shift(self, input_data: Dict) -> Dict:
        """
        Deliberately shift perspective on an existing analysis.
        
        Args:
            input_data: Input data dictionary with analysis and target perspective
            
        Returns:
            Re-analysis from the shifted perspective
        """
        logger.info("Performing perspective shift")
        
        # Get analysis to re-examine
        analysis = input_data.get("analysis", "")
        if not analysis:
            error_msg = "No analysis provided for perspective shift"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get target perspective
        target_perspective = input_data.get("target_perspective", "")
        if not target_perspective:
            # Select random perspective if not specified
            target_perspective = random.choice(list(self.personas.keys()))
            logger.info(f"No target perspective specified, randomly selected: {target_perspective}")
        
        # Check if perspective exists
        if target_perspective not in self.personas:
            error_msg = f"Unknown perspective: {target_perspective}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get original perspective if provided
        original_perspective = input_data.get("original_perspective", "unspecified")
        
        # Get persona details
        persona = self.personas[target_perspective]
        
        # Construct prompt for perspective shift
        system_prompt = f"""You are an expert analyst with a {target_perspective} thinking style. 
        {persona['description']}
        
        Your task is to re-examine an existing analysis from your {target_perspective} perspective, 
        highlighting new insights, challenging assumptions, and offering alternative interpretations."""
        
        prompt = f"ORIGINAL ANALYSIS:\n{analysis}\n\n"
        prompt += f"ORIGINAL PERSPECTIVE: {original_perspective}\n\n"
        prompt += f"NEW PERSPECTIVE: {target_perspective}\n\n"
        prompt += f"{persona['prompt_style']}\n\n"
        
        prompt += """Re-examine this analysis from your perspective. 
        Structure your response as JSON with the following fields:
        - perspective_shift: Overall re-assessment from your specific thinking perspective
        - challenged_assumptions: Assumptions in the original analysis that you would challenge
        - new_insights: New insights visible from your perspective
        - alternative_interpretations: Alternative interpretations of the evidence
        - additional_considerations: Important factors the original analysis may have overlooked
        - revised_recommendations: How recommendations might change from your perspective"""
        
        # Call LLM for perspective shift
        try:
            llm_response = call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                model=self.model,
                api_key=self.api_key,
                temperature=0.4  # Slightly higher temperature for perspective diversity
            )
            
            # Parse JSON response
            parsed_result = parse_json_response(llm_response)
            
            # Add metadata
            parsed_result["original_perspective"] = original_perspective
            parsed_result["new_perspective"] = target_perspective
            parsed_result["persona_description"] = persona["description"]
            parsed_result["timestamp"] = time.time()
            
            return {
                "operation": "perspective_shift",
                "original_perspective": original_perspective,
                "new_perspective": target_perspective,
                "input": analysis,
                "output": parsed_result
            }
            
        except (LLMCallError, LLMParsingError) as e:
            logger.error(f"Error in perspective shift: {str(e)}")
            return {
                "error": str(e),
                "operation": "perspective_shift",
                "original_perspective": original_perspective,
                "new_perspective": target_perspective,
                "input": analysis
            }
