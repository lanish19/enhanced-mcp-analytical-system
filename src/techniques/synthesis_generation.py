"""
Synthesis Generation Technique for the MCP Analytical System.
This module provides the AnalyticalTechnique class for generating synthesis from analysis results.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalyticalTechnique:
    """
    Synthesis Generation technique for creating a comprehensive synthesis of analysis results.
    
    This technique:
    1. Integrates findings from multiple analytical techniques
    2. Generates key judgments with supporting evidence
    3. Assesses confidence in the overall analysis
    4. Identifies alternative perspectives and limitations
    5. Provides indicators to monitor for future validation
    """
    
    def __init__(self):
        """Initialize the Synthesis Generation technique."""
        self.name = "synthesis_generation"
        self.description = "Generates a comprehensive synthesis of analysis results"
        self.category = "synthesis"
        self.suitable_for_question_types = ["predictive", "causal", "evaluative", "descriptive"]
        
        logger.info("Initialized Synthesis Generation technique")
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Synthesis Generation technique.
        
        Args:
            input_data: Dictionary containing:
                - question: The question being analyzed
                - context: The analysis context with all previous results
                - include_confidence: Whether to include confidence assessments
                
        Returns:
            Dictionary containing the synthesis results
        """
        logger.info("Executing Synthesis Generation technique")
        
        # Extract inputs
        question = input_data.get("question")
        context = input_data.get("context")
        include_confidence = input_data.get("include_confidence", True)
        
        if not question:
            error_msg = "No question provided for synthesis"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not context:
            error_msg = "No context provided for synthesis"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Collect all relevant results from the context
        logger.info("Collecting analysis results from context")
        analysis_results = self._collect_analysis_results(context)
        
        # Generate integrated assessment
        logger.info("Generating integrated assessment")
        integrated_assessment = self._generate_integrated_assessment(question, analysis_results)
        
        # Generate key judgments
        logger.info("Generating key judgments")
        key_judgments = self._generate_key_judgments(analysis_results)
        
        # Assess confidence if requested
        confidence_level = None
        confidence_explanation = None
        if include_confidence:
            logger.info("Assessing confidence in analysis")
            confidence_level, confidence_explanation = self._assess_confidence(analysis_results)
        
        # Identify alternative perspectives
        logger.info("Identifying alternative perspectives")
        alternative_perspectives = self._identify_alternative_perspectives(analysis_results)
        
        # Identify indicators to monitor
        logger.info("Identifying indicators to monitor")
        indicators_to_monitor = self._identify_indicators_to_monitor(question, analysis_results)
        
        # Prepare result
        result = {
            "status": "completed",
            "integrated_assessment": integrated_assessment,
            "key_judgments": key_judgments,
            "alternative_perspectives": alternative_perspectives,
            "indicators_to_monitor": indicators_to_monitor,
            "timestamp": time.time()
        }
        
        # Add confidence if included
        if include_confidence:
            result["confidence_level"] = confidence_level
            result["confidence_explanation"] = confidence_explanation
        
        # Update context with synthesis results
        if context:
            context.add("synthesis_integrated_assessment", integrated_assessment)
            context.add("synthesis_key_judgments", key_judgments)
            context.add("synthesis_alternative_perspectives", alternative_perspectives)
            context.add("synthesis_indicators_to_monitor", indicators_to_monitor)
            if include_confidence:
                context.add("synthesis_confidence_level", confidence_level)
                context.add("synthesis_confidence_explanation", confidence_explanation)
            context.add_event("info", "Synthesis Generation technique completed")
        
        logger.info("Synthesis Generation technique completed")
        return result
    
    def _collect_analysis_results(self, context) -> Dict[str, Any]:
        """
        Collect all relevant analysis results from the context.
        
        Args:
            context: The analysis context
            
        Returns:
            Dictionary of analysis results by technique
        """
        # Get all MCP results from context
        all_results = context.get_mcp_results()
        
        # Filter out internal results and organize by technique
        analysis_results = {}
        for technique_name, result in all_results.items():
            # Skip internal results like retries and fallbacks
            if "_retry_" in technique_name or "_fallback" in technique_name:
                continue
            
            # Skip the current technique to avoid circular reference
            if technique_name == self.name:
                continue
            
            # Add to analysis results
            analysis_results[technique_name] = result
        
        # Also add preliminary research if available
        preliminary_research = {
            "insights": context.get("preliminary_research_insights", []),
            "hypotheses": context.get("preliminary_research_hypotheses", []),
            "recommendations": context.get("preliminary_research_recommendations", [])
        }
        analysis_results["preliminary_research"] = preliminary_research
        
        # Add question analysis if available
        question_analysis = context.get("question_analysis")
        if question_analysis:
            analysis_results["question_analysis"] = question_analysis
        
        return analysis_results
    
    def _generate_integrated_assessment(self, question: str, analysis_results: Dict[str, Any]) -> str:
        """
        Generate an integrated assessment of the analysis results.
        
        Args:
            question: The question being analyzed
            analysis_results: Dictionary of analysis results by technique
            
        Returns:
            Integrated assessment text
        """
        # Extract key findings from research
        research_findings = []
        if "research_to_hypothesis" in analysis_results:
            research_result = analysis_results["research_to_hypothesis"]
            if isinstance(research_result, dict) and "research_findings" in research_result:
                research_findings = research_result["research_findings"]
        
        # Extract hypotheses from research
        hypotheses = []
        if "research_to_hypothesis" in analysis_results:
            research_result = analysis_results["research_to_hypothesis"]
            if isinstance(research_result, dict) and "hypotheses" in research_result:
                hypotheses = research_result["hypotheses"]
        
        # Extract scenarios if available
        scenarios = []
        if "scenario_triangulation" in analysis_results:
            scenario_result = analysis_results["scenario_triangulation"]
            if isinstance(scenario_result, dict) and "scenarios" in scenario_result:
                scenarios = scenario_result["scenarios"]
        
        # Generate assessment based on available information
        assessment_parts = []
        
        # Start with a general introduction
        assessment_parts.append(f"Based on comprehensive analysis of the question: '{question}', several key insights emerge.")
        
        # Add research findings summary if available
        if research_findings:
            finding_summary = "Research indicates that " + " Furthermore, ".join(research_findings[:3]) if len(research_findings) >= 3 else " ".join(research_findings)
            assessment_parts.append(finding_summary)
        
        # Add hypothesis summary if available
        if hypotheses:
            # Sort hypotheses by confidence if available
            sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("confidence", 0), reverse=True)
            top_hypothesis = sorted_hypotheses[0] if sorted_hypotheses else None
            
            if top_hypothesis:
                hypothesis_text = top_hypothesis.get("text", "")
                confidence_level = top_hypothesis.get("confidence_level", "")
                
                if hypothesis_text:
                    confidence_phrase = f"with {confidence_level} confidence" if confidence_level else ""
                    assessment_parts.append(f"The analysis suggests that {hypothesis_text} {confidence_phrase}.")
        
        # Add scenario summary if available
        if scenarios:
            scenario_summary = "Multiple future scenarios were considered, including: " + "; ".join([s.get("name", "Unnamed scenario") for s in scenarios[:3]])
            assessment_parts.append(scenario_summary)
        
        # Add a concluding statement
        assessment_parts.append("The following synthesis integrates these findings into a cohesive assessment with key judgments and recommendations.")
        
        # Combine all parts into a single assessment
        integrated_assessment = " ".join(assessment_parts)
        
        return integrated_assessment
    
    def _generate_key_judgments(self, analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate key judgments based on analysis results.
        
        Args:
            analysis_results: Dictionary of analysis results by technique
            
        Returns:
            List of key judgment dictionaries
        """
        judgments = []
        
        # Extract hypotheses from research
        hypotheses = []
        if "research_to_hypothesis" in analysis_results:
            research_result = analysis_results["research_to_hypothesis"]
            if isinstance(research_result, dict) and "hypotheses" in research_result:
                hypotheses = research_result["hypotheses"]
        
        # Convert top hypotheses to judgments
        if hypotheses:
            # Sort hypotheses by confidence if available
            sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("confidence", 0), reverse=True)
            
            # Take top 2-3 hypotheses as judgments
            for i, hypothesis in enumerate(sorted_hypotheses[:3]):
                hypothesis_text = hypothesis.get("text", "")
                confidence = hypothesis.get("confidence_level", "medium")
                supporting_findings = hypothesis.get("supporting_findings", [])
                
                if hypothesis_text:
                    judgment = {
                        "judgment": hypothesis_text,
                        "supporting_evidence": "; ".join(supporting_findings) if supporting_findings else "Based on available research",
                        "confidence": confidence,
                        "source": "research_to_hypothesis"
                    }
                    judgments.append(judgment)
        
        # Add judgments from other techniques if available
        if "analysis_of_competing_hypotheses" in analysis_results:
            ach_result = analysis_results["analysis_of_competing_hypotheses"]
            if isinstance(ach_result, dict) and "conclusions" in ach_result:
                conclusions = ach_result["conclusions"]
                for conclusion in conclusions:
                    judgment = {
                        "judgment": conclusion.get("text", ""),
                        "supporting_evidence": conclusion.get("evidence", "Based on competing hypotheses analysis"),
                        "confidence": conclusion.get("confidence", "medium"),
                        "source": "analysis_of_competing_hypotheses"
                    }
                    judgments.append(judgment)
        
        # Ensure we have at least 2 judgments
        if len(judgments) < 2:
            # Generate generic judgments based on question analysis
            question_analysis = analysis_results.get("question_analysis", {})
            question_type = question_analysis.get("type", "unknown")
            
            if question_type == "predictive":
                generic_judgment = {
                    "judgment": "Future developments are likely to follow established patterns with incremental changes.",
                    "supporting_evidence": "Based on historical precedent and current trends",
                    "confidence": "medium",
                    "source": "synthesis_generation"
                }
                judgments.append(generic_judgment)
            
            elif question_type == "causal":
                generic_judgment = {
                    "judgment": "Multiple factors contribute to the observed outcomes, with no single dominant cause.",
                    "supporting_evidence": "Based on complexity of causal relationships",
                    "confidence": "medium",
                    "source": "synthesis_generation"
                }
                judgments.append(generic_judgment)
            
            elif question_type == "evaluative":
                generic_judgment = {
                    "judgment": "The evidence suggests a mixed assessment with both positive and negative aspects.",
                    "supporting_evidence": "Based on balanced evaluation of available information",
                    "confidence": "medium",
                    "source": "synthesis_generation"
                }
                judgments.append(generic_judgment)
            
            else:
                generic_judgment = {
                    "judgment": "Available evidence provides a partial but incomplete understanding of the situation.",
                    "supporting_evidence": "Based on limitations in available data",
                    "confidence": "medium",
                    "source": "synthesis_generation"
                }
                judgments.append(generic_judgment)
        
        return judgments
    
    def _assess_confidence(self, analysis_results: Dict[str, Any]) -> tuple:
        """
        Assess overall confidence in the analysis.
        
        Args:
            analysis_results: Dictionary of analysis results by technique
            
        Returns:
            Tuple of (confidence_level, confidence_explanation)
        """
        # Collect confidence scores from all techniques
        confidence_scores = []
        
        # Check research confidence
        if "research_to_hypothesis" in analysis_results:
            research_result = analysis_results["research_to_hypothesis"]
            if isinstance(research_result, dict):
                if "overall_confidence" in research_result:
                    confidence_scores.append(research_result["overall_confidence"])
                elif "hypotheses" in research_result:
                    # Calculate average confidence from hypotheses
                    hypotheses = research_result["hypotheses"]
                    if hypotheses and "confidence" in hypotheses[0]:
                        avg_confidence = sum(h.get("confidence", 0) for h in hypotheses) / len(hypotheses)
                        confidence_scores.append(avg_confidence)
        
        # Check other techniques for confidence scores
        for technique, result in analysis_results.items():
            if technique == "research_to_hypothesis":
                continue  # Already processed
                
            if isinstance(result, dict) and "confidence" in result:
                confidence_scores.append(result["confidence"])
        
        # Calculate overall confidence
        if confidence_scores:
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.5  # Default to medium confidence
        
        # Determine confidence level
        if overall_confidence < 0.4:
            confidence_level = "low"
        elif overall_confidence < 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "high"
        
        # Generate explanation
        explanation_parts = []
        
        # Explain based on research depth and quality
        if "research_to_hypothesis" in analysis_results:
            research_result = analysis_results["research_to_hypothesis"]
            if isinstance(research_result, dict):
                depth = research_result.get("depth", "unknown")
                if depth == "deep":
                    explanation_parts.append("Deep research provides a strong foundation for the analysis.")
                elif depth == "quick":
                    explanation_parts.append("Limited research depth constrains confidence in some areas.")
                
                if research_result.get("conflicting_evidence_found", False):
                    explanation_parts.append("Conflicting evidence introduces uncertainty in some conclusions.")
                
                knowledge_gaps = research_result.get("knowledge_gaps", [])
                if knowledge_gaps:
                    explanation_parts.append(f"Identified knowledge gaps include: {knowledge_gaps[0]}")
        
        # Explain based on technique diversity
        technique_count = len(analysis_results)
        if technique_count >= 4:
            explanation_parts.append("Multiple analytical techniques provide cross-validation of findings.")
        elif technique_count <= 2:
            explanation_parts.append("Limited analytical techniques reduce robustness of conclusions.")
        
        # Add default explanation if needed
        if not explanation_parts:
            if confidence_level == "high":
                explanation_parts.append("High confidence based on consistent findings across multiple sources and techniques.")
            elif confidence_level == "low":
                explanation_parts.append("Low confidence due to limited data, conflicting evidence, or high uncertainty.")
            else:
                explanation_parts.append("Medium confidence reflects a balance of supporting evidence and remaining uncertainties.")
        
        # Combine explanations
        confidence_explanation = " ".join(explanation_parts)
        
        return confidence_level, confidence_explanation
    
    def _identify_alternative_perspectives(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Identify alternative perspectives or interpretations.
        
        Args:
            analysis_results: Dictionary of analysis results by technique
            
        Returns:
            List of alternative perspective statements
        """
        alternatives = []
        
        # Check for alternative hypotheses
        if "research_to_hypothesis" in analysis_results:
            research_result = analysis_results["research_to_hypothesis"]
            if isinstance(research_result, dict) and "hypotheses" in research_result:
                hypotheses = research_result["hypotheses"]
                
                # Use lower confidence hypotheses as alternatives
                if len(hypotheses) > 1:
                    # Sort by confidence ascending
                    sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("confidence", 0))
                    
                    # Take 1-2 lower confidence hypotheses as alternatives
                    for hypothesis in sorted_hypotheses[:2]:
                        hypothesis_text = hypothesis.get("text", "")
                        if hypothesis_text:
                            alternative = f"Alternative view: {hypothesis_text}"
                            alternatives.append(alternative)
        
        # Check for alternative scenarios
        if "scenario_triangulation" in analysis_results:
            scenario_result = analysis_results["scenario_triangulation"]
            if isinstance(scenario_result, dict) and "scenarios" in scenario_result:
                scenarios = scenario_result["scenarios"]
                
                # Use non-baseline scenarios as alternatives
                for scenario in scenarios:
                    if scenario.get("type") == "alternative" or scenario.get("probability", 0.5) < 0.3:
                        scenario_name = scenario.get("name", "Unnamed scenario")
                        scenario_desc = scenario.get("description", "")
                        
                        if scenario_desc:
                            alternative = f"Alternative scenario: {scenario_name} - {scenario_desc}"
                            alternatives.append(alternative)
        
        # Check for devil's advocate or red team analysis
        if "challenge_mechanism" in analysis_results:
            challenge_result = analysis_results["challenge_mechanism"]
            if isinstance(challenge_result, dict) and "challenges" in challenge_result:
                challenges = challenge_result["challenges"]
                
                for challenge in challenges:
                    challenge_text = challenge.get("text", "")
                    if challenge_text:
                        alternative = f"Challenge perspective: {challenge_text}"
                        alternatives.append(alternative)
        
        # Ensure we have at least 2 alternatives
        if len(alternatives) < 2:
            # Generate generic alternatives
            generic_alternatives = [
                "Alternative interpretation: The observed patterns may be temporary rather than indicative of a long-term trend.",
                "Alternative view: External factors not captured in the analysis may significantly influence outcomes.",
                "Alternative perspective: The relationships between key variables may be more complex than current models suggest.",
                "Alternative scenario: Unexpected disruptive events could fundamentally alter the trajectory of developments.",
                "Alternative interpretation: Cultural and regional factors may lead to different outcomes in different contexts."
            ]
            
            # Add generic alternatives until we have at least 2
            for alternative in generic_alternatives:
                if alternative not in alternatives:
                    alternatives.append(alternative)
                    if len(alternatives) >= 2:
                        break
        
        return alternatives
    
    def _identify_indicators_to_monitor(self, question: str, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Identify indicators to monitor for future validation.
        
        Args:
            question: The question being analyzed
            analysis_results: Dictionary of analysis results by technique
            
        Returns:
            List of indicators to monitor
        """
        indicators = []
        
        # Extract keywords from the question
        keywords = self._extract_keywords(question)
        
        # Generate indicators based on keywords
        for keyword in keywords[:3]:  # Use top 3 keywords
            indicator = self._generate_indicator_for_keyword(keyword)
            indicators.append(indicator)
        
        # Add indicators from uncertainty mapping if available
        if "uncertainty_mapping" in analysis_results:
            uncertainty_result = analysis_results["uncertainty_mapping"]
            if isinstance(uncertainty_result, dict) and "key_uncertainties" in uncertainty_result:
                uncertainties = uncertainty_result["key_uncertainties"]
                
                for uncertainty in uncertainties[:2]:  # Use top 2 uncertainties
                    uncertainty_text = uncertainty.get("text", "")
                    if uncertainty_text:
                        indicator = f"Monitor developments related to: {uncertainty_text}"
                        indicators.append(indicator)
        
        # Ensure we have at least 3 indicators
        if len(indicators) < 3:
            # Generate generic indicators
            generic_indicators = [
                "Monitor changes in regulatory frameworks that could impact key stakeholders.",
                "Track technological advancements that could disrupt current patterns.",
                "Observe shifts in public opinion or sentiment regarding key issues.",
                "Monitor economic indicators that may signal changes in underlying conditions.",
                "Track geopolitical developments that could alter the strategic landscape."
            ]
            
            # Add generic indicators until we have at least 3
            for indicator in generic_indicators:
                if indicator not in indicators:
                    indicators.append(indicator)
                    if len(indicators) >= 3:
                        break
        
        return indicators
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on common words
        # In a real implementation, this would use NLP techniques
        words = text.lower().split()
        stopwords = ["the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", "and", "or", "is", "are", "was", "were"]
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Deduplicate and limit to 5 keywords
        unique_keywords = list(set(keywords))[:5]
        
        return unique_keywords
    
    def _generate_indicator_for_keyword(self, keyword: str) -> str:
        """
        Generate an indicator to monitor based on a keyword.
        
        Args:
            keyword: Keyword to base the indicator on
            
        Returns:
            Indicator text
        """
        templates = [
            f"Monitor changes in {keyword} levels or metrics over the next 6-12 months.",
            f"Track policy developments related to {keyword} in key jurisdictions.",
            f"Observe how {keyword} is discussed in major media outlets and expert forums.",
            f"Monitor investment patterns in sectors related to {keyword}.",
            f"Track technological innovations that could impact {keyword}.",
            f"Observe shifts in public perception regarding {keyword}.",
            f"Monitor how key stakeholders are positioning themselves regarding {keyword}."
        ]
        
        # Select a template based on the keyword
        template_index = hash(keyword) % len(templates)
        return templates[template_index]
