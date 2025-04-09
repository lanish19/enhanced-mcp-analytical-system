"""
Historical Analogies Technique implementation.
This module provides the HistoricalAnalogiesTechnique class for comparative historical analysis.
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

class HistoricalAnalogiesTechnique(AnalyticalTechnique):
    """
    Identifies and analyzes historical situations analogous to the current question.
    
    This technique systematically compares the current situation to historical cases,
    extracting insights about patterns, outcomes, and contextual factors that might
    inform analysis of the present question.
    """
    
    def execute(self, context, parameters):
        """
        Execute the historical analogies technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing historical analogies results
        """
        logger.info(f"Executing HistoricalAnalogiesTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        num_analogies = parameters.get("num_analogies", 5)
        time_period = parameters.get("time_period", "Any")
        analogy_focus = parameters.get("analogy_focus", None)
        
        # Step 1: Identify key elements of current situation
        current_elements = self._identify_current_elements(context, analogy_focus)
        
        # Step 2: Identify historical analogies
        historical_analogies = self._identify_historical_analogies(context.question, current_elements, 
                                                                 num_analogies, time_period)
        
        # Step 3: Analyze each analogy
        analogy_analyses = self._analyze_analogies(context.question, historical_analogies, current_elements)
        
        # Step 4: Compare analogies
        comparative_analysis = self._compare_analogies(context.question, analogy_analyses)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, current_elements, 
                                           analogy_analyses, comparative_analysis)
        
        return {
            "technique": "Historical Analogies",
            "status": "Completed",
            "current_elements": current_elements,
            "historical_analogies": historical_analogies,
            "analogy_analyses": analogy_analyses,
            "comparative_analysis": comparative_analysis,
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
        return ["historical_research_mcp", "comparative_analysis_mcp"]
    
    def _identify_current_elements(self, context, analogy_focus):
        """
        Identify key elements of the current situation for analogy matching.
        
        Args:
            context: The analysis context
            analogy_focus: Optional specific focus for analogies
            
        Returns:
            Dictionary containing key elements of the current situation
        """
        logger.info("Identifying key elements of current situation...")
        
        # If analogy focus is provided, use it directly
        if analogy_focus:
            logger.info(f"Using provided analogy focus: {analogy_focus}")
            return {
                "focus": analogy_focus,
                "key_elements": ["Specified by user"],
                "context": "Provided analogy focus"
            }
        
        # Use historical research MCP if available
        historical_mcp = self.mcp_registry.get_mcp("historical_research_mcp")
        
        if historical_mcp:
            try:
                logger.info("Using historical research MCP")
                current_elements = historical_mcp.identify_current_elements(context.question)
                return current_elements
            except Exception as e:
                logger.error(f"Error using historical research MCP: {e}")
                # Fall through to LLM-based identification
        
        # Use LLM to identify current elements
        prompt = f"""
        Identify key elements of the current situation described in this analytical question:
        
        "{context.question}"
        
        For this analysis:
        1. Identify the core situation or phenomenon being analyzed
        2. Extract 5-7 key elements that characterize this situation
        3. For each element, explain why it's significant for historical comparison
        4. Identify any contextual factors that should be considered when seeking analogies
        
        Return your response as a JSON object with the following structure:
        {{
            "core_situation": "Brief description of the core situation or phenomenon",
            "key_elements": [
                {{
                    "element": "Name or description of the element",
                    "significance": "Why this element is significant for historical comparison"
                }},
                ...
            ],
            "contextual_factors": [
                {{
                    "factor": "Name or description of the contextual factor",
                    "relevance": "How this factor affects the applicability of historical analogies"
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
                logger.error(f"Error identifying current elements: {parsed_response.get('error')}")
                return self._generate_fallback_current_elements(context.question)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing current elements: {e}")
            return self._generate_fallback_current_elements(context.question)
    
    def _generate_fallback_current_elements(self, question):
        """
        Generate fallback current elements when identification fails.
        
        Args:
            question: The analytical question
            
        Returns:
            Dictionary containing fallback current elements
        """
        return {
            "core_situation": "The situation described in the analytical question",
            "key_elements": [
                {
                    "element": "Technological change",
                    "significance": "Rate and nature of technological evolution affects adaptation patterns"
                },
                {
                    "element": "Organizational structure",
                    "significance": "Institutional arrangements influence decision-making and implementation"
                },
                {
                    "element": "Resource constraints",
                    "significance": "Available resources shape options and strategies"
                },
                {
                    "element": "Stakeholder interests",
                    "significance": "Competing interests affect support and resistance"
                },
                {
                    "element": "External pressures",
                    "significance": "Environmental factors create urgency and constraints"
                }
            ],
            "contextual_factors": [
                {
                    "factor": "Technological context",
                    "relevance": "Modern technology differs significantly from historical periods"
                },
                {
                    "factor": "Globalization",
                    "relevance": "Interconnectedness creates different dynamics than in more isolated historical contexts"
                },
                {
                    "factor": "Information availability",
                    "relevance": "Modern information environment differs from historical information constraints"
                }
            ]
        }
    
    def _identify_historical_analogies(self, question, current_elements, num_analogies, time_period):
        """
        Identify historical analogies to the current situation.
        
        Args:
            question: The analytical question
            current_elements: Dictionary containing key elements of the current situation
            num_analogies: Number of analogies to identify
            time_period: Optional time period constraint
            
        Returns:
            List of historical analogy dictionaries
        """
        logger.info(f"Identifying {num_analogies} historical analogies...")
        
        # Use historical research MCP if available
        historical_mcp = self.mcp_registry.get_mcp("historical_research_mcp")
        
        if historical_mcp:
            try:
                logger.info("Using historical research MCP")
                historical_analogies = historical_mcp.identify_historical_analogies(
                    question, current_elements, num_analogies, time_period)
                return historical_analogies
            except Exception as e:
                logger.error(f"Error using historical research MCP: {e}")
                # Fall through to LLM-based identification
        
        # Extract key elements for the prompt
        core_situation = current_elements.get("core_situation", "")
        key_elements = [e.get("element", "") for e in current_elements.get("key_elements", [])]
        contextual_factors = [f.get("factor", "") for f in current_elements.get("contextual_factors", [])]
        
        # Use LLM to identify historical analogies
        prompt = f"""
        Identify {num_analogies} historical analogies to the following situation:
        
        Question: "{question}"
        
        Core Situation: "{core_situation}"
        
        Key Elements:
        {json.dumps(key_elements, indent=2)}
        
        Contextual Factors:
        {json.dumps(contextual_factors, indent=2)}
        
        Time Period Constraint: {time_period}
        
        For each historical analogy:
        1. Identify a specific historical situation that shares key similarities with the current situation
        2. Explain the key similarities that make this a relevant analogy
        3. Note important differences that might limit the analogy
        4. Provide basic factual information about the historical case
        
        Ensure the analogies:
        - Are historically accurate and well-documented
        - Span different contexts if possible (different regions, time periods, domains)
        - Have clear outcomes that can inform analysis
        - {"Are from the specified time period: " + time_period if time_period != "Any" else "Can be from any time period"}
        
        Return your response as a JSON object with the following structure:
        {{
            "historical_analogies": [
                {{
                    "name": "Brief name or title for the historical analogy",
                    "time_period": "When this historical situation occurred",
                    "description": "Brief description of the historical situation",
                    "key_similarities": ["Similarity 1", "Similarity 2", ...],
                    "key_differences": ["Difference 1", "Difference 2", ...],
                    "outcome": "What ultimately happened in this historical situation",
                    "relevance": "Why this analogy is particularly relevant to the current situation"
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
                logger.error(f"Error identifying historical analogies: {parsed_response.get('error')}")
                return self._generate_fallback_historical_analogies(num_analogies)
            
            historical_analogies = parsed_response.get("historical_analogies", [])
            
            if not historical_analogies or len(historical_analogies) < 1:
                logger.warning("No historical analogies identified")
                return self._generate_fallback_historical_analogies(num_analogies)
            
            return historical_analogies
        
        except Exception as e:
            logger.error(f"Error parsing historical analogies: {e}")
            return self._generate_fallback_historical_analogies(num_analogies)
    
    def _generate_fallback_historical_analogies(self, num_analogies):
        """
        Generate fallback historical analogies when identification fails.
        
        Args:
            num_analogies: Number of analogies to generate
            
        Returns:
            List of fallback historical analogy dictionaries
        """
        fallback_analogies = [
            {
                "name": "Industrial Revolution (1760-1840)",
                "time_period": "Late 18th to early 19th century",
                "description": "Transition from agrarian to industrial manufacturing economies with technological, socioeconomic, and cultural changes",
                "key_similarities": [
                    "Rapid technological transformation",
                    "Disruption of existing economic structures",
                    "Resistance from established interests",
                    "New skill requirements for workforce"
                ],
                "key_differences": [
                    "Slower pace of change compared to modern transitions",
                    "Less global interconnectedness",
                    "Different regulatory environment",
                    "Lower information availability"
                ],
                "outcome": "Fundamental transformation of economic and social structures with both positive (productivity, wealth creation) and negative (inequality, pollution) consequences",
                "relevance": "Demonstrates patterns of adaptation to technological disruption and institutional response"
            },
            {
                "name": "Post-WWII Economic Reconstruction (1945-1960)",
                "time_period": "Mid-20th century",
                "description": "Rebuilding of economic and political systems after major global conflict",
                "key_similarities": [
                    "Need for rapid transformation",
                    "Coordination across multiple stakeholders",
                    "Resource allocation challenges",
                    "Competing visions for the future"
                ],
                "key_differences": [
                    "Post-crisis context rather than proactive change",
                    "Stronger government direction",
                    "Different geopolitical dynamics",
                    "Less technological complexity"
                ],
                "outcome": "Successful economic revitalization in many regions, establishment of new international institutions, and emergence of new economic models",
                "relevance": "Illustrates how coordinated efforts can achieve rapid transformation when properly resourced and aligned"
            },
            {
                "name": "Digital Revolution (1980s-2000s)",
                "time_period": "Late 20th to early 21st century",
                "description": "Transition from analog to digital technology with widespread adoption of computers and the internet",
                "key_similarities": [
                    "Fundamental technological paradigm shift",
                    "Disruption of existing business models",
                    "Creation of new markets and opportunities",
                    "Changing skill requirements"
                ],
                "key_differences": [
                    "More recent historical context",
                    "Different regulatory approach",
                    "More gradual adoption curve",
                    "Different global power dynamics"
                ],
                "outcome": "Transformation of information access, business models, and social interaction patterns with both democratizing effects and new forms of concentration",
                "relevance": "Recent example of adapting to technological change with observable long-term consequences"
            },
            {
                "name": "Deregulation of Financial Markets (1980s-1990s)",
                "time_period": "Late 20th century",
                "description": "Removal of regulatory constraints on financial institutions and markets",
                "key_similarities": [
                    "Significant policy shift",
                    "Tension between innovation and stability",
                    "Complex stakeholder interests",
                    "Global ripple effects"
                ],
                "key_differences": [
                    "Sector-specific rather than economy-wide",
                    "Deregulatory rather than regulatory focus",
                    "Different technological context",
                    "Different political environment"
                ],
                "outcome": "Increased financial innovation and growth coupled with greater systemic risks, eventually contributing to the 2008 financial crisis",
                "relevance": "Demonstrates how policy changes can have long-term unintended consequences and the importance of appropriate safeguards"
            },
            {
                "name": "Green Revolution in Agriculture (1950s-1970s)",
                "time_period": "Mid-20th century",
                "description": "Introduction of new agricultural technologies, practices, and varieties to dramatically increase food production",
                "key_similarities": [
                    "Technology-driven transformation",
                    "Implementation across diverse contexts",
                    "Resistance to new approaches",
                    "Complex stakeholder ecosystem"
                ],
                "key_differences": [
                    "More centralized planning and implementation",
                    "Focus on a specific sector",
                    "Different global development context",
                    "Less digital component"
                ],
                "outcome": "Significant increases in agricultural productivity and food security, but with environmental consequences and uneven distribution of benefits",
                "relevance": "Shows how technological innovation can be deployed at scale with transformative effects, while highlighting the importance of considering long-term impacts"
            }
        ]
        
        # Return the requested number of analogies
        return fallback_analogies[:num_analogies]
    
    def _analyze_analogies(self, question, historical_analogies, current_elements):
        """
        Analyze each historical analogy in depth.
        
        Args:
            question: The analytical question
            historical_analogies: List of historical analogy dictionaries
            current_elements: Dictionary containing key elements of the current situation
            
        Returns:
            List of analogy analysis dictionaries
        """
        logger.info(f"Analyzing {len(historical_analogies)} historical analogies...")
        
        # Use comparative analysis MCP if available
        comparative_mcp = self.mcp_registry.get_mcp("comparative_analysis_mcp")
        
        if comparative_mcp:
            try:
                logger.info("Using comparative analysis MCP")
                analogy_analyses = comparative_mcp.analyze_historical_analogies(
                    question, historical_analogies, current_elements)
                return analogy_analyses
            except Exception as e:
                logger.error(f"Error using comparative analysis MCP: {e}")
                # Fall through to LLM-based analysis
        
        analogy_analyses = []
        
        for analogy in historical_analogies:
            analogy_name = analogy.get("name", "")
            logger.info(f"Analyzing analogy: {analogy_name}")
            
            # Use LLM to analyze this analogy
            prompt = f"""
            Analyze the following historical analogy in relation to this question:
            
            Question: "{question}"
            
            Historical Analogy:
            {json.dumps(analogy, indent=2)}
            
            Current Situation Elements:
            {json.dumps(current_elements, indent=2)}
            
            For this analysis:
            1. Evaluate the strength of the analogy (how well it maps to the current situation)
            2. Identify key lessons or insights from this historical case
            3. Analyze how contextual factors affected the historical outcome
            4. Assess how applicable these lessons are to the current situation
            5. Identify potential pitfalls in applying this analogy
            
            Return your analysis as a JSON object with the following structure:
            {{
                "analogy_name": "Name of the historical analogy",
                "analogy_strength": "Strong/Moderate/Weak",
                "strength_rationale": "Explanation of why the analogy is strong/moderate/weak",
                "key_lessons": [
                    {{
                        "lesson": "Description of the lesson or insight",
                        "evidence": "Historical evidence supporting this lesson",
                        "applicability": "High/Medium/Low",
                        "application_rationale": "Explanation of how this lesson applies to the current situation"
                    }},
                    ...
                ],
                "contextual_factors": [
                    {{
                        "factor": "Description of the contextual factor",
                        "historical_impact": "How this factor affected the historical outcome",
                        "current_relevance": "How this factor compares to the current situation"
                    }},
                    ...
                ],
                "potential_pitfalls": [
                    {{
                        "pitfall": "Description of the potential pitfall",
                        "mitigation": "How this pitfall could be mitigated"
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
                    logger.error(f"Error analyzing analogy: {parsed_response.get('error')}")
                    analogy_analyses.append(self._generate_fallback_analogy_analysis(analogy))
                else:
                    analogy_analyses.append(parsed_response)
            
            except Exception as e:
                logger.error(f"Error parsing analogy analysis: {e}")
                analogy_analyses.append(self._generate_fallback_analogy_analysis(analogy))
        
        return analogy_analyses
    
    def _generate_fallback_analogy_analysis(self, analogy):
        """
        Generate fallback analogy analysis when analysis fails.
        
        Args:
            analogy: Historical analogy dictionary
            
        Returns:
            Dictionary containing fallback analogy analysis
        """
        analogy_name = analogy.get("name", "Historical Analogy")
        
        return {
            "analogy_name": analogy_name,
            "analogy_strength": "Moderate",
            "strength_rationale": "The analogy shares some important similarities with the current situation but also has significant differences that limit direct comparison.",
            "key_lessons": [
                {
                    "lesson": "Adaptation requires both technological and institutional change",
                    "evidence": "Historical records show that successful transitions involved changes to both technologies and supporting institutions",
                    "applicability": "High",
                    "application_rationale": "The current situation similarly requires alignment between technological capabilities and institutional frameworks"
                },
                {
                    "lesson": "Stakeholder resistance can significantly delay implementation",
                    "evidence": "Historical documentation of opposition from groups whose interests were threatened",
                    "applicability": "Medium",
                    "application_rationale": "Similar stakeholder dynamics exist, though the specific groups and interests differ"
                }
            ],
            "contextual_factors": [
                {
                    "factor": "Technological capabilities",
                    "historical_impact": "Limited by the technological capabilities of the time",
                    "current_relevance": "Modern technological capabilities offer different possibilities and constraints"
                },
                {
                    "factor": "Information availability",
                    "historical_impact": "Decision-making constrained by limited information",
                    "current_relevance": "Greater information availability but also information overload challenges"
                }
            ],
            "potential_pitfalls": [
                {
                    "pitfall": "Overestimating the similarity of contextual factors",
                    "mitigation": "Carefully analyze differences in context and adjust expectations accordingly"
                },
                {
                    "pitfall": "Assuming similar causal relationships",
                    "mitigation": "Test causal assumptions rather than assuming they apply in the current context"
                }
            ]
        }
    
    def _compare_analogies(self, question, analogy_analyses):
        """
        Compare the different historical analogies.
        
        Args:
            question: The analytical question
            analogy_analyses: List of analogy analysis dictionaries
            
        Returns:
            Dictionary containing comparative analysis
        """
        logger.info("Comparing historical analogies...")
        
        # Use comparative analysis MCP if available
        comparative_mcp = self.mcp_registry.get_mcp("comparative_analysis_mcp")
        
        if comparative_mcp:
            try:
                logger.info("Using comparative analysis MCP")
                comparative_analysis = comparative_mcp.compare_historical_analogies(question, analogy_analyses)
                return comparative_analysis
            except Exception as e:
                logger.error(f"Error using comparative analysis MCP: {e}")
                # Fall through to LLM-based comparison
        
        # Use LLM to compare analogies
        prompt = f"""
        Compare the following historical analogies in relation to this question:
        
        Question: "{question}"
        
        Analogy Analyses:
        {json.dumps([{
            "analogy_name": a.get("analogy_name", ""),
            "analogy_strength": a.get("analogy_strength", ""),
            "key_lessons": [l.get("lesson", "") for l in a.get("key_lessons", [])]
        } for a in analogy_analyses], indent=2)}
        
        For this comparison:
        1. Identify patterns and common lessons across the analogies
        2. Highlight significant differences in the historical cases
        3. Assess which analogies are most relevant to the current situation
        4. Identify cross-cutting insights that emerge from considering multiple analogies
        
        Return your comparison as a JSON object with the following structure:
        {{
            "common_patterns": [
                {{
                    "pattern": "Description of the common pattern",
                    "analogies": ["Analogy 1", "Analogy 2", ...],
                    "significance": "Why this pattern is significant"
                }},
                ...
            ],
            "key_differences": [
                {{
                    "difference": "Description of the key difference",
                    "implications": "Implications of this difference for analysis"
                }},
                ...
            ],
            "most_relevant_analogies": [
                {{
                    "analogy": "Name of the analogy",
                    "relevance_rationale": "Why this analogy is particularly relevant"
                }},
                ...
            ],
            "cross_cutting_insights": [
                {{
                    "insight": "Description of the cross-cutting insight",
                    "supporting_evidence": "Evidence from multiple analogies that supports this insight",
                    "application": "How this insight applies to the current situation"
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
                logger.error(f"Error comparing analogies: {parsed_response.get('error')}")
                return self._generate_fallback_comparative_analysis(analogy_analyses)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing comparative analysis: {e}")
            return self._generate_fallback_comparative_analysis(analogy_analyses)
    
    def _generate_fallback_comparative_analysis(self, analogy_analyses):
        """
        Generate fallback comparative analysis when comparison fails.
        
        Args:
            analogy_analyses: List of analogy analysis dictionaries
            
        Returns:
            Dictionary containing fallback comparative analysis
        """
        # Extract analogy names
        analogy_names = [a.get("analogy_name", f"Analogy {i+1}") for i, a in enumerate(analogy_analyses)]
        
        return {
            "common_patterns": [
                {
                    "pattern": "Resistance to change from established interests",
                    "analogies": analogy_names,
                    "significance": "Suggests that managing stakeholder resistance is a critical success factor"
                },
                {
                    "pattern": "Importance of institutional adaptation alongside technological change",
                    "analogies": analogy_names[:3] if len(analogy_names) >= 3 else analogy_names,
                    "significance": "Indicates that technological solutions alone are insufficient without supporting institutional changes"
                }
            ],
            "key_differences": [
                {
                    "difference": "Varying levels of centralized coordination",
                    "implications": "Suggests that different governance approaches may be appropriate depending on context"
                },
                {
                    "difference": "Different paces of change and implementation",
                    "implications": "Highlights the importance of calibrating expectations about timeline and managing transition periods"
                }
            ],
            "most_relevant_analogies": [
                {
                    "analogy": analogy_names[0] if analogy_names else "Primary Analogy",
                    "relevance_rationale": "Closest match to current contextual factors and core dynamics"
                },
                {
                    "analogy": analogy_names[1] if len(analogy_names) >= 2 else "Secondary Analogy",
                    "relevance_rationale": "Provides important complementary insights about long-term consequences"
                }
            ],
            "cross_cutting_insights": [
                {
                    "insight": "Successful transformations require alignment of incentives across key stakeholders",
                    "supporting_evidence": "Multiple historical cases show that misaligned incentives led to implementation failures or unintended consequences",
                    "application": "Current situation similarly requires attention to incentive structures across the stakeholder ecosystem"
                },
                {
                    "insight": "Adaptability during implementation is more important than perfect initial design",
                    "supporting_evidence": "Historical cases consistently show that successful initiatives adapted to feedback and changing conditions",
                    "application": "Current approach should incorporate feedback mechanisms and flexibility rather than rigid implementation plans"
                }
            ]
        }
    
    def _generate_synthesis(self, question, current_elements, analogy_analyses, comparative_analysis):
        """
        Generate a synthesis of the historical analogies analysis.
        
        Args:
            question: The analytical question
            current_elements: Dictionary containing key elements of the current situation
            analogy_analyses: List of analogy analysis dictionaries
            comparative_analysis: Dictionary containing comparative analysis
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of historical analogies...")
        
        # Extract most relevant analogies
        most_relevant = [r.get("analogy", "") for r in comparative_analysis.get("most_relevant_analogies", [])]
        
        # Extract cross-cutting insights
        cross_cutting = [i.get("insight", "") for i in comparative_analysis.get("cross_cutting_insights", [])]
        
        # Use LLM to generate synthesis
        prompt = f"""
        Synthesize the following historical analogies analysis for the question:
        
        "{question}"
        
        Current Situation:
        {json.dumps(current_elements.get("core_situation", ""), indent=2)}
        
        Most Relevant Analogies:
        {json.dumps(most_relevant, indent=2)}
        
        Cross-Cutting Insights:
        {json.dumps(cross_cutting, indent=2)}
        
        Based on this historical analogies analysis:
        1. What key lessons from history are most applicable to the current situation?
        2. What patterns or principles emerge across multiple historical cases?
        3. What contextual differences limit the applicability of historical lessons?
        4. How should these historical insights inform the approach to the current situation?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the historical analogies
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "key_historical_lessons": ["Lesson 1", "Lesson 2", ...],
            "contextual_limitations": ["Limitation 1", "Limitation 2", ...],
            "recommended_approach": ["Recommendation 1", "Recommendation 2", ...],
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
                    "key_historical_lessons": ["Error in synthesis generation"],
                    "contextual_limitations": ["Error in synthesis generation"],
                    "recommended_approach": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_historical_lessons": ["Error in synthesis generation"],
                "contextual_limitations": ["Error in synthesis generation"],
                "recommended_approach": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
