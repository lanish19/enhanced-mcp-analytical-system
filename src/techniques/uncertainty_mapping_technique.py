"""
Uncertainty Mapping Technique for identifying and analyzing sources of uncertainty.
This module provides the UncertaintyMappingTechnique class for comprehensive uncertainty analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np

from src.analytical_technique import AnalyticalTechnique
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UncertaintyMappingTechnique(AnalyticalTechnique):
    """
    Uncertainty Mapping Technique for identifying and analyzing sources of uncertainty.
    
    This technique provides capabilities for:
    1. Identifying key sources of uncertainty
    2. Categorizing uncertainties by type and impact
    3. Assessing confidence levels and knowledge gaps
    4. Developing strategies for managing uncertainty
    5. Communicating uncertainty effectively
    """
    
    def __init__(self):
        """Initialize the Uncertainty Mapping Technique."""
        super().__init__(
            name="uncertainty_mapping",
            description="Identifies, categorizes, and analyzes sources of uncertainty in analysis",
            required_mcps=["llama4_scout", "research_mcp"],
            compatible_techniques=["key_assumptions_check", "premortem_analysis", "scenario_triangulation"],
            incompatible_techniques=[]
        )
        logger.info("Initialized UncertaintyMappingTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Uncertainty Mapping Technique")
        
        if parameters is None:
            parameters = {}
        
        if not self.validate_parameters(parameters):
            return {"error": "Invalid parameters"}
        
        try:
            # Get question from context
            question = context.get("question")
            if not question:
                return {"error": "No question found in context"}
            
            # Get research results from context
            research_results = context.get("research_results", {})
            
            # Identify uncertainties
            uncertainties = self._identify_uncertainties(question, research_results, context, parameters)
            
            # Categorize uncertainties
            categorized_uncertainties = self._categorize_uncertainties(uncertainties, context, parameters)
            
            # Assess confidence levels
            confidence_assessment = self._assess_confidence(uncertainties, categorized_uncertainties, context, parameters)
            
            # Develop uncertainty management strategies
            management_strategies = self._develop_management_strategies(uncertainties, categorized_uncertainties, confidence_assessment, context, parameters)
            
            # Compile results
            results = {
                "technique": "uncertainty_mapping",
                "timestamp": time.time(),
                "question": question,
                "uncertainties": uncertainties,
                "categorized_uncertainties": categorized_uncertainties,
                "confidence_assessment": confidence_assessment,
                "management_strategies": management_strategies,
                "findings": self._extract_findings(uncertainties, categorized_uncertainties, confidence_assessment),
                "assumptions": self._extract_assumptions(uncertainties, categorized_uncertainties),
                "uncertainties_meta": self._extract_meta_uncertainties(uncertainties, confidence_assessment)
            }
            
            # Add results to context
            context.add_technique_result("uncertainty_mapping", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Uncertainty Mapping Technique: {e}")
            return self.handle_error(e, context)
    
    def _identify_uncertainties(self, question: str, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Identify key sources of uncertainty.
        
        Args:
            question: The analytical question
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of uncertainties
        """
        logger.info("Identifying uncertainties")
        
        # Check if uncertainties are provided in parameters
        if "uncertainties" in parameters and isinstance(parameters["uncertainties"], list):
            return parameters["uncertainties"]
        
        # Collect uncertainties from previous technique results
        collected_uncertainties = []
        
        # Get all technique results from context
        technique_results = context.get_all_technique_results()
        
        # Extract uncertainties from each technique result
        for technique_name, result in technique_results.items():
            if "uncertainties" in result and isinstance(result["uncertainties"], list):
                for uncertainty in result["uncertainties"]:
                    # Add source information if not present
                    if "source" not in uncertainty:
                        uncertainty["source"] = technique_name
                    
                    collected_uncertainties.append(uncertainty)
        
        # If we have collected uncertainties, use them
        if collected_uncertainties:
            # Deduplicate uncertainties
            unique_uncertainties = []
            seen_descriptions = set()
            
            for uncertainty in collected_uncertainties:
                description = uncertainty.get("uncertainty", "")
                if description and description not in seen_descriptions:
                    seen_descriptions.add(description)
                    unique_uncertainties.append(uncertainty)
            
            # If we have enough uncertainties, return them
            if len(unique_uncertainties) >= 5:
                return unique_uncertainties
        
        # Use Llama4ScoutMCP to identify uncertainties
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Create prompt for uncertainty identification
            prompt = f"""
            Based on the following question, identify 8-12 key sources of uncertainty that affect the analysis.
            
            Question: {question}
            
            For each uncertainty:
            1. Provide a clear description of the uncertainty
            2. Explain why this creates uncertainty for the analysis
            3. Assess the potential impact of this uncertainty (high, medium, low)
            
            Consider different types of uncertainty:
            - Data uncertainty (missing, incomplete, or unreliable data)
            - Model uncertainty (limitations in analytical frameworks)
            - Parameter uncertainty (unknown values of key variables)
            - Structural uncertainty (unknown causal relationships)
            - Future uncertainty (unpredictable future developments)
            - Linguistic uncertainty (ambiguity or vagueness in key terms)
            
            Focus on uncertainties that:
            - Are specific and clearly defined
            - Have meaningful impact on the analysis
            - Represent different types of uncertainty
            - Include both known unknowns and potential unknown unknowns
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "uncertainty_identification",
                "context": {"prompt": grounded_prompt, "research_results": research_results}
            })
            
            # Extract uncertainties from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse uncertainties from content
                uncertainties = self._parse_uncertainties_from_text(content)
                if uncertainties:
                    return uncertainties
        
        # Fallback: Generate generic uncertainties
        return self._generate_generic_uncertainties(question)
    
    def _parse_uncertainties_from_text(self, text: str) -> List[Dict]:
        """
        Parse uncertainties from text.
        
        Args:
            text: Text containing uncertainty descriptions
            
        Returns:
            List of parsed uncertainties
        """
        uncertainties = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for uncertainty sections
        uncertainty_pattern = r'(?:^|\n)(?:Uncertainty|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Uncertainty|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        uncertainty_matches = re.findall(uncertainty_pattern, text, re.DOTALL)
        
        if not uncertainty_matches:
            # Try alternative pattern for numbered lists
            uncertainty_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            uncertainty_matches = re.findall(uncertainty_pattern, text, re.DOTALL)
            
            if uncertainty_matches:
                # Convert to expected format
                uncertainty_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(uncertainty_matches)]
        
        for match in uncertainty_matches:
            uncertainty_num = match[0].strip() if len(match) > 0 else ""
            uncertainty_description = match[1].strip() if len(match) > 1 else ""
            uncertainty_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract explanation
            explanation = ""
            explanation_pattern = r'(?:explanation|why|creates uncertainty|reason).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            explanation_match = re.search(explanation_pattern, uncertainty_content, re.IGNORECASE | re.DOTALL)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            else:
                # Use first paragraph as explanation
                paragraphs = uncertainty_content.split('\n\n')
                if paragraphs:
                    explanation = paragraphs[0].strip()
            
            # Extract impact
            impact = "medium"  # Default impact
            impact_pattern = r'(?:impact|effect|significance).*?(high|medium|low)'
            impact_match = re.search(impact_pattern, uncertainty_content, re.IGNORECASE)
            if impact_match:
                impact = impact_match.group(1).lower()
            
            uncertainties.append({
                "uncertainty": uncertainty_description,
                "explanation": explanation,
                "impact": impact,
                "source": "uncertainty_mapping"
            })
        
        return uncertainties
    
    def _generate_generic_uncertainties(self, question: str) -> List[Dict]:
        """
        Generate generic uncertainties based on the question.
        
        Args:
            question: The analytical question
            
        Returns:
            List of generic uncertainties
        """
        # Extract domain and type from question
        domain = self._extract_domain_from_question(question)
        question_type = self._extract_question_type(question)
        
        # Domain-specific uncertainty templates
        domain_uncertainties = {
            "economic": [
                {
                    "uncertainty": "Incomplete or unreliable economic data",
                    "explanation": "Economic data often has measurement errors, reporting lags, and methodological limitations that create uncertainty in baseline conditions.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Future policy changes and regulatory developments",
                    "explanation": "Economic outcomes are significantly influenced by policy decisions that are difficult to predict, especially across political cycles.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Market participant behavior and psychology",
                    "explanation": "Economic actors may not behave according to rational expectations, creating uncertainty in how they will respond to changing conditions.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "External economic shocks and global developments",
                    "explanation": "Unexpected external events can significantly impact economic outcomes in ways that are difficult to anticipate.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Technological disruption and innovation rates",
                    "explanation": "The pace and direction of technological change creates uncertainty about future productivity, business models, and labor markets.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Structural changes in economic relationships",
                    "explanation": "Historical relationships between economic variables may not hold in the future due to structural changes in the economy.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Distributional effects across sectors and groups",
                    "explanation": "Economic impacts are rarely uniform, creating uncertainty about which sectors or groups will be most affected.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Feedback loops and non-linear effects",
                    "explanation": "Economic systems often exhibit complex feedback mechanisms that can amplify or dampen initial effects in unpredictable ways.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                }
            ],
            "political": [
                {
                    "uncertainty": "Future electoral outcomes and leadership changes",
                    "explanation": "Political leadership changes can significantly alter policy directions and priorities in ways that are difficult to predict.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Public opinion shifts and social movements",
                    "explanation": "Public attitudes can change rapidly, creating uncertainty about political constraints and opportunities.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Implementation effectiveness of policies",
                    "explanation": "Even when policies are known, their implementation may vary in effectiveness, creating uncertainty about actual outcomes.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "International relations and geopolitical developments",
                    "explanation": "External political factors can significantly impact domestic politics in ways that are difficult to anticipate.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Institutional constraints and veto points",
                    "explanation": "Political systems have various constraints that may limit or enable action in ways that are difficult to predict precisely.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Crisis events and exogenous shocks",
                    "explanation": "Unexpected events can dramatically alter political dynamics and priorities.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Coalition dynamics and interest group influence",
                    "explanation": "The formation and stability of political coalitions creates uncertainty about policy outcomes.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Media coverage and information environment",
                    "explanation": "How issues are framed and covered in media can significantly impact political dynamics in unpredictable ways.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                }
            ],
            "technological": [
                {
                    "uncertainty": "Rate of technological advancement in key areas",
                    "explanation": "The pace of technological progress is inherently difficult to predict, especially for emerging technologies.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Adoption rates and diffusion patterns",
                    "explanation": "How quickly and widely new technologies will be adopted creates significant uncertainty about their impact.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Regulatory responses to new technologies",
                    "explanation": "How governments will regulate emerging technologies is uncertain and can significantly affect their development and impact.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Unexpected applications and use cases",
                    "explanation": "Technologies are often applied in ways not initially anticipated, creating uncertainty about their ultimate impact.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Technical limitations and unforeseen challenges",
                    "explanation": "Technological development often encounters unexpected technical barriers that may delay or redirect progress.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Social acceptance and resistance",
                    "explanation": "Public attitudes toward technologies can significantly affect their adoption and impact in unpredictable ways.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Competitive dynamics and strategic responses",
                    "explanation": "How companies and countries compete and respond strategically creates uncertainty about technological trajectories.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Second-order effects and system-wide impacts",
                    "explanation": "Technologies often have cascading effects throughout complex systems that are difficult to anticipate.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                }
            ],
            "social": [
                {
                    "uncertainty": "Demographic trends and population dynamics",
                    "explanation": "While some demographic changes are predictable, migration patterns, fertility decisions, and mortality developments create uncertainty.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Cultural value shifts and social norms",
                    "explanation": "Social values and norms can change in ways that are difficult to predict, affecting behaviors and institutions.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Social movement emergence and effectiveness",
                    "explanation": "The emergence, growth, and impact of social movements creates uncertainty about social and political change.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Institutional adaptation and resilience",
                    "explanation": "How social institutions adapt to changing conditions creates uncertainty about social outcomes.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Inequality dynamics and distributional effects",
                    "explanation": "How social changes affect different groups creates uncertainty about overall social impacts.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Media and information environment changes",
                    "explanation": "Evolving media landscapes and information consumption patterns create uncertainty about social discourse and cohesion.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Crisis events and social responses",
                    "explanation": "How societies respond to crises and unexpected events creates significant uncertainty about social trajectories.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Intergenerational dynamics and cohort effects",
                    "explanation": "Differences between generations create uncertainty about future social preferences and behaviors.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                }
            ],
            "general": [
                {
                    "uncertainty": "Data limitations and information gaps",
                    "explanation": "Incomplete, unreliable, or missing data creates fundamental uncertainty about baseline conditions and trends.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Future events and external developments",
                    "explanation": "Unpredictable future events can significantly impact outcomes in ways that cannot be anticipated.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Causal relationships and system dynamics",
                    "explanation": "Incomplete understanding of how factors interact creates uncertainty about how changes will propagate through systems.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Human behavior and decision-making",
                    "explanation": "How individuals and organizations will respond to changing conditions is inherently difficult to predict precisely.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Implementation and execution factors",
                    "explanation": "Even when plans are clear, their implementation may vary in ways that significantly affect outcomes.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Timeframes and development rates",
                    "explanation": "The pace at which changes occur creates uncertainty about when impacts will materialize.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Feedback loops and non-linear effects",
                    "explanation": "Complex systems often exhibit feedback mechanisms that can amplify or dampen initial effects in unpredictable ways.",
                    "impact": "high",
                    "source": "uncertainty_mapping"
                },
                {
                    "uncertainty": "Definitional and conceptual ambiguity",
                    "explanation": "Key terms and concepts may be understood differently by different stakeholders, creating uncertainty in communication and analysis.",
                    "impact": "medium",
                    "source": "uncertainty_mapping"
                }
            ]
        }
        
        # Question type modifications
        if question_type == "predictive":
            # Emphasize future uncertainties for predictive questions
            for uncertainty in domain_uncertainties.get(domain, domain_uncertainties["general"]):
                if "future" not in uncertainty["uncertainty"].lower():
                    uncertainty["impact"] = "high"
        
        # Return uncertainties for the identified domain
        if domain in domain_uncertainties:
            return domain_uncertainties[domain]
        else:
            return domain_uncertainties["general"]
    
    def _extract_domain_from_question(self, question: str) -> str:
        """
        Extract the domain from the question.
        
        Args:
            question: The analytical question
            
        Returns:
            Domain name
        """
        # Simple domain extraction based on keywords
        question_lower = question.lower()
        
        domains = {
            "economic": ["economic", "economy", "market", "financial", "growth", "recession", "inflation", "investment"],
            "political": ["political", "government", "policy", "regulation", "election", "democratic", "republican"],
            "technological": ["technology", "innovation", "digital", "ai", "automation", "tech", "software", "hardware"],
            "social": ["social", "cultural", "demographic", "education", "healthcare", "society", "community"]
        }
        
        # Count domain keywords
        domain_counts = {domain: sum(1 for term in terms if term in question_lower) for domain, terms in domains.items()}
        
        # Get domain with highest count
        if any(domain_counts.values()):
            primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0]
            return primary_domain
        
        # Default domain
        return "general"
    
    def _extract_question_type(self, question: str) -> str:
        """
        Extract the question type.
        
        Args:
            question: The analytical question
            
        Returns:
            Question type
        """
        question_lower = question.lower()
        
        # Check for predictive questions
        predictive_terms = ["will", "future", "next", "predict", "forecast", "outlook", "prospect", "projection"]
        if any(term in question_lower for term in predictive_terms):
            return "predictive"
        
        # Check for causal questions
        causal_terms = ["why", "cause", "reason", "factor", "driver", "lead to", "result in", "because"]
        if any(term in question_lower for term in causal_terms):
            return "causal"
        
        # Check for evaluative questions
        evaluative_terms = ["how effective", "how successful", "assess", "evaluate", "better", "worse", "should", "best"]
        if any(term in question_lower for term in evaluative_terms):
            return "evaluative"
        
        # Default to descriptive
        return "descriptive"
    
    def _categorize_uncertainties(self, uncertainties: List[Dict], context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Categorize uncertainties by type and impact.
        
        Args:
            uncertainties: List of uncertainties
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary of categorized uncertainties
        """
        logger.info("Categorizing uncertainties")
        
        # Initialize categorization structure
        categorized_uncertainties = {
            "by_type": {
                "data": [],
                "model": [],
                "parameter": [],
                "structural": [],
                "future": [],
                "linguistic": [],
                "other": []
            },
            "by_impact": {
                "high": [],
                "medium": [],
                "low": []
            },
            "by_reducibility": {
                "reducible": [],
                "partially_reducible": [],
                "irreducible": []
            }
        }
        
        # Use Llama4ScoutMCP to categorize uncertainties
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Format uncertainties for prompt
            uncertainties_text = ""
            for i, uncertainty in enumerate(uncertainties):
                uncertainty_description = uncertainty.get("uncertainty", "")
                uncertainty_explanation = uncertainty.get("explanation", "")
                uncertainties_text += f"{i+1}. {uncertainty_description}\n   {uncertainty_explanation}\n\n"
            
            # Create prompt for uncertainty categorization
            prompt = f"""
            Categorize the following uncertainties by type, impact, and reducibility.
            
            Uncertainties:
            {uncertainties_text}
            
            For each uncertainty, determine:
            
            1. Type category:
               - Data uncertainty (missing, incomplete, or unreliable data)
               - Model uncertainty (limitations in analytical frameworks)
               - Parameter uncertainty (unknown values of key variables)
               - Structural uncertainty (unknown causal relationships)
               - Future uncertainty (unpredictable future developments)
               - Linguistic uncertainty (ambiguity or vagueness in key terms)
               - Other (if it doesn't fit the above categories)
            
            2. Reducibility category:
               - Reducible (can be significantly reduced with more research or information)
               - Partially reducible (can be somewhat reduced but some uncertainty will remain)
               - Irreducible (inherently uncertain and cannot be significantly reduced)
            
            Provide your categorization in a structured format that clearly indicates the category assignments for each uncertainty.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "How should these uncertainties be categorized?",
                "analysis_type": "categorization",
                "context": {"prompt": prompt}
            })
            
            # Extract categorization from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse categorization from content
                categorization = self._parse_categorization_from_text(content, uncertainties)
                if categorization:
                    # Update uncertainties with categorization
                    for i, uncertainty in enumerate(uncertainties):
                        if i < len(categorization):
                            uncertainty_type = categorization[i].get("type", "other")
                            uncertainty_reducibility = categorization[i].get("reducibility", "partially_reducible")
                            
                            # Add categorization to uncertainty
                            uncertainty["type"] = uncertainty_type
                            uncertainty["reducibility"] = uncertainty_reducibility
                    
                    # Categorize uncertainties
                    for uncertainty in uncertainties:
                        # Categorize by type
                        uncertainty_type = uncertainty.get("type", "other")
                        if uncertainty_type in categorized_uncertainties["by_type"]:
                            categorized_uncertainties["by_type"][uncertainty_type].append(uncertainty)
                        else:
                            categorized_uncertainties["by_type"]["other"].append(uncertainty)
                        
                        # Categorize by impact
                        uncertainty_impact = uncertainty.get("impact", "medium")
                        if uncertainty_impact in categorized_uncertainties["by_impact"]:
                            categorized_uncertainties["by_impact"][uncertainty_impact].append(uncertainty)
                        else:
                            categorized_uncertainties["by_impact"]["medium"].append(uncertainty)
                        
                        # Categorize by reducibility
                        uncertainty_reducibility = uncertainty.get("reducibility", "partially_reducible")
                        if uncertainty_reducibility in categorized_uncertainties["by_reducibility"]:
                            categorized_uncertainties["by_reducibility"][uncertainty_reducibility].append(uncertainty)
                        else:
                            categorized_uncertainties["by_reducibility"]["partially_reducible"].append(uncertainty)
                    
                    return categorized_uncertainties
        
        # Fallback: Simple categorization based on keywords
        for uncertainty in uncertainties:
            uncertainty_description = uncertainty.get("uncertainty", "").lower()
            uncertainty_explanation = uncertainty.get("explanation", "").lower()
            uncertainty_impact = uncertainty.get("impact", "medium")
            
            # Determine type based on keywords
            uncertainty_type = "other"
            
            data_keywords = ["data", "information", "statistics", "measurement", "reporting", "incomplete", "unreliable"]
            model_keywords = ["model", "framework", "theory", "approach", "methodology", "assumption"]
            parameter_keywords = ["parameter", "variable", "value", "coefficient", "rate", "level"]
            structural_keywords = ["structural", "causal", "relationship", "mechanism", "interaction", "system"]
            future_keywords = ["future", "predict", "forecast", "projection", "scenario", "development"]
            linguistic_keywords = ["linguistic", "language", "definition", "term", "concept", "ambiguity", "vague"]
            
            # Check for keywords in description and explanation
            text_to_check = uncertainty_description + " " + uncertainty_explanation
            
            if any(keyword in text_to_check for keyword in data_keywords):
                uncertainty_type = "data"
            elif any(keyword in text_to_check for keyword in model_keywords):
                uncertainty_type = "model"
            elif any(keyword in text_to_check for keyword in parameter_keywords):
                uncertainty_type = "parameter"
            elif any(keyword in text_to_check for keyword in structural_keywords):
                uncertainty_type = "structural"
            elif any(keyword in text_to_check for keyword in future_keywords):
                uncertainty_type = "future"
            elif any(keyword in text_to_check for keyword in linguistic_keywords):
                uncertainty_type = "linguistic"
            
            # Determine reducibility based on type
            uncertainty_reducibility = "partially_reducible"  # Default
            
            if uncertainty_type == "data":
                uncertainty_reducibility = "reducible"
            elif uncertainty_type == "future":
                uncertainty_reducibility = "irreducible"
            elif uncertainty_type == "structural":
                uncertainty_reducibility = "partially_reducible"
            
            # Add categorization to uncertainty
            uncertainty["type"] = uncertainty_type
            uncertainty["reducibility"] = uncertainty_reducibility
            
            # Categorize by type
            if uncertainty_type in categorized_uncertainties["by_type"]:
                categorized_uncertainties["by_type"][uncertainty_type].append(uncertainty)
            else:
                categorized_uncertainties["by_type"]["other"].append(uncertainty)
            
            # Categorize by impact
            if uncertainty_impact in categorized_uncertainties["by_impact"]:
                categorized_uncertainties["by_impact"][uncertainty_impact].append(uncertainty)
            else:
                categorized_uncertainties["by_impact"]["medium"].append(uncertainty)
            
            # Categorize by reducibility
            if uncertainty_reducibility in categorized_uncertainties["by_reducibility"]:
                categorized_uncertainties["by_reducibility"][uncertainty_reducibility].append(uncertainty)
            else:
                categorized_uncertainties["by_reducibility"]["partially_reducible"].append(uncertainty)
        
        return categorized_uncertainties
    
    def _parse_categorization_from_text(self, text: str, uncertainties: List[Dict]) -> List[Dict]:
        """
        Parse uncertainty categorization from text.
        
        Args:
            text: Text containing categorization
            uncertainties: List of uncertainties
            
        Returns:
            List of categorization dictionaries
        """
        categorization = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for uncertainty categorizations
        for i, uncertainty in enumerate(uncertainties):
            uncertainty_description = uncertainty.get("uncertainty", "")
            
            # Look for section about this uncertainty
            uncertainty_pattern = f"{i+1}\..*?{re.escape(uncertainty_description[:30])}.*?(?:\n\n|\Z)"
            uncertainty_match = re.search(uncertainty_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if not uncertainty_match:
                # Try alternative pattern
                uncertainty_pattern = f"Uncertainty {i+1}:.*?(?:\n\n|\Z)"
                uncertainty_match = re.search(uncertainty_pattern, text, re.DOTALL)
            
            if uncertainty_match:
                uncertainty_text = uncertainty_match.group(0)
                
                # Extract type
                uncertainty_type = "other"  # Default
                type_pattern = r"Type.*?(?:data|model|parameter|structural|future|linguistic|other)"
                type_match = re.search(type_pattern, uncertainty_text, re.IGNORECASE)
                
                if type_match:
                    type_text = type_match.group(0).lower()
                    
                    if "data" in type_text:
                        uncertainty_type = "data"
                    elif "model" in type_text:
                        uncertainty_type = "model"
                    elif "parameter" in type_text:
                        uncertainty_type = "parameter"
                    elif "structural" in type_text:
                        uncertainty_type = "structural"
                    elif "future" in type_text:
                        uncertainty_type = "future"
                    elif "linguistic" in type_text:
                        uncertainty_type = "linguistic"
                
                # Extract reducibility
                uncertainty_reducibility = "partially_reducible"  # Default
                reducibility_pattern = r"Reducibility.*?(?:reducible|partially reducible|irreducible)"
                reducibility_match = re.search(reducibility_pattern, uncertainty_text, re.IGNORECASE)
                
                if reducibility_match:
                    reducibility_text = reducibility_match.group(0).lower()
                    
                    if "irreducible" in reducibility_text:
                        uncertainty_reducibility = "irreducible"
                    elif "partially" in reducibility_text:
                        uncertainty_reducibility = "partially_reducible"
                    elif "reducible" in reducibility_text:
                        uncertainty_reducibility = "reducible"
                
                categorization.append({
                    "type": uncertainty_type,
                    "reducibility": uncertainty_reducibility
                })
            else:
                # Default categorization if not found
                categorization.append({
                    "type": "other",
                    "reducibility": "partially_reducible"
                })
        
        return categorization
    
    def _assess_confidence(self, uncertainties: List[Dict], categorized_uncertainties: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Assess confidence levels and knowledge gaps.
        
        Args:
            uncertainties: List of uncertainties
            categorized_uncertainties: Dictionary of categorized uncertainties
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing confidence assessment
        """
        logger.info("Assessing confidence levels")
        
        # Initialize confidence assessment
        confidence_assessment = {
            "overall_confidence": "medium",
            "confidence_by_area": {},
            "knowledge_gaps": [],
            "confidence_factors": {
                "supporting": [],
                "limiting": []
            }
        }
        
        # Calculate overall confidence based on uncertainty impact
        high_impact_count = len(categorized_uncertainties["by_impact"].get("high", []))
        medium_impact_count = len(categorized_uncertainties["by_impact"].get("medium", []))
        low_impact_count = len(categorized_uncertainties["by_impact"].get("low", []))
        
        total_uncertainties = high_impact_count + medium_impact_count + low_impact_count
        
        if total_uncertainties > 0:
            # Calculate weighted impact score
            weighted_impact = (high_impact_count * 3 + medium_impact_count * 2 + low_impact_count * 1) / total_uncertainties
            
            # Determine overall confidence based on weighted impact
            if weighted_impact > 2.5:
                confidence_assessment["overall_confidence"] = "low"
            elif weighted_impact > 1.5:
                confidence_assessment["overall_confidence"] = "medium"
            else:
                confidence_assessment["overall_confidence"] = "high"
        
        # Identify confidence by area
        # Group uncertainties by domain areas
        domain_areas = self._extract_domain_areas(uncertainties)
        
        for area, area_uncertainties in domain_areas.items():
            # Calculate confidence for this area
            high_impact_area = sum(1 for u in area_uncertainties if u.get("impact", "medium") == "high")
            medium_impact_area = sum(1 for u in area_uncertainties if u.get("impact", "medium") == "medium")
            low_impact_area = sum(1 for u in area_uncertainties if u.get("impact", "medium") == "low")
            
            total_area = high_impact_area + medium_impact_area + low_impact_area
            
            if total_area > 0:
                # Calculate weighted impact score for this area
                weighted_area_impact = (high_impact_area * 3 + medium_impact_area * 2 + low_impact_area * 1) / total_area
                
                # Determine confidence for this area
                area_confidence = "medium"
                if weighted_area_impact > 2.5:
                    area_confidence = "low"
                elif weighted_area_impact > 1.5:
                    area_confidence = "medium"
                else:
                    area_confidence = "high"
                
                confidence_assessment["confidence_by_area"][area] = area_confidence
        
        # Identify knowledge gaps
        # Focus on high impact uncertainties that are reducible
        for uncertainty in uncertainties:
            impact = uncertainty.get("impact", "medium")
            reducibility = uncertainty.get("reducibility", "partially_reducible")
            
            if impact == "high" and reducibility in ["reducible", "partially_reducible"]:
                confidence_assessment["knowledge_gaps"].append({
                    "gap": uncertainty.get("uncertainty", ""),
                    "potential_resolution": self._generate_resolution_approach(uncertainty)
                })
        
        # Identify confidence factors
        # Supporting factors (areas with high confidence)
        for area, confidence in confidence_assessment["confidence_by_area"].items():
            if confidence == "high":
                confidence_assessment["confidence_factors"]["supporting"].append(
                    f"High confidence in {area} assessments"
                )
        
        # Add factor for reducible uncertainties
        reducible_count = len(categorized_uncertainties["by_reducibility"].get("reducible", []))
        if reducible_count > 0:
            confidence_assessment["confidence_factors"]["supporting"].append(
                f"{reducible_count} uncertainties are potentially reducible with additional research"
            )
        
        # Limiting factors (areas with low confidence)
        for area, confidence in confidence_assessment["confidence_by_area"].items():
            if confidence == "low":
                confidence_assessment["confidence_factors"]["limiting"].append(
                    f"Low confidence in {area} assessments due to significant uncertainties"
                )
        
        # Add factor for irreducible uncertainties
        irreducible_count = len(categorized_uncertainties["by_reducibility"].get("irreducible", []))
        if irreducible_count > 0:
            confidence_assessment["confidence_factors"]["limiting"].append(
                f"{irreducible_count} uncertainties are irreducible and will persist regardless of additional research"
            )
        
        return confidence_assessment
    
    def _extract_domain_areas(self, uncertainties: List[Dict]) -> Dict:
        """
        Extract domain areas from uncertainties.
        
        Args:
            uncertainties: List of uncertainties
            
        Returns:
            Dictionary mapping domain areas to uncertainties
        """
        domain_areas = {}
        
        # Define domain area keywords
        area_keywords = {
            "economic": ["economic", "market", "financial", "fiscal", "monetary", "trade", "investment"],
            "political": ["political", "policy", "government", "regulation", "governance", "election"],
            "social": ["social", "cultural", "demographic", "public", "society", "community"],
            "technological": ["technological", "technology", "innovation", "digital", "technical"],
            "environmental": ["environmental", "climate", "ecological", "sustainability", "resource"],
            "security": ["security", "defense", "military", "conflict", "threat", "risk"],
            "data_and_methodology": ["data", "methodology", "measurement", "analysis", "model", "framework"]
        }
        
        # Categorize uncertainties by domain area
        for uncertainty in uncertainties:
            uncertainty_text = uncertainty.get("uncertainty", "") + " " + uncertainty.get("explanation", "")
            uncertainty_text = uncertainty_text.lower()
            
            # Determine primary area
            area_matches = {}
            for area, keywords in area_keywords.items():
                area_matches[area] = sum(1 for keyword in keywords if keyword in uncertainty_text)
            
            # Get area with most matches
            if any(area_matches.values()):
                primary_area = max(area_matches.items(), key=lambda x: x[1])[0]
                
                # Add to domain areas
                if primary_area in domain_areas:
                    domain_areas[primary_area].append(uncertainty)
                else:
                    domain_areas[primary_area] = [uncertainty]
            else:
                # Default to "general" if no matches
                if "general" in domain_areas:
                    domain_areas["general"].append(uncertainty)
                else:
                    domain_areas["general"] = [uncertainty]
        
        return domain_areas
    
    def _generate_resolution_approach(self, uncertainty: Dict) -> str:
        """
        Generate an approach for resolving or reducing an uncertainty.
        
        Args:
            uncertainty: Uncertainty dictionary
            
        Returns:
            Resolution approach
        """
        uncertainty_type = uncertainty.get("type", "other")
        uncertainty_description = uncertainty.get("uncertainty", "")
        
        # Generate resolution approach based on uncertainty type
        if uncertainty_type == "data":
            return f"Gather additional data on {uncertainty_description.lower()} through targeted research, alternative data sources, or improved measurement techniques"
        
        elif uncertainty_type == "model":
            return f"Test alternative analytical frameworks for {uncertainty_description.lower()} and compare results to identify robust insights"
        
        elif uncertainty_type == "parameter":
            return f"Conduct sensitivity analysis to understand how different values of key parameters affect outcomes related to {uncertainty_description.lower()}"
        
        elif uncertainty_type == "structural":
            return f"Develop and test multiple causal models to better understand the relationships involved in {uncertainty_description.lower()}"
        
        elif uncertainty_type == "future":
            return f"Use scenario planning and horizon scanning to prepare for different possible developments related to {uncertainty_description.lower()}"
        
        elif uncertainty_type == "linguistic":
            return f"Develop clear operational definitions and ensure consistent understanding of key terms related to {uncertainty_description.lower()}"
        
        else:
            return f"Conduct targeted research to better understand the nature and implications of {uncertainty_description.lower()}"
    
    def _develop_management_strategies(self, uncertainties: List[Dict], categorized_uncertainties: Dict, confidence_assessment: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Develop strategies for managing uncertainty.
        
        Args:
            uncertainties: List of uncertainties
            categorized_uncertainties: Dictionary of categorized uncertainties
            confidence_assessment: Dictionary containing confidence assessment
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing management strategies
        """
        logger.info("Developing uncertainty management strategies")
        
        # Initialize management strategies
        management_strategies = {
            "research_strategies": [],
            "analytical_strategies": [],
            "decision_strategies": [],
            "communication_strategies": []
        }
        
        # Use Llama4ScoutMCP to develop management strategies
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Format uncertainties for prompt
            high_impact_uncertainties = categorized_uncertainties["by_impact"].get("high", [])
            high_impact_text = ""
            
            for i, uncertainty in enumerate(high_impact_uncertainties[:5]):  # Limit to top 5 high impact
                uncertainty_description = uncertainty.get("uncertainty", "")
                uncertainty_type = uncertainty.get("type", "other")
                uncertainty_reducibility = uncertainty.get("reducibility", "partially_reducible")
                
                high_impact_text += f"{i+1}. {uncertainty_description} (Type: {uncertainty_type}, Reducibility: {uncertainty_reducibility})\n"
            
            # Format knowledge gaps
            knowledge_gaps = confidence_assessment.get("knowledge_gaps", [])
            knowledge_gaps_text = ""
            
            for i, gap in enumerate(knowledge_gaps[:3]):  # Limit to top 3 gaps
                gap_description = gap.get("gap", "")
                knowledge_gaps_text += f"{i+1}. {gap_description}\n"
            
            # Create prompt for management strategies
            prompt = f"""
            Develop strategies for managing the following high-impact uncertainties and knowledge gaps in the analysis.
            
            High-Impact Uncertainties:
            {high_impact_text}
            
            Key Knowledge Gaps:
            {knowledge_gaps_text}
            
            Please provide specific strategies in the following categories:
            
            1. Research Strategies: How to gather additional information or reduce key uncertainties
            2. Analytical Strategies: How to structure analysis to account for uncertainties
            3. Decision Strategies: How to make robust decisions despite uncertainties
            4. Communication Strategies: How to effectively communicate uncertainties to stakeholders
            
            For each category, provide 3-5 specific, actionable strategies that directly address the identified uncertainties and knowledge gaps.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "How can we manage these uncertainties?",
                "analysis_type": "strategy_development",
                "context": {"prompt": prompt}
            })
            
            # Extract strategies from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse strategies from content
                strategies = self._parse_strategies_from_text(content)
                if strategies:
                    return strategies
        
        # Fallback: Generate generic management strategies
        return self._generate_generic_management_strategies(uncertainties, categorized_uncertainties, confidence_assessment)
    
    def _parse_strategies_from_text(self, text: str) -> Dict:
        """
        Parse management strategies from text.
        
        Args:
            text: Text containing strategies
            
        Returns:
            Dictionary of management strategies
        """
        strategies = {
            "research_strategies": [],
            "analytical_strategies": [],
            "decision_strategies": [],
            "communication_strategies": []
        }
        
        # Simple parsing based on patterns
        import re
        
        # Define section patterns
        section_patterns = {
            "research_strategies": r"(?:Research Strategies|1\.).*?(?:\n\n|\Z)",
            "analytical_strategies": r"(?:Analytical Strategies|2\.).*?(?:\n\n|\Z)",
            "decision_strategies": r"(?:Decision Strategies|3\.).*?(?:\n\n|\Z)",
            "communication_strategies": r"(?:Communication Strategies|4\.).*?(?:\n\n|\Z)"
        }
        
        # Extract strategies from each section
        for strategy_type, pattern in section_patterns.items():
            section_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if section_match:
                section_text = section_match.group(0)
                
                # Extract bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, section_text, re.DOTALL)
                
                if bullet_matches:
                    strategies[strategy_type] = [item.strip() for item in bullet_matches if item.strip()]
        
        return strategies
    
    def _generate_generic_management_strategies(self, uncertainties: List[Dict], categorized_uncertainties: Dict, confidence_assessment: Dict) -> Dict:
        """
        Generate generic management strategies.
        
        Args:
            uncertainties: List of uncertainties
            categorized_uncertainties: Dictionary of categorized uncertainties
            confidence_assessment: Dictionary containing confidence assessment
            
        Returns:
            Dictionary of generic management strategies
        """
        # Initialize management strategies
        management_strategies = {
            "research_strategies": [
                "Conduct targeted research on high-impact uncertainties that are reducible",
                "Develop improved data collection methods for key data uncertainties",
                "Consult domain experts to better understand structural uncertainties",
                "Monitor early indicators of future developments to reduce future uncertainties",
                "Establish a systematic process for ongoing uncertainty identification and tracking"
            ],
            "analytical_strategies": [
                "Use multiple analytical frameworks to test the robustness of findings",
                "Conduct sensitivity analysis to understand how key uncertainties affect outcomes",
                "Develop scenarios that account for different combinations of uncertainties",
                "Explicitly incorporate uncertainty into analytical models",
                "Use Bayesian methods to update assessments as new information becomes available"
            ],
            "decision_strategies": [
                "Identify robust options that perform well across different uncertainty scenarios",
                "Design adaptive strategies that can be adjusted as uncertainties resolve",
                "Establish clear decision triggers based on key indicators",
                "Maintain strategic flexibility to respond to unexpected developments",
                "Consider portfolio approaches that diversify across different uncertainty scenarios"
            ],
            "communication_strategies": [
                "Clearly communicate the nature and implications of key uncertainties",
                "Use consistent language and visual representations for uncertainty",
                "Distinguish between different types of uncertainty in communications",
                "Explain how uncertainties have been incorporated into the analysis",
                "Provide regular updates as uncertainties evolve or resolve"
            ]
        }
        
        # Customize strategies based on uncertainty types
        # Check if there are many data uncertainties
        data_uncertainties = categorized_uncertainties["by_type"].get("data", [])
        if len(data_uncertainties) >= 2:
            management_strategies["research_strategies"].append(
                f"Prioritize improving data quality for {data_uncertainties[0].get('uncertainty', 'key data issues')}"
            )
        
        # Check if there are many future uncertainties
        future_uncertainties = categorized_uncertainties["by_type"].get("future", [])
        if len(future_uncertainties) >= 2:
            management_strategies["analytical_strategies"].append(
                "Develop comprehensive scenario planning to address future uncertainties"
            )
            management_strategies["decision_strategies"].append(
                "Design contingency plans for different future scenarios"
            )
        
        # Check if there are many structural uncertainties
        structural_uncertainties = categorized_uncertainties["by_type"].get("structural", [])
        if len(structural_uncertainties) >= 2:
            management_strategies["analytical_strategies"].append(
                "Test multiple causal models to address structural uncertainties"
            )
        
        return management_strategies
    
    def _extract_findings(self, uncertainties: List[Dict], categorized_uncertainties: Dict, confidence_assessment: Dict) -> List[Dict]:
        """
        Extract key findings from uncertainty mapping.
        
        Args:
            uncertainties: List of uncertainties
            categorized_uncertainties: Dictionary of categorized uncertainties
            confidence_assessment: Dictionary containing confidence assessment
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about overall confidence
        overall_confidence = confidence_assessment.get("overall_confidence", "medium")
        findings.append({
            "finding": f"Overall confidence in the analysis is {overall_confidence} due to the identified uncertainties",
            "confidence": "high",
            "source": "uncertainty_mapping"
        })
        
        # Add finding about high impact uncertainties
        high_impact_uncertainties = categorized_uncertainties["by_impact"].get("high", [])
        if high_impact_uncertainties:
            findings.append({
                "finding": f"Key high-impact uncertainty: {high_impact_uncertainties[0].get('uncertainty', '')}",
                "confidence": "high",
                "source": "uncertainty_mapping"
            })
        
        # Add finding about reducible uncertainties
        reducible_uncertainties = categorized_uncertainties["by_reducibility"].get("reducible", [])
        if reducible_uncertainties:
            findings.append({
                "finding": f"Priority uncertainty for research: {reducible_uncertainties[0].get('uncertainty', '')}",
                "confidence": "medium",
                "source": "uncertainty_mapping"
            })
        
        # Add finding about irreducible uncertainties
        irreducible_uncertainties = categorized_uncertainties["by_reducibility"].get("irreducible", [])
        if irreducible_uncertainties:
            findings.append({
                "finding": f"Key irreducible uncertainty that must be accepted: {irreducible_uncertainties[0].get('uncertainty', '')}",
                "confidence": "medium",
                "source": "uncertainty_mapping"
            })
        
        # Add finding about confidence by area
        confidence_by_area = confidence_assessment.get("confidence_by_area", {})
        low_confidence_areas = [area for area, confidence in confidence_by_area.items() if confidence == "low"]
        if low_confidence_areas:
            findings.append({
                "finding": f"Low confidence in assessments related to {low_confidence_areas[0]} due to significant uncertainties",
                "confidence": "medium",
                "source": "uncertainty_mapping"
            })
        
        return findings
    
    def _extract_assumptions(self, uncertainties: List[Dict], categorized_uncertainties: Dict) -> List[Dict]:
        """
        Extract assumptions from uncertainty mapping.
        
        Args:
            uncertainties: List of uncertainties
            categorized_uncertainties: Dictionary of categorized uncertainties
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about uncertainty identification
        assumptions.append({
            "assumption": "The identified uncertainties represent the most significant sources of uncertainty for this analysis",
            "criticality": "high",
            "source": "uncertainty_mapping"
        })
        
        # Add assumption about uncertainty impact
        assumptions.append({
            "assumption": "The assessed impact levels of uncertainties accurately reflect their importance to the analysis",
            "criticality": "medium",
            "source": "uncertainty_mapping"
        })
        
        # Add assumption about uncertainty reducibility
        assumptions.append({
            "assumption": "The categorization of uncertainties as reducible or irreducible is accurate",
            "criticality": "medium",
            "source": "uncertainty_mapping"
        })
        
        # Add assumption from high impact uncertainties
        high_impact_uncertainties = categorized_uncertainties["by_impact"].get("high", [])
        if high_impact_uncertainties:
            high_impact_uncertainty = high_impact_uncertainties[0]
            uncertainty_type = high_impact_uncertainty.get("type", "other")
            
            if uncertainty_type == "data":
                assumptions.append({
                    "assumption": "Available data is sufficient for meaningful analysis despite identified data uncertainties",
                    "criticality": "high",
                    "source": "uncertainty_mapping"
                })
            elif uncertainty_type == "model":
                assumptions.append({
                    "assumption": "Selected analytical frameworks are appropriate despite identified model uncertainties",
                    "criticality": "high",
                    "source": "uncertainty_mapping"
                })
            elif uncertainty_type == "future":
                assumptions.append({
                    "assumption": "Analysis can provide valuable insights despite inherent unpredictability of future developments",
                    "criticality": "high",
                    "source": "uncertainty_mapping"
                })
        
        return assumptions
    
    def _extract_meta_uncertainties(self, uncertainties: List[Dict], confidence_assessment: Dict) -> List[Dict]:
        """
        Extract meta-uncertainties about the uncertainty mapping itself.
        
        Args:
            uncertainties: List of uncertainties
            confidence_assessment: Dictionary containing confidence assessment
            
        Returns:
            List of meta-uncertainties
        """
        meta_uncertainties = []
        
        # Add meta-uncertainty about completeness
        meta_uncertainties.append({
            "uncertainty": "There may be important uncertainties that have not been identified",
            "impact": "medium",
            "source": "uncertainty_mapping"
        })
        
        # Add meta-uncertainty about impact assessment
        meta_uncertainties.append({
            "uncertainty": "The assessed impact of uncertainties may not accurately reflect their true importance",
            "impact": "medium",
            "source": "uncertainty_mapping"
        })
        
        # Add meta-uncertainty about reducibility assessment
        meta_uncertainties.append({
            "uncertainty": "Some uncertainties categorized as irreducible may be more reducible than anticipated, or vice versa",
            "impact": "low",
            "source": "uncertainty_mapping"
        })
        
        # Add meta-uncertainty about confidence assessment
        meta_uncertainties.append({
            "uncertainty": "The overall confidence assessment may not fully capture the implications of all uncertainties",
            "impact": "medium",
            "source": "uncertainty_mapping"
        })
        
        return meta_uncertainties
