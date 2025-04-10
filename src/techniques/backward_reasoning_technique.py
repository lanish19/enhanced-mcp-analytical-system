"""
Backward Reasoning Technique for analyzing problems by working backward from outcomes.
This module provides the BackwardReasoningTechnique class for outcome-based analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.analytical_technique import AnalyticalTechnique
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackwardReasoningTechnique(AnalyticalTechnique):
    """
    Backward Reasoning Technique for analyzing problems by working backward from outcomes.
    
    This technique provides capabilities for:
    1. Defining potential end states or outcomes
    2. Working backward to identify necessary conditions and causal pathways
    3. Assessing the likelihood and requirements of different pathways
    4. Identifying key indicators and decision points
    """
    
    def __init__(self):
        """Initialize the Backward Reasoning Technique."""
        super().__init__(
            name="backward_reasoning",
            description="Analyzes problems by working backward from potential outcomes to identify causal pathways",
            required_mcps=["llama4_scout", "research_mcp"],
            compatible_techniques=["scenario_triangulation", "causal_network_analysis", "indicators_development"],
            incompatible_techniques=[]
        )
        logger.info("Initialized BackwardReasoningTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Backward Reasoning Technique")
        
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
            research_results = context.get("research_results")
            
            # Define end states
            end_states = self._define_end_states(question, research_results, context, parameters)
            
            # Identify causal pathways
            causal_pathways = self._identify_causal_pathways(end_states, research_results, context, parameters)
            
            # Assess pathway likelihoods
            pathway_assessment = self._assess_pathway_likelihoods(causal_pathways, research_results, context, parameters)
            
            # Identify key indicators
            key_indicators = self._identify_key_indicators(causal_pathways, pathway_assessment, research_results, context, parameters)
            
            # Compile results
            results = {
                "technique": "backward_reasoning",
                "timestamp": time.time(),
                "question": question,
                "end_states": end_states,
                "causal_pathways": causal_pathways,
                "pathway_assessment": pathway_assessment,
                "key_indicators": key_indicators,
                "findings": self._extract_findings(end_states, causal_pathways, pathway_assessment, key_indicators),
                "assumptions": self._extract_assumptions(causal_pathways),
                "uncertainties": self._extract_uncertainties(causal_pathways, pathway_assessment)
            }
            
            # Add results to context
            context.add_technique_result("backward_reasoning", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Backward Reasoning Technique: {e}")
            return self.handle_error(e, context)
    
    def _define_end_states(self, question: str, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Define potential end states or outcomes.
        
        Args:
            question: The analytical question
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of end states
        """
        logger.info("Defining end states")
        
        # Check if end states are provided in parameters
        if "end_states" in parameters and isinstance(parameters["end_states"], list):
            return parameters["end_states"]
        
        # Use Llama4ScoutMCP to define end states
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Create prompt for end state definition
            prompt = f"""
            Based on the following question, define 3-5 distinct potential end states or outcomes that could occur.
            
            Question: {question}
            
            For each end state:
            1. Provide a clear name and description
            2. Explain the key characteristics of this outcome
            3. Identify the implications of this outcome
            4. Assess the relative desirability of this outcome (from different perspectives if relevant)
            
            Ensure the end states cover a range of possibilities, from favorable to unfavorable, and include both expected and less likely but important outcomes.
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "predictive",
                "context": {"prompt": grounded_prompt, "research_results": research_results}
            })
            
            # Extract end states from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse end states from content
                end_states = self._parse_end_states_from_text(content)
                if end_states:
                    return end_states
        
        # Check for scenario results in context
        scenario_results = context.get_technique_result("scenario_triangulation")
        if scenario_results and "scenarios" in scenario_results:
            scenarios = scenario_results.get("scenarios", [])
            if scenarios:
                # Convert scenarios to end states
                end_states = []
                for scenario in scenarios[:5]:  # Limit to top 5
                    end_states.append({
                        "name": scenario.get("name", "Unknown scenario"),
                        "description": scenario.get("narrative", ""),
                        "characteristics": [f"Characterized by {outcome}" for factor, outcome in scenario.get("uncertainty_outcomes", {}).items()],
                        "implications": [f"Impact on {factor}" for factor in scenario.get("uncertainty_outcomes", {}).keys()],
                        "desirability": "mixed"  # Default value
                    })
                return end_states
        
        # Fallback: Generate generic end states
        return self._generate_generic_end_states(question)
    
    def _parse_end_states_from_text(self, text: str) -> List[Dict]:
        """
        Parse end states from text.
        
        Args:
            text: Text containing end state descriptions
            
        Returns:
            List of parsed end states
        """
        end_states = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for end state sections
        end_state_pattern = r'(?:^|\n)(?:End State|Outcome|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:End State|Outcome|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        end_state_matches = re.findall(end_state_pattern, text, re.DOTALL)
        
        if not end_state_matches:
            # Try alternative pattern for numbered lists
            end_state_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            end_state_matches = re.findall(end_state_pattern, text, re.DOTALL)
            
            if end_state_matches:
                # Convert to expected format
                end_state_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(end_state_matches)]
        
        for match in end_state_matches:
            end_state_num = match[0].strip() if len(match) > 0 else ""
            end_state_name = match[1].strip() if len(match) > 1 else ""
            end_state_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract description
            description = ""
            description_pattern = r'(?:description|overview).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            description_match = re.search(description_pattern, end_state_content, re.IGNORECASE | re.DOTALL)
            if description_match:
                description = description_match.group(1).strip()
            else:
                # Use first paragraph as description
                paragraphs = end_state_content.split('\n\n')
                if paragraphs:
                    description = paragraphs[0].strip()
            
            # Extract characteristics
            characteristics = []
            characteristics_pattern = r'(?:characteristics|key characteristics|features|attributes).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            characteristics_match = re.search(characteristics_pattern, end_state_content, re.IGNORECASE | re.DOTALL)
            if characteristics_match:
                characteristics_text = characteristics_match.group(1).strip()
                # Look for bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, characteristics_text, re.DOTALL)
                if bullet_matches:
                    characteristics = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', characteristics_text)
                    characteristics = [item.strip() for item in items if item.strip()]
            
            # Extract implications
            implications = []
            implications_pattern = r'(?:implications|consequences|effects|impacts).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            implications_match = re.search(implications_pattern, end_state_content, re.IGNORECASE | re.DOTALL)
            if implications_match:
                implications_text = implications_match.group(1).strip()
                # Look for bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, implications_text, re.DOTALL)
                if bullet_matches:
                    implications = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', implications_text)
                    implications = [item.strip() for item in items if item.strip()]
            
            # Extract desirability
            desirability = "mixed"
            desirability_pattern = r'(?:desirability|desirable|favorable).*?(favorable|unfavorable|positive|negative|mixed|neutral)'
            desirability_match = re.search(desirability_pattern, end_state_content, re.IGNORECASE)
            if desirability_match:
                desirability_value = desirability_match.group(1).lower()
                if desirability_value in ["favorable", "positive"]:
                    desirability = "favorable"
                elif desirability_value in ["unfavorable", "negative"]:
                    desirability = "unfavorable"
                elif desirability_value in ["mixed"]:
                    desirability = "mixed"
                elif desirability_value in ["neutral"]:
                    desirability = "neutral"
            
            end_states.append({
                "name": end_state_name,
                "description": description,
                "characteristics": characteristics,
                "implications": implications,
                "desirability": desirability
            })
        
        return end_states
    
    def _generate_generic_end_states(self, question: str) -> List[Dict]:
        """
        Generate generic end states based on the question.
        
        Args:
            question: The analytical question
            
        Returns:
            List of generic end states
        """
        # Extract domain from question
        domain = self._extract_domain_from_question(question)
        
        # Domain-specific end state templates
        domain_end_states = {
            "economic": [
                {
                    "name": "Robust Growth Scenario",
                    "description": "A scenario characterized by strong economic growth, innovation, and expanding opportunities",
                    "characteristics": [
                        "GDP growth above historical averages",
                        "Low unemployment and rising wages",
                        "Technological innovation driving productivity gains",
                        "Expanding global trade and investment"
                    ],
                    "implications": [
                        "Improved living standards for many segments of society",
                        "Increased tax revenues supporting public services",
                        "Greater resources for addressing social and environmental challenges",
                        "Potential for widening inequality if growth is not inclusive"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Stagnation Scenario",
                    "description": "A scenario of economic stagnation with low growth, limited innovation, and persistent challenges",
                    "characteristics": [
                        "GDP growth below historical averages",
                        "Stagnant wages and employment opportunities",
                        "Limited productivity improvements",
                        "Reduced business investment and risk-taking"
                    ],
                    "implications": [
                        "Fiscal pressures on government budgets",
                        "Intergenerational tensions over limited resources",
                        "Political pressure for structural reforms",
                        "Potential for social unrest if conditions persist"
                    ],
                    "desirability": "unfavorable"
                },
                {
                    "name": "Transformation Scenario",
                    "description": "A scenario of economic transformation with significant structural changes to the economy",
                    "characteristics": [
                        "Rapid technological disruption of traditional industries",
                        "Emergence of new business models and sectors",
                        "Significant labor market shifts",
                        "Changes in patterns of global economic integration"
                    ],
                    "implications": [
                        "Uneven impacts across different regions and sectors",
                        "Need for significant workforce reskilling and adaptation",
                        "Opportunities for new forms of economic value creation",
                        "Challenges in managing transition costs and distributional impacts"
                    ],
                    "desirability": "mixed"
                }
            ],
            "political": [
                {
                    "name": "Stability and Reform Scenario",
                    "description": "A scenario of political stability with gradual, consensus-based reforms",
                    "characteristics": [
                        "Functional political institutions with broad legitimacy",
                        "Incremental policy changes based on evidence and compromise",
                        "Effective management of social tensions",
                        "Stable international relations"
                    ],
                    "implications": [
                        "Predictable policy environment for long-term planning",
                        "Gradual progress on addressing structural challenges",
                        "Maintained social cohesion and trust in institutions",
                        "Risk of insufficient response to rapidly evolving challenges"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Polarization and Gridlock Scenario",
                    "description": "A scenario of increasing political polarization and institutional dysfunction",
                    "characteristics": [
                        "Deep partisan divisions preventing effective governance",
                        "Declining trust in political institutions",
                        "Policy instability and reversals",
                        "Increasing social tensions and conflict"
                    ],
                    "implications": [
                        "Inability to address major policy challenges",
                        "Economic uncertainty affecting investment and growth",
                        "Erosion of democratic norms and processes",
                        "Potential for political realignment or institutional reform"
                    ],
                    "desirability": "unfavorable"
                },
                {
                    "name": "Transformation and Renewal Scenario",
                    "description": "A scenario of political transformation leading to renewed institutions and approaches",
                    "characteristics": [
                        "Significant political realignment and new coalitions",
                        "Reform of key political institutions and processes",
                        "Emergence of new political movements and leaders",
                        "Shift in dominant political paradigms"
                    ],
                    "implications": [
                        "Period of uncertainty during transition",
                        "Potential for addressing previously intractable problems",
                        "Disruption of existing power structures and interests",
                        "Opportunity for more responsive and effective governance"
                    ],
                    "desirability": "mixed"
                }
            ],
            "technological": [
                {
                    "name": "Accelerated Innovation Scenario",
                    "description": "A scenario of rapid technological advancement with widespread benefits",
                    "characteristics": [
                        "Breakthrough innovations in multiple domains",
                        "Effective governance of technology development",
                        "Broad diffusion of benefits across society",
                        "Successful management of disruption costs"
                    ],
                    "implications": [
                        "Significant productivity and economic growth",
                        "Solutions to major social and environmental challenges",
                        "Improved quality of life and new opportunities",
                        "Transformation of work and social institutions"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Uneven Progress Scenario",
                    "description": "A scenario of technological advancement with uneven distribution of benefits and costs",
                    "characteristics": [
                        "Significant innovation in select domains",
                        "Concentration of benefits among certain groups",
                        "Inadequate management of transition costs",
                        "Growing digital and technological divides"
                    ],
                    "implications": [
                        "Increasing inequality and social tensions",
                        "Mixed economic impacts across sectors and regions",
                        "Political pressure for redistributive policies",
                        "Ethical challenges around access and control"
                    ],
                    "desirability": "mixed"
                },
                {
                    "name": "Technological Backlash Scenario",
                    "description": "A scenario where technological advancement faces significant social and political resistance",
                    "characteristics": [
                        "Growing public concern about technology impacts",
                        "Restrictive regulatory responses",
                        "Declining trust in technology companies and experts",
                        "Fragmentation of technology governance globally"
                    ],
                    "implications": [
                        "Slowed pace of innovation in regulated domains",
                        "Competitive disadvantages for restrictive jurisdictions",
                        "Emergence of gray/black market technological development",
                        "Missed opportunities to address challenges through innovation"
                    ],
                    "desirability": "unfavorable"
                }
            ],
            "social": [
                {
                    "name": "Cohesion and Inclusion Scenario",
                    "description": "A scenario of strengthened social cohesion and increased inclusion",
                    "characteristics": [
                        "Reduced inequality and expanded opportunity",
                        "Strong social institutions and community bonds",
                        "Effective integration of diverse groups",
                        "High levels of social trust and cooperation"
                    ],
                    "implications": [
                        "Improved social outcomes across multiple dimensions",
                        "Greater resilience to social and economic shocks",
                        "More effective collective action on shared challenges",
                        "Potential for innovation through diverse perspectives"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Fragmentation Scenario",
                    "description": "A scenario of increasing social fragmentation and division",
                    "characteristics": [
                        "Deepening social divides along multiple dimensions",
                        "Declining trust in shared institutions",
                        "Formation of isolated social and cultural enclaves",
                        "Reduced sense of common identity or purpose"
                    ],
                    "implications": [
                        "Difficulty addressing collective challenges",
                        "Increased social conflict and tension",
                        "Erosion of social safety nets and public goods",
                        "Vulnerability to exploitation of divisions by political actors"
                    ],
                    "desirability": "unfavorable"
                },
                {
                    "name": "Adaptation and Resilience Scenario",
                    "description": "A scenario where society adapts to significant changes through new social arrangements",
                    "characteristics": [
                        "Evolution of social institutions to address new challenges",
                        "Development of new forms of community and belonging",
                        "Changing norms around work, family, and civic engagement",
                        "Diverse approaches to social organization across communities"
                    ],
                    "implications": [
                        "Uneven transition experiences across different groups",
                        "Period of social experimentation and learning",
                        "Emergence of innovative social practices and arrangements",
                        "Tension between traditional and emerging social forms"
                    ],
                    "desirability": "mixed"
                }
            ],
            "environmental": [
                {
                    "name": "Sustainable Transition Scenario",
                    "description": "A scenario of successful transition to environmental sustainability",
                    "characteristics": [
                        "Rapid decarbonization of energy systems",
                        "Circular economy principles widely adopted",
                        "Restoration of key ecosystems and biodiversity",
                        "Adaptation measures reducing climate vulnerability"
                    ],
                    "implications": [
                        "Avoidance of worst climate impacts",
                        "New economic opportunities in green sectors",
                        "Improved public health and quality of life",
                        "More resilient communities and infrastructure"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Incremental Response Scenario",
                    "description": "A scenario of partial and uneven progress on environmental challenges",
                    "characteristics": [
                        "Some progress on emissions reduction but insufficient to meet targets",
                        "Uneven implementation of sustainability measures globally",
                        "Continued degradation of some ecosystems alongside protection of others",
                        "Reactive rather than proactive approach to environmental challenges"
                    ],
                    "implications": [
                        "Significant but not catastrophic climate impacts",
                        "Increasing costs of adaptation and disaster response",
                        "Distributional conflicts over environmental resources",
                        "Gradual transition to more sustainable practices over longer timeframe"
                    ],
                    "desirability": "mixed"
                },
                {
                    "name": "Environmental Crisis Scenario",
                    "description": "A scenario of environmental degradation leading to multiple crises",
                    "characteristics": [
                        "Failure to significantly reduce emissions or other environmental pressures",
                        "Crossing of critical ecological thresholds",
                        "Accelerating impacts from climate change and biodiversity loss",
                        "Inadequate adaptation measures"
                    ],
                    "implications": [
                        "Severe economic costs from environmental disasters",
                        "Mass displacement and migration from vulnerable areas",
                        "Conflict over increasingly scarce resources",
                        "Potential for eventual radical response after significant damage"
                    ],
                    "desirability": "unfavorable"
                }
            ],
            "security": [
                {
                    "name": "Cooperative Security Scenario",
                    "description": "A scenario of enhanced international cooperation on security challenges",
                    "characteristics": [
                        "Strengthened multilateral security institutions",
                        "Effective arms control and non-proliferation regimes",
                        "Collaborative approaches to transnational threats",
                        "Reduced interstate tensions and conflict"
                    ],
                    "implications": [
                        "Lower defense spending allowing resource reallocation",
                        "More effective management of global security challenges",
                        "Reduced risk of major power conflict",
                        "Greater predictability in international relations"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Competitive Multipolarity Scenario",
                    "description": "A scenario of strategic competition between major powers in a multipolar world",
                    "characteristics": [
                        "Intensified great power rivalry across multiple domains",
                        "Weakened global governance institutions",
                        "Formation of competing security blocs or alignments",
                        "Increased military spending and capabilities"
                    ],
                    "implications": [
                        "Higher risk of conflict through miscalculation",
                        "Economic costs of security competition",
                        "Difficulty addressing transnational challenges requiring cooperation",
                        "Potential for regional proxy conflicts"
                    ],
                    "desirability": "unfavorable"
                },
                {
                    "name": "Fragmented Security Scenario",
                    "description": "A scenario of diverse security challenges with varied regional responses",
                    "characteristics": [
                        "Different security dynamics across regions",
                        "Mix of cooperation and competition between powers",
                        "Varied effectiveness of security institutions by region",
                        "Multiple security challenges from state and non-state actors"
                    ],
                    "implications": [
                        "Complex security environment requiring tailored approaches",
                        "Uneven security outcomes across different regions",
                        "Opportunities for positive security developments in some areas",
                        "Persistent challenges in fragile regions or domains"
                    ],
                    "desirability": "mixed"
                }
            ],
            "general": [
                {
                    "name": "Positive Transformation Scenario",
                    "description": "A scenario of successful adaptation to challenges leading to positive outcomes",
                    "characteristics": [
                        "Effective problem-solving across multiple domains",
                        "Balanced approach to competing priorities",
                        "Innovation leading to new solutions",
                        "Inclusive distribution of benefits"
                    ],
                    "implications": [
                        "Improved outcomes across multiple dimensions",
                        "Enhanced capacity to address future challenges",
                        "Strengthened social cohesion and trust",
                        "New opportunities emerging from successful transitions"
                    ],
                    "desirability": "favorable"
                },
                {
                    "name": "Muddling Through Scenario",
                    "description": "A scenario of partial and uneven progress with mixed outcomes",
                    "characteristics": [
                        "Some challenges addressed while others persist",
                        "Incremental rather than transformative change",
                        "Varied outcomes across different domains and groups",
                        "Reactive rather than proactive approaches"
                    ],
                    "implications": [
                        "Continued progress in some areas alongside stagnation in others",
                        "Persistent problems requiring ongoing management",
                        "Uneven distribution of benefits and costs",
                        "Gradual adaptation rather than rapid transformation"
                    ],
                    "desirability": "mixed"
                },
                {
                    "name": "Crisis and Disruption Scenario",
                    "description": "A scenario where multiple challenges overwhelm existing capacities",
                    "characteristics": [
                        "Cascading crises across interconnected systems",
                        "Inadequate institutional responses",
                        "Erosion of trust and cooperation",
                        "Disruptive impacts on established patterns"
                    ],
                    "implications": [
                        "Significant costs and negative impacts in the near term",
                        "Potential for eventual renewal and reform after crisis",
                        "Uneven distribution of impacts with vulnerable groups most affected",
                        "Unpredictable secondary effects across systems"
                    ],
                    "desirability": "unfavorable"
                }
            ]
        }
        
        # Return end states for the identified domain
        if domain in domain_end_states:
            return domain_end_states[domain]
        else:
            return domain_end_states["general"]
    
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
            "social": ["social", "cultural", "demographic", "education", "healthcare", "society", "community"],
            "environmental": ["environmental", "climate", "sustainability", "renewable", "carbon", "pollution"],
            "security": ["security", "defense", "military", "conflict", "war", "terrorism", "cyber", "intelligence"]
        }
        
        # Count domain keywords
        domain_counts = {domain: sum(1 for term in terms if term in question_lower) for domain, terms in domains.items()}
        
        # Get domain with highest count
        if any(domain_counts.values()):
            primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0]
            return primary_domain
        
        # Default domain
        return "general"
    
    def _identify_causal_pathways(self, end_states: List[Dict], research_results: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Identify causal pathways leading to each end state.
        
        Args:
            end_states: List of end states
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary of causal pathways for each end state
        """
        logger.info("Identifying causal pathways")
        
        causal_pathways = {}
        
        # Use Llama4ScoutMCP to identify causal pathways
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            for end_state in end_states:
                end_state_name = end_state.get("name", "")
                end_state_description = end_state.get("description", "")
                end_state_characteristics = "\n".join([f"- {c}" for c in end_state.get("characteristics", [])])
                
                # Create prompt for causal pathway identification
                prompt = f"""
                Working backward from the following end state, identify the causal pathway that could lead to this outcome.
                
                End State: {end_state_name}
                
                Description: {end_state_description}
                
                Key Characteristics:
                {end_state_characteristics}
                
                Please identify:
                
                1. Necessary preconditions - What conditions must exist for this outcome to be possible
                2. Key causal factors - The most important factors that would drive this outcome
                3. Critical events - Specific events or developments that would be part of this pathway
                4. Branching points - Decision points or contingencies that could alter the pathway
                5. Timeframe - Approximate timeline for this pathway to unfold
                
                Focus on creating a logical chain of causation working backward from the end state to present conditions.
                """
                
                # Ground prompt with research results
                grounded_prompt = self.ground_llm_with_context(prompt, context)
                
                # Call Llama4ScoutMCP
                llama_response = llama4_scout.process({
                    "question": f"What causal pathway could lead to {end_state_name}?",
                    "analysis_type": "causal",
                    "context": {"prompt": grounded_prompt, "research_results": research_results}
                })
                
                # Extract causal pathway from response
                if isinstance(llama_response, dict) and "sections" in llama_response:
                    content = ""
                    for section_name, section_content in llama_response["sections"].items():
                        content += section_content + "\n\n"
                    
                    # Parse causal pathway from content
                    causal_pathway = self._parse_causal_pathway_from_text(content)
                    causal_pathways[end_state_name] = causal_pathway
                else:
                    # Fallback: Generate generic causal pathway
                    causal_pathways[end_state_name] = self._generate_generic_causal_pathway(end_state)
        else:
            # Generate generic causal pathways for all end states
            for end_state in end_states:
                end_state_name = end_state.get("name", "")
                causal_pathways[end_state_name] = self._generate_generic_causal_pathway(end_state)
        
        return causal_pathways
    
    def _parse_causal_pathway_from_text(self, text: str) -> Dict:
        """
        Parse causal pathway from text.
        
        Args:
            text: Text containing causal pathway description
            
        Returns:
            Dictionary containing parsed causal pathway
        """
        # Initialize causal pathway structure
        causal_pathway = {
            "preconditions": [],
            "key_factors": [],
            "critical_events": [],
            "branching_points": [],
            "timeframe": ""
        }
        
        # Simple parsing based on patterns
        import re
        
        # Map of section names to keys
        section_map = {
            "preconditions": ["preconditions", "necessary conditions", "prerequisites", "required conditions"],
            "key_factors": ["key factors", "causal factors", "driving factors", "key drivers"],
            "critical_events": ["critical events", "key events", "significant developments", "milestone events"],
            "branching_points": ["branching points", "decision points", "contingencies", "key junctures"],
            "timeframe": ["timeframe", "timeline", "time horizon", "temporal sequence"]
        }
        
        # Look for sections
        for key, section_names in section_map.items():
            section_pattern = '|'.join(section_names)
            section_regex = rf'(?:^|\n)(?:{section_pattern}).*?(?::|$)(.*?)(?=(?:\n\n|\n(?:{"|".join([s for sublist in section_map.values() for s in sublist])})|$))'
            section_match = re.search(section_regex, text, re.IGNORECASE | re.DOTALL)
            
            if section_match:
                section_text = section_match.group(1)
                
                if key == "timeframe":
                    causal_pathway[key] = section_text.strip()
                else:
                    # Extract bullet points
                    bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                    bullet_matches = re.findall(bullet_pattern, section_text, re.DOTALL)
                    
                    if bullet_matches:
                        causal_pathway[key] = [item.strip() for item in bullet_matches]
                    else:
                        # Split by newlines or sentences
                        items = re.split(r'(?:\n|\.(?:\s+|$))', section_text)
                        causal_pathway[key] = [item.strip() for item in items if item.strip()]
        
        return causal_pathway
    
    def _generate_generic_causal_pathway(self, end_state: Dict) -> Dict:
        """
        Generate a generic causal pathway for an end state.
        
        Args:
            end_state: End state definition
            
        Returns:
            Dictionary containing generic causal pathway
        """
        end_state_name = end_state.get("name", "").lower()
        desirability = end_state.get("desirability", "mixed")
        
        # Base causal pathway
        causal_pathway = {
            "preconditions": [
                "Current trends continue without major disruption",
                "Key stakeholders maintain their current priorities and approaches",
                "No unexpected external shocks significantly alter the trajectory"
            ],
            "key_factors": [
                "Institutional capacity to address emerging challenges",
                "Technological developments and their implementation",
                "Social and political responses to changing conditions",
                "Economic incentives and constraints"
            ],
            "critical_events": [
                "Policy decisions at national and international levels",
                "Technological breakthroughs or failures",
                "Shifts in public opinion and social priorities",
                "Market developments and economic transitions"
            ],
            "branching_points": [
                "Key policy decisions that could accelerate or impede progress",
                "Technological adoption rates and directions",
                "Social acceptance or resistance to changes",
                "Responses to early warning signals or emerging trends"
            ],
            "timeframe": "This pathway would likely unfold over 5-10 years, with initial developments visible in 1-2 years and more significant changes emerging in 3-5 years."
        }
        
        # Customize based on desirability
        if desirability == "favorable":
            causal_pathway["preconditions"].append("Early successes create positive feedback loops")
            causal_pathway["key_factors"].append("Effective coordination among key stakeholders")
            causal_pathway["critical_events"].append("Successful demonstration projects that build momentum")
            causal_pathway["branching_points"].append("Early investment decisions that enable later opportunities")
        
        elif desirability == "unfavorable":
            causal_pathway["preconditions"].append("Early warning signs are ignored or downplayed")
            causal_pathway["key_factors"].append("Failure to address underlying vulnerabilities")
            causal_pathway["critical_events"].append("Trigger events that accelerate negative trends")
            causal_pathway["branching_points"].append("Missed opportunities for preventive action")
        
        # Customize based on end state name
        if "growth" in end_state_name or "expansion" in end_state_name:
            causal_pathway["key_factors"].append("Productivity-enhancing innovations")
            causal_pathway["critical_events"].append("Investment surges in key sectors")
        
        elif "stagnation" in end_state_name or "decline" in end_state_name:
            causal_pathway["key_factors"].append("Declining investment and innovation")
            causal_pathway["critical_events"].append("Failure of key reform initiatives")
        
        elif "transformation" in end_state_name or "transition" in end_state_name:
            causal_pathway["key_factors"].append("Disruptive innovations that challenge status quo")
            causal_pathway["critical_events"].append("Tipping points in adoption of new approaches")
        
        elif "crisis" in end_state_name or "disruption" in end_state_name:
            causal_pathway["key_factors"].append("Accumulating systemic vulnerabilities")
            causal_pathway["critical_events"].append("Trigger events that expose underlying weaknesses")
        
        return causal_pathway
    
    def _assess_pathway_likelihoods(self, causal_pathways: Dict, research_results: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Assess the likelihood and requirements of different pathways.
        
        Args:
            causal_pathways: Dictionary of causal pathways
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary of pathway assessments
        """
        logger.info("Assessing pathway likelihoods")
        
        pathway_assessment = {}
        
        # Use Llama4ScoutMCP to assess pathway likelihoods
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            for end_state_name, pathway in causal_pathways.items():
                # Format pathway components
                preconditions = "\n".join([f"- {p}" for p in pathway.get("preconditions", [])])
                key_factors = "\n".join([f"- {f}" for f in pathway.get("key_factors", [])])
                critical_events = "\n".join([f"- {e}" for e in pathway.get("critical_events", [])])
                
                # Create prompt for pathway assessment
                prompt = f"""
                Assess the likelihood and requirements of the following causal pathway leading to the end state: {end_state_name}
                
                Preconditions:
                {preconditions}
                
                Key Factors:
                {key_factors}
                
                Critical Events:
                {critical_events}
                
                Please provide:
                
                1. Overall likelihood assessment - How probable is this pathway (high, medium, low)
                2. Key enablers - Factors that would increase the likelihood of this pathway
                3. Key obstacles - Factors that would decrease the likelihood of this pathway
                4. Required resources - What resources would be necessary for this pathway
                5. Early indicators - What early signs would suggest this pathway is developing
                
                Base your assessment on current conditions and trends, considering both supporting and contradicting evidence.
                """
                
                # Call Llama4ScoutMCP
                llama_response = llama4_scout.process({
                    "question": f"How likely is the causal pathway leading to {end_state_name}?",
                    "analysis_type": "evaluative",
                    "context": {"prompt": prompt, "research_results": research_results}
                })
                
                # Extract assessment from response
                if isinstance(llama_response, dict) and "sections" in llama_response:
                    content = ""
                    for section_name, section_content in llama_response["sections"].items():
                        content += section_content + "\n\n"
                    
                    # Parse assessment from content
                    assessment = self._parse_assessment_from_text(content)
                    pathway_assessment[end_state_name] = assessment
                else:
                    # Fallback: Generate generic assessment
                    pathway_assessment[end_state_name] = self._generate_generic_assessment(end_state_name, pathway)
        else:
            # Generate generic assessments for all pathways
            for end_state_name, pathway in causal_pathways.items():
                pathway_assessment[end_state_name] = self._generate_generic_assessment(end_state_name, pathway)
        
        return pathway_assessment
    
    def _parse_assessment_from_text(self, text: str) -> Dict:
        """
        Parse pathway assessment from text.
        
        Args:
            text: Text containing assessment
            
        Returns:
            Dictionary containing parsed assessment
        """
        # Initialize assessment structure
        assessment = {
            "likelihood": "medium",
            "enablers": [],
            "obstacles": [],
            "required_resources": [],
            "early_indicators": []
        }
        
        # Simple parsing based on patterns
        import re
        
        # Extract likelihood
        likelihood_pattern = r'(?:likelihood|probability|how probable).*?(high|medium|low)'
        likelihood_match = re.search(likelihood_pattern, text, re.IGNORECASE)
        if likelihood_match:
            assessment["likelihood"] = likelihood_match.group(1).lower()
        
        # Map of section names to keys
        section_map = {
            "enablers": ["enablers", "key enablers", "supporting factors", "factors that would increase"],
            "obstacles": ["obstacles", "key obstacles", "hindering factors", "factors that would decrease"],
            "required_resources": ["required resources", "necessary resources", "resource requirements", "resources needed"],
            "early_indicators": ["early indicators", "early signs", "indicators", "warning signs"]
        }
        
        # Look for sections
        for key, section_names in section_map.items():
            section_pattern = '|'.join(section_names)
            section_regex = rf'(?:^|\n)(?:{section_pattern}).*?(?::|$)(.*?)(?=(?:\n\n|\n(?:{"|".join([s for sublist in section_map.values() for s in sublist])})|$))'
            section_match = re.search(section_regex, text, re.IGNORECASE | re.DOTALL)
            
            if section_match:
                section_text = section_match.group(1)
                
                # Extract bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, section_text, re.DOTALL)
                
                if bullet_matches:
                    assessment[key] = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', section_text)
                    assessment[key] = [item.strip() for item in items if item.strip()]
        
        return assessment
    
    def _generate_generic_assessment(self, end_state_name: str, pathway: Dict) -> Dict:
        """
        Generate a generic assessment for a causal pathway.
        
        Args:
            end_state_name: Name of the end state
            pathway: Causal pathway
            
        Returns:
            Dictionary containing generic assessment
        """
        end_state_lower = end_state_name.lower()
        
        # Determine generic likelihood based on end state name
        likelihood = "medium"  # Default
        
        if "positive" in end_state_lower or "favorable" in end_state_lower or "robust" in end_state_lower:
            likelihood = "medium"  # Positive outcomes are possible but not guaranteed
        elif "crisis" in end_state_lower or "disruption" in end_state_lower or "collapse" in end_state_lower:
            likelihood = "low"  # Extreme negative outcomes are less likely
        elif "transformation" in end_state_lower or "transition" in end_state_lower:
            likelihood = "medium"  # Transformative changes are plausible
        elif "muddling" in end_state_lower or "stagnation" in end_state_lower or "incremental" in end_state_lower:
            likelihood = "high"  # Continuation of mixed trends is often most likely
        
        # Base assessment
        assessment = {
            "likelihood": likelihood,
            "enablers": [
                "Strong leadership and institutional capacity",
                "Alignment of incentives among key stakeholders",
                "Technological developments that support the pathway",
                "Public support and social acceptance"
            ],
            "obstacles": [
                "Institutional inertia and resistance to change",
                "Misaligned incentives among key stakeholders",
                "Resource constraints and competing priorities",
                "Unexpected external shocks or developments"
            ],
            "required_resources": [
                "Financial resources for key investments",
                "Human capital and expertise in relevant domains",
                "Institutional capacity for implementation",
                "Political capital and stakeholder support"
            ],
            "early_indicators": [
                "Policy developments and regulatory changes",
                "Investment patterns and resource allocation",
                "Technological developments and adoption rates",
                "Shifts in public discourse and stakeholder positions"
            ]
        }
        
        # Customize based on pathway components
        key_factors = pathway.get("key_factors", [])
        for factor in key_factors:
            factor_lower = factor.lower()
            
            # Add factor-specific enablers and obstacles
            if "technology" in factor_lower or "innovation" in factor_lower:
                assessment["enablers"].append("Accelerated technological innovation and adoption")
                assessment["obstacles"].append("Technical challenges or slower than expected innovation")
                assessment["early_indicators"].append("R&D breakthroughs and early-stage technology demonstrations")
            
            elif "policy" in factor_lower or "regulation" in factor_lower:
                assessment["enablers"].append("Political consensus on policy direction")
                assessment["obstacles"].append("Political gridlock or policy reversals")
                assessment["early_indicators"].append("Early policy proposals and stakeholder positioning")
            
            elif "economic" in factor_lower or "market" in factor_lower:
                assessment["enablers"].append("Favorable economic conditions and market signals")
                assessment["obstacles"].append("Economic downturns or adverse market conditions")
                assessment["early_indicators"].append("Market trends and economic leading indicators")
            
            elif "social" in factor_lower or "public" in factor_lower:
                assessment["enablers"].append("Strong public support and social movements")
                assessment["obstacles"].append("Public resistance or competing social priorities")
                assessment["early_indicators"].append("Shifts in public opinion and social discourse")
        
        return assessment
    
    def _identify_key_indicators(self, causal_pathways: Dict, pathway_assessment: Dict, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Identify key indicators and decision points.
        
        Args:
            causal_pathways: Dictionary of causal pathways
            pathway_assessment: Dictionary of pathway assessments
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of key indicators
        """
        logger.info("Identifying key indicators")
        
        # Use Llama4ScoutMCP to identify key indicators
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Compile early indicators from all pathways
            all_indicators = []
            for end_state_name, assessment in pathway_assessment.items():
                early_indicators = assessment.get("early_indicators", [])
                for indicator in early_indicators:
                    all_indicators.append({
                        "indicator": indicator,
                        "related_pathway": end_state_name
                    })
            
            # Compile branching points from all pathways
            all_branching_points = []
            for end_state_name, pathway in causal_pathways.items():
                branching_points = pathway.get("branching_points", [])
                for point in branching_points:
                    all_branching_points.append({
                        "decision_point": point,
                        "related_pathway": end_state_name
                    })
            
            # Create prompt for key indicators synthesis
            indicators_text = "\n".join([f"- {i['indicator']} (Pathway: {i['related_pathway']})" for i in all_indicators])
            branching_points_text = "\n".join([f"- {p['decision_point']} (Pathway: {p['related_pathway']})" for p in all_branching_points])
            
            prompt = f"""
            Based on the following early indicators and decision points from different pathways, identify and prioritize the most important indicators to monitor.
            
            Early Indicators:
            {indicators_text}
            
            Decision Points:
            {branching_points_text}
            
            For each key indicator you identify:
            1. Provide a clear name and description
            2. Explain what pathway(s) it relates to
            3. Describe how to monitor or measure it
            4. Explain its significance and what changes would indicate
            
            Focus on indicators that are:
            - Observable in the near term
            - Clearly linked to important pathways
            - Practical to monitor
            - Provide early warning of significant developments
            
            Prioritize 5-8 of the most important indicators.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "What are the key indicators to monitor for these pathways?",
                "analysis_type": "indicators",
                "context": {"prompt": prompt, "research_results": research_results}
            })
            
            # Extract key indicators from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse key indicators from content
                key_indicators = self._parse_key_indicators_from_text(content)
                if key_indicators:
                    return key_indicators
        
        # Fallback: Generate key indicators from pathway assessments
        return self._generate_key_indicators_from_assessments(causal_pathways, pathway_assessment)
    
    def _parse_key_indicators_from_text(self, text: str) -> List[Dict]:
        """
        Parse key indicators from text.
        
        Args:
            text: Text containing key indicator descriptions
            
        Returns:
            List of parsed key indicators
        """
        key_indicators = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for indicator sections
        indicator_pattern = r'(?:^|\n)(?:Indicator|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Indicator|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        indicator_matches = re.findall(indicator_pattern, text, re.DOTALL)
        
        if not indicator_matches:
            # Try alternative pattern for numbered lists
            indicator_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            indicator_matches = re.findall(indicator_pattern, text, re.DOTALL)
            
            if indicator_matches:
                # Convert to expected format
                indicator_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(indicator_matches)]
        
        for match in indicator_matches:
            indicator_num = match[0].strip() if len(match) > 0 else ""
            indicator_name = match[1].strip() if len(match) > 1 else ""
            indicator_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract related pathways
            related_pathways = []
            pathways_pattern = r'(?:related|relates to|pathway|pathways).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            pathways_match = re.search(pathways_pattern, indicator_content, re.IGNORECASE | re.DOTALL)
            if pathways_match:
                pathways_text = pathways_match.group(1).strip()
                # Split by commas or "and"
                related_pathways = [p.strip() for p in re.split(r',|\sand\s', pathways_text) if p.strip()]
            
            # Extract monitoring method
            monitoring = ""
            monitoring_pattern = r'(?:monitor|measure|tracking|assessment).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            monitoring_match = re.search(monitoring_pattern, indicator_content, re.IGNORECASE | re.DOTALL)
            if monitoring_match:
                monitoring = monitoring_match.group(1).strip()
            
            # Extract significance
            significance = ""
            significance_pattern = r'(?:significance|importance|indicates|signal).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            significance_match = re.search(significance_pattern, indicator_content, re.IGNORECASE | re.DOTALL)
            if significance_match:
                significance = significance_match.group(1).strip()
            
            key_indicators.append({
                "name": indicator_name,
                "related_pathways": related_pathways,
                "monitoring_method": monitoring,
                "significance": significance
            })
        
        return key_indicators
    
    def _generate_key_indicators_from_assessments(self, causal_pathways: Dict, pathway_assessment: Dict) -> List[Dict]:
        """
        Generate key indicators from pathway assessments.
        
        Args:
            causal_pathways: Dictionary of causal pathways
            pathway_assessment: Dictionary of pathway assessments
            
        Returns:
            List of generated key indicators
        """
        key_indicators = []
        
        # Collect all early indicators from assessments
        all_indicators = {}
        for end_state_name, assessment in pathway_assessment.items():
            early_indicators = assessment.get("early_indicators", [])
            for indicator in early_indicators:
                if indicator in all_indicators:
                    all_indicators[indicator].append(end_state_name)
                else:
                    all_indicators[indicator] = [end_state_name]
        
        # Prioritize indicators that appear in multiple pathways
        prioritized_indicators = sorted(all_indicators.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Generate key indicators from prioritized list
        for i, (indicator, pathways) in enumerate(prioritized_indicators[:6]):  # Limit to top 6
            # Generate monitoring method based on indicator content
            monitoring_method = ""
            if "policy" in indicator.lower() or "regulation" in indicator.lower():
                monitoring_method = "Track policy announcements, regulatory changes, and legislative developments"
            elif "investment" in indicator.lower() or "funding" in indicator.lower():
                monitoring_method = "Monitor investment flows, funding announcements, and capital allocation patterns"
            elif "technology" in indicator.lower() or "innovation" in indicator.lower():
                monitoring_method = "Track technology announcements, patent filings, and R&D developments"
            elif "public" in indicator.lower() or "opinion" in indicator.lower():
                monitoring_method = "Monitor public opinion surveys, media coverage, and social media sentiment"
            elif "market" in indicator.lower() or "economic" in indicator.lower():
                monitoring_method = "Track market indicators, economic data, and industry reports"
            else:
                monitoring_method = "Establish a systematic monitoring process with regular data collection and analysis"
            
            # Generate significance based on related pathways
            significance = f"Changes in this indicator could signal development toward {', '.join(pathways[:2])} pathways"
            if len(pathways) > 2:
                significance += f" and {len(pathways) - 2} other pathways"
            
            key_indicators.append({
                "name": indicator,
                "related_pathways": pathways,
                "monitoring_method": monitoring_method,
                "significance": significance
            })
        
        return key_indicators
    
    def _extract_findings(self, end_states: List[Dict], causal_pathways: Dict, pathway_assessment: Dict, key_indicators: List[Dict]) -> List[Dict]:
        """
        Extract key findings from backward reasoning analysis.
        
        Args:
            end_states: List of end states
            causal_pathways: Dictionary of causal pathways
            pathway_assessment: Dictionary of pathway assessments
            key_indicators: List of key indicators
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about end states
        findings.append({
            "finding": f"Analysis identified {len(end_states)} distinct potential end states with different implications",
            "confidence": "high",
            "source": "backward_reasoning"
        })
        
        # Add finding about most likely pathway
        most_likely_pathway = None
        most_likely_end_state = None
        highest_likelihood = "low"
        
        likelihood_rank = {"high": 3, "medium": 2, "low": 1}
        
        for end_state_name, assessment in pathway_assessment.items():
            likelihood = assessment.get("likelihood", "medium")
            if likelihood_rank.get(likelihood, 0) > likelihood_rank.get(highest_likelihood, 0):
                highest_likelihood = likelihood
                most_likely_end_state = end_state_name
                most_likely_pathway = causal_pathways.get(end_state_name, {})
        
        if most_likely_end_state and most_likely_pathway:
            findings.append({
                "finding": f"The pathway leading to '{most_likely_end_state}' appears most likely based on current conditions",
                "confidence": "medium",
                "source": "backward_reasoning"
            })
        
        # Add finding about key enablers
        if most_likely_end_state:
            assessment = pathway_assessment.get(most_likely_end_state, {})
            enablers = assessment.get("enablers", [])
            if enablers:
                findings.append({
                    "finding": f"Key enabler for the most likely pathway: {enablers[0]}",
                    "confidence": "medium",
                    "source": "backward_reasoning"
                })
        
        # Add finding about key indicators
        if key_indicators:
            findings.append({
                "finding": f"Critical indicator to monitor: {key_indicators[0].get('name')}",
                "confidence": "medium",
                "source": "backward_reasoning"
            })
        
        return findings
    
    def _extract_assumptions(self, causal_pathways: Dict) -> List[Dict]:
        """
        Extract assumptions from backward reasoning analysis.
        
        Args:
            causal_pathways: Dictionary of causal pathways
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about causal relationships
        assumptions.append({
            "assumption": "The identified causal relationships will operate as expected without unforeseen interactions",
            "criticality": "high",
            "source": "backward_reasoning"
        })
        
        # Add assumption about preconditions
        all_preconditions = []
        for end_state_name, pathway in causal_pathways.items():
            preconditions = pathway.get("preconditions", [])
            all_preconditions.extend(preconditions)
        
        if all_preconditions:
            assumptions.append({
                "assumption": f"Key precondition: {all_preconditions[0]}",
                "criticality": "high",
                "source": "backward_reasoning"
            })
        
        # Add assumption about timeframe
        assumptions.append({
            "assumption": "The timeframes identified for these pathways are reasonable estimates",
            "criticality": "medium",
            "source": "backward_reasoning"
        })
        
        return assumptions
    
    def _extract_uncertainties(self, causal_pathways: Dict, pathway_assessment: Dict) -> List[Dict]:
        """
        Extract uncertainties from backward reasoning analysis.
        
        Args:
            causal_pathways: Dictionary of causal pathways
            pathway_assessment: Dictionary of pathway assessments
            
        Returns:
            List of uncertainties
        """
        uncertainties = []
        
        # Add uncertainty about pathway likelihood
        uncertainties.append({
            "uncertainty": "The relative likelihoods of different pathways may change as conditions evolve",
            "impact": "high",
            "source": "backward_reasoning"
        })
        
        # Add uncertainties from obstacles
        all_obstacles = []
        for end_state_name, assessment in pathway_assessment.items():
            obstacles = assessment.get("obstacles", [])
            all_obstacles.extend(obstacles)
        
        # Select top uncertainties from obstacles
        seen = set()
        for obstacle in all_obstacles:
            obstacle_key = obstacle.lower()
            if obstacle_key not in seen and len(uncertainties) < 4:
                seen.add(obstacle_key)
                uncertainties.append({
                    "uncertainty": f"Uncertainty regarding obstacle: {obstacle}",
                    "impact": "medium",
                    "source": "backward_reasoning"
                })
        
        # Add uncertainty about external factors
        uncertainties.append({
            "uncertainty": "External factors not included in the analysis could significantly alter these pathways",
            "impact": "high",
            "source": "backward_reasoning"
        })
        
        return uncertainties
