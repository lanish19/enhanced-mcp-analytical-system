"""
Analysis of Competing Hypotheses (ACH) Technique for structured evaluation of alternative explanations.
This module provides the ACHTechnique class for systematic hypothesis evaluation.
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

class ACHTechnique(AnalyticalTechnique):
    """
    Analysis of Competing Hypotheses (ACH) Technique for structured evaluation of alternative explanations.
    
    This technique provides capabilities for:
    1. Generating multiple competing hypotheses
    2. Identifying key evidence and diagnostic value
    3. Systematically evaluating hypotheses against evidence
    4. Assessing consistency, inconsistency, and relevance
    5. Determining most likely explanations based on evidence
    """
    
    def __init__(self):
        """Initialize the Analysis of Competing Hypotheses Technique."""
        
        super().__init__(
            name="analysis_of_competing_hypotheses",
            description="Systematically evaluates multiple competing hypotheses against evidence",
            required_mcps=["llama4_scout", "research_mcp", "perplexity_sonar", "economics_mcp", "geopolitics_mcp"],
            compatible_techniques=["key_assumptions_check", "multi_persona", "red_teaming"],
            incompatible_techniques=[]
        )


        # Ensure all required MCPs are initialized and available
        required_mcps = ["llama4_scout", "research_mcp", "perplexity_sonar", "economics_mcp", "geopolitics_mcp"]
        for mcp_name in required_mcps:
            if mcp_name not in self.mcp_registry:
                raise ValueError(f"{mcp_name} must be initialized and available in the mcp_registry")

        # Ensure research_mcp is initialized and available
        # if "research_mcp" not in self.mcp_registry:
        #     raise ValueError("research_mcp must be initialized and available in the mcp_registry")
        #
        # if "llama4_scout" not in self.mcp_registry:
        #     raise ValueError("llama4_scout must be initialized and available in the mcp_registry")
        # if "perplexity_sonar" not in self.mcp_registry:
        #     raise ValueError("perplexity_sonar must be initialized and available in the mcp_registry")
        # if "economics_mcp" not in self.mcp_registry:
        #     raise ValueError("economics_mcp must be initialized and available in the mcp_registry")
        # if "geopolitics_mcp" not in self.mcp_registry:
        #     raise ValueError("geopolitics_mcp must be initialized and available in the mcp_registry")

        logger.info("Initialized ACHTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Analysis of Competing Hypotheses Technique")
        
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
            
            # Fetch economic and geopolitical data
            relevant_economic_data = self._fetch_relevant_economic_data(question, context)

            # Gather evidence, including research findings
            self._gather_evidence(question, context)
            relevant_geopolitical_data = self._fetch_relevant_geopolitical_data(question, context)

            
            # Generate hypotheses
            hypotheses = self._generate_hypotheses(question, research_results, context, parameters)
            
            # Identify evidence
            evidence = self._identify_evidence(question, hypotheses, research_results, context, parameters)
            
            # Evaluate hypotheses against evidence
            evaluation_matrix = self._evaluate_hypotheses(hypotheses, evidence, research_results, relevant_economic_data, relevant_geopolitical_data, context, parameters)
            
            # Analyze results
            analysis_results = self._analyze_results(hypotheses, evidence, evaluation_matrix, context, parameters)
            
            # Compile results
            results = {
                "technique": "analysis_of_competing_hypotheses",
                "timestamp": time.time(),
                "question": question,
                "hypotheses": hypotheses,
                "evidence": evidence,
                "evaluation_matrix": evaluation_matrix,
                "analysis_results": analysis_results,
                "findings": self._extract_findings(hypotheses, evidence, analysis_results),
                "assumptions": self._extract_assumptions(hypotheses, evidence, analysis_results),
                "uncertainties": self._extract_uncertainties(hypotheses, evidence, analysis_results)
            }
            
            # Add results to context
            context.add_technique_result("analysis_of_competing_hypotheses", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Analysis of Competing Hypotheses Technique: {e}")
            return self.handle_error(e, context)
    
    def _gather_evidence(self, question: str, context: AnalysisContext) -> None:
        """
        Gather evidence, including research findings, using the research_mcp.

        Args:
            question: The analytical question.
            context: Analysis context.
        """
        logger.info("Gathering evidence using research_mcp")

        # Get the research_mcp from the mcp_registry
        research_mcp = self.get_mcp("research_mcp")

        # Check if research_mcp is available
        if research_mcp:
            # Fetch research findings
            research_results = research_mcp.get_research(query=question)
            # Add research findings to the context
            context.add("research_results", research_results)
            logger.info(f"Added research results to context: {research_results}")
        else:
            logger.warning("research_mcp not available for gathering evidence.")
    
    def _fetch_relevant_economic_data(self, question: str, context: AnalysisContext) -> List[Dict]:
        """
        Fetch relevant economic data from EconomicsMCP.
        
        Args:
            question: The analytical question
            context: Analysis context
        
        Returns:
            List of economic data items
        """
        logger.info("Fetching relevant economic data")
        
        economics_mcp = self.get_mcp("economics_mcp")
        if not economics_mcp:
            logger.warning("EconomicsMCP not available for data fetching.")
            return []
        
        # Determine relevant economic indicators based on the question
        relevant_indicators = []
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["gdp", "growth", "economy"]):
            relevant_indicators.append("GDP")
        if any(keyword in question_lower for keyword in ["inflation", "cpi", "prices"]):
            relevant_indicators.append("CPI")
        if any(keyword in question_lower for keyword in ["unemployment", "jobs", "labor"]):
            relevant_indicators.append("UNEMPLOYMENT")
        # Add more indicators based on common keywords...
        
        if not relevant_indicators:
            logger.info("No specific economic indicators identified, fetching default data.")
            relevant_indicators = ["GDP", "CPI"]  # Default indicators
        
        economic_data = []
        for indicator in relevant_indicators:
            try:
                # Fetch data from EconomicsMCP
                data = economics_mcp.get_data(series_id=indicator, start_date="2020-01-01", end_date="2023-12-31")
                if data:
                    economic_data.extend(data)
            except Exception as e:
                logger.error(f"Error fetching economic data for {indicator}: {e}")
        
        return economic_data

    def _fetch_relevant_geopolitical_data(self, question: str, context: AnalysisContext) -> List[Dict]:
        """
        Fetch relevant geopolitical event data from GeopoliticsMCP.
        
        Args:
            question: The analytical question
            context: Analysis context
            
        Returns:
            List of geopolitical data items
        """
        logger.info("Fetching relevant geopolitical data")
        
        geopolitics_mcp = self.get_mcp("geopolitics_mcp")
        if not geopolitics_mcp:
            logger.warning("GeopoliticsMCP not available for data fetching.")
            return []
        
        # Determine relevant location based on the question
        relevant_location = "Global"  # Default to global if no location specified
        question_lower = question.lower()
        
        if "ukraine" in question_lower:
            relevant_location = "Ukraine"
        elif "middle east" in question_lower:
            relevant_location = "Middle East"
        
        try:
            # Fetch data from GeopoliticsMCP
            data = geopolitics_mcp.get_event_data(location=relevant_location, start_date="2023-01-01", end_date="2023-12-31")
            return data if data else []
        except Exception as e:
            logger.error(f"Error fetching geopolitical data for {relevant_location}: {e}")
            return []
        
    
    def _generate_hypotheses(self, question: str, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Generate competing hypotheses.
        
        Args:
            question: The analytical question
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of hypotheses
        """
        logger.info("Generating hypotheses")
        
        # Check if hypotheses are provided in parameters
        # Get research results from context
        research_results = context.get("research_results", {})
        research_results_content = ""
        for result in research_results:
            research_results_content+= f"{result}\n"
        if "hypotheses" in parameters and isinstance(parameters["hypotheses"], list):
            return parameters["hypotheses"]
        
        # Check if hypotheses exist in context from other techniques
        hypothesis_results = context.get_technique_result("research_to_hypothesis")
        if hypothesis_results and "hypotheses" in hypothesis_results:
            return hypothesis_results.get("hypotheses", [])
        
        # Use PerplexitySonarMCP for initial research-based hypotheses
        perplexity_sonar = self.get_mcp("perplexity_sonar")
        if perplexity_sonar:
            # Create prompt for hypothesis generation
            prompt = f"""
            Based on the following question, generate 4-6 competing hypotheses that could explain or answer it.
            
            Question: {question}
            
            For each hypothesis:
            1. Provide a clear statement of the hypothesis
            2. Explain the key reasoning behind this hypothesis
            3. Identify the main assumptions underlying this hypothesis
            
            Ensure the hypotheses:
            - Are mutually exclusive where possible
            - Cover the range of plausible explanations
            - Include both conventional and less conventional possibilities
            - Are specific and testable against evidence
            """
            
            # Call PerplexitySonarMCP
            sonar_response = perplexity_sonar.process({
                "question": f"What are the competing hypotheses for: {question}?",
                "search_type": "deep_research",
                "context": {"prompt": prompt}
            })
            
            # Extract hypotheses from response
            if isinstance(sonar_response, dict) and "research_results" in sonar_response:
                research_content = sonar_response["research_results"].get("content", "")
                
                # Parse hypotheses from research content
                hypotheses = self._parse_hypotheses_from_text(research_content)
                if hypotheses:
                    return hypotheses
        
        # Fallback to Llama4ScoutMCP
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Create prompt for hypothesis generation
            prompt = f"""
            Based on the following question, generate 4-6 competing hypotheses that could explain or answer it.
            
            Question: {question}
            
            For each hypothesis:
            1. Provide a clear statement of the hypothesis
            2. Explain the key reasoning behind this hypothesis
            3. Identify the main assumptions underlying this hypothesis
            
            Ensure the hypotheses:
            - Are mutually exclusive where possible
            - Cover the range of plausible explanations
            - Include both conventional and less conventional possibilities
            - Are specific and testable against evidence
            """

            prompt += f"""
            Research Findings:
            {research_results_content}

            Consider the above research findings when generating hypotheses.
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "hypothesis_generation",
                "context": {"prompt": grounded_prompt, "research_results": research_results}
            })
            
            # Extract hypotheses from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"

                # Check for relevance of each hypothesis to the research findings
                relevant_hypotheses = []
                for hypothesis in hypotheses:
                    perplexity_sonar = self.get_mcp("perplexity_sonar")
                    if perplexity_sonar:
                        relevance_prompt = f"""
                        Assess the relevance of the following hypothesis to the research findings provided below.

                        Hypothesis: {hypothesis["statement"]}

                        Research Findings:
                        {research_results_content}

                        Is this hypothesis relevant to the research findings? Answer with one of the following:
                        - "relevant"
                        - "related"
                        - "supports"
                        - "not relevant"
                        """
                        relevance_response = perplexity_sonar.process({"prompt": relevance_prompt})
                        if isinstance(relevance_response, dict) and "results" in relevance_response:
                            relevance_text = relevance_response["results"].get("text", "").lower()
                            if any(keyword in relevance_text for keyword in ["relevant", "related", "supports"]):
                                relevant_hypotheses.append(hypothesis)
                            else:
                                logger.info(f"Hypothesis '{hypothesis['statement']}' deemed not relevant to research findings.")

                # Filter hypotheses based on relevance
                if relevant_hypotheses:
                    hypotheses = relevant_hypotheses

                # Parse hypotheses from content
                hypotheses = self._parse_hypotheses_from_text(content)
                if hypotheses:
                    return hypotheses
        
        # Fallback: Generate generic hypotheses
        return self._generate_generic_hypotheses(question)
    
    def _parse_hypotheses_from_text(self, text: str) -> List[Dict]:
        """
        Parse hypotheses from text.
        
        Args:
            text: Text containing hypothesis descriptions
            
        Returns:
            List of parsed hypotheses
        """
        hypotheses = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for hypothesis sections
        hypothesis_pattern = r'(?:^|\n)(?:Hypothesis|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Hypothesis|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        hypothesis_matches = re.findall(hypothesis_pattern, text, re.DOTALL)
        
        if not hypothesis_matches:
            # Try alternative pattern for numbered lists
            hypothesis_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            hypothesis_matches = re.findall(hypothesis_pattern, text, re.DOTALL)
            
            if hypothesis_matches:
                # Convert to expected format
                hypothesis_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(hypothesis_matches)]
        
        for match in hypothesis_matches:
            hypothesis_num = match[0].strip() if len(match) > 0 else ""
            hypothesis_statement = match[1].strip() if len(match) > 1 else ""
            hypothesis_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract reasoning
            reasoning = ""
            reasoning_pattern = r'(?:reasoning|rationale|explanation).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            reasoning_match = re.search(reasoning_pattern, hypothesis_content, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                # Use first paragraph as reasoning
                paragraphs = hypothesis_content.split('\n\n')
                if paragraphs:
                    reasoning = paragraphs[0].strip()
            
            # Extract assumptions
            assumptions = []
            assumptions_pattern = r'(?:assumptions|assumes|assumption).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            assumptions_match = re.search(assumptions_pattern, hypothesis_content, re.IGNORECASE | re.DOTALL)
            if assumptions_match:
                assumptions_text = assumptions_match.group(1).strip()
                # Look for bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, assumptions_text, re.DOTALL)
                if bullet_matches:
                    assumptions = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', assumptions_text)
                    assumptions = [item.strip() for item in items if item.strip()]
            
            hypotheses.append({
                "id": f"H{hypothesis_num}",
                "statement": hypothesis_statement,
                "reasoning": reasoning,
                "assumptions": assumptions
            })
        
        return hypotheses
    
    def _generate_generic_hypotheses(self, question: str) -> List[Dict]:
        """
        Generate generic hypotheses based on the question.
        
        Args:
            question: The analytical question
            
        Returns:
            List of generic hypotheses
        """
        # Extract domain and type from question
        domain = self._extract_domain_from_question(question)
        question_type = self._extract_question_type(question)
        
        # Domain-specific hypothesis templates
        domain_hypotheses = {
            "economic": [
                {
                    "id": "H1",
                    "statement": "Market-driven factors are the primary cause",
                    "reasoning": "Economic outcomes are primarily determined by market forces including supply, demand, and price mechanisms operating with minimal distortion.",
                    "assumptions": [
                        "Markets are functioning efficiently",
                        "Economic actors are primarily rational",
                        "Information asymmetries are limited"
                    ]
                },
                {
                    "id": "H2",
                    "statement": "Policy and regulatory factors are the primary cause",
                    "reasoning": "Government policies, regulations, and interventions are the dominant factors shaping economic outcomes through incentives, constraints, and direct actions.",
                    "assumptions": [
                        "Government has significant influence over economic outcomes",
                        "Policy implementation is effective",
                        "Economic actors respond predictably to policy signals"
                    ]
                },
                {
                    "id": "H3",
                    "statement": "Structural and institutional factors are the primary cause",
                    "reasoning": "Long-term structural factors and institutional arrangements fundamentally determine economic outcomes by setting the rules of the game and shaping incentives.",
                    "assumptions": [
                        "Institutional quality significantly impacts economic performance",
                        "Path dependency is important in economic development",
                        "Formal and informal institutions shape economic behavior"
                    ]
                },
                {
                    "id": "H4",
                    "statement": "External and global factors are the primary cause",
                    "reasoning": "International forces, global trends, and external shocks are the main drivers of economic outcomes through trade, capital flows, and spillover effects.",
                    "assumptions": [
                        "The economy is highly integrated with global markets",
                        "External developments significantly impact domestic conditions",
                        "Global economic forces outweigh domestic factors"
                    ]
                }
            ],
            "political": [
                {
                    "id": "H1",
                    "statement": "Interest-based factors are the primary cause",
                    "reasoning": "Political outcomes are primarily determined by the competition between organized interest groups pursuing their rational self-interest.",
                    "assumptions": [
                        "Political actors are primarily motivated by self-interest",
                        "Interest groups have significant influence on policy",
                        "Political processes aggregate competing interests"
                    ]
                },
                {
                    "id": "H2",
                    "statement": "Institutional and structural factors are the primary cause",
                    "reasoning": "Political outcomes are shaped by institutional arrangements, constitutional structures, and procedural rules that constrain and channel political behavior.",
                    "assumptions": [
                        "Institutions significantly constrain political actors",
                        "Institutional design shapes political outcomes",
                        "Path dependency is important in political development"
                    ]
                },
                {
                    "id": "H3",
                    "statement": "Ideational and cultural factors are the primary cause",
                    "reasoning": "Political outcomes are driven by ideas, values, norms, and cultural factors that shape preferences, identities, and political discourse.",
                    "assumptions": [
                        "Ideas and values significantly influence political behavior",
                        "Cultural factors shape political preferences",
                        "Ideational change can drive political change"
                    ]
                },
                {
                    "id": "H4",
                    "statement": "Leadership and agency factors are the primary cause",
                    "reasoning": "Political outcomes are significantly influenced by the decisions, strategies, and characteristics of key political leaders and their ability to mobilize support.",
                    "assumptions": [
                        "Individual leaders can significantly impact political outcomes",
                        "Leadership qualities matter for political effectiveness",
                        "Political agency can overcome structural constraints"
                    ]
                }
            ],
            "technological": [
                {
                    "id": "H1",
                    "statement": "Market-driven innovation is the primary factor",
                    "reasoning": "Technological developments are primarily driven by market incentives, commercial applications, and profit-seeking behavior by firms and entrepreneurs.",
                    "assumptions": [
                        "Market incentives effectively drive innovation",
                        "Commercial viability determines technological trajectories",
                        "Private sector is the primary source of innovation"
                    ]
                },
                {
                    "id": "H2",
                    "statement": "Public investment and policy are the primary factors",
                    "reasoning": "Technological developments are significantly shaped by government funding, public research, and policy frameworks that enable and direct innovation.",
                    "assumptions": [
                        "Public investment is crucial for fundamental research",
                        "Government policy significantly shapes innovation",
                        "Public-private partnerships are important for technology development"
                    ]
                },
                {
                    "id": "H3",
                    "statement": "Scientific advancement and knowledge factors are primary",
                    "reasoning": "Technological developments follow from scientific discoveries and the expansion of fundamental knowledge, with applications emerging from basic research.",
                    "assumptions": [
                        "Scientific progress drives technological innovation",
                        "Basic research leads to applied developments",
                        "Knowledge accumulation is the foundation of technological change"
                    ]
                },
                {
                    "id": "H4",
                    "statement": "Social and institutional factors are primary",
                    "reasoning": "Technological developments are shaped by social needs, cultural values, and institutional arrangements that influence which technologies are developed and adopted.",
                    "assumptions": [
                        "Social context significantly shapes technological trajectories",
                        "Institutional factors influence technology adoption",
                        "Cultural values affect technological priorities"
                    ]
                }
            ],
            "social": [
                {
                    "id": "H1",
                    "statement": "Economic and material factors are primary",
                    "reasoning": "Social outcomes are primarily determined by economic conditions, material resources, and class relations that shape opportunities and constraints.",
                    "assumptions": [
                        "Economic factors significantly influence social outcomes",
                        "Material conditions shape social behavior",
                        "Resource distribution affects social relations"
                    ]
                },
                {
                    "id": "H2",
                    "statement": "Cultural and ideational factors are primary",
                    "reasoning": "Social outcomes are shaped by cultural values, norms, beliefs, and ideas that influence behavior, preferences, and social organization.",
                    "assumptions": [
                        "Cultural factors significantly shape social behavior",
                        "Values and norms influence social outcomes",
                        "Ideational change drives social change"
                    ]
                },
                {
                    "id": "H3",
                    "statement": "Institutional and policy factors are primary",
                    "reasoning": "Social outcomes are determined by institutional arrangements, policies, and formal rules that structure incentives and opportunities.",
                    "assumptions": [
                        "Institutions significantly shape social outcomes",
                        "Policy design affects social behavior",
                        "Formal rules influence social organization"
                    ]
                },
                {
                    "id": "H4",
                    "statement": "Network and relational factors are primary",
                    "reasoning": "Social outcomes emerge from patterns of relationships, social networks, and interactions between individuals and groups.",
                    "assumptions": [
                        "Social networks significantly influence outcomes",
                        "Relational patterns shape social behavior",
                        "Emergent properties arise from social interactions"
                    ]
                }
            ],
            "general": [
                {
                    "id": "H1",
                    "statement": "Conventional explanation with established factors",
                    "reasoning": "The outcome can be explained by well-established causal factors that are widely recognized in this domain, following expected patterns.",
                    "assumptions": [
                        "Conventional wisdom in this area is generally correct",
                        "Established explanatory factors are sufficient",
                        "The situation follows typical patterns"
                    ]
                },
                {
                    "id": "H2",
                    "statement": "Alternative explanation with overlooked factors",
                    "reasoning": "The outcome is better explained by factors that are often overlooked or underestimated, revealing limitations in conventional explanations.",
                    "assumptions": [
                        "Conventional explanations miss important factors",
                        "Overlooked variables have significant explanatory power",
                        "Alternative perspectives offer valuable insights"
                    ]
                },
                {
                    "id": "H3",
                    "statement": "Complex interaction of multiple factors",
                    "reasoning": "The outcome results from the complex interaction of multiple factors that cannot be reduced to a single primary cause or simple explanation.",
                    "assumptions": [
                        "Multiple factors interact in complex ways",
                        "Simple mono-causal explanations are insufficient",
                        "Emergent properties arise from factor interactions"
                    ]
                },
                {
                    "id": "H4",
                    "statement": "Contingent and context-specific factors",
                    "reasoning": "The outcome is highly contingent on specific contextual factors and historical circumstances that make generalization difficult.",
                    "assumptions": [
                        "Context-specific factors are decisive",
                        "Historical contingency plays a major role",
                        "Generalized explanations have limited applicability"
                    ]
                }
            ]
        }
        
        # Question type modifications
        if question_type == "predictive":
            for i, hypothesis in enumerate(domain_hypotheses.get(domain, domain_hypotheses["general"])):
                hypothesis["statement"] = hypothesis["statement"].replace("are the primary cause", "will be the primary driver")
                hypothesis["reasoning"] = hypothesis["reasoning"].replace("are primarily determined by", "will be primarily determined by")
                hypothesis["reasoning"] = hypothesis["reasoning"].replace("are shaped by", "will be shaped by")
                hypothesis["reasoning"] = hypothesis["reasoning"].replace("are driven by", "will be driven by")
                hypothesis["reasoning"] = hypothesis["reasoning"].replace("are significantly influenced by", "will be significantly influenced by")
        
        # Return hypotheses for the identified domain
        if domain in domain_hypotheses:
            return domain_hypotheses[domain]
        else:
            return domain_hypotheses["general"]
    
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
    
    def _identify_evidence(self, question: str, hypotheses: List[Dict], research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Identify relevant evidence for evaluating hypotheses.
        
        Args:
            question: The analytical question
            hypotheses: List of hypotheses
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of evidence items
        """
        logger.info("Identifying evidence")
        
        # Check if evidence is provided in parameters
        if "evidence" in parameters and isinstance(parameters["evidence"], list):
            return parameters["evidence"]
        
        # Use ResearchMCP to gather evidence
        research_mcp = self.get_mcp("research_mcp")
        if research_mcp:
            # Create evidence gathering queries based on hypotheses
            evidence_queries = []
            for hypothesis in hypotheses:
                hypothesis_statement = hypothesis.get("statement", "")
                evidence_queries.append(f"Evidence supporting or contradicting: {hypothesis_statement}")
            
            # Call ResearchMCP
            research_response = research_mcp.process({
                "question": question,
                "search_queries": evidence_queries,
                "search_depth": "medium"
            })
            
            # Extract evidence from research response
            if isinstance(research_response, dict) and "search_results" in research_response:
                search_results = research_response.get("search_results", [])
                
                # Process search results into evidence items
                evidence = []
                for i, result in enumerate(search_results):
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        source = result.get("source", "")
                        
                        # Extract evidence items from content
                        evidence_items = self._extract_evidence_from_text(content, source)
                        evidence.extend(evidence_items)
                
                # Deduplicate evidence
                unique_evidence = []
                seen_descriptions = set()
                for item in evidence:
                    description = item.get("description", "")
                    if description and description not in seen_descriptions:
                        seen_descriptions.add(description)
                        unique_evidence.append(item)
                
                if unique_evidence:
                    # Assign diagnostic value to evidence
                    return self._assign_diagnostic_value(unique_evidence, hypotheses)
        
        # Fallback to Llama4ScoutMCP
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Format hypotheses for prompt
            hypotheses_text = ""
            for hypothesis in hypotheses:
                hypothesis_id = hypothesis.get("id", "")
                hypothesis_statement = hypothesis.get("statement", "")
                hypotheses_text += f"{hypothesis_id}: {hypothesis_statement}\n"
            
            # Create prompt for evidence identification
            prompt = f"""
            Based on the following question and competing hypotheses, identify 8-12 key pieces of evidence that would help evaluate these hypotheses.
            
            Question: {question}
            
            Competing Hypotheses:
            {hypotheses_text}
            
            For each evidence item:
            1. Provide a clear description of the evidence
            2. Identify the source or basis for this evidence
            3. Explain why this evidence is relevant to evaluating the hypotheses
            
            Focus on evidence that is:
            - Specific and factual rather than interpretive
            - Relevant to multiple hypotheses where possible
            - Varied in type (e.g., statistical data, expert assessments, historical patterns)
            - Diagnostic (helps distinguish between hypotheses)
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "evidence_identification",
                "context": {"prompt": grounded_prompt, "research_results": research_results}
            })
            
            # Extract evidence from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse evidence from content
                evidence = self._parse_evidence_from_text(content)
                if evidence:
                    # Assign diagnostic value to evidence
                    return self._assign_diagnostic_value(evidence, hypotheses)
        
        # Fallback: Generate generic evidence
        return self._generate_generic_evidence(question, hypotheses)
    
    def _extract_evidence_from_text(self, text: str, source: str) -> List[Dict]:
        """
        Extract evidence items from text.
        
        Args:
            text: Text containing evidence
            source: Source of the text
            
        Returns:
            List of evidence items
        """
        evidence_items = []
        
        # Simple extraction based on sentences
        import re
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Process sentences to extract factual statements
        for sentence in sentences:
            # Skip short sentences
            if len(sentence.split()) < 5:
                continue
            
            # Skip sentences that are questions
            if sentence.strip().endswith("?"):
                continue
            
            # Skip sentences with first-person pronouns
            if re.search(r'\b(I|we|our|my)\b', sentence, re.IGNORECASE):
                continue
            
            # Skip sentences that seem like opinions
            opinion_phrases = ["believe", "think", "feel", "opinion", "suggest", "may", "might", "could", "possibly"]
            if any(phrase in sentence.lower() for phrase in opinion_phrases):
                continue
            
            # Add as evidence item
            evidence_items.append({
                "id": f"E{len(evidence_items) + 1}",
                "description": sentence.strip(),
                "source": source,
                "relevance": "medium"  # Default relevance
            })
            
            # Limit to reasonable number of evidence items per source
            if len(evidence_items) >= 3:
                break
        
        return evidence_items
    
    def _parse_evidence_from_text(self, text: str) -> List[Dict]:
        """
        Parse evidence items from text.
        
        Args:
            text: Text containing evidence descriptions
            
        Returns:
            List of parsed evidence items
        """
        evidence_items = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for evidence sections
        evidence_pattern = r'(?:^|\n)(?:Evidence|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Evidence|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        evidence_matches = re.findall(evidence_pattern, text, re.DOTALL)
        
        if not evidence_matches:
            # Try alternative pattern for numbered lists
            evidence_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            evidence_matches = re.findall(evidence_pattern, text, re.DOTALL)
            
            if evidence_matches:
                # Convert to expected format
                evidence_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(evidence_matches)]
        
        for match in evidence_matches:
            evidence_num = match[0].strip() if len(match) > 0 else ""
            evidence_description = match[1].strip() if len(match) > 1 else ""
            evidence_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract source
            source = "Analysis"  # Default source
            source_pattern = r'(?:source|from|according to).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            source_match = re.search(source_pattern, evidence_content, re.IGNORECASE | re.DOTALL)
            if source_match:
                source = source_match.group(1).strip()
            
            # Extract relevance
            relevance = "medium"  # Default relevance
            relevance_pattern = r'(?:relevance|relevant|importance).*?(high|medium|low)'
            relevance_match = re.search(relevance_pattern, evidence_content, re.IGNORECASE)
            if relevance_match:
                relevance = relevance_match.group(1).lower()
            
            evidence_items.append({
                "id": f"E{evidence_num}",
                "description": evidence_description,
                "source": source,
                "relevance": relevance
            })
        
        return evidence_items
    
    def _generate_generic_evidence(self, question: str, hypotheses: List[Dict]) -> List[Dict]:
        """
        Generate generic evidence based on the question and hypotheses.
        
        Args:
            question: The analytical question
            hypotheses: List of hypotheses
            
        Returns:
            List of generic evidence items
        """
        # Extract domain from question
        domain = self._extract_domain_from_question(question)
        
        # Domain-specific evidence templates
        domain_evidence = {
            "economic": [
                {
                    "id": "E1",
                    "description": "Recent economic growth rates have shown significant variation across sectors",
                    "source": "Economic analysis",
                    "relevance": "high"
                },
                {
                    "id": "E2",
                    "description": "Policy changes in the past year have altered regulatory frameworks for key industries",
                    "source": "Policy analysis",
                    "relevance": "high"
                },
                {
                    "id": "E3",
                    "description": "Market indicators show changing patterns of investment across different asset classes",
                    "source": "Market data",
                    "relevance": "medium"
                },
                {
                    "id": "E4",
                    "description": "International trade patterns have shifted in response to global economic conditions",
                    "source": "Trade statistics",
                    "relevance": "medium"
                },
                {
                    "id": "E5",
                    "description": "Consumer behavior metrics indicate changing preferences and spending patterns",
                    "source": "Consumer surveys",
                    "relevance": "medium"
                },
                {
                    "id": "E6",
                    "description": "Institutional quality measures show significant differences across regions",
                    "source": "Governance indicators",
                    "relevance": "high"
                },
                {
                    "id": "E7",
                    "description": "Technology adoption rates vary significantly across different economic sectors",
                    "source": "Industry reports",
                    "relevance": "medium"
                },
                {
                    "id": "E8",
                    "description": "Labor market data shows changing patterns of employment and wage growth",
                    "source": "Employment statistics",
                    "relevance": "high"
                }
            ],
            "political": [
                {
                    "id": "E1",
                    "description": "Voting patterns in recent elections show significant shifts in key demographics",
                    "source": "Electoral data",
                    "relevance": "high"
                },
                {
                    "id": "E2",
                    "description": "Policy implementation has varied significantly across different jurisdictions",
                    "source": "Policy analysis",
                    "relevance": "high"
                },
                {
                    "id": "E3",
                    "description": "Public opinion surveys indicate changing attitudes toward key political issues",
                    "source": "Opinion polls",
                    "relevance": "medium"
                },
                {
                    "id": "E4",
                    "description": "Interest group activity and lobbying expenditures have focused on specific policy areas",
                    "source": "Lobbying disclosures",
                    "relevance": "medium"
                },
                {
                    "id": "E5",
                    "description": "Leadership changes have occurred in key political institutions and organizations",
                    "source": "Political reporting",
                    "relevance": "high"
                },
                {
                    "id": "E6",
                    "description": "Institutional constraints have limited policy options in several key areas",
                    "source": "Governance analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E7",
                    "description": "Political discourse analysis shows changing framing of key issues",
                    "source": "Media analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E8",
                    "description": "International relations have influenced domestic political calculations",
                    "source": "Foreign policy analysis",
                    "relevance": "medium"
                }
            ],
            "technological": [
                {
                    "id": "E1",
                    "description": "R&D investment patterns show significant focus on specific technological domains",
                    "source": "Investment data",
                    "relevance": "high"
                },
                {
                    "id": "E2",
                    "description": "Patent filings indicate accelerating innovation in key technology areas",
                    "source": "Patent databases",
                    "relevance": "high"
                },
                {
                    "id": "E3",
                    "description": "Technology adoption rates vary significantly across different sectors and regions",
                    "source": "Industry surveys",
                    "relevance": "medium"
                },
                {
                    "id": "E4",
                    "description": "Public research funding has prioritized specific technological domains",
                    "source": "Funding data",
                    "relevance": "high"
                },
                {
                    "id": "E5",
                    "description": "Regulatory frameworks for emerging technologies have evolved in different ways across jurisdictions",
                    "source": "Regulatory analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E6",
                    "description": "Scientific publications show increasing focus on interdisciplinary research",
                    "source": "Bibliometric analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E7",
                    "description": "Technology companies have made strategic acquisitions in specific domains",
                    "source": "Market analysis",
                    "relevance": "high"
                },
                {
                    "id": "E8",
                    "description": "Public attitudes toward technology adoption show significant variation across applications",
                    "source": "Public opinion surveys",
                    "relevance": "medium"
                }
            ],
            "social": [
                {
                    "id": "E1",
                    "description": "Demographic trends show changing population composition and distribution",
                    "source": "Demographic data",
                    "relevance": "high"
                },
                {
                    "id": "E2",
                    "description": "Social mobility indicators reveal varying opportunities across different groups",
                    "source": "Socioeconomic analysis",
                    "relevance": "high"
                },
                {
                    "id": "E3",
                    "description": "Cultural values surveys indicate shifting priorities across generations",
                    "source": "Values surveys",
                    "relevance": "medium"
                },
                {
                    "id": "E4",
                    "description": "Educational outcomes show persistent disparities across different communities",
                    "source": "Education statistics",
                    "relevance": "medium"
                },
                {
                    "id": "E5",
                    "description": "Social network analysis reveals changing patterns of connection and influence",
                    "source": "Network analysis",
                    "relevance": "high"
                },
                {
                    "id": "E6",
                    "description": "Health indicators show varying outcomes related to social determinants",
                    "source": "Health statistics",
                    "relevance": "medium"
                },
                {
                    "id": "E7",
                    "description": "Media consumption patterns indicate fragmentation of information sources",
                    "source": "Media analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E8",
                    "description": "Community organization and civic engagement levels vary significantly across regions",
                    "source": "Civil society research",
                    "relevance": "high"
                }
            ],
            "general": [
                {
                    "id": "E1",
                    "description": "Historical patterns show similar situations have developed in predictable ways",
                    "source": "Historical analysis",
                    "relevance": "high"
                },
                {
                    "id": "E2",
                    "description": "Expert assessments indicate consensus on some aspects but disagreement on others",
                    "source": "Expert surveys",
                    "relevance": "high"
                },
                {
                    "id": "E3",
                    "description": "Quantitative data shows significant trends that have persisted over time",
                    "source": "Statistical analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E4",
                    "description": "Case studies of similar situations reveal important contextual factors",
                    "source": "Comparative analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E5",
                    "description": "Recent developments indicate acceleration of previously identified trends",
                    "source": "Current reporting",
                    "relevance": "high"
                },
                {
                    "id": "E6",
                    "description": "Stakeholder positions show alignment on some issues but conflict on others",
                    "source": "Stakeholder analysis",
                    "relevance": "medium"
                },
                {
                    "id": "E7",
                    "description": "Theoretical models predict different outcomes based on varying assumptions",
                    "source": "Academic research",
                    "relevance": "medium"
                },
                {
                    "id": "E8",
                    "description": "Implementation challenges have affected similar initiatives in the past",
                    "source": "Implementation studies",
                    "relevance": "high"
                }
            ]
        }
        
        # Get evidence for the identified domain
        evidence = domain_evidence.get(domain, domain_evidence["general"])
        
        # Assign diagnostic value
        return self._assign_diagnostic_value(evidence, hypotheses)
    
    def _assign_diagnostic_value(self, evidence: List[Dict], hypotheses: List[Dict]) -> List[Dict]:
        """
        Assign diagnostic value to evidence items.
        
        Args:
            evidence: List of evidence items
            hypotheses: List of hypotheses
            
        Returns:
            Evidence items with diagnostic value
        """
        # Use Llama4ScoutMCP to assign diagnostic value
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Format hypotheses and evidence for prompt
            hypotheses_text = ""
            for hypothesis in hypotheses:
                hypothesis_id = hypothesis.get("id", "")
                hypothesis_statement = hypothesis.get("statement", "")
                hypotheses_text += f"{hypothesis_id}: {hypothesis_statement}\n"
            
            evidence_text = ""
            for item in evidence:
                item_id = item.get("id", "")
                item_description = item.get("description", "")
                evidence_text += f"{item_id}: {item_description}\n"
            
            # Create prompt for diagnostic value assignment
            prompt = f"""
            Assess the diagnostic value of each evidence item for evaluating the competing hypotheses.
            
            Competing Hypotheses:
            {hypotheses_text}
            
            Evidence Items:
            {evidence_text}
            
            For each evidence item, determine its diagnostic value by assessing:
            1. How strongly it supports or contradicts each hypothesis
            2. Whether it helps distinguish between hypotheses
            
            Provide your assessment in a structured format that indicates the relationship between each evidence item and each hypothesis.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "What is the diagnostic value of the evidence?",
                "analysis_type": "diagnostic_assessment",
                "context": {"prompt": prompt}
            })
            
            # Extract diagnostic value from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse diagnostic value from content
                diagnostic_values = self._parse_diagnostic_value_from_text(content, evidence, hypotheses)
                if diagnostic_values:
                    # Update evidence with diagnostic value
                    for item in evidence:
                        item_id = item.get("id", "")
                        if item_id in diagnostic_values:
                            item["diagnostic_value"] = diagnostic_values[item_id]
                    
                    return evidence
        
        # Fallback: Assign random diagnostic value
        for item in evidence:
            item["diagnostic_value"] = self._generate_random_diagnostic_value(hypotheses)
        
        return evidence
    
    def _parse_diagnostic_value_from_text(self, text: str, evidence: List[Dict], hypotheses: List[Dict]) -> Dict:
        """
        Parse diagnostic value from text.
        
        Args:
            text: Text containing diagnostic value assessment
            evidence: List of evidence items
            hypotheses: List of hypotheses
            
        Returns:
            Dictionary mapping evidence IDs to diagnostic value
        """
        diagnostic_values = {}
        
        # Simple parsing based on patterns
        import re
        
        # Get evidence IDs
        evidence_ids = [item.get("id", "") for item in evidence]
        
        # Get hypothesis IDs
        hypothesis_ids = [hypothesis.get("id", "") for hypothesis in hypotheses]
        
        # Look for evidence assessments
        for evidence_id in evidence_ids:
            # Look for section about this evidence item
            evidence_pattern = f"{evidence_id}.*?(?:\n\n|$)"
            evidence_match = re.search(evidence_pattern, text, re.DOTALL)
            
            if evidence_match:
                evidence_text = evidence_match.group(0)
                
                # Initialize diagnostic value for this evidence item
                diagnostic_value = {}
                
                # Look for assessments of each hypothesis
                for hypothesis_id in hypothesis_ids:
                    # Look for relationship with this hypothesis
                    hypothesis_pattern = f"{hypothesis_id}.*?(strongly supports|supports|neutral|contradicts|strongly contradicts)"
                    hypothesis_match = re.search(hypothesis_pattern, evidence_text, re.IGNORECASE)
                    
                    if hypothesis_match:
                        relationship = hypothesis_match.group(1).lower()
                        
                        # Map relationship to consistency value
                        consistency_map = {
                            "strongly supports": "CC",  # Consistent
                            "supports": "C",           # Consistent
                            "neutral": "NA",           # Not Applicable
                            "contradicts": "I",        # Inconsistent
                            "strongly contradicts": "II"  # Inconsistent
                        }
                        
                        diagnostic_value[hypothesis_id] = consistency_map.get(relationship, "NA")
                    else:
                        # Default to Not Applicable
                        diagnostic_value[hypothesis_id] = "NA"
                
                diagnostic_values[evidence_id] = diagnostic_value
        
        return diagnostic_values
    
    def _generate_random_diagnostic_value(self, hypotheses: List[Dict]) -> Dict:
        """
        Generate random diagnostic value for evidence.
        
        Args:
            hypotheses: List of hypotheses
            
        Returns:
            Dictionary mapping hypothesis IDs to consistency values
        """
        import random
        
        diagnostic_value = {}
        consistency_values = ["CC", "C", "NA", "I", "II"]
        
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis.get("id", "")
            # Randomly select consistency value with bias toward NA
            weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Weights for CC, C, NA, I, II
            diagnostic_value[hypothesis_id] = random.choices(consistency_values, weights=weights)[0]
        
        return diagnostic_value
    
    def _evaluate_hypotheses(self, hypotheses: List[Dict], evidence: List[Dict], research_results: Dict, relevant_economic_data: List[Dict], relevant_geopolitical_data: List[Dict], context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Evaluate hypotheses against evidence.
        
        Args:
            hypotheses: List of hypotheses
            evidence: List of evidence items
            research_results: Research results
            relevant_economic_data: relevant economic data.
            relevant_geopolitical_data: relevant geopolitical data.
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing evaluation matrix
        """
        logger.info("Evaluating hypotheses against evidence")
        
        # Initialize evaluation matrix
        evaluation_matrix = {
            "hypotheses": [hypothesis.get("id", "") for hypothesis in hypotheses],
            "evidence": [item.get("id", "") for item in evidence],
            "matrix": {},
            "scores": {}
        }
        
        # Format economic and geopolitical data for prompt
        economic_data_text = ""
        if relevant_economic_data:
            economic_data_text += "Relevant Economic Data:\n"
            for item in relevant_economic_data:
                date = item.get("date", "")
                value = item.get("value", "")
                series_id = item.get("series_id","")

                economic_data_text += f"- {date}: {series_id} = {value}\n"

        geopolitical_data_text = ""
        if relevant_geopolitical_data:
            geopolitical_data_text += "Relevant Geopolitical Event Data:\n"
            for item in relevant_geopolitical_data:
                date = item.get("date", "")
                event_code = item.get("event_code", "")
                actor1 = item.get("actor1", "")
                actor2 = item.get("actor2", "")

                geopolitical_data_text += f"- {date}: Event Code {event_code} between {actor1} and {actor2}\n"

        #create a prompt to the llm
        prompt = f"""
        Consider the following economic data: {economic_data_text}
        Consider the following geopolitical event data: {geopolitical_data_text}
        Now assess the evidence against the hypotheses. 
        """
        context.add_metadata("data_grounding_prompt", prompt)

        # Fill matrix with consistency values from evidence diagnostic value
        for item in evidence:
            item_id = item.get("id", "")
            diagnostic_value = item.get("diagnostic_value", {})
            
            if item_id and diagnostic_value:
                evaluation_matrix["matrix"][item_id] = diagnostic_value

        # Incorporate uncertainty information from UncertaintyMappingTechnique
        uncertainty_results = context.get_technique_result("uncertainty_mapping")
        if uncertainty_results and "uncertainties" in uncertainty_results:
            for uncertainty in uncertainty_results["uncertainties"]:
                uncertainty_description = uncertainty.get("uncertainty", "")
                uncertainty_impact = uncertainty.get("impact", "low")

                for item in evidence:
                    item_description = item.get("description", "")
                    # Check if evidence is related to uncertainty
                    if uncertainty_description.lower() in item_description.lower() or item_description.lower() in uncertainty_description.lower():
                        # Discount the consistency of this evidence based on uncertainty impact
                        for hypothesis_id in evaluation_matrix["hypotheses"]:
                            if item.get("id", "") in evaluation_matrix["matrix"]:
                                consistency = evaluation_matrix["matrix"][item.get("id", "")].get(hypothesis_id, "NA")
                                if consistency in ["CC", "C"]:
                                    if uncertainty_impact == "high":
                                        evaluation_matrix["matrix"][item.get("id", "")][hypothesis_id] = "NA"
                                    elif uncertainty_impact == "medium":
                                        evaluation_matrix["matrix"][item.get("id", "")][hypothesis_id] = "NA"

        
        # Calculate scores
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis.get("id", "")
            
            if hypothesis_id:
                # Count inconsistencies
                inconsistency_count = 0
                weighted_inconsistency_count = 0
                
                for item in evidence:
                    item_id = item.get("id", "")
                    relevance = item.get("relevance", "medium")
                    
                    # Convert relevance to weight
                    weight = 1.0
                    if relevance == "high":
                        weight = 2.0
                    elif relevance == "low":
                        weight = 0.5
                    
                    # Get consistency value for this hypothesis
                    if item_id in evaluation_matrix["matrix"]:
                        consistency = evaluation_matrix["matrix"][item_id].get(hypothesis_id, "NA")
                        
                        # Count inconsistencies
                        if consistency == "I":
                            inconsistency_count += 1
                            weighted_inconsistency_count += weight
                        elif consistency == "II":
                            inconsistency_count += 2
                            weighted_inconsistency_count += 2 * weight
                
                # Store scores
                evaluation_matrix["scores"][hypothesis_id] = {
                    "inconsistency_count": inconsistency_count,
                    "weighted_inconsistency_count": weighted_inconsistency_count
                }
        
        return evaluation_matrix
    
    def _analyze_results(self, hypotheses: List[Dict], evidence: List[Dict], evaluation_matrix: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Analyze evaluation results.
        
        Args:
            hypotheses: List of hypotheses
            evidence: List of evidence items
            evaluation_matrix: Evaluation matrix
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing results")
        
        # Get scores
        scores = evaluation_matrix.get("scores", {})
        
        
        # Rank hypotheses by weighted inconsistency count (lower is better)
        ranked_hypotheses = []
        
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis.get("id", "")
            
            if hypothesis_id and hypothesis_id in scores:
                # Get score
                score = scores[hypothesis_id]
                
                # Add to ranked list
                ranked_hypotheses.append({
                    "id": hypothesis_id,
                    "statement": hypothesis.get("statement", ""),
                    "inconsistency_count": score.get("inconsistency_count", 0),
                    "weighted_inconsistency_count": score.get("weighted_inconsistency_count", 0)
                })
        
        # Sort by weighted inconsistency count (ascending)
        ranked_hypotheses.sort(key=lambda x: x["weighted_inconsistency_count"])
        
        # Identify most likely hypothesis
        most_likely_hypothesis = ranked_hypotheses[0] if ranked_hypotheses else None
        
        # Identify key evidence
        key_evidence = []
        
        for item in evidence:
            item_id = item.get("id", "")
            relevance = item.get("relevance", "medium")
            
            if item_id and relevance == "high":
                # Check if this evidence discriminates between hypotheses
                diagnostic_value = item.get("diagnostic_value", {})
                
                if diagnostic_value:
                    # Count different consistency values
                    consistency_counts = {}
                    
                    for hypothesis_id, consistency in diagnostic_value.items():
                        if consistency in consistency_counts:
                            consistency_counts[consistency] += 1
                        else:
                            consistency_counts[consistency] = 1
                    
                    # If evidence has different consistency values for different hypotheses, it's discriminating
                    if len(consistency_counts) > 1 and "NA" not in consistency_counts:
                        key_evidence.append({
                            "id": item_id,
                            "description": item.get("description", ""),
                            "source": item.get("source", "")
                        })
        
        # Identify sensitivity points
        sensitivity_points = []
        
        # If top two hypotheses are close in score, identify evidence that distinguishes them
        if len(ranked_hypotheses) >= 2:
            top_hypothesis = ranked_hypotheses[0]
            second_hypothesis = ranked_hypotheses[1]
            
            # Check if scores are close
            score_difference = second_hypothesis["weighted_inconsistency_count"] - top_hypothesis["weighted_inconsistency_count"]
            
            if score_difference < 2.0:
                # Identify evidence that distinguishes between top two hypotheses
                for item in evidence:
                    item_id = item.get("id", "")
                    diagnostic_value = item.get("diagnostic_value", {})
                    
                    if item_id and diagnostic_value:
                        # Get consistency values for top two hypotheses
                        top_consistency = diagnostic_value.get(top_hypothesis["id"], "NA")
                        second_consistency = diagnostic_value.get(second_hypothesis["id"], "NA")
                        
                        # Check if evidence distinguishes between them
                        if top_consistency != second_consistency and top_consistency != "NA" and second_consistency != "NA":
                            sensitivity_points.append({
                                "evidence_id": item_id,
                                "description": item.get("description", ""),
                                "impact": "Distinguishes between top hypotheses"
                            })
        
        # Incorporate uncertainty information
        uncertainty_results = context.get_technique_result("uncertainty_mapping")
        if uncertainty_results and "uncertainties" in uncertainty_results:
            for uncertainty in uncertainty_results["uncertainties"]:
                uncertainty_description = uncertainty.get("uncertainty", "")
                uncertainty_impact = uncertainty.get("impact", "low")

                # Check if top-ranked hypothesis is highly sensitive to a specific uncertainty
                for item in evidence:
                    item_description = item.get("description", "")
                    if uncertainty_description.lower() in item_description.lower() or item_description.lower() in uncertainty_description.lower():
                        for hypothesis_id in evaluation_matrix["hypotheses"]:
                            if hypothesis_id == most_likely_hypothesis["id"] and uncertainty_impact == "high":
                                sensitivity_points.append({
                                    "evidence_id": item.get("id"),
                                    "description": item_description,
                                    "impact": f"Highly sensitive to uncertainty: {uncertainty_description}"
                                })
                                most_likely_hypothesis["confidence"] = "low"

        # Compile analysis results
        analysis_results = {
            "ranked_hypotheses": ranked_hypotheses,
            "most_likely_hypothesis": most_likely_hypothesis,
            "key_evidence": key_evidence,
            "sensitivity_points": sensitivity_points
        }
        
        return analysis_results
    
    def _extract_findings(self, hypotheses: List[Dict], evidence: List[Dict], analysis_results: Dict) -> List[Dict]:
        """
        Extract key findings from ACH analysis.
        
        Args:
            hypotheses: List of hypotheses
            evidence: List of evidence items
            analysis_results: Analysis results
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about most likely hypothesis
        most_likely_hypothesis = analysis_results.get("most_likely_hypothesis")
        if most_likely_hypothesis:
            findings.append({
                    "finding": f"The most likely hypothesis based on evidence is: {most_likely_hypothesis.get('statement', '')}",
                    "confidence": most_likely_hypothesis.get("confidence", "medium"),  # Use the confidence level from the hypothesis
                    "source": "analysis_of_competing_hypotheses"
                })

        # Adjust confidence based on sensitivity points
        if most_likely_hypothesis:
            sensitivity_points = analysis_results.get("sensitivity_points", [])
            for sensitivity in sensitivity_points:
                if "Highly sensitive" in sensitivity.get("impact", ""):
                    if findings:
                        # find the finding that corresponds to this hypothesis and adjust the confidence accordingly
                        for finding in findings:
                            if most_likely_hypothesis.get("statement", "") in finding.get("finding", ""):
                                finding["confidence"] = "low"
                                break

        # Add finding about key evidence
        key_evidence = analysis_results.get("key_evidence", [])
        if key_evidence:
            findings.append({
                "finding": f"Key discriminating evidence: {key_evidence[0].get('description', '')}",
                "confidence": "high",
                "source": "analysis_of_competing_hypotheses"
            })
        
        # Add finding about sensitivity points
        sensitivity_points = analysis_results.get("sensitivity_points", [])
        if sensitivity_points:
            findings.append({
                "finding": f"The analysis is sensitive to evidence regarding: {sensitivity_points[0].get('description', '')}",
                "confidence": "medium",
                "source": "analysis_of_competing_hypotheses"
            })
        
        # Add finding about hypothesis ranking
        ranked_hypotheses = analysis_results.get("ranked_hypotheses", [])
        if len(ranked_hypotheses) >= 2:
            findings.append({
                "finding": f"The second most likely hypothesis is: {ranked_hypotheses[1].get('statement', '')}",
                "confidence": "medium",
                "source": "analysis_of_competing_hypotheses"
            })
        
        return findings
    
    def _extract_assumptions(self, hypotheses: List[Dict], evidence: List[Dict], analysis_results: Dict) -> List[Dict]:
        """
        Extract assumptions from ACH analysis.
        
        Args:
            hypotheses: List of hypotheses
            evidence: List of evidence items
            analysis_results: Analysis results
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about evidence reliability
        assumptions.append({
            "assumption": "The evidence items are reliable and accurately represented",
            "criticality": "high",
            "source": "analysis_of_competing_hypotheses"
        })
        
        # Add assumption about evidence completeness
        assumptions.append({
            "assumption": "The set of evidence is sufficiently complete to evaluate the hypotheses",
            "criticality": "high",
            "source": "analysis_of_competing_hypotheses"
        })
        
        # Add assumption about hypothesis set
        assumptions.append({
            "assumption": "The set of hypotheses includes all reasonable explanations",
            "criticality": "medium",
            "source": "analysis_of_competing_hypotheses"
        })
        
        # Add assumption from most likely hypothesis
        most_likely_hypothesis = analysis_results.get("most_likely_hypothesis")
        if most_likely_hypothesis:
            # Find the full hypothesis object
            for hypothesis in hypotheses:
                if hypothesis.get("id") == most_likely_hypothesis.get("id"):
                    # Get assumptions from this hypothesis
                    hypothesis_assumptions = hypothesis.get("assumptions", [])
                    if hypothesis_assumptions:
                        assumptions.append({
                            "assumption": f"From leading hypothesis: {hypothesis_assumptions[0]}",
                            "criticality": "high",
                            "source": "analysis_of_competing_hypotheses"
                        })
                    break
        
        return assumptions
    
    def _extract_uncertainties(self, hypotheses: List[Dict], evidence: List[Dict], analysis_results: Dict) -> List[Dict]:
        """
        Extract uncertainties from ACH analysis.
        
        Args:
            hypotheses: List of hypotheses
            evidence: List of evidence items
            analysis_results: Analysis results
            
        Returns:
            List of uncertainties
        """
        uncertainties = []
        
        # Add uncertainty about evidence interpretation
        uncertainties.append({
            "uncertainty": "The interpretation of evidence and its consistency with hypotheses involves subjective judgment",
            "impact": "high",
            "source": "analysis_of_competing_hypotheses"
        })
        
        # Add uncertainty about missing evidence
        uncertainties.append({
            "uncertainty": "There may be important evidence not included in the analysis",
            "impact": "high",
            "source": "analysis_of_competing_hypotheses"
        })
        
        # Add uncertainty about sensitivity points
        sensitivity_points = analysis_results.get("sensitivity_points", [])
        if sensitivity_points:
            uncertainties.append({
                "uncertainty": f"The analysis is sensitive to evidence regarding: {sensitivity_points[0].get('description', '')}",
                "impact": "medium",
                "source": "analysis_of_competing_hypotheses"
            })
        
        # Add uncertainty about hypothesis formulation
        uncertainties.append({
            "uncertainty": "The specific formulation of hypotheses may affect their evaluation against evidence",
            "impact": "medium",
            "source": "analysis_of_competing_hypotheses"
        })
        
        return uncertainties
