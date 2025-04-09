"""
Consensus Challenge Technique for identifying and challenging consensus views.
This module provides the ConsensusChallengeTechnique class for consensus analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.analytical_technique import AnalyticalTechnique
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsensusChallengeTechnique(AnalyticalTechnique):
    """
    Consensus Challenge Technique for identifying and challenging consensus views.
    
    This technique provides capabilities for:
    1. Identifying prevailing consensus views on a topic
    2. Systematically challenging these views with alternative perspectives
    3. Evaluating the strength of consensus and potential blind spots
    4. Generating alternative hypotheses that challenge conventional wisdom
    """
    
    def __init__(self):
        """Initialize the Consensus Challenge Technique."""
        super().__init__(
            name="consensus_challenge",
            description="Identifies and systematically challenges prevailing consensus views",
            required_mcps=["llama4_scout", "research_mcp"],
            compatible_techniques=["red_teaming", "key_assumptions_check", "multi_persona"],
            incompatible_techniques=[]
        )
        logger.info("Initialized ConsensusChallengeTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Consensus Challenge Technique")
        
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
            
            # Identify consensus views
            consensus_views = self._identify_consensus_views(question, research_results, context, parameters)
            
            # Challenge consensus views
            challenges = self._challenge_consensus_views(consensus_views, research_results, context, parameters)
            
            # Evaluate consensus strength
            consensus_evaluation = self._evaluate_consensus_strength(consensus_views, challenges, research_results, parameters)
            
            # Generate alternative hypotheses
            alternative_hypotheses = self._generate_alternative_hypotheses(consensus_views, challenges, research_results, context, parameters)
            
            # Compile results
            results = {
                "technique": "consensus_challenge",
                "timestamp": time.time(),
                "question": question,
                "consensus_views": consensus_views,
                "challenges": challenges,
                "consensus_evaluation": consensus_evaluation,
                "alternative_hypotheses": alternative_hypotheses,
                "findings": self._extract_findings(consensus_views, challenges, alternative_hypotheses),
                "assumptions": self._extract_assumptions(consensus_views),
                "uncertainties": self._extract_uncertainties(consensus_views, challenges)
            }
            
            # Add results to context
            context.add_technique_result("consensus_challenge", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Consensus Challenge Technique: {e}")
            return self.handle_error(e, context)
    
    def _identify_consensus_views(self, question: str, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Identify prevailing consensus views on the topic.
        
        Args:
            question: The analytical question
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of consensus views
        """
        logger.info("Identifying consensus views")
        
        # Check if consensus views are provided in parameters
        if "consensus_views" in parameters and isinstance(parameters["consensus_views"], list):
            return parameters["consensus_views"]
        
        # Use Llama4ScoutMCP to identify consensus views
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Create prompt for consensus view identification
            prompt = f"""
            Based on the following question, identify the 3-5 most prevalent consensus views or widely accepted beliefs on this topic.
            
            Question: {question}
            
            For each consensus view:
            1. Clearly state the consensus position
            2. Identify the key proponents or sources of this view
            3. Summarize the main evidence or arguments supporting this view
            4. Estimate how widely accepted this view is (dominant, majority, common, emerging)
            
            Focus on identifying what most experts or mainstream sources believe about this topic.
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "standard",
                "context": {"prompt": grounded_prompt, "research_results": research_results}
            })
            
            # Extract consensus views from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse consensus views from content
                consensus_views = self._parse_consensus_views_from_text(content)
                if consensus_views:
                    return consensus_views
        
        # If no consensus views identified or no LLM available, use research results
        research_mcp = self.get_mcp("research_mcp")
        if research_mcp and research_results:
            # Extract key findings from research results
            key_findings = ""
            if isinstance(research_results, dict):
                if "research_results" in research_results and isinstance(research_results["research_results"], dict):
                    key_findings = research_results["research_results"].get("key_findings", "")
            
            # Generate consensus views based on key findings
            if key_findings:
                return self._generate_consensus_views_from_findings(key_findings)
        
        # Fallback: Generate generic consensus views
        return self._generate_generic_consensus_views(question)
    
    def _parse_consensus_views_from_text(self, text: str) -> List[Dict]:
        """
        Parse consensus views from text.
        
        Args:
            text: Text containing consensus view descriptions
            
        Returns:
            List of parsed consensus views
        """
        consensus_views = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for numbered or bulleted items
        view_pattern = r'(?:^|\n)(?:Consensus View|View|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Consensus View|View|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        view_matches = re.findall(view_pattern, text, re.DOTALL)
        
        if not view_matches:
            # Try alternative pattern for numbered lists
            view_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            view_matches = re.findall(view_pattern, text, re.DOTALL)
            
            if view_matches:
                # Convert to expected format
                view_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(view_matches)]
        
        for match in view_matches:
            view_num = match[0].strip() if len(match) > 0 else ""
            view_statement = match[1].strip() if len(match) > 1 else ""
            view_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract proponents
            proponents = []
            proponents_pattern = r'(?:proponents|sources|advocates|supporters).*?(?::|include|are)(.*?)(?=(?:\n\n|\Z))'
            proponents_match = re.search(proponents_pattern, view_content, re.IGNORECASE | re.DOTALL)
            if proponents_match:
                proponents_text = proponents_match.group(1).strip()
                # Split by commas or "and"
                proponents = [p.strip() for p in re.split(r',|\sand\s', proponents_text) if p.strip()]
            
            # Extract evidence
            evidence = ""
            evidence_pattern = r'(?:evidence|arguments|support|reasoning).*?(?::|include|are)(.*?)(?=(?:\n\n|\Z))'
            evidence_match = re.search(evidence_pattern, view_content, re.IGNORECASE | re.DOTALL)
            if evidence_match:
                evidence = evidence_match.group(1).strip()
            
            # Extract acceptance level
            acceptance = "common"
            acceptance_pattern = r'(?:accepted|acceptance|prevalence|widespread).*?(dominant|majority|common|emerging|widespread|general)'
            acceptance_match = re.search(acceptance_pattern, view_content, re.IGNORECASE)
            if acceptance_match:
                acceptance = acceptance_match.group(1).lower()
                if acceptance == "widespread" or acceptance == "general":
                    acceptance = "dominant"
            
            consensus_views.append({
                "statement": view_statement,
                "proponents": proponents,
                "evidence": evidence,
                "acceptance_level": acceptance
            })
        
        return consensus_views
    
    def _generate_consensus_views_from_findings(self, key_findings: str) -> List[Dict]:
        """
        Generate consensus views based on key findings from research.
        
        Args:
            key_findings: Key findings from research
            
        Returns:
            List of generated consensus views
        """
        # Split findings into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', key_findings)
        
        # Filter for sentences that sound like consensus statements
        consensus_indicators = ["widely", "generally", "commonly", "typically", "most", "many", "experts", "consensus", "accepted", "believed", "understood", "recognized"]
        
        potential_consensus = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in consensus_indicators):
                potential_consensus.append(sentence)
        
        # Generate consensus views
        consensus_views = []
        for i, statement in enumerate(potential_consensus[:3]):  # Limit to top 3
            consensus_views.append({
                "statement": statement,
                "proponents": ["Research literature", "Expert consensus"],
                "evidence": "Based on research findings",
                "acceptance_level": "common"
            })
        
        return consensus_views
    
    def _generate_generic_consensus_views(self, question: str) -> List[Dict]:
        """
        Generate generic consensus views based on the question.
        
        Args:
            question: The analytical question
            
        Returns:
            List of generic consensus views
        """
        # Extract key terms from question
        import re
        question_lower = question.lower()
        
        # Check for domain indicators in question
        domains = {
            "economic": ["economic", "economy", "market", "financial", "growth", "recession", "inflation", "investment"],
            "political": ["political", "government", "policy", "regulation", "election", "democratic", "republican"],
            "technological": ["technology", "innovation", "digital", "ai", "automation", "tech", "software", "hardware"],
            "social": ["social", "cultural", "demographic", "education", "healthcare", "society", "community"],
            "environmental": ["environmental", "climate", "sustainability", "renewable", "carbon", "pollution"]
        }
        
        # Determine primary domain
        domain_counts = {domain: sum(1 for term in terms if term in question_lower) for domain, terms in domains.items()}
        primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0] if any(domain_counts.values()) else "general"
        
        # Generate domain-specific consensus views
        if primary_domain == "economic":
            return [
                {
                    "statement": "Economic growth is primarily driven by productivity improvements and innovation",
                    "proponents": ["Mainstream economists", "Central banks", "Economic research institutions"],
                    "evidence": "Historical correlation between productivity growth and economic expansion",
                    "acceptance_level": "dominant"
                },
                {
                    "statement": "Market-based solutions generally lead to more efficient outcomes than centralized planning",
                    "proponents": ["Neoclassical economists", "Financial institutions", "Business schools"],
                    "evidence": "Comparative economic performance of market vs. planned economies",
                    "acceptance_level": "majority"
                },
                {
                    "statement": "Central bank independence is essential for maintaining price stability",
                    "proponents": ["Monetary economists", "Central bankers", "International financial institutions"],
                    "evidence": "Lower inflation rates in countries with independent central banks",
                    "acceptance_level": "common"
                }
            ]
        elif primary_domain == "political":
            return [
                {
                    "statement": "Democratic governance, despite its flaws, provides better outcomes than authoritarian alternatives",
                    "proponents": ["Political scientists", "Democratic institutions", "Human rights organizations"],
                    "evidence": "Comparative studies of human development indicators across regime types",
                    "acceptance_level": "dominant"
                },
                {
                    "statement": "Effective governance requires balancing centralized authority with local autonomy",
                    "proponents": ["Public policy experts", "Governance researchers", "International development organizations"],
                    "evidence": "Case studies of successful governance reforms",
                    "acceptance_level": "common"
                },
                {
                    "statement": "Political polarization is increasing in many democratic societies",
                    "proponents": ["Political analysts", "Social scientists", "Media researchers"],
                    "evidence": "Voting pattern analysis and public opinion surveys",
                    "acceptance_level": "majority"
                }
            ]
        elif primary_domain == "technological":
            return [
                {
                    "statement": "Technological innovation is accelerating and will continue to transform industries",
                    "proponents": ["Technology analysts", "Industry leaders", "Research institutions"],
                    "evidence": "Historical adoption curves of new technologies",
                    "acceptance_level": "dominant"
                },
                {
                    "statement": "AI and automation will significantly impact labor markets and require workforce adaptation",
                    "proponents": ["Economists", "Technology researchers", "Industry associations"],
                    "evidence": "Early impact studies of automation on employment patterns",
                    "acceptance_level": "majority"
                },
                {
                    "statement": "Data privacy and security concerns will increasingly shape technology development",
                    "proponents": ["Cybersecurity experts", "Privacy advocates", "Regulatory bodies"],
                    "evidence": "Growing regulatory frameworks like GDPR and increasing consumer concern",
                    "acceptance_level": "common"
                }
            ]
        elif primary_domain == "social":
            return [
                {
                    "statement": "Education and skill development are critical for social mobility",
                    "proponents": ["Education researchers", "Sociologists", "Economic mobility experts"],
                    "evidence": "Correlation between educational attainment and lifetime earnings",
                    "acceptance_level": "dominant"
                },
                {
                    "statement": "Demographic shifts are creating new social challenges and opportunities",
                    "proponents": ["Demographers", "Social policy experts", "Urban planners"],
                    "evidence": "Population aging trends and migration patterns",
                    "acceptance_level": "majority"
                },
                {
                    "statement": "Social media is significantly changing social interaction patterns",
                    "proponents": ["Social psychologists", "Media researchers", "Behavioral scientists"],
                    "evidence": "Studies on social media usage and its effects on relationships",
                    "acceptance_level": "common"
                }
            ]
        elif primary_domain == "environmental":
            return [
                {
                    "statement": "Climate change is primarily caused by human activities and requires urgent action",
                    "proponents": ["Climate scientists", "Environmental organizations", "International bodies like IPCC"],
                    "evidence": "Scientific consensus on anthropogenic climate change",
                    "acceptance_level": "dominant"
                },
                {
                    "statement": "Transitioning to renewable energy is both necessary and economically viable",
                    "proponents": ["Energy economists", "Environmental scientists", "Renewable industry analysts"],
                    "evidence": "Declining costs of renewable technologies and carbon impact studies",
                    "acceptance_level": "majority"
                },
                {
                    "statement": "Biodiversity loss poses significant risks to ecosystem services and human wellbeing",
                    "proponents": ["Ecologists", "Conservation biologists", "Environmental economists"],
                    "evidence": "Studies on ecosystem service valuation and biodiversity decline",
                    "acceptance_level": "common"
                }
            ]
        else:  # general
            return [
                {
                    "statement": "Complex problems typically require multidisciplinary approaches",
                    "proponents": ["Systems thinkers", "Academic researchers", "Policy experts"],
                    "evidence": "Case studies of successful complex problem solving",
                    "acceptance_level": "common"
                },
                {
                    "statement": "Long-term trends are often more significant than short-term fluctuations",
                    "proponents": ["Strategic analysts", "Forecasters", "Historical researchers"],
                    "evidence": "Historical analysis of major transitions and changes",
                    "acceptance_level": "majority"
                },
                {
                    "statement": "Cognitive biases significantly influence human decision making",
                    "proponents": ["Behavioral economists", "Cognitive psychologists", "Decision scientists"],
                    "evidence": "Experimental studies demonstrating systematic biases",
                    "acceptance_level": "dominant"
                }
            ]
    
    def _challenge_consensus_views(self, consensus_views: List[Dict], research_results: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Systematically challenge consensus views with alternative perspectives.
        
        Args:
            consensus_views: List of consensus views
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary of challenges to each consensus view
        """
        logger.info("Challenging consensus views")
        
        challenges = {}
        
        # Use Llama4ScoutMCP to challenge consensus views
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            for view in consensus_views:
                view_statement = view.get("statement", "")
                
                # Create prompt for challenging the consensus view
                prompt = f"""
                The following represents a consensus view on a topic:
                
                "{view_statement}"
                
                Your task is to systematically challenge this consensus view by:
                
                1.  Identifying potential weaknesses or blind spots in the consensus position.
                2.  Presenting alternative perspectives or contrarian viewpoints.
                3.  Highlighting evidence, data, or logical reasoning that contradicts or complicates the consensus view.
                4.  Exploring assumptions that underlie the consensus position.

                Each challenge should be supported by specific evidence, data, or logical reasoning. Do not rely on general or unsubstantiated criticisms. Focus on challenges that can be supported with concrete evidence or logical arguments. Avoid making vague or unsubstantiated claims.

                Provide a balanced, thoughtful, evidence-based critique that doesn't simply dismiss the consensus but critically examines it.
                """

                
                # Call Llama4ScoutMCP
                llama_response = llama4_scout.process({
                    "question": view_statement,
                    "analysis_type": "evaluative",
                    "context": {"prompt": prompt, "research_results": research_results}
                })
                
                # Extract challenge from response
                if isinstance(llama_response, dict) and "sections" in llama_response:
                    content = ""
                    for section_name, section_content in llama_response["sections"].items():
                        content += section_content + "\n\n"
                    
                    # Parse challenge from content
                    challenge = self._parse_challenge_from_text(content)
                    challenges[view_statement] = challenge
                else:
                    # Fallback: Generate generic challenge
                    challenges[view_statement] = self._generate_generic_challenge(view)
        else:
            # Generate generic challenges for all views
            for view in consensus_views:
                view_statement = view.get("statement", "")
                challenges[view_statement] = self._generate_generic_challenge(view)
        
        return challenges
    
    def _parse_challenge_from_text(self, text: str) -> Dict:
        """
        Parse challenge from text.
        
        Args:
            text: Text containing challenge
            
        Returns:
            Dictionary containing parsed challenge
        """
        # Initialize challenge structure
        challenge = {
            "weaknesses": [],
            "alternative_perspectives": [],
            "contradictory_evidence": [],
            "underlying_assumptions": []
        }
        
        # Simple parsing based on patterns
        import re
        
        # Map of section names to keys
        section_map = {
            "weaknesses": ["weaknesses", "blind spots", "limitations", "flaws"],
            "alternative_perspectives": ["alternative perspectives", "contrarian viewpoints", "different views", "other perspectives"],
            "contradictory_evidence": ["contradictory evidence", "contrary evidence", "evidence against", "counterevidence"],
            "underlying_assumptions": ["underlying assumptions", "assumptions", "presumptions", "presuppositions"]
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
                    challenge[key] = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', section_text)
                    challenge[key] = [item.strip() for item in items if item.strip()]
        
        # If no structured sections found, try to extract from whole text
        if not any(challenge.values()):
            # Look for bullet points in entire text
            bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
            bullet_matches = re.findall(bullet_pattern, text, re.DOTALL)
            
            if bullet_matches:
                # Distribute bullet points across categories
                for i, item in enumerate(bullet_matches):
                    item = item.strip()
                    if "assum" in item.lower():
                        challenge["underlying_assumptions"].append(item)
                    elif "evidence" in item.lower() or "data" in item.lower() or "research" in item.lower():
                        challenge["contradictory_evidence"].append(item)
                    elif "perspective" in item.lower() or "view" in item.lower() or "approach" in item.lower():
                        challenge["alternative_perspectives"].append(item)
                    else:
                        challenge["weaknesses"].append(item)
        
        return challenge
    
    def _generate_generic_challenge(self, view: Dict) -> Dict:
        """
        Generate a generic challenge for a consensus view.
        
        Args:
            view: Consensus view to challenge
            
        Returns:
            Dictionary containing generic challenge
        """
        view_statement = view.get("statement", "").lower()
        acceptance = view.get("acceptance_level", "common")
        
        # Base challenge structure
        challenge = {
            "weaknesses": [
                "The consensus view may oversimplify a complex issue with multiple dimensions",
                "The evidence supporting this view may suffer from selection bias or methodological limitations",
                "The consensus may reflect groupthink rather than rigorous critical evaluation"
            ],
            "alternative_perspectives": [
                "Alternative frameworks might provide different insights on this issue",
                "Minority viewpoints that challenge this consensus deserve consideration",
                "Cross-disciplinary perspectives might reveal blind spots in the dominant view"
            ],
            "contradictory_evidence": [
                "Some empirical findings appear inconsistent with this consensus position",
                "Edge cases and exceptions challenge the universality of this view",
                "Recent developments may not have been fully incorporated into the consensus"
            ],
            "underlying_assumptions": [
                "The consensus view assumes a stability that may not persist in changing conditions",
                "There are implicit value judgments embedded in this seemingly objective position",
                "The consensus may assume universal applicability when context matters significantly"
            ]
        }
        
        # Customize based on acceptance level
        if acceptance == "dominant":
            challenge["weaknesses"].append("Dominant consensus views are particularly vulnerable to confirmation bias")
            challenge["alternative_perspectives"].append("Systematically marginalized perspectives may offer valuable insights")
        elif acceptance == "emerging":
            challenge["weaknesses"].append("Emerging consensus may be driven by novelty rather than robust evidence")
            challenge["contradictory_evidence"].append("Established evidence from previous paradigms may contradict this view")
        
        # Customize based on content
        if "economic" in view_statement or "market" in view_statement or "growth" in view_statement:
            challenge["alternative_perspectives"].append("Alternative economic frameworks like ecological economics or complexity economics offer different perspectives")
            challenge["underlying_assumptions"].append("The view assumes rational actors and efficient markets which may not reflect reality")
        
        elif "technology" in view_statement or "innovation" in view_statement or "digital" in view_statement:
            challenge["alternative_perspectives"].append("Critical technology studies offer counterpoints to techno-optimist narratives")
            challenge["underlying_assumptions"].append("The view may assume technology determinism rather than social shaping of technology")
        
        elif "climate" in view_statement or "environment" in view_statement or "sustainable" in view_statement:
            challenge["contradictory_evidence"].append("Some localized studies show different patterns than global models predict")
            challenge["underlying_assumptions"].append("The consensus may underestimate adaptive capacity or technological solutions")
        
        elif "social" in view_statement or "cultural" in view_statement or "demographic" in view_statement:
            challenge["alternative_perspectives"].append("Perspectives from different cultural contexts challenge universalist assumptions")
            challenge["underlying_assumptions"].append("The view may normalize specific cultural patterns as universal")
        
        return challenge
    
    def _evaluate_consensus_strength(self, consensus_views: List[Dict], challenges: Dict, research_results: Dict, parameters: Dict) -> Dict:
        """
        Evaluate the strength of consensus and potential blind spots.
        
        Args:
            consensus_views: List of consensus views
            challenges: Dictionary of challenges to consensus views
            research_results: Research results
            parameters: Technique parameters
            
        Returns:
            Dictionary containing consensus evaluation
        """
        logger.info("Evaluating consensus strength")
        
        # Initialize evaluation
        evaluation = {
            "overall_assessment": "",
            "consensus_strength": {},
            "blind_spots": [],
            "diversity_of_perspectives": "",
            "evidence_quality": ""
        }
        
        # Evaluate each consensus view
        for view in consensus_views:
            view_statement = view.get("statement", "")
            acceptance = view.get("acceptance_level", "common")
            
            # Get challenge for this view
            view_challenges = challenges.get(view_statement, {})
            
            # Assess strength based on challenges
            strength = "strong"
            if view_challenges:
                contradictory_evidence = view_challenges.get("contradictory_evidence", [])
                if len(contradictory_evidence) >= 3:
                    strength = "weak"
                elif len(contradictory_evidence) >= 1:
                    strength = "moderate"
                
                # Adjust based on acceptance level
                if acceptance == "dominant" and strength == "moderate":
                    strength = "strong"
                elif acceptance == "emerging" and strength == "moderate":
                    strength = "weak"
            
            evaluation["consensus_strength"][view_statement] = strength
        
        # Identify blind spots
        all_assumptions = []
        for view_statement, view_challenges in challenges.items():
            assumptions = view_challenges.get("underlying_assumptions", [])
            all_assumptions.extend(assumptions)
        
        # Select top blind spots (deduplicated)
        seen = set()
        for assumption in all_assumptions:
            assumption_key = assumption.lower()
            if assumption_key not in seen and len(evaluation["blind_spots"]) < 5:
                seen.add(assumption_key)
                evaluation["blind_spots"].append(assumption)
        
        # Assess diversity of perspectives
        unique_proponents = set()
        for view in consensus_views:
            proponents = view.get("proponents", [])
            unique_proponents.update(proponents)
        
        if len(unique_proponents) <= 2:
            evaluation["diversity_of_perspectives"] = "low"
        elif len(unique_proponents) <= 5:
            evaluation["diversity_of_perspectives"] = "moderate"
        else:
            evaluation["diversity_of_perspectives"] = "high"
        
        # Assess evidence quality
        evidence_mentions = 0
        for view in consensus_views:
            evidence = view.get("evidence", "")
            if "study" in evidence.lower() or "research" in evidence.lower() or "data" in evidence.lower():
                evidence_mentions += 1
        
        if evidence_mentions <= 1:
            evaluation["evidence_quality"] = "low"
        elif evidence_mentions <= len(consensus_views) // 2:
            evaluation["evidence_quality"] = "moderate"
        else:
            evaluation["evidence_quality"] = "high"
        
        # Overall assessment
        strength_values = list(evaluation["consensus_strength"].values())
        strong_count = strength_values.count("strong")
        weak_count = strength_values.count("weak")
        
        if strong_count > weak_count and evaluation["diversity_of_perspectives"] != "low":
            evaluation["overall_assessment"] = "The consensus appears generally robust, though with some identified blind spots"
        elif weak_count > strong_count or evaluation["diversity_of_perspectives"] == "low":
            evaluation["overall_assessment"] = "The consensus has significant weaknesses and may reflect limited perspectives"
        else:
            evaluation["overall_assessment"] = "The consensus has mixed strength with both supporting evidence and notable challenges"
        
        return evaluation
    
    def _generate_alternative_hypotheses(self, consensus_views: List[Dict], challenges: Dict, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Generate alternative hypotheses that challenge conventional wisdom.
        
        Args:
            consensus_views: List of consensus views
            challenges: Dictionary of challenges to consensus views
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of alternative hypotheses
        """
        logger.info("Generating alternative hypotheses")
        
        # Use Llama4ScoutMCP to generate alternative hypotheses
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Get question from context
            question = context.get("question")
            
            # Create summaries of consensus views and challenges
            views_summary = ""
            for i, view in enumerate(consensus_views):
                view_statement = view.get("statement", "")
                views_summary += f"{i+1}. {view_statement}\n"
                
                # Add key challenges
                view_challenges = challenges.get(view_statement, {})
                weaknesses = view_challenges.get("weaknesses", [])
                if weaknesses:
                    views_summary += f"   Key weakness: {weaknesses[0]}\n"
                
                contradictory_evidence = view_challenges.get("contradictory_evidence", [])
                if contradictory_evidence:
                    views_summary += f"   Contradictory evidence: {contradictory_evidence[0]}\n"
            
            # Create prompt for alternative hypotheses
            prompt = f"""
            Based on the following question and consensus views (along with their challenges), generate 3-5 alternative hypotheses that challenge conventional wisdom.
            
            Question: {question}
            
            Consensus Views and Challenges:
            {views_summary}
            
            For each alternative hypothesis:
            1. Provide a clear statement of the hypothesis
            2. Explain how it challenges or reframes conventional thinking
            3. Identify supporting evidence or reasoning
            4. Assess its plausibility (low, medium, high)
            
            Focus on generating thoughtful alternatives that are plausible but meaningfully different from the consensus.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "creative",
                "context": {"prompt": prompt, "research_results": research_results}
            })
            
            # Extract alternative hypotheses from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse alternative hypotheses from content
                hypotheses = self._parse_hypotheses_from_text(content)
                if hypotheses:
                    return hypotheses
        
        # Fallback: Generate alternative hypotheses from challenges
        return self._generate_hypotheses_from_challenges(consensus_views, challenges)
    
    def _parse_hypotheses_from_text(self, text: str) -> List[Dict]:
        """
        Parse alternative hypotheses from text.
        
        Args:
            text: Text containing hypothesis descriptions
            
        Returns:
            List of parsed hypotheses
        """
        hypotheses = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for hypothesis sections
        hypothesis_pattern = r'(?:^|\n)(?:Hypothesis|Alternative|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Hypothesis|Alternative|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
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
            
            # Extract challenge to conventional thinking
            challenge = ""
            challenge_pattern = r'(?:challenges|reframes|differs|contrasts).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            challenge_match = re.search(challenge_pattern, hypothesis_content, re.IGNORECASE | re.DOTALL)
            if challenge_match:
                challenge = challenge_match.group(1).strip()
            
            # Extract supporting evidence
            evidence = ""
            evidence_pattern = r'(?:supporting evidence|reasoning|support|evidence).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            evidence_match = re.search(evidence_pattern, hypothesis_content, re.IGNORECASE | re.DOTALL)
            if evidence_match:
                evidence = evidence_match.group(1).strip()
            
            # Extract plausibility
            plausibility = "medium"
            plausibility_pattern = r'(?:plausibility|likelihood).*?(low|medium|high)'
            plausibility_match = re.search(plausibility_pattern, hypothesis_content, re.IGNORECASE)
            if plausibility_match:
                plausibility = plausibility_match.group(1).lower()
            
            hypotheses.append({
                "statement": hypothesis_statement,
                "challenges_consensus": challenge,
                "supporting_evidence": evidence,
                "plausibility": plausibility
            })
        
        return hypotheses
    
    def _generate_hypotheses_from_challenges(self, consensus_views: List[Dict], challenges: Dict) -> List[Dict]:
        """
        Generate alternative hypotheses based on challenges to consensus views.
        
        Args:
            consensus_views: List of consensus views
            challenges: Dictionary of challenges to consensus views
            
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        # Generate hypotheses from alternative perspectives and contradictory evidence
        for view in consensus_views:
            view_statement = view.get("statement", "")
            view_challenges = challenges.get(view_statement, {})
            
            alternative_perspectives = view_challenges.get("alternative_perspectives", [])
            contradictory_evidence = view_challenges.get("contradictory_evidence", [])
            
            # Generate hypothesis from alternative perspective
            if alternative_perspectives:
                for perspective in alternative_perspectives[:1]:  # Take first perspective
                    # Create an alternative hypothesis by inverting or qualifying the consensus view
                    if "not" in view_statement or "no" in view_statement:
                        hypothesis_statement = view_statement.replace("not", "").replace("No", "").replace("no", "")
                    else:
                        hypothesis_statement = "It is not necessarily true that " + view_statement.lower()
                    
                    hypotheses.append({
                        "statement": hypothesis_statement,
                        "challenges_consensus": f"Challenges the consensus view that {view_statement}",
                        "supporting_evidence": perspective,
                        "plausibility": "medium"
                    })
            
            # Generate hypothesis from contradictory evidence
            if contradictory_evidence:
                for evidence in contradictory_evidence[:1]:  # Take first evidence
                    # Create a qualified hypothesis
                    hypothesis_statement = f"While partially valid, the view that {view_statement.lower()} requires significant qualification"
                    
                    hypotheses.append({
                        "statement": hypothesis_statement,
                        "challenges_consensus": f"Qualifies and limits the scope of the consensus view",
                        "supporting_evidence": evidence,
                        "plausibility": "medium"
                    })
        
        # If we don't have enough hypotheses, add generic ones
        if len(hypotheses) < 3:
            generic_hypotheses = [
                {
                    "statement": "The relationship between variables may be more complex than the linear model suggested by consensus views",
                    "challenges_consensus": "Challenges the simplistic causal models in conventional thinking",
                    "supporting_evidence": "Complex systems research suggests non-linear relationships and emergent properties",
                    "plausibility": "high"
                },
                {
                    "statement": "Contextual factors may be more important than universal principles in this domain",
                    "challenges_consensus": "Challenges the universalist assumptions in consensus views",
                    "supporting_evidence": "Comparative studies show significant variation across different contexts",
                    "plausibility": "medium"
                },
                {
                    "statement": "The direction of causality may be reversed from what is commonly assumed",
                    "challenges_consensus": "Inverts the standard causal explanation",
                    "supporting_evidence": "Some time-series analyses suggest different temporal relationships",
                    "plausibility": "low"
                }
            ]
            
            # Add generic hypotheses until we have at least 3
            for hypothesis in generic_hypotheses:
                if len(hypotheses) >= 3:
                    break
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _extract_findings(self, consensus_views: List[Dict], challenges: Dict, alternative_hypotheses: List[Dict]) -> List[Dict]:
        """
        Extract key findings from consensus challenge analysis.
        
        Args:
            consensus_views: List of consensus views
            challenges: Dictionary of challenges to consensus views
            alternative_hypotheses: List of alternative hypotheses
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about consensus views
        if consensus_views:
            findings.append({
                "finding": f"Analysis identified {len(consensus_views)} prevailing consensus views on the topic",
                "confidence": "high",
                "source": "consensus_challenge"
            })
        
        # Add finding about strongest consensus view
        strongest_view = None
        for view in consensus_views:
            if view.get("acceptance_level") == "dominant":
                strongest_view = view
                break
        
        if strongest_view:
            findings.append({
                "finding": f"The dominant consensus view is: {strongest_view.get('statement')}",
                "confidence": "high",
                "source": "consensus_challenge"
            })
        
        # Add finding about most significant challenge
        most_significant_challenge = None
        most_contradictory_evidence = 0
        
        for view_statement, view_challenges in challenges.items():
            contradictory_evidence = view_challenges.get("contradictory_evidence", [])
            if len(contradictory_evidence) > most_contradictory_evidence:
                most_contradictory_evidence = len(contradictory_evidence)
                most_significant_challenge = {
                    "view": view_statement,
                    "evidence": contradictory_evidence[0] if contradictory_evidence else ""
                }
        
        if most_significant_challenge:
            findings.append({
                "finding": f"The consensus view that '{most_significant_challenge['view']}' faces significant challenges based on contradictory evidence",
                "confidence": "medium",
                "source": "consensus_challenge"
            })
        
        # Add finding about most plausible alternative hypothesis
        most_plausible = None
        for hypothesis in alternative_hypotheses:
            if hypothesis.get("plausibility") == "high":
                most_plausible = hypothesis
                break
        
        if most_plausible:
            findings.append({
                "finding": f"A highly plausible alternative hypothesis is: {most_plausible.get('statement')}",
                "confidence": "medium",
                "source": "consensus_challenge"
            })
        
        return findings
    
    def _extract_assumptions(self, consensus_views: List[Dict]) -> List[Dict]:
        """
        Extract assumptions from consensus challenge analysis.
        
        Args:
            consensus_views: List of consensus views
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about consensus identification
        assumptions.append({
            "assumption": "The identified consensus views accurately represent prevailing expert opinion",
            "criticality": "high",
            "source": "consensus_challenge"
        })
        
        # Add assumption about consensus formation
        assumptions.append({
            "assumption": "Consensus views form through evidence-based reasoning rather than social dynamics alone",
            "criticality": "medium",
            "source": "consensus_challenge"
        })
        
        # Add assumption about alternative perspectives
        assumptions.append({
            "assumption": "Alternative perspectives exist that may provide valuable insights not captured by consensus",
            "criticality": "high",
            "source": "consensus_challenge"
        })
        
        return assumptions
    
    def _extract_uncertainties(self, consensus_views: List[Dict], challenges: Dict) -> List[Dict]:
        """
        Extract uncertainties from consensus challenge analysis.
        
        Args:
            consensus_views: List of consensus views
            challenges: Dictionary of challenges to consensus views
            
        Returns:
            List of uncertainties
        """
        uncertainties = []
        
        # Add uncertainty about consensus strength
        uncertainties.append({
            "uncertainty": "The actual strength and breadth of consensus among experts may differ from what is publicly visible",
            "impact": "high",
            "source": "consensus_challenge"
        })
        
        # Add uncertainties from underlying assumptions
        all_assumptions = []
        for view_statement, view_challenges in challenges.items():
            assumptions = view_challenges.get("underlying_assumptions", [])
            all_assumptions.extend(assumptions)
        
        # Select top uncertainties from assumptions
        seen = set()
        for assumption in all_assumptions:
            assumption_key = assumption.lower()
            if assumption_key not in seen and len(uncertainties) < 4:
                seen.add(assumption_key)
                uncertainties.append({
                    "uncertainty": f"Uncertainty regarding assumption: {assumption}",
                    "impact": "medium",
                    "source": "consensus_challenge"
                })
        
        # Add uncertainty about evidence quality
        uncertainties.append({
            "uncertainty": "The quality and comprehensiveness of evidence supporting consensus views may vary significantly",
            "impact": "high",
            "source": "consensus_challenge"
        })
        
        return uncertainties
