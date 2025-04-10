"""
Red Teaming Technique for challenging analysis from adversarial perspectives.
This module provides the RedTeamingTechnique class for comprehensive adversarial analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.analytical_technique import AnalyticalTechnique
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedTeamingTechnique(AnalyticalTechnique):
    """
    Red Teaming Technique for challenging analysis from adversarial perspectives.
    
    This technique provides capabilities for:
    1. Identifying potential blind spots and biases in analysis
    2. Challenging key assumptions from adversarial perspectives
    3. Exploring alternative interpretations of evidence
    4. Stress-testing conclusions against adversarial scenarios
    5. Identifying potential vulnerabilities in strategies
    """
    
    def __init__(self):
        """Initialize the Red Teaming Technique."""
        super().__init__(
            name="red_teaming",
            description="Challenges analysis from adversarial perspectives to identify blind spots and vulnerabilities",
            required_mcps=["llama4_scout", "research_mcp"],
            compatible_techniques=["key_assumptions_check", "analysis_of_competing_hypotheses", "premortem_analysis"],
            incompatible_techniques=[]
        )
        logger.info("Initialized RedTeamingTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Red Teaming Technique")
        
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
            
            # Get findings from previous techniques
            findings = self._collect_findings_from_context(context)
            
            # Get assumptions from previous techniques
            assumptions = self._collect_assumptions_from_context(context)
            
            # Define adversarial perspectives
            adversarial_perspectives = self._define_adversarial_perspectives(question, findings, assumptions, context, parameters)
            
            # Challenge assumptions from adversarial perspectives
            challenged_assumptions = self._challenge_assumptions(assumptions, adversarial_perspectives, context, parameters)
            
            # Challenge findings from adversarial perspectives
            challenged_findings = self._challenge_findings(findings, adversarial_perspectives, context, parameters)
            
            # Generate alternative interpretations
            alternative_interpretations = self._generate_alternative_interpretations(findings, assumptions, adversarial_perspectives, context, parameters)
            
            # Identify vulnerabilities
            vulnerabilities = self._identify_vulnerabilities(findings, assumptions, adversarial_perspectives, context, parameters)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(challenged_assumptions, challenged_findings, alternative_interpretations, vulnerabilities, context, parameters)
            
            # Compile results
            results = {
                "technique": "red_teaming",
                "timestamp": time.time(),
                "question": question,
                "adversarial_perspectives": adversarial_perspectives,
                "challenged_assumptions": challenged_assumptions,
                "challenged_findings": challenged_findings,
                "alternative_interpretations": alternative_interpretations,
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations,
                "findings": self._extract_findings(challenged_assumptions, challenged_findings, alternative_interpretations, vulnerabilities),
                "assumptions": self._extract_assumptions(challenged_assumptions)
            }
            
            # Add results to context
            context.add_technique_result("red_teaming", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Red Teaming Technique: {e}")
            return self.handle_error(e, context)
    
    def _collect_findings_from_context(self, context: AnalysisContext) -> List[Dict]:
        """
        Collect findings from previous techniques.
        
        Args:
            context: Analysis context
            
        Returns:
            List of findings
        """
        logger.info("Collecting findings from context")
        
        # Initialize findings list
        findings = []
        
        # Get all technique results from context
        technique_results = context.get_all_technique_results()
        
        # Extract findings from each technique result
        for technique_name, result in technique_results.items():
            if "findings" in result and isinstance(result["findings"], list):
                for finding in result["findings"]:
                    # Add source information if not present
                    if "source" not in finding:
                        finding["source"] = technique_name
                    
                    findings.append(finding)
        
        return findings
    
    def _collect_assumptions_from_context(self, context: AnalysisContext) -> List[Dict]:
        """
        Collect assumptions from previous techniques.
        
        Args:
            context: Analysis context
            
        Returns:
            List of assumptions
        """
        logger.info("Collecting assumptions from context")
        
        # Initialize assumptions list
        assumptions = []
        
        # Get all technique results from context
        technique_results = context.get_all_technique_results()
        
        # Extract assumptions from each technique result
        for technique_name, result in technique_results.items():
            if "assumptions" in result and isinstance(result["assumptions"], list):
                for assumption in result["assumptions"]:
                    # Add source information if not present
                    if "source" not in assumption:
                        assumption["source"] = technique_name
                    
                    assumptions.append(assumption)
        
        return assumptions
    
    def _define_adversarial_perspectives(self, question: str, findings: List[Dict], assumptions: List[Dict], context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Define adversarial perspectives for red teaming.
        
        Args:
            question: The analytical question
            findings: List of findings
            assumptions: List of assumptions
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of adversarial perspectives
        """
        logger.info("Defining adversarial perspectives")
        
        # Check if adversarial perspectives are provided in parameters
        if "adversarial_perspectives" in parameters and isinstance(parameters["adversarial_perspectives"], list):
            return parameters["adversarial_perspectives"]
        
        # Use Llama4ScoutMCP to define adversarial perspectives
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Format findings and assumptions for prompt
            findings_text = ""
            for i, finding in enumerate(findings[:5]):  # Limit to top 5 findings
                finding_text = finding.get("finding", "")
                findings_text += f"{i+1}. {finding_text}\n"
            
            assumptions_text = ""
            for i, assumption in enumerate(assumptions[:5]):  # Limit to top 5 assumptions
                assumption_text = assumption.get("assumption", "")
                assumptions_text += f"{i+1}. {assumption_text}\n"
            
            # Create prompt for adversarial perspectives
            prompt = f"""
            Define 4-6 distinct adversarial perspectives to challenge the following analysis on this question:
            
            Question: {question}
            
            Key Findings:
            {findings_text}
            
            Key Assumptions:
            {assumptions_text}
            
            For each adversarial perspective:
            1. Provide a clear name/identity for the perspective
            2. Describe the core worldview or mental model of this perspective
            3. Explain key biases or priorities that would shape this perspective's analysis
            4. Identify what this perspective would be most likely to challenge in the analysis
            
            Create diverse perspectives that:
            - Represent fundamentally different worldviews or mental models
            - Challenge different aspects of the analysis
            - Include both conventional and unconventional perspectives
            - Are grounded in realistic alternative viewpoints
            
            Each perspective should be coherent, realistic, and genuinely challenging to the analysis.
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "adversarial_perspectives",
                "context": {"prompt": grounded_prompt}
            })
            
            # Extract adversarial perspectives from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse adversarial perspectives from content
                perspectives = self._parse_perspectives_from_text(content)
                if perspectives:
                    return perspectives
        
        # Fallback: Generate generic adversarial perspectives
        return self._generate_generic_adversarial_perspectives(question)
    
    def _parse_perspectives_from_text(self, text: str) -> List[Dict]:
        """
        Parse adversarial perspectives from text.
        
        Args:
            text: Text containing perspective descriptions
            
        Returns:
            List of parsed perspectives
        """
        perspectives = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for perspective sections
        perspective_pattern = r'(?:^|\n)(?:Perspective|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Perspective|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        perspective_matches = re.findall(perspective_pattern, text, re.DOTALL)
        
        if not perspective_matches:
            # Try alternative pattern for numbered lists
            perspective_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            perspective_matches = re.findall(perspective_pattern, text, re.DOTALL)
            
            if perspective_matches:
                # Convert to expected format
                perspective_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(perspective_matches)]
        
        for match in perspective_matches:
            perspective_num = match[0].strip() if len(match) > 0 else ""
            perspective_name = match[1].strip() if len(match) > 1 else ""
            perspective_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract worldview
            worldview = ""
            worldview_pattern = r'(?:worldview|mental model|core belief|perspective).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            worldview_match = re.search(worldview_pattern, perspective_content, re.IGNORECASE | re.DOTALL)
            if worldview_match:
                worldview = worldview_match.group(1).strip()
            else:
                # Use first paragraph as worldview
                paragraphs = perspective_content.split('\n\n')
                if paragraphs:
                    worldview = paragraphs[0].strip()
            
            # Extract biases
            biases = ""
            biases_pattern = r'(?:biases|priorities|focus|emphasis).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            biases_match = re.search(biases_pattern, perspective_content, re.IGNORECASE | re.DOTALL)
            if biases_match:
                biases = biases_match.group(1).strip()
            
            # Extract challenges
            challenges = ""
            challenges_pattern = r'(?:challenge|question|criticize|dispute).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            challenges_match = re.search(challenges_pattern, perspective_content, re.IGNORECASE | re.DOTALL)
            if challenges_match:
                challenges = challenges_match.group(1).strip()
            
            perspectives.append({
                "name": perspective_name,
                "worldview": worldview,
                "biases": biases,
                "challenges": challenges
            })
        
        return perspectives
    
    def _generate_generic_adversarial_perspectives(self, question: str) -> List[Dict]:
        """
        Generate generic adversarial perspectives based on the question.
        
        Args:
            question: The analytical question
            
        Returns:
            List of generic adversarial perspectives
        """
        # Extract domain from question
        domain = self._extract_domain_from_question(question)
        
        # Domain-specific adversarial perspectives
        domain_perspectives = {
            "economic": [
                {
                    "name": "Market Fundamentalist",
                    "worldview": "Markets are inherently efficient and self-correcting. Government intervention almost always creates more problems than it solves.",
                    "biases": "Overemphasizes market efficiency, undervalues externalities and market failures, skeptical of regulatory solutions.",
                    "challenges": "Will challenge any analysis that suggests market failures require intervention or that downplays market self-correction mechanisms."
                },
                {
                    "name": "Economic Nationalist",
                    "worldview": "National economic interests should be prioritized over global efficiency. Economic policy should protect domestic industries and workers.",
                    "biases": "Prioritizes national economic sovereignty, suspicious of globalization, emphasizes short-term domestic impacts over long-term global efficiency.",
                    "challenges": "Will challenge analyses that assume free trade is optimal or that downplay national economic security concerns."
                },
                {
                    "name": "Structural Critic",
                    "worldview": "Economic outcomes are primarily determined by power structures and institutional arrangements, not individual choices or market forces.",
                    "biases": "Focuses on power imbalances and structural inequalities, skeptical of market-based solutions, emphasizes historical context.",
                    "challenges": "Will challenge analyses that attribute outcomes to individual choices or market forces rather than structural factors."
                },
                {
                    "name": "Technological Disruptor",
                    "worldview": "Technological change is the primary driver of economic transformation, making traditional economic models increasingly obsolete.",
                    "biases": "Overemphasizes technological disruption, dismisses historical patterns, focuses on discontinuities rather than continuities.",
                    "challenges": "Will challenge analyses that rely on historical patterns or that underestimate the transformative impact of new technologies."
                },
                {
                    "name": "Ecological Economist",
                    "worldview": "The economy is a subsystem of the finite biosphere, and perpetual growth is impossible within planetary boundaries.",
                    "biases": "Prioritizes ecological sustainability, skeptical of growth-oriented economics, emphasizes long-term resource constraints.",
                    "challenges": "Will challenge analyses that assume continued economic growth is possible or desirable without addressing ecological limits."
                }
            ],
            "political": [
                {
                    "name": "Realpolitik Strategist",
                    "worldview": "Politics is fundamentally about power and national interest, not values or norms. International relations is a zero-sum competition.",
                    "biases": "Emphasizes power dynamics and security concerns, dismisses normative considerations, focuses on relative rather than absolute gains.",
                    "challenges": "Will challenge analyses that emphasize cooperation, shared values, or international norms over power politics."
                },
                {
                    "name": "Institutional Formalist",
                    "worldview": "Political outcomes are primarily determined by formal institutional structures and rules, which create predictable incentives and constraints.",
                    "biases": "Overemphasizes formal rules and procedures, undervalues informal norms and cultural factors, focuses on institutional design.",
                    "challenges": "Will challenge analyses that attribute outcomes to individual leaders or cultural factors rather than institutional incentives."
                },
                {
                    "name": "Popular Sovereignty Advocate",
                    "worldview": "Political legitimacy derives exclusively from the will of the people, and elite consensus often diverges from popular preferences.",
                    "biases": "Prioritizes popular opinion over expert judgment, suspicious of technocratic solutions, emphasizes democratic accountability.",
                    "challenges": "Will challenge analyses that prioritize expert consensus or that dismiss populist movements as irrational."
                },
                {
                    "name": "Historical Determinist",
                    "worldview": "Political developments follow predictable historical patterns shaped by material conditions and structural forces.",
                    "biases": "Emphasizes historical continuity and structural factors, downplays individual agency and contingency, focuses on long-term trends.",
                    "challenges": "Will challenge analyses that emphasize individual leadership or that treat current developments as unprecedented."
                },
                {
                    "name": "Cultural Identity Theorist",
                    "worldview": "Political behavior is primarily driven by cultural identity and values, not material interests or rational calculation.",
                    "biases": "Prioritizes cultural and identity factors, skeptical of interest-based explanations, emphasizes symbolic politics.",
                    "challenges": "Will challenge analyses that focus on material interests or rational choice explanations rather than identity and values."
                }
            ],
            "technological": [
                {
                    "name": "Technological Determinist",
                    "worldview": "Technology develops according to its own internal logic, largely independent of social choices, and society must adapt to technological imperatives.",
                    "biases": "Overemphasizes technological momentum, undervalues social shaping of technology, focuses on adaptation rather than governance.",
                    "challenges": "Will challenge analyses that suggest technology can be effectively governed or directed through social choice."
                },
                {
                    "name": "Precautionary Advocate",
                    "worldview": "New technologies pose significant risks that are often underestimated, and caution should be the default approach to technological innovation.",
                    "biases": "Emphasizes potential harms and unintended consequences, prioritizes risk avoidance over potential benefits, focuses on worst-case scenarios.",
                    "challenges": "Will challenge analyses that emphasize benefits of new technologies or that downplay potential risks and unintended consequences."
                },
                {
                    "name": "Innovation Optimist",
                    "worldview": "Technological innovation is the primary driver of human progress, and barriers to innovation should be minimized.",
                    "biases": "Overemphasizes benefits of innovation, undervalues transition costs and disruption, focuses on long-term gains over short-term impacts.",
                    "challenges": "Will challenge analyses that emphasize risks or transition costs of new technologies or that suggest slowing the pace of innovation."
                },
                {
                    "name": "Digital Rights Advocate",
                    "worldview": "Digital technologies are reshaping power relationships, often in ways that threaten individual rights and democratic values.",
                    "biases": "Prioritizes privacy and individual autonomy, suspicious of surveillance and centralized control, emphasizes power asymmetries.",
                    "challenges": "Will challenge analyses that prioritize efficiency or security over privacy and autonomy or that downplay power implications of technology."
                },
                {
                    "name": "Technological Pragmatist",
                    "worldview": "Technologies are tools that can be used for multiple purposes, and their impacts depend primarily on how they are deployed and governed.",
                    "biases": "Emphasizes implementation and governance over inherent properties of technologies, focuses on specific use cases rather than general impacts.",
                    "challenges": "Will challenge analyses that attribute inherent properties or impacts to technologies rather than focusing on specific applications and governance."
                }
            ],
            "social": [
                {
                    "name": "Social Traditionalist",
                    "worldview": "Traditional social institutions and norms evolved for good reasons and should be preserved. Rapid social change often has unintended negative consequences.",
                    "biases": "Emphasizes stability and continuity, suspicious of rapid social change, prioritizes traditional institutions and values.",
                    "challenges": "Will challenge analyses that present social change as inherently progressive or that dismiss concerns about social disruption."
                },
                {
                    "name": "Social Justice Advocate",
                    "worldview": "Social outcomes are shaped by systemic inequalities and power imbalances that require active intervention to address.",
                    "biases": "Focuses on structural inequalities and marginalized perspectives, emphasizes historical injustices, prioritizes equity over formal equality.",
                    "challenges": "Will challenge analyses that ignore power differentials or that attribute disparate outcomes to individual choices rather than structural factors."
                },
                {
                    "name": "Individualist",
                    "worldview": "Social outcomes primarily reflect individual choices and responsibilities, not structural factors or collective phenomena.",
                    "biases": "Emphasizes individual agency and responsibility, skeptical of structural explanations, focuses on incentives and personal choice.",
                    "challenges": "Will challenge analyses that emphasize structural factors or collective phenomena over individual agency and choice."
                },
                {
                    "name": "Communitarian",
                    "worldview": "Social cohesion and shared values are essential for well-functioning societies, and excessive individualism undermines social fabric.",
                    "biases": "Prioritizes community bonds and social cohesion, concerned about fragmentation and polarization, emphasizes shared values and identity.",
                    "challenges": "Will challenge analyses that prioritize individual autonomy over social cohesion or that ignore the importance of shared values."
                },
                {
                    "name": "Generational Theorist",
                    "worldview": "Social change is driven by generational replacement, with each generation shaped by formative experiences that create distinct values and priorities.",
                    "biases": "Emphasizes generational differences, focuses on cohort effects rather than life-cycle or period effects, prioritizes youth perspectives.",
                    "challenges": "Will challenge analyses that ignore generational differences or that assume current trends will continue as younger generations age."
                }
            ],
            "general": [
                {
                    "name": "Status Quo Defender",
                    "worldview": "Current systems and arrangements exist for good reasons and have proven their resilience. Change should be incremental and carefully considered.",
                    "biases": "Emphasizes stability and continuity, risk-averse, prioritizes known trade-offs over uncertain benefits of change.",
                    "challenges": "Will challenge analyses that advocate significant changes to existing systems or that underestimate transition costs and risks."
                },
                {
                    "name": "Radical Reformer",
                    "worldview": "Current systems are fundamentally flawed and require transformative change rather than incremental reform.",
                    "biases": "Emphasizes structural problems with status quo, dismisses incremental solutions, focuses on transformative possibilities.",
                    "challenges": "Will challenge analyses that propose incremental reforms or that assume current systems can be effectively improved without fundamental change."
                },
                {
                    "name": "Complexity Theorist",
                    "worldview": "Complex systems behave in non-linear ways that defy simple causal explanations or predictions.",
                    "biases": "Emphasizes emergent properties and feedback loops, skeptical of linear causal explanations, focuses on system dynamics rather than individual components.",
                    "challenges": "Will challenge analyses that rely on simple causal models or that make confident predictions about complex system behavior."
                },
                {
                    "name": "Contrarian Skeptic",
                    "worldview": "Conventional wisdom and expert consensus are often wrong, especially when they align with dominant interests or narratives.",
                    "biases": "Suspicious of consensus views, emphasizes historical examples of expert failure, prioritizes alternative explanations and minority viewpoints.",
                    "challenges": "Will challenge analyses that rely heavily on expert consensus or conventional wisdom without considering alternative perspectives."
                },
                {
                    "name": "Long-Term Futurist",
                    "worldview": "Current decisions should prioritize long-term impacts over short-term considerations, which are typically overweighted in decision-making.",
                    "biases": "Emphasizes long-term consequences, dismisses short-term concerns, focuses on intergenerational impacts and existential considerations.",
                    "challenges": "Will challenge analyses that prioritize short-term impacts or that fail to consider very long-term consequences of current decisions."
                }
            ]
        }
        
        # Return perspectives for the identified domain
        if domain in domain_perspectives:
            return domain_perspectives[domain]
        else:
            return domain_perspectives["general"]
    
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
    
    def _challenge_assumptions(self, assumptions: List[Dict], adversarial_perspectives: List[Dict], context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Challenge assumptions from adversarial perspectives.
        
        Args:
            assumptions: List of assumptions
            adversarial_perspectives: List of adversarial perspectives
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of challenged assumptions
        """
        logger.info("Challenging assumptions")
        
        # Initialize challenged assumptions
        challenged_assumptions = []
        
        # Use Llama4ScoutMCP to challenge assumptions
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout and assumptions:
            # Prioritize assumptions by criticality
            prioritized_assumptions = sorted(
                assumptions,
                key=lambda x: 0 if x.get("criticality", "medium") == "high" else (1 if x.get("criticality", "medium") == "medium" else 2)
            )
            
            # Limit to top 5 assumptions
            top_assumptions = prioritized_assumptions[:5]
            
            # Format assumptions for prompt
            assumptions_text = ""
            for i, assumption in enumerate(top_assumptions):
                assumption_text = assumption.get("assumption", "")
                criticality = assumption.get("criticality", "medium")
                assumptions_text += f"{i+1}. {assumption_text} (Criticality: {criticality})\n"
            
            # Format perspectives for prompt
            perspectives_text = ""
            for i, perspective in enumerate(adversarial_perspectives[:3]):  # Limit to top 3 perspectives
                perspective_name = perspective.get("name", "")
                perspective_worldview = perspective.get("worldview", "")
                perspectives_text += f"{i+1}. {perspective_name}: {perspective_worldview}\n"
            
            # Create prompt for challenging assumptions
            prompt = f"""
            Challenge the following key assumptions from multiple adversarial perspectives.
            
            Key Assumptions:
            {assumptions_text}
            
            Adversarial Perspectives:
            {perspectives_text}
            
            For each assumption, provide:
            1. The original assumption
            2. Challenges from each adversarial perspective
            3. Alternative assumptions that could replace the original
            4. Implications if the original assumption is wrong
            
            Focus on substantive challenges that:
            - Identify specific weaknesses or blind spots in the assumption
            - Provide concrete alternative viewpoints
            - Are grounded in the worldview of each adversarial perspective
            - Highlight meaningful implications if the assumption is wrong
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "How might these assumptions be challenged?",
                "analysis_type": "assumption_challenge",
                "context": {"prompt": prompt}
            })
            
            # Extract challenged assumptions from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse challenged assumptions from content
                challenged_assumptions = self._parse_challenged_assumptions(content, top_assumptions, adversarial_perspectives)
                if challenged_assumptions:
                    return challenged_assumptions
        
        # Fallback: Generate generic challenges to assumptions
        return self._generate_generic_assumption_challenges(assumptions, adversarial_perspectives)
    
    def _parse_challenged_assumptions(self, text: str, assumptions: List[Dict], perspectives: List[Dict]) -> List[Dict]:
        """
        Parse challenged assumptions from text.
        
        Args:
            text: Text containing challenged assumptions
            assumptions: List of original assumptions
            perspectives: List of adversarial perspectives
            
        Returns:
            List of challenged assumptions
        """
        challenged_assumptions = []
        
        # Simple parsing based on patterns
        import re
        
        # Process each assumption
        for i, assumption in enumerate(assumptions):
            original_assumption = assumption.get("assumption", "")
            
            # Look for section about this assumption
            assumption_pattern = f"(?:Assumption|#)\s*{i+1}.*?(?:\n\n|\Z)"
            assumption_match = re.search(assumption_pattern, text, re.DOTALL)
            
            if not assumption_match:
                # Try alternative pattern
                assumption_pattern = f"{i+1}\.\s*.*?{re.escape(original_assumption[:30])}.*?(?:\n\n|\Z)"
                assumption_match = re.search(assumption_pattern, text, re.DOTALL)
            
            if assumption_match:
                assumption_text = assumption_match.group(0)
                
                # Extract challenges
                challenges = []
                
                for perspective in perspectives:
                    perspective_name = perspective.get("name", "")
                    
                    # Look for challenge from this perspective
                    perspective_pattern = f"{perspective_name}.*?(?::|$)(.*?)(?=(?:{perspectives[0].get('name', '')}|\n\n|\Z))"
                    perspective_match = re.search(perspective_pattern, assumption_text, re.IGNORECASE | re.DOTALL)
                    
                    if perspective_match:
                        challenge_text = perspective_match.group(1).strip()
                        challenges.append({
                            "perspective": perspective_name,
                            "challenge": challenge_text
                        })
                
                # Extract alternative assumptions
                alternatives = []
                alternatives_pattern = r"(?:Alternative|Alternatives).*?(?::|$)(.*?)(?=(?:Implications|\n\n|\Z))"
                alternatives_match = re.search(alternatives_pattern, assumption_text, re.IGNORECASE | re.DOTALL)
                
                if alternatives_match:
                    alternatives_text = alternatives_match.group(1).strip()
                    
                    # Split into individual alternatives
                    alternative_items = re.findall(r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))', alternatives_text, re.DOTALL)
                    
                    if alternative_items:
                        alternatives = [item.strip() for item in alternative_items if item.strip()]
                    else:
                        # If no bullet points, use the whole text
                        alternatives = [alternatives_text]
                
                # Extract implications
                implications = ""
                implications_pattern = r"(?:Implications|If wrong).*?(?::|$)(.*?)(?=(?:\n\n|\Z))"
                implications_match = re.search(implications_pattern, assumption_text, re.IGNORECASE | re.DOTALL)
                
                if implications_match:
                    implications = implications_match.group(1).strip()
                
                challenged_assumptions.append({
                    "original_assumption": original_assumption,
                    "criticality": assumption.get("criticality", "medium"),
                    "source": assumption.get("source", "unknown"),
                    "challenges": challenges,
                    "alternative_assumptions": alternatives,
                    "implications_if_wrong": implications
                })
            else:
                # If no match found, add the original assumption with generic challenges
                challenged_assumptions.append({
                    "original_assumption": original_assumption,
                    "criticality": assumption.get("criticality", "medium"),
                    "source": assumption.get("source", "unknown"),
                    "challenges": self._generate_generic_challenges_for_assumption(assumption, perspectives),
                    "alternative_assumptions": ["The assumption may be partially or completely incorrect."],
                    "implications_if_wrong": "Analysis conclusions may be significantly affected if this assumption is incorrect."
                })
        
        return challenged_assumptions
    
    def _generate_generic_challenges_for_assumption(self, assumption: Dict, perspectives: List[Dict]) -> List[Dict]:
        """
        Generate generic challenges for an assumption.
        
        Args:
            assumption: Assumption dictionary
            perspectives: List of adversarial perspectives
            
        Returns:
            List of generic challenges
        """
        challenges = []
        
        assumption_text = assumption.get("assumption", "")
        
        for perspective in perspectives[:3]:  # Limit to top 3 perspectives
            perspective_name = perspective.get("name", "")
            perspective_worldview = perspective.get("worldview", "")
            
            # Generate a generic challenge based on the perspective
            challenge = f"This assumption may not hold true when considering that {perspective_worldview}"
            
            challenges.append({
                "perspective": perspective_name,
                "challenge": challenge
            })
        
        return challenges
    
    def _generate_generic_assumption_challenges(self, assumptions: List[Dict], adversarial_perspectives: List[Dict]) -> List[Dict]:
        """
        Generate generic challenges to assumptions.
        
        Args:
            assumptions: List of assumptions
            adversarial_perspectives: List of adversarial perspectives
            
        Returns:
            List of challenged assumptions
        """
        challenged_assumptions = []
        
        # Prioritize assumptions by criticality
        prioritized_assumptions = sorted(
            assumptions,
            key=lambda x: 0 if x.get("criticality", "medium") == "high" else (1 if x.get("criticality", "medium") == "medium" else 2)
        )
        
        # Limit to top 5 assumptions
        top_assumptions = prioritized_assumptions[:5]
        
        for assumption in top_assumptions:
            original_assumption = assumption.get("assumption", "")
            criticality = assumption.get("criticality", "medium")
            source = assumption.get("source", "unknown")
            
            # Generate challenges from each perspective
            challenges = self._generate_generic_challenges_for_assumption(assumption, adversarial_perspectives)
            
            # Generate generic alternative assumptions
            alternative_assumptions = [
                f"The opposite of the original assumption may be true.",
                f"The assumption may be true only under specific conditions that are not currently met.",
                f"The assumption may be partially true but missing important nuances."
            ]
            
            # Generate generic implications
            implications = f"If this {criticality}-criticality assumption is wrong, the analysis conclusions may be significantly affected, particularly those derived from the {source} technique."
            
            challenged_assumptions.append({
                "original_assumption": original_assumption,
                "criticality": criticality,
                "source": source,
                "challenges": challenges,
                "alternative_assumptions": alternative_assumptions,
                "implications_if_wrong": implications
            })
        
        return challenged_assumptions
    
    def _challenge_findings(self, findings: List[Dict], adversarial_perspectives: List[Dict], context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Challenge findings from adversarial perspectives.
        
        Args:
            findings: List of findings
            adversarial_perspectives: List of adversarial perspectives
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of challenged findings
        """
        logger.info("Challenging findings")
        
        # Initialize challenged findings
        challenged_findings = []
        
        # Use Llama4ScoutMCP to challenge findings
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout and findings:
            # Prioritize findings by confidence
            prioritized_findings = sorted(
                findings,
                key=lambda x: 0 if x.get("confidence", "medium") == "high" else (1 if x.get("confidence", "medium") == "medium" else 2),
                reverse=True
            )
            
            # Limit to top 5 findings
            top_findings = prioritized_findings[:5]
            
            # Format findings for prompt
            findings_text = ""
            for i, finding in enumerate(top_findings):
                finding_text = finding.get("finding", "")
                confidence = finding.get("confidence", "medium")
                findings_text += f"{i+1}. {finding_text} (Confidence: {confidence})\n"
            
            # Format perspectives for prompt
            perspectives_text = ""
            for i, perspective in enumerate(adversarial_perspectives[:3]):  # Limit to top 3 perspectives
                perspective_name = perspective.get("name", "")
                perspective_worldview = perspective.get("worldview", "")
                perspectives_text += f"{i+1}. {perspective_name}: {perspective_worldview}\n"
            
            # Create prompt for challenging findings
            prompt = f"""
            Challenge the following key findings from multiple adversarial perspectives.
            
            Key Findings:
            {findings_text}
            
            Adversarial Perspectives:
            {perspectives_text}
            
            For each finding, provide:
            1. The original finding
            2. Challenges from each adversarial perspective
            3. Alternative interpretations of the evidence
            4. Potential weaknesses in the finding's logic or evidence
            
            Focus on substantive challenges that:
            - Identify specific weaknesses or blind spots in the finding
            - Provide concrete alternative interpretations
            - Are grounded in the worldview of each adversarial perspective
            - Highlight meaningful implications if the finding is wrong
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "How might these findings be challenged?",
                "analysis_type": "finding_challenge",
                "context": {"prompt": prompt}
            })
            
            # Extract challenged findings from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse challenged findings from content
                challenged_findings = self._parse_challenged_findings(content, top_findings, adversarial_perspectives)
                if challenged_findings:
                    return challenged_findings
        
        # Fallback: Generate generic challenges to findings
        return self._generate_generic_finding_challenges(findings, adversarial_perspectives)
    
    def _parse_challenged_findings(self, text: str, findings: List[Dict], perspectives: List[Dict]) -> List[Dict]:
        """
        Parse challenged findings from text.
        
        Args:
            text: Text containing challenged findings
            findings: List of original findings
            perspectives: List of adversarial perspectives
            
        Returns:
            List of challenged findings
        """
        challenged_findings = []
        
        # Simple parsing based on patterns
        import re
        
        # Process each finding
        for i, finding in enumerate(findings):
            original_finding = finding.get("finding", "")
            
            # Look for section about this finding
            finding_pattern = f"(?:Finding|#)\s*{i+1}.*?(?:\n\n|\Z)"
            finding_match = re.search(finding_pattern, text, re.DOTALL)
            
            if not finding_match:
                # Try alternative pattern
                finding_pattern = f"{i+1}\.\s*.*?{re.escape(original_finding[:30])}.*?(?:\n\n|\Z)"
                finding_match = re.search(finding_pattern, text, re.DOTALL)
            
            if finding_match:
                finding_text = finding_match.group(0)
                
                # Extract challenges
                challenges = []
                
                for perspective in perspectives:
                    perspective_name = perspective.get("name", "")
                    
                    # Look for challenge from this perspective
                    perspective_pattern = f"{perspective_name}.*?(?::|$)(.*?)(?=(?:{perspectives[0].get('name', '')}|\n\n|\Z))"
                    perspective_match = re.search(perspective_pattern, finding_text, re.IGNORECASE | re.DOTALL)
                    
                    if perspective_match:
                        challenge_text = perspective_match.group(1).strip()
                        challenges.append({
                            "perspective": perspective_name,
                            "challenge": challenge_text
                        })
                
                # Extract alternative interpretations
                alternatives = []
                alternatives_pattern = r"(?:Alternative|Alternatives).*?(?::|$)(.*?)(?=(?:Weaknesses|\n\n|\Z))"
                alternatives_match = re.search(alternatives_pattern, finding_text, re.IGNORECASE | re.DOTALL)
                
                if alternatives_match:
                    alternatives_text = alternatives_match.group(1).strip()
                    
                    # Split into individual alternatives
                    alternative_items = re.findall(r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))', alternatives_text, re.DOTALL)
                    
                    if alternative_items:
                        alternatives = [item.strip() for item in alternative_items if item.strip()]
                    else:
                        # If no bullet points, use the whole text
                        alternatives = [alternatives_text]
                
                # Extract weaknesses
                weaknesses = []
                weaknesses_pattern = r"(?:Weaknesses|Limitations).*?(?::|$)(.*?)(?=(?:\n\n|\Z))"
                weaknesses_match = re.search(weaknesses_pattern, finding_text, re.IGNORECASE | re.DOTALL)
                
                if weaknesses_match:
                    weaknesses_text = weaknesses_match.group(1).strip()
                    
                    # Split into individual weaknesses
                    weakness_items = re.findall(r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))', weaknesses_text, re.DOTALL)
                    
                    if weakness_items:
                        weaknesses = [item.strip() for item in weakness_items if item.strip()]
                    else:
                        # If no bullet points, use the whole text
                        weaknesses = [weaknesses_text]
                
                challenged_findings.append({
                    "original_finding": original_finding,
                    "confidence": finding.get("confidence", "medium"),
                    "source": finding.get("source", "unknown"),
                    "challenges": challenges,
                    "alternative_interpretations": alternatives,
                    "weaknesses": weaknesses
                })
            else:
                # If no match found, add the original finding with generic challenges
                challenged_findings.append({
                    "original_finding": original_finding,
                    "confidence": finding.get("confidence", "medium"),
                    "source": finding.get("source", "unknown"),
                    "challenges": self._generate_generic_challenges_for_finding(finding, perspectives),
                    "alternative_interpretations": ["The evidence may support alternative conclusions."],
                    "weaknesses": ["The finding may rely on incomplete evidence or questionable assumptions."]
                })
        
        return challenged_findings
    
    def _generate_generic_challenges_for_finding(self, finding: Dict, perspectives: List[Dict]) -> List[Dict]:
        """
        Generate generic challenges for a finding.
        
        Args:
            finding: Finding dictionary
            perspectives: List of adversarial perspectives
            
        Returns:
            List of generic challenges
        """
        challenges = []
        
        finding_text = finding.get("finding", "")
        
        for perspective in perspectives[:3]:  # Limit to top 3 perspectives
            perspective_name = perspective.get("name", "")
            perspective_worldview = perspective.get("worldview", "")
            
            # Generate a generic challenge based on the perspective
            challenge = f"This finding may be questioned when considering that {perspective_worldview}"
            
            challenges.append({
                "perspective": perspective_name,
                "challenge": challenge
            })
        
        return challenges
    
    def _generate_generic_finding_challenges(self, findings: List[Dict], adversarial_perspectives: List[Dict]) -> List[Dict]:
        """
        Generate generic challenges to findings.
        
        Args:
            findings: List of findings
            adversarial_perspectives: List of adversarial perspectives
            
        Returns:
            List of challenged findings
        """
        challenged_findings = []
        
        # Prioritize findings by confidence
        prioritized_findings = sorted(
            findings,
            key=lambda x: 0 if x.get("confidence", "medium") == "high" else (1 if x.get("confidence", "medium") == "medium" else 2),
            reverse=True
        )
        
        # Limit to top 5 findings
        top_findings = prioritized_findings[:5]
        
        for finding in top_findings:
            original_finding = finding.get("finding", "")
            confidence = finding.get("confidence", "medium")
            source = finding.get("source", "unknown")
            
            # Generate challenges from each perspective
            challenges = self._generate_generic_challenges_for_finding(finding, adversarial_perspectives)
            
            # Generate generic alternative interpretations
            alternative_interpretations = [
                "The evidence may support the opposite conclusion.",
                "The finding may be true only under specific conditions that are not generally applicable.",
                "The finding may be partially true but missing important nuances."
            ]
            
            # Generate generic weaknesses
            weaknesses = [
                "The finding may rely on incomplete or biased evidence.",
                "The finding may not adequately account for alternative explanations.",
                "The finding may overstate the certainty of the conclusion given the available evidence."
            ]
            
            challenged_findings.append({
                "original_finding": original_finding,
                "confidence": confidence,
                "source": source,
                "challenges": challenges,
                "alternative_interpretations": alternative_interpretations,
                "weaknesses": weaknesses
            })
        
        return challenged_findings
    
    def _generate_alternative_interpretations(self, findings: List[Dict], assumptions: List[Dict], adversarial_perspectives: List[Dict], context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Generate alternative interpretations of evidence.
        
        Args:
            findings: List of findings
            assumptions: List of assumptions
            adversarial_perspectives: List of adversarial perspectives
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of alternative interpretations
        """
        logger.info("Generating alternative interpretations")
        
        # Initialize alternative interpretations
        alternative_interpretations = []
        
        # Use Llama4ScoutMCP to generate alternative interpretations
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout and findings:
            # Get question from context
            question = context.get("question", "")
            
            # Format findings for prompt
            findings_text = ""
            for i, finding in enumerate(findings[:3]):  # Limit to top 3 findings
                finding_text = finding.get("finding", "")
                findings_text += f"{i+1}. {finding_text}\n"
            
            # Format perspectives for prompt
            perspectives_text = ""
            for i, perspective in enumerate(adversarial_perspectives[:3]):  # Limit to top 3 perspectives
                perspective_name = perspective.get("name", "")
                perspective_worldview = perspective.get("worldview", "")
                perspectives_text += f"{i+1}. {perspective_name}: {perspective_worldview}\n"
            
            # Create prompt for alternative interpretations
            prompt = f"""
            Generate alternative interpretations of the evidence for the following question from multiple adversarial perspectives.
            
            Question: {question}
            
            Key Findings:
            {findings_text}
            
            Adversarial Perspectives:
            {perspectives_text}
            
            For each adversarial perspective, provide:
            1. An alternative overall interpretation of the evidence
            2. Key evidence that supports this alternative interpretation
            3. Implications of this alternative interpretation
            
            Focus on substantive alternatives that:
            - Provide coherent alternative explanations for the available evidence
            - Are grounded in the worldview of each adversarial perspective
            - Highlight meaningful implications if the alternative is correct
            - Challenge the dominant narrative in significant ways
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "What alternative interpretations might explain the evidence?",
                "analysis_type": "alternative_interpretations",
                "context": {"prompt": prompt}
            })
            
            # Extract alternative interpretations from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse alternative interpretations from content
                alternative_interpretations = self._parse_alternative_interpretations(content, adversarial_perspectives)
                if alternative_interpretations:
                    return alternative_interpretations
        
        # Fallback: Generate generic alternative interpretations
        return self._generate_generic_alternative_interpretations(findings, adversarial_perspectives)
    
    def _parse_alternative_interpretations(self, text: str, perspectives: List[Dict]) -> List[Dict]:
        """
        Parse alternative interpretations from text.
        
        Args:
            text: Text containing alternative interpretations
            perspectives: List of adversarial perspectives
            
        Returns:
            List of alternative interpretations
        """
        alternative_interpretations = []
        
        # Simple parsing based on patterns
        import re
        
        # Process each perspective
        for perspective in perspectives[:3]:  # Limit to top 3 perspectives
            perspective_name = perspective.get("name", "")
            
            # Look for section about this perspective
            perspective_pattern = f"(?:Perspective|#)\s*{perspective_name}.*?(?:\n\n|\Z)"
            perspective_match = re.search(perspective_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if not perspective_match:
                # Try alternative pattern
                perspective_pattern = f"{perspective_name}.*?(?:\n\n|\Z)"
                perspective_match = re.search(perspective_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if perspective_match:
                perspective_text = perspective_match.group(0)
                
                # Extract interpretation
                interpretation = ""
                interpretation_pattern = r"(?:Interpretation|Alternative).*?(?::|$)(.*?)(?=(?:Evidence|Key evidence|\n\n|\Z))"
                interpretation_match = re.search(interpretation_pattern, perspective_text, re.IGNORECASE | re.DOTALL)
                
                if interpretation_match:
                    interpretation = interpretation_match.group(1).strip()
                else:
                    # Use first paragraph as interpretation
                    paragraphs = perspective_text.split('\n\n')
                    if len(paragraphs) > 1:
                        interpretation = paragraphs[1].strip()
                    elif paragraphs:
                        interpretation = paragraphs[0].strip()
                
                # Extract supporting evidence
                supporting_evidence = []
                evidence_pattern = r"(?:Evidence|Key evidence|Supporting).*?(?::|$)(.*?)(?=(?:Implications|\n\n|\Z))"
                evidence_match = re.search(evidence_pattern, perspective_text, re.IGNORECASE | re.DOTALL)
                
                if evidence_match:
                    evidence_text = evidence_match.group(1).strip()
                    
                    # Split into individual evidence items
                    evidence_items = re.findall(r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))', evidence_text, re.DOTALL)
                    
                    if evidence_items:
                        supporting_evidence = [item.strip() for item in evidence_items if item.strip()]
                    else:
                        # If no bullet points, use the whole text
                        supporting_evidence = [evidence_text]
                
                # Extract implications
                implications = ""
                implications_pattern = r"(?:Implications|If correct).*?(?::|$)(.*?)(?=(?:\n\n|\Z))"
                implications_match = re.search(implications_pattern, perspective_text, re.IGNORECASE | re.DOTALL)
                
                if implications_match:
                    implications = implications_match.group(1).strip()
                
                alternative_interpretations.append({
                    "perspective": perspective_name,
                    "interpretation": interpretation,
                    "supporting_evidence": supporting_evidence,
                    "implications": implications
                })
            else:
                # If no match found, add a generic alternative interpretation
                alternative_interpretations.append({
                    "perspective": perspective_name,
                    "interpretation": f"From the {perspective_name} perspective, the evidence may support an alternative interpretation based on {perspective.get('worldview', '')}",
                    "supporting_evidence": ["The same evidence can often be interpreted in multiple ways depending on one's perspective and assumptions."],
                    "implications": "If this alternative interpretation is correct, it would suggest different conclusions and recommendations."
                })
        
        return alternative_interpretations
    
    def _generate_generic_alternative_interpretations(self, findings: List[Dict], adversarial_perspectives: List[Dict]) -> List[Dict]:
        """
        Generate generic alternative interpretations.
        
        Args:
            findings: List of findings
            adversarial_perspectives: List of adversarial perspectives
            
        Returns:
            List of alternative interpretations
        """
        alternative_interpretations = []
        
        for perspective in adversarial_perspectives[:3]:  # Limit to top 3 perspectives
            perspective_name = perspective.get("name", "")
            perspective_worldview = perspective.get("worldview", "")
            perspective_biases = perspective.get("biases", "")
            
            # Generate a generic interpretation based on the perspective
            interpretation = f"From the {perspective_name} perspective, the evidence may be interpreted through the lens that {perspective_worldview}"
            
            # Generate generic supporting evidence
            supporting_evidence = [
                "The same evidence can be interpreted differently based on different underlying assumptions.",
                f"When prioritizing {perspective_biases.lower()}, the evidence takes on different significance.",
                "Selective emphasis of certain facts over others can lead to different conclusions."
            ]
            
            # Generate generic implications
            implications = f"If this alternative interpretation from the {perspective_name} perspective is correct, it would suggest fundamentally different conclusions and recommendations than the primary analysis."
            
            alternative_interpretations.append({
                "perspective": perspective_name,
                "interpretation": interpretation,
                "supporting_evidence": supporting_evidence,
                "implications": implications
            })
        
        return alternative_interpretations
    
    def _identify_vulnerabilities(self, findings: List[Dict], assumptions: List[Dict], adversarial_perspectives: List[Dict], context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Identify vulnerabilities in the analysis.
        
        Args:
            findings: List of findings
            assumptions: List of assumptions
            adversarial_perspectives: List of adversarial perspectives
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of vulnerabilities
        """
        logger.info("Identifying vulnerabilities")
        
        # Initialize vulnerabilities
        vulnerabilities = []
        
        # Use Llama4ScoutMCP to identify vulnerabilities
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Get question from context
            question = context.get("question", "")
            
            # Format findings for prompt
            findings_text = ""
            for i, finding in enumerate(findings[:3]):  # Limit to top 3 findings
                finding_text = finding.get("finding", "")
                findings_text += f"{i+1}. {finding_text}\n"
            
            # Format assumptions for prompt
            assumptions_text = ""
            for i, assumption in enumerate(assumptions[:3]):  # Limit to top 3 assumptions
                assumption_text = assumption.get("assumption", "")
                assumptions_text += f"{i+1}. {assumption_text}\n"
            
            # Create prompt for vulnerabilities
            prompt = f"""
            Identify key vulnerabilities in the analysis for the following question.
            
            Question: {question}
            
            Key Findings:
            {findings_text}
            
            Key Assumptions:
            {assumptions_text}
            
            For each vulnerability, provide:
            1. A clear description of the vulnerability
            2. Why it matters for the analysis
            3. How it could be exploited or lead to errors
            4. Potential mitigations
            
            Focus on substantive vulnerabilities that:
            - Represent significant weaknesses in the analytical process or conclusions
            - Could lead to major errors or misinterpretations
            - Are not easily addressed without significant changes to the analysis
            - Represent different types of vulnerabilities (e.g., data, logic, scope, bias)
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "What are the key vulnerabilities in this analysis?",
                "analysis_type": "vulnerability_identification",
                "context": {"prompt": prompt}
            })
            
            # Extract vulnerabilities from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse vulnerabilities from content
                vulnerabilities = self._parse_vulnerabilities_from_text(content)
                if vulnerabilities:
                    return vulnerabilities
        
        # Fallback: Generate generic vulnerabilities
        return self._generate_generic_vulnerabilities(findings, assumptions)
    
    def _parse_vulnerabilities_from_text(self, text: str) -> List[Dict]:
        """
        Parse vulnerabilities from text.
        
        Args:
            text: Text containing vulnerabilities
            
        Returns:
            List of vulnerabilities
        """
        vulnerabilities = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for vulnerability sections
        vulnerability_pattern = r'(?:^|\n)(?:Vulnerability|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Vulnerability|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        vulnerability_matches = re.findall(vulnerability_pattern, text, re.DOTALL)
        
        if not vulnerability_matches:
            # Try alternative pattern for numbered lists
            vulnerability_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            vulnerability_matches = re.findall(vulnerability_pattern, text, re.DOTALL)
            
            if vulnerability_matches:
                # Convert to expected format
                vulnerability_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(vulnerability_matches)]
        
        for match in vulnerability_matches:
            vulnerability_num = match[0].strip() if len(match) > 0 else ""
            vulnerability_description = match[1].strip() if len(match) > 1 else ""
            vulnerability_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract why it matters
            why_matters = ""
            why_pattern = r'(?:Why it matters|Importance|Significance).*?(?::|$)(.*?)(?=(?:How it could|Exploitation|\n\n|\Z))'
            why_match = re.search(why_pattern, vulnerability_content, re.IGNORECASE | re.DOTALL)
            if why_match:
                why_matters = why_match.group(1).strip()
            
            # Extract how it could be exploited
            exploitation = ""
            exploitation_pattern = r'(?:How it could|Exploitation|Could lead to).*?(?::|$)(.*?)(?=(?:Potential mitigations|Mitigations|\n\n|\Z))'
            exploitation_match = re.search(exploitation_pattern, vulnerability_content, re.IGNORECASE | re.DOTALL)
            if exploitation_match:
                exploitation = exploitation_match.group(1).strip()
            
            # Extract potential mitigations
            mitigations = []
            mitigations_pattern = r'(?:Potential mitigations|Mitigations|Addressing).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            mitigations_match = re.search(mitigations_pattern, vulnerability_content, re.IGNORECASE | re.DOTALL)
            if mitigations_match:
                mitigations_text = mitigations_match.group(1).strip()
                
                # Split into individual mitigations
                mitigation_items = re.findall(r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))', mitigations_text, re.DOTALL)
                
                if mitigation_items:
                    mitigations = [item.strip() for item in mitigation_items if item.strip()]
                else:
                    # If no bullet points, use the whole text
                    mitigations = [mitigations_text]
            
            vulnerabilities.append({
                "vulnerability": vulnerability_description,
                "why_it_matters": why_matters,
                "exploitation": exploitation,
                "mitigations": mitigations
            })
        
        return vulnerabilities
    
    def _generate_generic_vulnerabilities(self, findings: List[Dict], assumptions: List[Dict]) -> List[Dict]:
        """
        Generate generic vulnerabilities.
        
        Args:
            findings: List of findings
            assumptions: List of assumptions
            
        Returns:
            List of vulnerabilities
        """
        vulnerabilities = [
            {
                "vulnerability": "Overreliance on high-confidence assumptions",
                "why_it_matters": "Even high-confidence assumptions may be incorrect, and building analysis on these creates a fragile foundation.",
                "exploitation": "If key assumptions are wrong, the entire analysis may be invalidated, leading to incorrect conclusions and recommendations.",
                "mitigations": [
                    "Systematically challenge all assumptions, especially those deemed high-confidence",
                    "Develop contingency analyses for scenarios where key assumptions are wrong",
                    "Explicitly acknowledge the dependency of conclusions on assumptions"
                ]
            },
            {
                "vulnerability": "Insufficient consideration of alternative hypotheses",
                "why_it_matters": "Focusing too narrowly on a preferred hypothesis can lead to confirmation bias and missed insights.",
                "exploitation": "Important alternative explanations may be overlooked, leading to incomplete or misleading conclusions.",
                "mitigations": [
                    "Systematically generate and evaluate multiple competing hypotheses",
                    "Assign team members to advocate for alternative explanations",
                    "Explicitly consider what evidence would support alternative hypotheses"
                ]
            },
            {
                "vulnerability": "Data limitations and quality issues",
                "why_it_matters": "Analysis is only as good as the data it's based on, and data limitations can significantly affect conclusions.",
                "exploitation": "Conclusions may be drawn from incomplete, biased, or low-quality data, leading to unreliable results.",
                "mitigations": [
                    "Explicitly acknowledge data limitations and their implications",
                    "Seek multiple data sources to triangulate findings",
                    "Consider how conclusions would change with different data quality assumptions"
                ]
            },
            {
                "vulnerability": "Cognitive biases in analysis",
                "why_it_matters": "Analysts are subject to various cognitive biases that can systematically distort reasoning and conclusions.",
                "exploitation": "Biases can lead to systematic errors in judgment and interpretation of evidence.",
                "mitigations": [
                    "Use structured analytical techniques specifically designed to counter biases",
                    "Involve diverse analysts with different perspectives",
                    "Explicitly identify and address potential biases in the analysis"
                ]
            },
            {
                "vulnerability": "Scope limitations and boundary conditions",
                "why_it_matters": "Analysis may not adequately account for factors outside its defined scope that could significantly impact conclusions.",
                "exploitation": "External factors not considered in the analysis could invalidate conclusions when applied in the real world.",
                "mitigations": [
                    "Explicitly define and justify scope boundaries",
                    "Consider how external factors might impact conclusions",
                    "Acknowledge the limitations of applying conclusions beyond the defined scope"
                ]
            }
        ]
        
        return vulnerabilities
    
    def _generate_recommendations(self, challenged_assumptions: List[Dict], challenged_findings: List[Dict], alternative_interpretations: List[Dict], vulnerabilities: List[Dict], context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Generate recommendations based on red teaming results.
        
        Args:
            challenged_assumptions: List of challenged assumptions
            challenged_findings: List of challenged findings
            alternative_interpretations: List of alternative interpretations
            vulnerabilities: List of vulnerabilities
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of recommendations
        """
        logger.info("Generating recommendations")
        
        # Initialize recommendations
        recommendations = []
        
        # Use Llama4ScoutMCP to generate recommendations
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Get question from context
            question = context.get("question", "")
            
            # Format vulnerabilities for prompt
            vulnerabilities_text = ""
            for i, vulnerability in enumerate(vulnerabilities[:3]):  # Limit to top 3 vulnerabilities
                vulnerability_text = vulnerability.get("vulnerability", "")
                vulnerabilities_text += f"{i+1}. {vulnerability_text}\n"
            
            # Format alternative interpretations for prompt
            interpretations_text = ""
            for i, interpretation in enumerate(alternative_interpretations[:2]):  # Limit to top 2 interpretations
                perspective = interpretation.get("perspective", "")
                interpretation_text = interpretation.get("interpretation", "")
                interpretations_text += f"{i+1}. {perspective}: {interpretation_text}\n"
            
            # Create prompt for recommendations
            prompt = f"""
            Generate recommendations to strengthen the analysis based on red teaming results.
            
            Question: {question}
            
            Key Vulnerabilities:
            {vulnerabilities_text}
            
            Alternative Interpretations:
            {interpretations_text}
            
            Provide recommendations in the following categories:
            1. Analytical Process Improvements: How to strengthen the analytical process
            2. Alternative Perspectives: How to better incorporate diverse viewpoints
            3. Assumption Testing: How to better test and validate key assumptions
            4. Communication Enhancements: How to better communicate uncertainties and limitations
            
            For each recommendation:
            - Provide a clear, actionable recommendation
            - Explain how it addresses specific vulnerabilities or challenges identified in the red teaming
            - Consider feasibility and potential trade-offs
            
            Focus on substantive recommendations that would meaningfully improve the analysis.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": "How can the analysis be strengthened based on red teaming?",
                "analysis_type": "recommendation_generation",
                "context": {"prompt": prompt}
            })
            
            # Extract recommendations from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse recommendations from content
                recommendations = self._parse_recommendations_from_text(content)
                if recommendations:
                    return recommendations
        
        # Fallback: Generate generic recommendations
        return self._generate_generic_recommendations(vulnerabilities)
    
    def _parse_recommendations_from_text(self, text: str) -> List[Dict]:
        """
        Parse recommendations from text.
        
        Args:
            text: Text containing recommendations
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Simple parsing based on patterns
        import re
        
        # Define category patterns
        category_patterns = {
            "analytical_process": r"(?:Analytical Process|Process Improvements).*?(?:\n\n|\Z)",
            "alternative_perspectives": r"(?:Alternative Perspectives|Diverse Viewpoints).*?(?:\n\n|\Z)",
            "assumption_testing": r"(?:Assumption Testing|Testing Assumptions).*?(?:\n\n|\Z)",
            "communication": r"(?:Communication|Communication Enhancements).*?(?:\n\n|\Z)"
        }
        
        # Extract recommendations from each category
        for category, pattern in category_patterns.items():
            category_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if category_match:
                category_text = category_match.group(0)
                
                # Extract bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, category_text, re.DOTALL)
                
                if bullet_matches:
                    for bullet in bullet_matches:
                        bullet_text = bullet.strip()
                        
                        # Split into recommendation and explanation if possible
                        parts = bullet_text.split(":", 1)
                        
                        if len(parts) > 1:
                            recommendation = parts[0].strip()
                            explanation = parts[1].strip()
                        else:
                            # Try splitting by period
                            parts = bullet_text.split(".", 1)
                            
                            if len(parts) > 1:
                                recommendation = parts[0].strip() + "."
                                explanation = parts[1].strip()
                            else:
                                recommendation = bullet_text
                                explanation = ""
                        
                        recommendations.append({
                            "category": category,
                            "recommendation": recommendation,
                            "explanation": explanation
                        })
        
        return recommendations
    
    def _generate_generic_recommendations(self, vulnerabilities: List[Dict]) -> List[Dict]:
        """
        Generate generic recommendations.
        
        Args:
            vulnerabilities: List of vulnerabilities
            
        Returns:
            List of recommendations
        """
        recommendations = [
            {
                "category": "analytical_process",
                "recommendation": "Implement a more structured approach to hypothesis generation and testing",
                "explanation": "Using techniques like Analysis of Competing Hypotheses can help ensure that multiple explanations are systematically considered and evaluated against evidence."
            },
            {
                "category": "analytical_process",
                "recommendation": "Conduct pre-mortem analysis for key judgments",
                "explanation": "Imagining that conclusions have proven wrong and working backward to identify potential causes can help identify blind spots and weaknesses in the analysis."
            },
            {
                "category": "analytical_process",
                "recommendation": "Implement quality control checks at key stages of analysis",
                "explanation": "Regular review points can help identify and address analytical weaknesses before they become embedded in final conclusions."
            },
            {
                "category": "alternative_perspectives",
                "recommendation": "Incorporate diverse analytical perspectives throughout the process",
                "explanation": "Involving analysts with different backgrounds, expertise, and viewpoints can help identify blind spots and challenge groupthink."
            },
            {
                "category": "alternative_perspectives",
                "recommendation": "Assign devil's advocate roles for key judgments",
                "explanation": "Designating specific team members to challenge mainstream views can help ensure that alternative perspectives are thoroughly considered."
            },
            {
                "category": "alternative_perspectives",
                "recommendation": "Consult external experts with different viewpoints",
                "explanation": "External perspectives can provide valuable challenges to internal consensus and identify new considerations."
            },
            {
                "category": "assumption_testing",
                "recommendation": "Explicitly identify and document all key assumptions",
                "explanation": "Making assumptions explicit allows them to be systematically evaluated and challenged."
            },
            {
                "category": "assumption_testing",
                "recommendation": "Conduct sensitivity analysis for key assumptions",
                "explanation": "Testing how conclusions would change if key assumptions were different can identify which assumptions are most critical to validate."
            },
            {
                "category": "assumption_testing",
                "recommendation": "Regularly revisit and update assumptions as new information becomes available",
                "explanation": "Treating assumptions as dynamic rather than static can help ensure that analysis remains valid as circumstances change."
            },
            {
                "category": "communication",
                "recommendation": "Clearly communicate key assumptions and their impact on conclusions",
                "explanation": "Transparency about assumptions helps stakeholders understand the foundations of the analysis and its potential limitations."
            },
            {
                "category": "communication",
                "recommendation": "Use standardized language to express confidence and uncertainty",
                "explanation": "Consistent terminology for expressing confidence levels and uncertainties can help prevent misinterpretation of analytical judgments."
            },
            {
                "category": "communication",
                "recommendation": "Present alternative scenarios or interpretations alongside primary conclusions",
                "explanation": "Acknowledging plausible alternatives can provide a more complete picture and highlight the conditional nature of analytical conclusions."
            }
        ]
        
        return recommendations
    
    def _extract_findings(self, challenged_assumptions: List[Dict], challenged_findings: List[Dict], alternative_interpretations: List[Dict], vulnerabilities: List[Dict]) -> List[Dict]:
        """
        Extract key findings from red teaming.
        
        Args:
            challenged_assumptions: List of challenged assumptions
            challenged_findings: List of challenged findings
            alternative_interpretations: List of alternative interpretations
            vulnerabilities: List of vulnerabilities
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about most critical challenged assumption
        if challenged_assumptions:
            # Sort by criticality
            sorted_assumptions = sorted(
                challenged_assumptions,
                key=lambda x: 0 if x.get("criticality", "medium") == "high" else (1 if x.get("criticality", "medium") == "medium" else 2)
            )
            
            most_critical = sorted_assumptions[0]
            original_assumption = most_critical.get("original_assumption", "")
            
            findings.append({
                "finding": f"Critical assumption requiring further validation: {original_assumption}",
                "confidence": "high",
                "source": "red_teaming"
            })
        
        # Add finding about most significant vulnerability
        if vulnerabilities:
            most_significant = vulnerabilities[0]
            vulnerability = most_significant.get("vulnerability", "")
            
            findings.append({
                "finding": f"Key vulnerability in the analysis: {vulnerability}",
                "confidence": "high",
                "source": "red_teaming"
            })
        
        # Add finding about alternative interpretation
        if alternative_interpretations:
            alternative = alternative_interpretations[0]
            perspective = alternative.get("perspective", "")
            interpretation = alternative.get("interpretation", "")
            
            findings.append({
                "finding": f"Alternative interpretation from {perspective} perspective: {interpretation}",
                "confidence": "medium",
                "source": "red_teaming"
            })
        
        # Add finding about challenged finding
        if challenged_findings:
            challenged = challenged_findings[0]
            original_finding = challenged.get("original_finding", "")
            
            findings.append({
                "finding": f"Finding requiring reconsideration: {original_finding}",
                "confidence": "medium",
                "source": "red_teaming"
            })
        
        # Add general finding about red teaming
        findings.append({
            "finding": "Red teaming identified significant alternative perspectives that should be considered in the final analysis",
            "confidence": "high",
            "source": "red_teaming"
        })
        
        return findings
    
    def _extract_assumptions(self, challenged_assumptions: List[Dict]) -> List[Dict]:
        """
        Extract assumptions from red teaming.
        
        Args:
            challenged_assumptions: List of challenged assumptions
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about red teaming process
        assumptions.append({
            "assumption": "The adversarial perspectives used in red teaming adequately represent the range of relevant alternative viewpoints",
            "criticality": "medium",
            "source": "red_teaming"
        })
        
        # Add assumption about alternative interpretations
        assumptions.append({
            "assumption": "The most significant challenges to the analysis have been identified through the red teaming process",
            "criticality": "medium",
            "source": "red_teaming"
        })
        
        # Add assumption about vulnerability assessment
        assumptions.append({
            "assumption": "The identified vulnerabilities represent the most significant weaknesses in the analysis",
            "criticality": "high",
            "source": "red_teaming"
        })
        
        return assumptions
    
    def ground_llm_with_context(self, prompt: str, context: AnalysisContext) -> str:
        """
        Ground LLM prompt with relevant context.
        
        Args:
            prompt: Original prompt
            context: Analysis context
            
        Returns:
            Grounded prompt
        """
        # Get research results from context
        research_results = context.get("research_results", {})
        
        # Extract relevant information from research results
        relevant_info = ""
        
        if isinstance(research_results, dict):
            # Extract key facts if available
            if "key_facts" in research_results and isinstance(research_results["key_facts"], list):
                relevant_info += "Key Facts:\n"
                for fact in research_results["key_facts"][:10]:  # Limit to top 10 facts
                    relevant_info += f"- {fact}\n"
                relevant_info += "\n"
            
            # Extract key sources if available
            if "sources" in research_results and isinstance(research_results["sources"], list):
                relevant_info += "Key Sources:\n"
                for source in research_results["sources"][:5]:  # Limit to top 5 sources
                    if isinstance(source, dict):
                        title = source.get("title", "")
                        url = source.get("url", "")
                        relevant_info += f"- {title} ({url})\n"
                    elif isinstance(source, str):
                        relevant_info += f"- {source}\n"
                relevant_info += "\n"
        
        # Add relevant information to prompt if available
        if relevant_info:
            grounded_prompt = f"""
            {prompt}
            
            Consider the following relevant information from research:
            
            {relevant_info}
            """
            return grounded_prompt
        
        return prompt
    
    def handle_error(self, error: Exception, context: AnalysisContext) -> Dict:
        """
        Handle errors during technique execution.
        
        Args:
            error: The exception that occurred
            context: Analysis context
            
        Returns:
            Error result dictionary
        """
        error_message = str(error)
        logger.error(f"Error in RedTeamingTechnique: {error_message}")
        
        # Create error result
        error_result = {
            "technique": "red_teaming",
            "timestamp": time.time(),
            "error": error_message,
            "partial_results": {}
        }
        
        # Try to add question to error result
        question = context.get("question")
        if question:
            error_result["question"] = question
        
        # Add error to context
        context.add_error("red_teaming", error_message)
        
        return error_result
