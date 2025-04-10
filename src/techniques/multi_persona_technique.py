"""
Multi-Persona Technique for analyzing problems from multiple perspectives.
This module provides the MultiPersonaTechnique class for multi-perspective analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.analytical_technique import AnalyticalTechnique
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiPersonaTechnique(AnalyticalTechnique):
    """
    Multi-Persona Technique for analyzing problems from multiple perspectives.
    
    This technique provides capabilities for:
    1. Defining diverse personas with different expertise, biases, and viewpoints
    2. Analyzing the problem through each persona's perspective
    3. Identifying areas of agreement and disagreement across personas
    4. Synthesizing insights from multiple perspectives
    """
    
    def __init__(self):
        """Initialize the Multi-Persona Technique."""
        super().__init__(
            name="multi_persona",
            description="Analyzes problems from multiple perspectives using diverse personas",
            required_mcps=["llama4_scout", "research_mcp"],
            compatible_techniques=["red_teaming", "consensus_challenge", "key_assumptions_check"],
            incompatible_techniques=[]
        )
        logger.info("Initialized MultiPersonaTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Multi-Persona Technique")
        
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
            
            # Define personas
            personas = self._define_personas(question, research_results, context, parameters)
            
            # Generate persona analyses
            persona_analyses = self._generate_persona_analyses(question, personas, research_results, context, parameters)
            
            # Identify agreements and disagreements
            comparison = self._identify_agreements_disagreements(persona_analyses, parameters)
            
            # Synthesize insights
            synthesis = self._synthesize_insights(persona_analyses, comparison, research_results, context, parameters)
            
            # Compile results
            results = {
                "technique": "multi_persona",
                "timestamp": time.time(),
                "question": question,
                "personas": personas,
                "persona_analyses": persona_analyses,
                "comparison": comparison,
                "synthesis": synthesis,
                "findings": self._extract_findings(persona_analyses, comparison, synthesis),
                "assumptions": self._extract_assumptions(personas, persona_analyses),
                "uncertainties": self._extract_uncertainties(persona_analyses, comparison)
            }
            
            # Add results to context
            context.add_technique_result("multi_persona", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Multi-Persona Technique: {e}")
            return self.handle_error(e, context)
    
    def _define_personas(self, question: str, research_results: Dict, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Define diverse personas with different expertise, biases, and viewpoints.
        
        Args:
            question: The analytical question
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of persona definitions
        """
        logger.info("Defining personas")
        
        # Check if personas are provided in parameters
        if "personas" in parameters and isinstance(parameters["personas"], list):
            return parameters["personas"]
        
        # Use Llama4ScoutMCP to define personas
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Extract domain from question
            domain = self._extract_domain_from_question(question)
            
            # Create prompt for persona definition
            prompt = f"""
            Based on the following question, define 4-6 diverse personas with different expertise, biases, and viewpoints who would have valuable perspectives on this topic.
            
            Question: {question}
            
            Domain: {domain}
            
            For each persona:
            1. Provide a name and brief description (role, background)
            2. Identify their key expertise and knowledge areas
            3. Describe their potential biases or viewpoint tendencies
            4. Explain why their perspective is valuable for this question
            
            Ensure the personas represent diverse viewpoints, expertise areas, and thinking styles relevant to the question.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "creative",
                "context": {"prompt": prompt, "research_results": research_results}
            })
            
            # Extract personas from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse personas from content
                personas = self._parse_personas_from_text(content)
                if personas:
                    return personas
        
        # Fallback: Generate domain-specific personas
        return self._generate_domain_specific_personas(question)
    
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
            "security": ["security", "defense", "military", "conflict", "war", "terrorism", "cyber", "intelligence"],
            "healthcare": ["health", "medical", "disease", "treatment", "patient", "doctor", "hospital", "pharmaceutical"],
            "business": ["business", "corporate", "industry", "company", "startup", "entrepreneur", "management"]
        }
        
        # Count domain keywords
        domain_counts = {domain: sum(1 for term in terms if term in question_lower) for domain, terms in domains.items()}
        
        # Get domain with highest count
        if any(domain_counts.values()):
            primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0]
            return primary_domain
        
        # Default domain
        return "general"
    
    def _parse_personas_from_text(self, text: str) -> List[Dict]:
        """
        Parse personas from text.
        
        Args:
            text: Text containing persona descriptions
            
        Returns:
            List of parsed personas
        """
        personas = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for persona sections
        persona_pattern = r'(?:^|\n)(?:Persona|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Persona|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        persona_matches = re.findall(persona_pattern, text, re.DOTALL)
        
        if not persona_matches:
            # Try alternative pattern for numbered lists
            persona_pattern = r'(?:^|\n)(?:\d+\.)\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n\d+\.|\Z))'
            persona_matches = re.findall(persona_pattern, text, re.DOTALL)
            
            if persona_matches:
                # Convert to expected format
                persona_matches = [(str(i+1), match[0], match[1]) for i, match in enumerate(persona_matches)]
        
        for match in persona_matches:
            persona_num = match[0].strip() if len(match) > 0 else ""
            persona_name = match[1].strip() if len(match) > 1 else ""
            persona_content = match[2].strip() if len(match) > 2 else ""
            
            # Extract description
            description = ""
            description_pattern = r'(?:description|background|role).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            description_match = re.search(description_pattern, persona_content, re.IGNORECASE | re.DOTALL)
            if description_match:
                description = description_match.group(1).strip()
            else:
                # Use first paragraph as description
                paragraphs = persona_content.split('\n\n')
                if paragraphs:
                    description = paragraphs[0].strip()
            
            # Extract expertise
            expertise = []
            expertise_pattern = r'(?:expertise|knowledge|skills|specializes).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            expertise_match = re.search(expertise_pattern, persona_content, re.IGNORECASE | re.DOTALL)
            if expertise_match:
                expertise_text = expertise_match.group(1).strip()
                # Look for bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, expertise_text, re.DOTALL)
                if bullet_matches:
                    expertise = [item.strip() for item in bullet_matches]
                else:
                    # Split by commas or semicolons
                    expertise = [item.strip() for item in re.split(r',|;', expertise_text) if item.strip()]
            
            # Extract biases
            biases = []
            biases_pattern = r'(?:biases|viewpoint|tendencies|perspective).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            biases_match = re.search(biases_pattern, persona_content, re.IGNORECASE | re.DOTALL)
            if biases_match:
                biases_text = biases_match.group(1).strip()
                # Look for bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, biases_text, re.DOTALL)
                if bullet_matches:
                    biases = [item.strip() for item in bullet_matches]
                else:
                    # Split by commas or semicolons
                    biases = [item.strip() for item in re.split(r',|;', biases_text) if item.strip()]
            
            # Extract value
            value = ""
            value_pattern = r'(?:value|contribution|valuable|useful).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            value_match = re.search(value_pattern, persona_content, re.IGNORECASE | re.DOTALL)
            if value_match:
                value = value_match.group(1).strip()
            
            personas.append({
                "name": persona_name,
                "description": description,
                "expertise": expertise,
                "biases": biases,
                "value": value
            })
        
        return personas
    
    def _generate_domain_specific_personas(self, question: str) -> List[Dict]:
        """
        Generate domain-specific personas based on the question.
        
        Args:
            question: The analytical question
            
        Returns:
            List of domain-specific personas
        """
        # Extract domain from question
        domain = self._extract_domain_from_question(question)
        
        # Domain-specific persona templates
        domain_personas = {
            "economic": [
                {
                    "name": "Dr. Emily Chen, Macroeconomist",
                    "description": "Professor of Economics at a prestigious university with expertise in macroeconomic modeling and policy analysis",
                    "expertise": ["Macroeconomic theory", "Economic forecasting", "Monetary policy", "International economics"],
                    "biases": ["Tends to favor evidence-based policy", "Skeptical of extreme market interventions", "Values academic rigor"],
                    "value": "Provides theoretical frameworks and empirical evidence from economic research"
                },
                {
                    "name": "Michael Rodriguez, Investment Strategist",
                    "description": "Senior investment strategist at a global asset management firm with 20+ years of market experience",
                    "expertise": ["Market analysis", "Investment trends", "Risk assessment", "Financial markets"],
                    "biases": ["Market-oriented perspective", "Focuses on practical investment implications", "Attentive to market sentiment"],
                    "value": "Offers practical market insights and real-world investment implications"
                },
                {
                    "name": "Sarah Johnson, Small Business Owner",
                    "description": "Entrepreneur who has built and scaled multiple small businesses across different economic cycles",
                    "expertise": ["Small business operations", "Local economic impacts", "Practical business challenges", "Entrepreneurship"],
                    "biases": ["Skeptical of academic theories without practical application", "Concerned with regulatory burdens", "Values business autonomy"],
                    "value": "Provides ground-level perspective on how economic factors affect small businesses"
                },
                {
                    "name": "Dr. Kwame Osei, Development Economist",
                    "description": "Economist specializing in emerging markets and economic development with experience at international organizations",
                    "expertise": ["Development economics", "Emerging markets", "Economic inequality", "International economic institutions"],
                    "biases": ["Concerned with distributional impacts", "Attentive to global south perspectives", "Critical of one-size-fits-all approaches"],
                    "value": "Brings perspective on how economic issues affect developing economies and marginalized communities"
                }
            ],
            "political": [
                {
                    "name": "Dr. Robert Miller, Political Scientist",
                    "description": "Professor of Political Science specializing in institutional analysis and comparative politics",
                    "expertise": ["Political institutions", "Comparative politics", "Electoral systems", "Political theory"],
                    "biases": ["Academic perspective", "Institutionalist approach", "Values systematic analysis"],
                    "value": "Provides theoretical frameworks and comparative insights from political science research"
                },
                {
                    "name": "Maria Gonzalez, Policy Analyst",
                    "description": "Senior policy analyst at a think tank with experience in government and legislative affairs",
                    "expertise": ["Policy analysis", "Legislative processes", "Regulatory frameworks", "Public administration"],
                    "biases": ["Pragmatic policy orientation", "Attentive to implementation challenges", "Values evidence-based policy"],
                    "value": "Offers practical insights on policy design, implementation, and political feasibility"
                },
                {
                    "name": "James Wilson, Political Strategist",
                    "description": "Veteran political consultant who has advised campaigns across the political spectrum",
                    "expertise": ["Electoral strategy", "Public opinion", "Political messaging", "Campaign operations"],
                    "biases": ["Strategic political perspective", "Attentive to public opinion", "Focuses on electoral implications"],
                    "value": "Provides insights on political dynamics, public reception, and strategic implications"
                },
                {
                    "name": "Aisha Patel, Civil Society Advocate",
                    "description": "Director of a civil society organization focused on civic engagement and democratic participation",
                    "expertise": ["Civil society", "Grassroots organizing", "Democratic participation", "Social movements"],
                    "biases": ["Citizen-centered perspective", "Concerned with democratic values", "Attentive to marginalized voices"],
                    "value": "Brings perspective on how political issues affect ordinary citizens and civil society"
                }
            ],
            "technological": [
                {
                    "name": "Dr. David Kim, Technology Researcher",
                    "description": "Computer scientist and researcher at a leading technology research lab",
                    "expertise": ["AI and machine learning", "Computer science theory", "Technology R&D", "Emerging technologies"],
                    "biases": ["Technical perspective", "Innovation-oriented", "Values technical feasibility and scientific progress"],
                    "value": "Provides deep technical insights and understanding of technological capabilities and limitations"
                },
                {
                    "name": "Jennifer Liu, Tech Industry Executive",
                    "description": "Senior executive at a major technology company with experience scaling technology products",
                    "expertise": ["Technology product development", "Tech business models", "Industry trends", "Technology strategy"],
                    "biases": ["Business-oriented perspective", "Pragmatic about technology adoption", "Attentive to market dynamics"],
                    "value": "Offers insights on commercial applications, market adoption, and business implications"
                },
                {
                    "name": "Marcus Johnson, Technology Ethicist",
                    "description": "Philosopher and ethicist specializing in technology ethics and social impacts",
                    "expertise": ["Technology ethics", "Social impacts of technology", "Digital rights", "Ethical frameworks"],
                    "biases": ["Concerned with ethical implications", "Critical of techno-solutionism", "Values human-centered design"],
                    "value": "Brings perspective on ethical considerations, social impacts, and normative questions"
                },
                {
                    "name": "Sophia Rodriguez, User Experience Researcher",
                    "description": "UX researcher who studies how people interact with and are affected by technology",
                    "expertise": ["User experience", "Human-computer interaction", "User research", "Technology adoption"],
                    "biases": ["User-centered perspective", "Attentive to accessibility and usability", "Values inclusive design"],
                    "value": "Provides insights on how technologies affect different user groups and real-world usage patterns"
                }
            ],
            "social": [
                {
                    "name": "Dr. James Thompson, Sociologist",
                    "description": "Professor of Sociology specializing in social structures and institutional analysis",
                    "expertise": ["Social theory", "Institutional analysis", "Social structures", "Sociological research methods"],
                    "biases": ["Structural perspective", "Attentive to social patterns", "Values empirical social research"],
                    "value": "Provides theoretical frameworks and empirical insights from sociological research"
                },
                {
                    "name": "Elena Martinez, Community Organizer",
                    "description": "Experienced community organizer working with diverse urban communities",
                    "expertise": ["Community dynamics", "Grassroots organizing", "Urban social issues", "Community development"],
                    "biases": ["Community-centered perspective", "Attentive to power dynamics", "Values local knowledge"],
                    "value": "Offers ground-level insights on how social issues affect communities and practical approaches"
                },
                {
                    "name": "Dr. Michael Okafor, Cultural Anthropologist",
                    "description": "Anthropologist who studies cultural practices and social norms across different societies",
                    "expertise": ["Cultural analysis", "Cross-cultural comparison", "Ethnographic methods", "Social norms"],
                    "biases": ["Cultural relativist perspective", "Attentive to cultural context", "Values deep understanding of social practices"],
                    "value": "Brings perspective on cultural dimensions and comparative insights across different societies"
                },
                {
                    "name": "Sarah Chen, Social Policy Analyst",
                    "description": "Policy researcher specializing in social programs and their impacts",
                    "expertise": ["Social policy", "Program evaluation", "Social indicators", "Policy implementation"],
                    "biases": ["Evidence-based policy orientation", "Pragmatic about interventions", "Values measurable outcomes"],
                    "value": "Provides insights on policy design, implementation challenges, and evaluation of social interventions"
                }
            ],
            "environmental": [
                {
                    "name": "Dr. Emma Wilson, Environmental Scientist",
                    "description": "Climate scientist with expertise in environmental systems and climate modeling",
                    "expertise": ["Climate science", "Environmental systems", "Scientific modeling", "Data analysis"],
                    "biases": ["Scientific perspective", "Evidence-oriented", "Values peer-reviewed research"],
                    "value": "Provides scientific understanding of environmental processes and evidence-based analysis"
                },
                {
                    "name": "Carlos Mendoza, Environmental Policy Expert",
                    "description": "Policy specialist with experience in environmental regulation and international agreements",
                    "expertise": ["Environmental policy", "Regulatory frameworks", "International environmental governance", "Policy implementation"],
                    "biases": ["Policy-oriented perspective", "Pragmatic about governance", "Values institutional approaches"],
                    "value": "Offers insights on policy design, regulatory frameworks, and governance challenges"
                },
                {
                    "name": "Linda Johnson, Industry Sustainability Director",
                    "description": "Corporate sustainability executive with experience implementing environmental initiatives in industry",
                    "expertise": ["Corporate sustainability", "Environmental management", "Green business practices", "Industry transitions"],
                    "biases": ["Business-oriented perspective", "Pragmatic about economic constraints", "Values market-based solutions"],
                    "value": "Brings perspective on industry challenges, economic feasibility, and business implementation"
                },
                {
                    "name": "Amir Hassan, Environmental Justice Advocate",
                    "description": "Activist and advocate focusing on environmental justice and community impacts",
                    "expertise": ["Environmental justice", "Community impacts", "Grassroots organizing", "Social dimensions of environmental issues"],
                    "biases": ["Justice-oriented perspective", "Attentive to distributional impacts", "Values community voices"],
                    "value": "Provides insights on social and equity dimensions of environmental challenges"
                }
            ],
            "security": [
                {
                    "name": "Dr. Richard Chen, Security Analyst",
                    "description": "Former intelligence officer now working as an academic security researcher",
                    "expertise": ["Security theory", "Intelligence analysis", "Threat assessment", "Strategic studies"],
                    "biases": ["Analytical perspective", "Evidence-oriented", "Values systematic security analysis"],
                    "value": "Provides theoretical frameworks and analytical approaches from security studies"
                },
                {
                    "name": "Colonel Sarah Johnson (Ret.)",
                    "description": "Retired military officer with experience in defense planning and operations",
                    "expertise": ["Military strategy", "Defense planning", "Operational experience", "Security operations"],
                    "biases": ["Operational perspective", "Attentive to practical security challenges", "Values preparedness"],
                    "value": "Offers insights from military experience and operational security considerations"
                },
                {
                    "name": "Miguel Rodriguez, Cybersecurity Expert",
                    "description": "Cybersecurity professional with experience in both government and private sector",
                    "expertise": ["Cybersecurity", "Digital threats", "Security technology", "Cyber defense"],
                    "biases": ["Technical security perspective", "Attentive to emerging threats", "Values defensive capabilities"],
                    "value": "Brings perspective on technological dimensions of security and cyber threats"
                },
                {
                    "name": "Dr. Amina Osei, Human Security Specialist",
                    "description": "Researcher focusing on human security, peace studies, and conflict resolution",
                    "expertise": ["Human security", "Peace studies", "Conflict resolution", "Non-traditional security"],
                    "biases": ["Human-centered security perspective", "Attentive to root causes", "Values preventive approaches"],
                    "value": "Provides insights on broader security concepts beyond traditional military approaches"
                }
            ],
            "healthcare": [
                {
                    "name": "Dr. James Wilson, Medical Researcher",
                    "description": "Physician-scientist conducting clinical research at a major medical center",
                    "expertise": ["Clinical medicine", "Medical research", "Evidence-based medicine", "Healthcare innovation"],
                    "biases": ["Scientific medical perspective", "Evidence-oriented", "Values clinical research"],
                    "value": "Provides scientific understanding of medical issues and evidence-based approaches"
                },
                {
                    "name": "Maria Chen, Healthcare Administrator",
                    "description": "Hospital administrator with experience managing healthcare systems and operations",
                    "expertise": ["Healthcare administration", "Health systems", "Operational management", "Healthcare economics"],
                    "biases": ["Systems perspective", "Pragmatic about healthcare delivery", "Values operational efficiency"],
                    "value": "Offers insights on healthcare delivery, system constraints, and implementation challenges"
                },
                {
                    "name": "Dr. Robert Johnson, Public Health Expert",
                    "description": "Epidemiologist with experience in public health programs and population health",
                    "expertise": ["Public health", "Epidemiology", "Population health", "Health policy"],
                    "biases": ["Population health perspective", "Prevention-oriented", "Values public health approaches"],
                    "value": "Brings perspective on population-level health issues and preventive approaches"
                },
                {
                    "name": "Sarah Patel, Patient Advocate",
                    "description": "Healthcare advocate representing patient interests and experiences",
                    "expertise": ["Patient experience", "Healthcare access", "Patient rights", "Healthcare navigation"],
                    "biases": ["Patient-centered perspective", "Attentive to access issues", "Values patient autonomy"],
                    "value": "Provides insights on patient experiences, healthcare access, and patient-centered approaches"
                }
            ],
            "business": [
                {
                    "name": "Michael Chen, Corporate Executive",
                    "description": "CEO with experience leading large corporations across multiple industries",
                    "expertise": ["Corporate strategy", "Executive leadership", "Business operations", "Organizational management"],
                    "biases": ["Shareholder-oriented perspective", "Pragmatic about business realities", "Values profitability and growth"],
                    "value": "Provides insights from corporate leadership experience and strategic business thinking"
                },
                {
                    "name": "Dr. Sarah Johnson, Business Academic",
                    "description": "Professor of Business Administration specializing in organizational theory and management",
                    "expertise": ["Management theory", "Organizational behavior", "Business research", "Corporate governance"],
                    "biases": ["Academic perspective", "Evidence-oriented", "Values theoretical frameworks"],
                    "value": "Offers theoretical frameworks and research-based insights on business practices"
                },
                {
                    "name": "David Rodriguez, Entrepreneur",
                    "description": "Serial entrepreneur who has founded multiple startups in different sectors",
                    "expertise": ["Entrepreneurship", "Startup dynamics", "Innovation", "Business development"],
                    "biases": ["Innovation-oriented perspective", "Risk-tolerant", "Values disruptive thinking"],
                    "value": "Brings perspective on entrepreneurial approaches, innovation, and emerging business models"
                },
                {
                    "name": "Emma Wilson, Sustainability Consultant",
                    "description": "Business consultant specializing in sustainable business practices and ESG",
                    "expertise": ["Sustainable business", "ESG frameworks", "Corporate responsibility", "Stakeholder management"],
                    "biases": ["Stakeholder-oriented perspective", "Attentive to long-term impacts", "Values sustainable practices"],
                    "value": "Provides insights on sustainable business approaches and stakeholder considerations"
                }
            ],
            "general": [
                {
                    "name": "Dr. Michael Chen, Academic Researcher",
                    "description": "Professor with interdisciplinary background in social sciences and systems thinking",
                    "expertise": ["Systems analysis", "Research methodology", "Theoretical frameworks", "Interdisciplinary approaches"],
                    "biases": ["Academic perspective", "Evidence-oriented", "Values theoretical rigor"],
                    "value": "Provides theoretical frameworks and research-based insights with methodological rigor"
                },
                {
                    "name": "Sarah Johnson, Practical Implementer",
                    "description": "Professional with extensive experience implementing solutions in real-world contexts",
                    "expertise": ["Practical implementation", "Operational challenges", "Project management", "Stakeholder engagement"],
                    "biases": ["Pragmatic perspective", "Solution-oriented", "Values practical feasibility"],
                    "value": "Offers insights on practical implementation challenges and real-world constraints"
                },
                {
                    "name": "David Wilson, Critical Thinker",
                    "description": "Analyst known for challenging conventional wisdom and identifying blind spots",
                    "expertise": ["Critical analysis", "Logical reasoning", "Contrarian thinking", "Assumption identification"],
                    "biases": ["Skeptical perspective", "Question-oriented", "Values intellectual rigor"],
                    "value": "Brings perspective that challenges assumptions and identifies potential flaws in reasoning"
                },
                {
                    "name": "Maria Rodriguez, Integrative Synthesizer",
                    "description": "Consultant specializing in synthesizing diverse perspectives into coherent frameworks",
                    "expertise": ["Synthesis", "Pattern recognition", "Integrative thinking", "Complexity navigation"],
                    "biases": ["Holistic perspective", "Connection-oriented", "Values comprehensive understanding"],
                    "value": "Provides insights that connect different viewpoints and identify emergent patterns"
                }
            ]
        }
        
        # Return personas for the identified domain
        if domain in domain_personas:
            return domain_personas[domain]
        else:
            return domain_personas["general"]
    
    def _generate_persona_analyses(self, question: str, personas: List[Dict], research_results: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Generate analyses from each persona's perspective.
        
        Args:
            question: The analytical question
            personas: List of personas
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary of persona analyses
        """
        logger.info("Generating persona analyses")
        
        persona_analyses = {}
        
        # Use Llama4ScoutMCP to generate persona analyses
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            for persona in personas:
                persona_name = persona.get("name", "")
                persona_description = persona.get("description", "")
                persona_expertise = ", ".join(persona.get("expertise", []))
                persona_biases = ", ".join(persona.get("biases", []))
                
                # Create prompt for persona analysis
                prompt = f"""
                Analyze the following question from the perspective of {persona_name}.
                
                Question: {question}
                
                Persona Information:
                - Description: {persona_description}
                - Expertise: {persona_expertise}
                - Perspective tendencies: {persona_biases}
                
                Provide a comprehensive analysis that reflects how this persona would approach the question, including:
                
                1. Initial assessment - How they would frame and understand the question
                2. Key considerations - What factors they would prioritize in their analysis
                3. Main arguments - Their central points and reasoning
                4. Recommendations - What actions or conclusions they would suggest
                5. Potential blind spots - What this persona might overlook or undervalue
                
                Ensure the analysis authentically reflects this persona's expertise, viewpoint, and analytical style.
                """
                
                # Ground prompt with research results
                grounded_prompt = self.ground_llm_with_context(prompt, context)
                
                # Call Llama4ScoutMCP
                llama_response = llama4_scout.process({
                    "question": question,
                    "analysis_type": "perspective",
                    "context": {
                        "prompt": grounded_prompt, 
                        "research_results": research_results,
                        "persona": persona
                    }
                })
                
                # Extract analysis from response
                if isinstance(llama_response, dict) and "sections" in llama_response:
                    content = ""
                    for section_name, section_content in llama_response["sections"].items():
                        content += section_content + "\n\n"
                    
                    # Parse analysis from content
                    analysis = self._parse_analysis_from_text(content)
                    persona_analyses[persona_name] = analysis
                else:
                    # Fallback: Generate generic analysis
                    persona_analyses[persona_name] = self._generate_generic_analysis(persona, question)
        else:
            # Generate generic analyses for all personas
            for persona in personas:
                persona_name = persona.get("name", "")
                persona_analyses[persona_name] = self._generate_generic_analysis(persona, question)
        
        return persona_analyses
    
    def _parse_analysis_from_text(self, text: str) -> Dict:
        """
        Parse analysis from text.
        
        Args:
            text: Text containing analysis
            
        Returns:
            Dictionary containing parsed analysis
        """
        # Initialize analysis structure
        analysis = {
            "initial_assessment": "",
            "key_considerations": [],
            "main_arguments": [],
            "recommendations": [],
            "potential_blind_spots": []
        }
        
        # Simple parsing based on patterns
        import re
        
        # Map of section names to keys
        section_map = {
            "initial_assessment": ["initial assessment", "framing", "understanding", "perspective", "approach"],
            "key_considerations": ["key considerations", "factors", "priorities", "important aspects", "key factors"],
            "main_arguments": ["main arguments", "central points", "key arguments", "reasoning", "analysis"],
            "recommendations": ["recommendations", "suggestions", "actions", "conclusions", "proposed solutions"],
            "potential_blind_spots": ["potential blind spots", "limitations", "weaknesses", "overlooked aspects", "biases"]
        }
        
        # Look for sections
        for key, section_names in section_map.items():
            section_pattern = '|'.join(section_names)
            section_regex = rf'(?:^|\n)(?:{section_pattern}).*?(?::|$)(.*?)(?=(?:\n\n|\n(?:{"|".join([s for sublist in section_map.values() for s in sublist])})|$))'
            section_match = re.search(section_regex, text, re.IGNORECASE | re.DOTALL)
            
            if section_match:
                section_text = section_match.group(1)
                
                if key == "initial_assessment":
                    analysis[key] = section_text.strip()
                else:
                    # Extract bullet points
                    bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                    bullet_matches = re.findall(bullet_pattern, section_text, re.DOTALL)
                    
                    if bullet_matches:
                        analysis[key] = [item.strip() for item in bullet_matches]
                    else:
                        # Split by newlines or sentences
                        items = re.split(r'(?:\n|\.(?:\s+|$))', section_text)
                        analysis[key] = [item.strip() for item in items if item.strip()]
        
        return analysis
    
    def _generate_generic_analysis(self, persona: Dict, question: str) -> Dict:
        """
        Generate a generic analysis for a persona.
        
        Args:
            persona: Persona definition
            question: The analytical question
            
        Returns:
            Dictionary containing generic analysis
        """
        persona_name = persona.get("name", "")
        persona_expertise = persona.get("expertise", [])
        persona_biases = persona.get("biases", [])
        
        # Extract domain from persona name and expertise
        domains = ["economic", "political", "technological", "social", "environmental", "security", "healthcare", "business"]
        
        persona_text = persona_name.lower() + " " + " ".join(persona_expertise).lower()
        domain_matches = [domain for domain in domains if domain in persona_text]
        domain = domain_matches[0] if domain_matches else "general"
        
        # Base analysis structure
        analysis = {
            "initial_assessment": f"From my perspective as {persona_name.split(',')[0]}, this question requires careful analysis of {domain} factors and their interactions.",
            "key_considerations": [
                f"The {domain} context and current trends are essential to consider",
                "Multiple stakeholders with different interests are involved",
                "Both short-term and long-term implications should be evaluated"
            ],
            "main_arguments": [
                f"Based on my expertise in {', '.join(persona_expertise[:2]) if persona_expertise else domain} analysis, the situation presents both challenges and opportunities",
                "The evidence suggests multiple possible interpretations requiring careful evaluation",
                "A balanced approach considering multiple factors will yield the most robust analysis"
            ],
            "recommendations": [
                "Gather additional data to reduce key uncertainties",
                "Consider multiple scenarios rather than a single prediction",
                "Engage diverse stakeholders in the decision-making process"
            ],
            "potential_blind_spots": [
                f"My perspective may underemphasize {domain} factors outside my core expertise",
                f"As someone with {persona_biases[0] if persona_biases else 'certain biases'}, I might not fully account for alternative viewpoints",
                "Technical or specialized aspects beyond my expertise may require additional input"
            ]
        }
        
        # Customize based on domain
        if domain == "economic":
            analysis["key_considerations"].append("Economic incentives and market dynamics")
            analysis["main_arguments"].append("Market mechanisms will likely play a crucial role in addressing this challenge")
            analysis["recommendations"].append("Consider economic incentives to align stakeholder interests")
        
        elif domain == "political":
            analysis["key_considerations"].append("Political feasibility and stakeholder interests")
            analysis["main_arguments"].append("Political dynamics and institutional constraints will significantly shape outcomes")
            analysis["recommendations"].append("Develop strategies that account for political realities and build necessary coalitions")
        
        elif domain == "technological":
            analysis["key_considerations"].append("Technological capabilities and limitations")
            analysis["main_arguments"].append("Technological innovation offers promising solutions but requires careful implementation")
            analysis["recommendations"].append("Invest in technological capabilities while addressing potential unintended consequences")
        
        elif domain == "social":
            analysis["key_considerations"].append("Social dynamics and community impacts")
            analysis["main_arguments"].append("Social factors and human behaviors will significantly influence outcomes")
            analysis["recommendations"].append("Ensure solutions are socially acceptable and address community needs")
        
        elif domain == "environmental":
            analysis["key_considerations"].append("Environmental impacts and sustainability")
            analysis["main_arguments"].append("Environmental factors must be integrated into any comprehensive solution")
            analysis["recommendations"].append("Prioritize sustainable approaches that minimize environmental harm")
        
        return analysis
    
    def _identify_agreements_disagreements(self, persona_analyses: Dict, parameters: Dict) -> Dict:
        """
        Identify areas of agreement and disagreement across personas.
        
        Args:
            persona_analyses: Dictionary of persona analyses
            parameters: Technique parameters
            
        Returns:
            Dictionary of agreements and disagreements
        """
        logger.info("Identifying agreements and disagreements")
        
        # Initialize comparison structure
        comparison = {
            "agreements": {
                "key_considerations": [],
                "main_arguments": [],
                "recommendations": []
            },
            "disagreements": {
                "key_considerations": [],
                "main_arguments": [],
                "recommendations": []
            },
            "unique_insights": {}
        }
        
        # Extract all items by category
        all_considerations = []
        all_arguments = []
        all_recommendations = []
        
        for persona_name, analysis in persona_analyses.items():
            considerations = analysis.get("key_considerations", [])
            arguments = analysis.get("main_arguments", [])
            recommendations = analysis.get("recommendations", [])
            
            all_considerations.extend([(item, persona_name) for item in considerations])
            all_arguments.extend([(item, persona_name) for item in arguments])
            all_recommendations.extend([(item, persona_name) for item in recommendations])
            
            # Track unique insights by persona
            unique_insights = []
            for category in ["key_considerations", "main_arguments", "recommendations"]:
                items = analysis.get(category, [])
                for item in items:
                    if self._is_unique_insight(item, persona_analyses, persona_name, category):
                        unique_insights.append(f"{category.replace('_', ' ').title()}: {item}")
            
            if unique_insights:
                comparison["unique_insights"][persona_name] = unique_insights
        
        # Identify agreements and disagreements
        comparison["agreements"]["key_considerations"] = self._find_agreements(all_considerations)
        comparison["agreements"]["main_arguments"] = self._find_agreements(all_arguments)
        comparison["agreements"]["recommendations"] = self._find_agreements(all_recommendations)
        
        comparison["disagreements"]["key_considerations"] = self._find_disagreements(all_considerations)
        comparison["disagreements"]["main_arguments"] = self._find_disagreements(all_arguments)
        comparison["disagreements"]["recommendations"] = self._find_disagreements(all_recommendations)
        
        return comparison
    
    def _is_unique_insight(self, item: str, persona_analyses: Dict, persona_name: str, category: str) -> bool:
        """
        Check if an insight is unique to a persona.
        
        Args:
            item: The insight to check
            persona_analyses: Dictionary of persona analyses
            persona_name: Name of the persona with this insight
            category: Category of the insight
            
        Returns:
            True if the insight is unique, False otherwise
        """
        # Simple similarity function
        def similarity(a: str, b: str) -> float:
            a_words = set(a.lower().split())
            b_words = set(b.lower().split())
            
            if not a_words or not b_words:
                return 0.0
            
            intersection = len(a_words.intersection(b_words))
            union = len(a_words.union(b_words))
            
            return intersection / union
        
        # Check if this insight is similar to any insight from other personas
        for other_persona, analysis in persona_analyses.items():
            if other_persona == persona_name:
                continue
            
            other_items = analysis.get(category, [])
            for other_item in other_items:
                if similarity(item, other_item) > 0.5:  # Similarity threshold
                    return False
        
        return True
    
    def _find_agreements(self, items: List[tuple]) -> List[Dict]:
        """
        Find agreements among items from different personas.
        
        Args:
            items: List of (item, persona_name) tuples
            
        Returns:
            List of agreement dictionaries
        """
        # Simple similarity function
        def similarity(a: str, b: str) -> float:
            a_words = set(a.lower().split())
            b_words = set(b.lower().split())
            
            if not a_words or not b_words:
                return 0.0
            
            intersection = len(a_words.intersection(b_words))
            union = len(a_words.union(b_words))
            
            return intersection / union
        
        # Group similar items
        groups = []
        for item, persona in items:
            found_group = False
            for group in groups:
                for group_item, _ in group:
                    if similarity(item, group_item) > 0.5:  # Similarity threshold
                        group.append((item, persona))
                        found_group = True
                        break
                if found_group:
                    break
            
            if not found_group:
                groups.append([(item, persona)])
        
        # Select groups with multiple personas
        agreement_groups = [group for group in groups if len(set(persona for _, persona in group)) > 1]
        
        # Format agreements
        agreements = []
        for group in agreement_groups:
            personas = list(set(persona for _, persona in group))
            # Use the shortest item as representative
            representative = min([item for item, _ in group], key=len)
            
            agreements.append({
                "statement": representative,
                "personas": personas
            })
        
        return agreements
    
    def _find_disagreements(self, items: List[tuple]) -> List[Dict]:
        """
        Find disagreements among items from different personas.
        
        Args:
            items: List of (item, persona_name) tuples
            
        Returns:
            List of disagreement dictionaries
        """
        # Look for contradictory statements
        disagreements = []
        
        # Simple contradiction detection based on negation
        negation_terms = ["not", "no", "never", "unlikely", "disagree", "oppose", "contrary", "against", "reject"]
        
        for i, (item1, persona1) in enumerate(items):
            item1_lower = item1.lower()
            
            for item2, persona2 in items[i+1:]:
                if persona1 == persona2:
                    continue
                
                item2_lower = item2.lower()
                
                # Check for direct contradictions
                contradiction = False
                
                # Check if one contains negation of the other
                for term in negation_terms:
                    if (term in item1_lower and self._contains_opposite(item1_lower, item2_lower, term)) or \
                       (term in item2_lower and self._contains_opposite(item2_lower, item1_lower, term)):
                        contradiction = True
                        break
                
                # Check for opposing directional terms
                directional_pairs = [
                    ("increase", "decrease"),
                    ("expand", "contract"),
                    ("positive", "negative"),
                    ("benefit", "harm"),
                    ("advantage", "disadvantage"),
                    ("support", "oppose"),
                    ("agree", "disagree"),
                    ("high", "low"),
                    ("more", "less"),
                    ("strengthen", "weaken")
                ]
                
                for term1, term2 in directional_pairs:
                    if (term1 in item1_lower and term2 in item2_lower) or \
                       (term2 in item1_lower and term1 in item2_lower):
                        contradiction = True
                        break
                
                if contradiction:
                    disagreements.append({
                        "statements": [item1, item2],
                        "personas": [persona1, persona2]
                    })
        
        return disagreements
    
    def _contains_opposite(self, text_with_negation: str, other_text: str, negation_term: str) -> bool:
        """
        Check if a text with negation contains the opposite of another text.
        
        Args:
            text_with_negation: Text containing negation term
            other_text: Text to compare against
            negation_term: The negation term
            
        Returns:
            True if text_with_negation contains opposite of other_text, False otherwise
        """
        # Simple check for whether key terms in other_text appear near negation in text_with_negation
        other_words = set(other_text.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "that", "this", "these", "those"}
        other_words = other_words - common_words
        
        # Check if key words from other_text appear near negation in text_with_negation
        negation_index = text_with_negation.find(negation_term)
        if negation_index >= 0:
            # Look at context around negation (10 words before and after)
            context_start = max(0, text_with_negation.rfind(".", 0, negation_index))
            if context_start == 0:
                context_start = 0
            else:
                context_start += 1  # Skip the period
            
            context_end = text_with_negation.find(".", negation_index)
            if context_end < 0:
                context_end = len(text_with_negation)
            
            context = text_with_negation[context_start:context_end].lower()
            
            # Check if any significant words from other_text appear in this context
            for word in other_words:
                if len(word) > 3 and word in context:  # Only check significant words
                    return True
        
        return False
    
    def _synthesize_insights(self, persona_analyses: Dict, comparison: Dict, research_results: Dict, context: AnalysisContext, parameters: Dict) -> Dict:
        """
        Synthesize insights from multiple perspectives.
        
        Args:
            persona_analyses: Dictionary of persona analyses
            comparison: Dictionary of agreements and disagreements
            research_results: Research results
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing synthesis
        """
        logger.info("Synthesizing insights")
        
        # Use Llama4ScoutMCP to synthesize insights
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Get question from context
            question = context.get("question")
            
            # Create summaries of agreements and disagreements
            agreements_summary = ""
            for category in ["key_considerations", "main_arguments", "recommendations"]:
                agreements = comparison["agreements"].get(category, [])
                if agreements:
                    agreements_summary += f"\n## {category.replace('_', ' ').title()} Agreements:\n"
                    for agreement in agreements:
                        statement = agreement.get("statement", "")
                        personas = ", ".join(agreement.get("personas", []))
                        agreements_summary += f"- {statement} (Agreed by: {personas})\n"
            
            disagreements_summary = ""
            for category in ["key_considerations", "main_arguments", "recommendations"]:
                disagreements = comparison["disagreements"].get(category, [])
                if disagreements:
                    disagreements_summary += f"\n## {category.replace('_', ' ').title()} Disagreements:\n"
                    for disagreement in disagreements:
                        statements = disagreement.get("statements", [])
                        personas = disagreement.get("personas", [])
                        disagreements_summary += f"- {personas[0]}: {statements[0]}\n  vs.\n  {personas[1]}: {statements[1]}\n"
            
            unique_insights_summary = ""
            for persona, insights in comparison["unique_insights"].items():
                if insights:
                    unique_insights_summary += f"\n## Unique insights from {persona}:\n"
                    for insight in insights:
                        unique_insights_summary += f"- {insight}\n"
            
            # Create prompt for synthesis
            prompt = f"""
            Based on the multiple persona analyses of the following question, synthesize key insights that emerge from considering these diverse perspectives.
            
            Question: {question}
            
            Areas of Agreement:
            {agreements_summary}
            
            Areas of Disagreement:
            {disagreements_summary}
            
            Unique Insights:
            {unique_insights_summary}
            
            Please provide:
            
            1. Integrated understanding - How the different perspectives collectively enhance understanding
            2. Key tensions - The most significant tensions between different viewpoints
            3. Complementary insights - How different perspectives complement each other
            4. Synthesis conclusions - Overall conclusions that emerge from integrating perspectives
            5. Next steps - Recommended actions based on this multi-perspective analysis
            
            Focus on how the integration of these perspectives provides a more robust and nuanced understanding than any single viewpoint.
            """
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "synthesis",
                "context": {"prompt": prompt, "research_results": research_results}
            })
            
            # Extract synthesis from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse synthesis from content
                synthesis = self._parse_synthesis_from_text(content)
                if synthesis:
                    return synthesis
        
        # Fallback: Generate generic synthesis
        return self._generate_generic_synthesis(persona_analyses, comparison)
    
    def _parse_synthesis_from_text(self, text: str) -> Dict:
        """
        Parse synthesis from text.
        
        Args:
            text: Text containing synthesis
            
        Returns:
            Dictionary containing parsed synthesis
        """
        # Initialize synthesis structure
        synthesis = {
            "integrated_understanding": "",
            "key_tensions": [],
            "complementary_insights": [],
            "synthesis_conclusions": [],
            "next_steps": []
        }
        
        # Simple parsing based on patterns
        import re
        
        # Map of section names to keys
        section_map = {
            "integrated_understanding": ["integrated understanding", "collective understanding", "enhanced understanding", "holistic view"],
            "key_tensions": ["key tensions", "tensions", "conflicts", "disagreements", "opposing views"],
            "complementary_insights": ["complementary insights", "complementary perspectives", "complementary views", "synergies"],
            "synthesis_conclusions": ["synthesis conclusions", "overall conclusions", "integrated conclusions", "key takeaways"],
            "next_steps": ["next steps", "recommended actions", "path forward", "recommendations", "suggested approach"]
        }
        
        # Look for sections
        for key, section_names in section_map.items():
            section_pattern = '|'.join(section_names)
            section_regex = rf'(?:^|\n)(?:{section_pattern}).*?(?::|$)(.*?)(?=(?:\n\n|\n(?:{"|".join([s for sublist in section_map.values() for s in sublist])})|$))'
            section_match = re.search(section_regex, text, re.IGNORECASE | re.DOTALL)
            
            if section_match:
                section_text = section_match.group(1)
                
                if key == "integrated_understanding":
                    synthesis[key] = section_text.strip()
                else:
                    # Extract bullet points
                    bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                    bullet_matches = re.findall(bullet_pattern, section_text, re.DOTALL)
                    
                    if bullet_matches:
                        synthesis[key] = [item.strip() for item in bullet_matches]
                    else:
                        # Split by newlines or sentences
                        items = re.split(r'(?:\n|\.(?:\s+|$))', section_text)
                        synthesis[key] = [item.strip() for item in items if item.strip()]
        
        return synthesis
    
    def _generate_generic_synthesis(self, persona_analyses: Dict, comparison: Dict) -> Dict:
        """
        Generate a generic synthesis.
        
        Args:
            persona_analyses: Dictionary of persona analyses
            comparison: Dictionary of agreements and disagreements
            
        Returns:
            Dictionary containing generic synthesis
        """
        # Count agreements and disagreements
        agreement_count = sum(len(agreements) for category, agreements in comparison["agreements"].items())
        disagreement_count = sum(len(disagreements) for category, disagreements in comparison["disagreements"].items())
        
        # Determine overall pattern
        if agreement_count > disagreement_count * 2:
            pattern = "high_agreement"
        elif disagreement_count > agreement_count:
            pattern = "high_disagreement"
        else:
            pattern = "mixed"
        
        # Base synthesis
        synthesis = {
            "integrated_understanding": "The multiple perspectives reveal that this issue has both technical and social dimensions, with different stakeholders prioritizing different aspects based on their expertise and values.",
            "key_tensions": [
                "Short-term pragmatic solutions versus long-term systemic approaches",
                "Technical/objective analysis versus social/subjective considerations",
                "Centralized versus distributed approaches to addressing the challenge"
            ],
            "complementary_insights": [
                "Technical experts provide implementation feasibility while social perspectives highlight acceptance requirements",
                "Theoretical frameworks offer structure while practical perspectives ground analysis in reality",
                "Different disciplinary approaches reveal different aspects of the same problem"
            ],
            "synthesis_conclusions": [
                "A comprehensive approach must integrate multiple perspectives rather than privileging any single viewpoint",
                "The most robust solutions will address both technical and social dimensions of the challenge",
                "Ongoing dialogue between different perspectives is essential for adaptive responses"
            ],
            "next_steps": [
                "Develop an integrated framework that incorporates insights from multiple perspectives",
                "Identify specific areas where additional expertise or research is needed",
                "Create processes for ongoing multi-perspective evaluation as implementation proceeds"
            ]
        }
        
        # Customize based on pattern
        if pattern == "high_agreement":
            synthesis["integrated_understanding"] = "The multiple perspectives show substantial agreement on core aspects of this issue, suggesting a robust consensus across different domains of expertise and viewpoints."
            synthesis["synthesis_conclusions"].append("The high level of agreement across diverse perspectives suggests these conclusions are particularly robust")
        
        elif pattern == "high_disagreement":
            synthesis["integrated_understanding"] = "The multiple perspectives reveal significant disagreements on this issue, indicating it is complex and multifaceted with legitimate differences in how it can be approached."
            synthesis["key_tensions"].append("Fundamental differences in how the problem is framed and understood")
            synthesis["next_steps"].append("Facilitate structured dialogue between competing perspectives to better understand points of disagreement")
        
        return synthesis
    
    def _extract_findings(self, persona_analyses: Dict, comparison: Dict, synthesis: Dict) -> List[Dict]:
        """
        Extract key findings from multi-persona analysis.
        
        Args:
            persona_analyses: Dictionary of persona analyses
            comparison: Dictionary of agreements and disagreements
            synthesis: Dictionary containing synthesis
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about multiple perspectives
        findings.append({
            "finding": f"Analysis incorporated {len(persona_analyses)} distinct perspectives with different expertise and viewpoints",
            "confidence": "high",
            "source": "multi_persona"
        })
        
        # Add finding about agreements
        agreement_count = sum(len(agreements) for category, agreements in comparison["agreements"].items())
        if agreement_count > 0:
            findings.append({
                "finding": f"Despite different perspectives, significant agreement was found on {agreement_count} key points",
                "confidence": "medium",
                "source": "multi_persona"
            })
        
        # Add finding about disagreements
        disagreement_count = sum(len(disagreements) for category, disagreements in comparison["disagreements"].items())
        if disagreement_count > 0:
            findings.append({
                "finding": f"Important disagreements were identified on {disagreement_count} issues, highlighting areas of uncertainty",
                "confidence": "medium",
                "source": "multi_persona"
            })
        
        # Add findings from synthesis conclusions
        synthesis_conclusions = synthesis.get("synthesis_conclusions", [])
        for i, conclusion in enumerate(synthesis_conclusions):
            if i < 2:  # Limit to top 2 conclusions
                findings.append({
                    "finding": conclusion,
                    "confidence": "medium",
                    "source": "multi_persona"
                })
        
        return findings
    
    def _extract_assumptions(self, personas: List[Dict], persona_analyses: Dict) -> List[Dict]:
        """
        Extract assumptions from multi-persona analysis.
        
        Args:
            personas: List of personas
            persona_analyses: Dictionary of persona analyses
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about persona representation
        assumptions.append({
            "assumption": "The selected personas adequately represent the range of relevant perspectives on this issue",
            "criticality": "high",
            "source": "multi_persona"
        })
        
        # Add assumption about persona simulation
        assumptions.append({
            "assumption": "The simulated persona analyses accurately reflect how real experts with these backgrounds would reason",
            "criticality": "medium",
            "source": "multi_persona"
        })
        
        # Add assumption about perspective integration
        assumptions.append({
            "assumption": "Integrating multiple perspectives provides a more robust analysis than any single perspective",
            "criticality": "high",
            "source": "multi_persona"
        })
        
        # Add persona-specific assumptions
        for persona in personas[:2]:  # Limit to first 2 personas
            persona_name = persona.get("name", "").split(",")[0]
            biases = persona.get("biases", [])
            if biases:
                assumptions.append({
                    "assumption": f"The {persona_name} perspective may be influenced by {biases[0]}",
                    "criticality": "medium",
                    "source": "multi_persona"
                })
        
        return assumptions
    
    def _extract_uncertainties(self, persona_analyses: Dict, comparison: Dict) -> List[Dict]:
        """
        Extract uncertainties from multi-persona analysis.
        
        Args:
            persona_analyses: Dictionary of persona analyses
            comparison: Dictionary of agreements and disagreements
            
        Returns:
            List of uncertainties
        """
        uncertainties = []
        
        # Add uncertainty about perspective completeness
        uncertainties.append({
            "uncertainty": "Additional perspectives not included in this analysis might yield different insights",
            "impact": "medium",
            "source": "multi_persona"
        })
        
        # Add uncertainties from disagreements
        for category in ["key_considerations", "main_arguments"]:
            disagreements = comparison["disagreements"].get(category, [])
            for i, disagreement in enumerate(disagreements):
                if i < 2:  # Limit to top 2 disagreements per category
                    statements = disagreement.get("statements", [])
                    if statements:
                        uncertainties.append({
                            "uncertainty": f"Uncertainty regarding {statements[0]}",
                            "impact": "high",
                            "source": "multi_persona"
                        })
        
        # Add uncertainty about blind spots
        all_blind_spots = []
        for persona_name, analysis in persona_analyses.items():
            blind_spots = analysis.get("potential_blind_spots", [])
            all_blind_spots.extend(blind_spots)
        
        if all_blind_spots:
            uncertainties.append({
                "uncertainty": f"Potential blind spot: {all_blind_spots[0]}",
                "impact": "medium",
                "source": "multi_persona"
            })
        
        return uncertainties
