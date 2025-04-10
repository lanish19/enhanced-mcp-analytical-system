"""
Scenario Triangulation Technique for generating and analyzing multiple scenarios.
This module provides the ScenarioTriangulationTechnique class for scenario-based analysis.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.analytical_technique import AnalyticalTechnique
from src.analysis_context import AnalysisContext
from utils.mcp_utils import get_relevant_data_from_mcps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScenarioTriangulationTechnique(AnalyticalTechnique):
    """
    Scenario Triangulation Technique for generating and analyzing multiple scenarios.
    
    This technique provides capabilities for:
    1. Generating diverse scenarios based on key uncertainties
    2. Analyzing implications of each scenario
    3. Identifying common elements across scenarios
    4. Determining robust strategies that work across scenarios
    """
    
    def __init__(self):
        """Initialize the Scenario Triangulation Technique."""
        super().__init__(
            name="scenario_triangulation",
            description="Generates and analyzes multiple scenarios to identify robust strategies",
            required_mcps=["llama4_scout", "research_mcp"],
            compatible_techniques=["cross_impact_analysis", "indicators_development", "system_dynamics_modeling"],
            incompatible_techniques=[]
        )
        logger.info("Initialized ScenarioTriangulationTechnique")
    
    def execute(self, context: AnalysisContext, parameters: Dict = None) -> Dict:
        """
        Execute the technique with the given context and parameters.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            Dictionary containing technique results
        """
        logger.info("Executing Scenario Triangulation Technique")
        
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
            
            # Extract key uncertainties
            uncertainties = self._extract_key_uncertainties(context, parameters)
            
            # Generate scenarios
            scenarios = self._generate_scenarios(question, uncertainties, research_results, parameters)
            
            # Get data from MCPs
            relevant_data = get_relevant_data_from_mcps(self.mcp_registry, question, ["EconomicsMCP", "GeopoliticsMCP"])

             # Analyze scenario implications
            implications = self._analyze_scenario_implications(question, scenarios, research_results, parameters)
            
            # Identify common elements
            common_elements = self._identify_common_elements(scenarios, implications)
            
            # Determine robust strategies
            robust_strategies = self._determine_robust_strategies(question, scenarios, implications, parameters)
            
            # Compile results
            results = {
                "technique": "scenario_triangulation",
                "timestamp": time.time(),
                "question": question,
                "key_uncertainties": uncertainties,
                "scenarios": scenarios,
                "implications": implications,
                "common_elements": common_elements,
                "robust_strategies": robust_strategies,
                "findings": self._extract_findings(scenarios, implications, robust_strategies),
                "assumptions": self._extract_assumptions(scenarios),
                "uncertainties": self._extract_scenario_uncertainties(scenarios, uncertainties)
            }
            
            # Add results to context
            context.add_technique_result("scenario_triangulation", results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing Scenario Triangulation Technique: {e}")
            return self.handle_error(e, context)
    
    def validate_parameters(self, parameters: Dict) -> bool:
        """
        Validate the parameters for the technique.
        
        Args:
            parameters: Technique parameters
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Basic validation from parent class
        if not super().validate_parameters(parameters):
            return False
        
        # Specific validation for this technique
        if "num_scenarios" in parameters and not isinstance(parameters["num_scenarios"], int):
            logger.error(f"Invalid num_scenarios parameter: {parameters['num_scenarios']}, expected int")
            return False
        
        if "time_horizon" in parameters and not isinstance(parameters["time_horizon"], str):
            logger.error(f"Invalid time_horizon parameter: {parameters['time_horizon']}, expected str")
            return False
        
        return True
    
    def _extract_key_uncertainties(self, context: AnalysisContext, parameters: Dict) -> List[Dict]:
        """
        Extract key uncertainties from context or generate them.
        
        Args:
            context: Analysis context
            parameters: Technique parameters
            
        Returns:
            List of key uncertainties
        """
        logger.info("Extracting key uncertainties")
        
        # Check if uncertainties are provided in parameters
        if "uncertainties" in parameters and isinstance(parameters["uncertainties"], list):
            return parameters["uncertainties"]
        
        # Check if uncertainties are in context
        context_uncertainties = context.uncertainties
        if context_uncertainties:
            # Format uncertainties
            formatted_uncertainties = []
            for uncertainty in context_uncertainties[:5]:  # Limit to top 5
                if isinstance(uncertainty, dict):
                    formatted_uncertainties.append({
                        "factor": uncertainty.get("description", "Unknown factor"),
                        "description": uncertainty.get("details", ""),
                        "range": uncertainty.get("range", ["Low", "High"])
                    })
            
            if formatted_uncertainties:
                return formatted_uncertainties
        
        # Generate uncertainties using Llama4ScoutMCP
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            question = context.get("question")
            research_results = context.get("research_results")
            
            # Create prompt for uncertainty extraction
            prompt = f"""
            Based on the following question and research results, identify the 3-5 most critical uncertainties that could significantly impact future outcomes.
            
            Question: {question}
            
            For each uncertainty, provide:
            1. The uncertain factor
            2. A brief description of why it's uncertain
            3. The range of possible outcomes (e.g., "Low growth to High growth")
            
            Format your response as a structured list of uncertainties.
            """
            
            # Ground prompt with research results
            grounded_prompt = self.ground_llm_with_context(prompt, context)
            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "standard",
                "context": {"prompt": grounded_prompt}
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
        return [
            {
                "factor": "Economic Growth",
                "description": "Uncertainty about the rate of economic growth in relevant markets",
                "range": ["Recession", "Slow Growth", "Rapid Growth"]
            },
            {
                "factor": "Technological Change",
                "description": "Uncertainty about the pace and direction of technological innovation",
                "range": ["Incremental", "Disruptive"]
            },
            {
                "factor": "Regulatory Environment",
                "description": "Uncertainty about future regulatory changes",
                "range": ["Relaxed", "Stringent"]
            },
            {
                "factor": "Competitive Landscape",
                "description": "Uncertainty about competitive dynamics and market structure",
                "range": ["Consolidated", "Fragmented"]
            }
        ]
    
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
        
        # Look for numbered or bulleted items
        patterns = [
            r'(?:^|\n)(?:\d+\.|[-•*]\s+)([^:\n]+):\s*(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))',
            r'(?:^|\n)(?:Uncertainty|Factor)\s*\d*:\s*([^:\n]+)(?::|(?:\n|$))(?:\s*Description:\s*(.*?))?(?:\s*Range:\s*(.*?))?(?=(?:\n(?:Uncertainty|Factor)|$))'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    if len(match) >= 2:
                        factor = match[0].strip()
                        description = match[1].strip() if len(match) > 1 else ""
                        range_str = match[2].strip() if len(match) > 2 else ""
                        
                        # Parse range
                        range_values = []
                        if range_str:
                            range_values = [r.strip() for r in re.split(r'(?:to|,|;|\n)', range_str)]
                        else:
                            # Look for range in description
                            range_match = re.search(r'range.*?(?:from\s+)?([^,;]+)(?:\s+to\s+|\s*-\s*)([^,;]+)', description, re.IGNORECASE)
                            if range_match:
                                range_values = [range_match.group(1).strip(), range_match.group(2).strip()]
                        
                        # Default range if none found
                        if not range_values:
                            range_values = ["Low", "High"]
                        
                        uncertainties.append({
                            "factor": factor,
                            "description": description,
                            "range": range_values
                        })
        
        return uncertainties
    
    def _generate_scenarios(self, question: str, uncertainties: List[Dict], research_results: Dict, parameters: Dict) -> List[Dict]:
        """
        Generate scenarios based on key uncertainties.
        
        Args:
            question: The analytical question
            uncertainties: List of key uncertainties
            research_results: Research results
            parameters: Technique parameters
            
        Returns:
            List of generated scenarios
        """
        logger.info("Generating scenarios")
        
        # Get parameters
        num_scenarios = parameters.get("num_scenarios", 4)
        time_horizon = parameters.get("time_horizon", "5 years")
        
        # Limit number of scenarios
        if num_scenarios > 6:
            num_scenarios = 6
            logger.warning(f"Limiting number of scenarios to {num_scenarios}")
        
        # Generate scenarios using Llama4ScoutMCP
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Create prompt for scenario generation
            uncertainties_text = ""
            for i, uncertainty in enumerate(uncertainties):
                factor = uncertainty.get("factor", f"Uncertainty {i+1}")
                description = uncertainty.get("description", "")
                range_values = uncertainty.get("range", ["Low", "High"])
                range_text = " to ".join(range_values)
                
                uncertainties_text += f"{i+1}. {factor}: {description} (Range: {range_text})\n"
            
            prompt = f"""
            Based on the following question and key uncertainties, generate {num_scenarios} distinct scenarios for a {time_horizon} time horizon.
            
            Question: {question}
            
            Key Uncertainties:
            {uncertainties_text}
            
            For each scenario:
            1. Provide a descriptive name that captures its essence
            2. Describe how each uncertainty plays out in this scenario
            3. Provide a narrative description of how this scenario unfolds over time
            4. Identify key indicators that would suggest this scenario is developing
            
            Make the scenarios diverse and cover different combinations of uncertainty outcomes.
            """
             # Include relevant data in prompt
            if relevant_data:
                prompt += "\n\nRelevant Economic and Geopolitical Data:\n"
                for data_source, data_items in relevant_data.items():
                    prompt += f"{data_source}:\n" + "\n".join(data_items) + "\n\n"

            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "predictive",
                "context": {"prompt": prompt, "research_results": research_results}
            })
            
            # Extract scenarios from response
            if isinstance(llama_response, dict) and "sections" in llama_response:
                content = ""
                for section_name, section_content in llama_response["sections"].items():
                    content += section_content + "\n\n"
                
                # Parse scenarios from content
                scenarios = self._parse_scenarios_from_text(content, uncertainties)
                if scenarios:
                    return scenarios
        
        # Fallback: Generate generic scenarios
        return self._generate_generic_scenarios(uncertainties, num_scenarios, time_horizon)
    
    def _parse_scenarios_from_text(self, text: str, uncertainties: List[Dict]) -> List[Dict]:
        """
        Parse scenarios from text.
        
        Args:
            text: Text containing scenario descriptions
            uncertainties: List of key uncertainties
            
        Returns:
            List of parsed scenarios
        """
        scenarios = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for scenario sections
        scenario_pattern = r'(?:^|\n)(?:Scenario|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Scenario|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        scenario_matches = re.findall(scenario_pattern, text, re.DOTALL)
        
        for i, match in enumerate(scenario_matches):
            scenario_num = match[0].strip()
            scenario_name = match[1].strip()
            scenario_content = match[2].strip()
            
            # Extract narrative
            narrative = scenario_content
            
            # Extract uncertainty outcomes
            uncertainty_outcomes = {}
            for uncertainty in uncertainties:
                factor = uncertainty.get("factor")
                if factor:
                    # Look for factor in content
                    factor_pattern = rf'{re.escape(factor)}.*?([^.\n]+)'
                    factor_match = re.search(factor_pattern, scenario_content, re.IGNORECASE)
                    if factor_match:
                        uncertainty_outcomes[factor] = factor_match.group(1).strip()
            
            # Extract indicators
            indicators = []
            indicators_pattern = r'(?:Indicators|Signs|Signals).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            indicators_match = re.search(indicators_pattern, scenario_content, re.IGNORECASE | re.DOTALL)
            if indicators_match:
                indicators_text = indicators_match.group(1)
                # Extract bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, indicators_text, re.DOTALL)
                if bullet_matches:
                    indicators = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', indicators_text)
                    indicators = [item.strip() for item in items if item.strip()]
            
            scenarios.append({
                "name": scenario_name,
                "narrative": narrative,
                "uncertainty_outcomes": uncertainty_outcomes,
                "indicators": indicators,
                "probability": self._estimate_scenario_probability(i, len(scenario_matches))
            })
        
        return scenarios
    
    def _generate_generic_scenarios(self, uncertainties: List[Dict], num_scenarios: int, time_horizon: str) -> List[Dict]:
        """
        Generate generic scenarios when other methods fail.
        
        Args:
            uncertainties: List of key uncertainties
            num_scenarios: Number of scenarios to generate
            time_horizon: Time horizon for scenarios
            
        Returns:
            List of generic scenarios
        """
        scenarios = []
        
        # Scenario templates
        templates = [
            {
                "name": "Baseline Continuation",
                "narrative": f"Current trends continue for the next {time_horizon} with minimal disruption. Incremental changes occur but the fundamental structure remains intact.",
                "indicators": ["Stable growth rates", "Continuation of current policies", "Gradual technology adoption"]
            },
            {
                "name": "Accelerated Change",
                "narrative": f"The pace of change accelerates dramatically over the next {time_horizon}. Disruptive technologies and new entrants reshape the landscape.",
                "indicators": ["Rapid technology adoption", "New market entrants gaining share", "Changing consumer behaviors"]
            },
            {
                "name": "Structural Disruption",
                "narrative": f"Major structural changes occur within the next {time_horizon}, fundamentally altering the existing paradigm. Traditional models are challenged.",
                "indicators": ["Regulatory overhaul", "Industry consolidation", "Emergence of new business models"]
            },
            {
                "name": "Crisis and Response",
                "narrative": f"A significant crisis occurs within the next {time_horizon}, followed by adaptation and new equilibrium. Resilience is tested.",
                "indicators": ["Early warning signals of instability", "Rapid policy responses", "Emergence of new leadership"]
            }
        ]
        
        # Generate scenarios based on templates and uncertainties
        for i in range(min(num_scenarios, len(templates))):
            template = templates[i]
            
            # Generate uncertainty outcomes
            uncertainty_outcomes = {}
            for j, uncertainty in enumerate(uncertainties):
                factor = uncertainty.get("factor")
                range_values = uncertainty.get("range", ["Low", "High"])
                
                # Alternate between low and high values based on scenario
                if i % 2 == 0:
                    outcome = range_values[0] if j % 2 == 0 else range_values[-1]
                else:
                    outcome = range_values[-1] if j % 2 == 0 else range_values[0]
                
                uncertainty_outcomes[factor] = outcome
            
            scenarios.append({
                "name": template["name"],
                "narrative": template["narrative"],
                "uncertainty_outcomes": uncertainty_outcomes,
                "indicators": template["indicators"],
                "probability": self._estimate_scenario_probability(i, num_scenarios)
            })
        
        return scenarios
    
    def _estimate_scenario_probability(self, index: int, total: int) -> float:
        """
        Estimate scenario probability based on index.
        
        Args:
            index: Scenario index
            total: Total number of scenarios
            
        Returns:
            Estimated probability
        """
        # Simple probability estimation
        if total <= 1:
            return 1.0
        
        # Base probability
        base_prob = 1.0 / total
        
        # Adjust based on index (first scenarios slightly more probable)
        if index == 0:
            return min(base_prob * 1.5, 0.5)
        elif index == 1:
            return min(base_prob * 1.2, 0.4)
        else:
            return max(base_prob * 0.8, 0.1)
    
    def _analyze_scenario_implications(self, question: str, scenarios: List[Dict], research_results: Dict, parameters: Dict) -> Dict:
        """
        Analyze implications of each scenario.
        
        Args:
            question: The analytical question
            scenarios: List of scenarios
            research_results: Research results
            parameters: Technique parameters
            
        Returns:
            Dictionary of scenario implications
        """
        logger.info("Analyzing scenario implications")
        
        implications = {}
        
        # Analyze implications using Llama4ScoutMCP
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            for scenario in scenarios:
                scenario_name = scenario.get("name")
                narrative = scenario.get("narrative")
                
                # Create prompt for implications analysis
                prompt = f"""
                Based on the following scenario and question, analyze the key implications.
                
                Question: {question}
                
                Scenario: {scenario_name}
                
                Scenario Description:
                {narrative}
                
                Please analyze:
                1. Strategic implications - How this scenario affects strategic decisions
                2. Operational implications - How this scenario affects operations
                3. Risk implications - Key risks that emerge in this scenario
                4. Opportunity implications - Key opportunities that emerge in this scenario
                
                Provide specific, actionable insights for each category.
                """
                # Include relevant data in prompt
                if relevant_data:
                    prompt += "\n\nRelevant Economic and Geopolitical Data:\n"
                    for data_source, data_items in relevant_data.items():
                        prompt += f"{data_source}:\n" + "\n".join(data_items) + "\n\n"


                
                # Call Llama4ScoutMCP
                llama_response = llama4_scout.process({
                    "question": question,
                    "analysis_type": "evaluative",
                    "context": {"prompt": prompt, "research_results": research_results}
                })
                
                # Extract implications from response
                if isinstance(llama_response, dict) and "sections" in llama_response:
                    content = ""
                    for section_name, section_content in llama_response["sections"].items():
                        content += section_content + "\n\n"
                    
                    # Parse implications from content
                    scenario_implications = self._parse_implications_from_text(content)
                    implications[scenario_name] = scenario_implications
                else:
                    # Fallback: Generate generic implications
                    implications[scenario_name] = self._generate_generic_implications(scenario)
        else:
            # Generate generic implications for all scenarios
            for scenario in scenarios:
                scenario_name = scenario.get("name")
                implications[scenario_name] = self._generate_generic_implications(scenario)
        
        return implications
    
    def _parse_implications_from_text(self, text: str) -> Dict:
        """
        Parse implications from text.
        
        Args:
            text: Text containing implications
            
        Returns:
            Dictionary of parsed implications
        """
        implications = {
            "strategic": [],
            "operational": [],
            "risks": [],
            "opportunities": []
        }
        
        # Simple parsing based on patterns
        import re
        
        # Map of section names to keys
        section_map = {
            "strategic": ["strategic", "strategy", "strategic implications"],
            "operational": ["operational", "operations", "operational implications"],
            "risks": ["risk", "risks", "risk implications"],
            "opportunities": ["opportunity", "opportunities", "opportunity implications"]
        }
        
        # Look for sections
        for key, section_names in section_map.items():
            section_pattern = '|'.join(section_names)
            section_regex = rf'(?:^|\n)(?:{section_pattern}).*?(?::|$)(.*?)(?=(?:\n\n|\n(?:{"|".join(section_map.values())})|$))'
            section_match = re.search(section_regex, text, re.IGNORECASE | re.DOTALL)
            
            if section_match:
                section_text = section_match.group(1)
                
                # Extract bullet points
                bullet_pattern = r'(?:^|\n)(?:\d+\.|[-•*]\s+)(.*?)(?=(?:\n(?:\d+\.|[-•*]\s+)|\Z))'
                bullet_matches = re.findall(bullet_pattern, section_text, re.DOTALL)
                
                if bullet_matches:
                    implications[key] = [item.strip() for item in bullet_matches]
                else:
                    # Split by newlines or sentences
                    items = re.split(r'(?:\n|\.(?:\s+|$))', section_text)
                    implications[key] = [item.strip() for item in items if item.strip()]
        
        return implications
    
    def _generate_generic_implications(self, scenario: Dict) -> Dict:
        """
        Generate generic implications for a scenario.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            Dictionary of generic implications
        """
        scenario_name = scenario.get("name", "").lower()
        
        # Base implications
        implications = {
            "strategic": [
                "Reassess long-term strategic goals and alignment",
                "Consider portfolio diversification to manage uncertainty",
                "Evaluate partnership and alliance opportunities"
            ],
            "operational": [
                "Review operational flexibility and adaptability",
                "Assess supply chain resilience",
                "Consider workforce skills and capabilities needed"
            ],
            "risks": [
                "Identify emerging competitive threats",
                "Monitor regulatory and compliance changes",
                "Prepare for potential disruptions to business model"
            ],
            "opportunities": [
                "Explore new market segments that may emerge",
                "Identify potential first-mover advantages",
                "Consider innovation opportunities aligned with scenario"
            ]
        }
        
        # Customize based on scenario name
        if "continuation" in scenario_name or "baseline" in scenario_name:
            implications["strategic"].append("Optimize current strategy for incremental improvements")
            implications["operational"].append("Focus on operational excellence and efficiency")
            implications["risks"].append("Guard against complacency and disruption blindness")
            implications["opportunities"].append("Leverage stability for long-term investments")
        
        elif "accelerated" in scenario_name or "rapid" in scenario_name:
            implications["strategic"].append("Prepare for faster decision cycles and market changes")
            implications["operational"].append("Increase agility and responsiveness in operations")
            implications["risks"].append("Monitor for signs of market overheating or bubbles")
            implications["opportunities"].append("Position for early adoption of emerging technologies")
        
        elif "disruption" in scenario_name or "transform" in scenario_name:
            implications["strategic"].append("Develop contingency plans for structural industry changes")
            implications["operational"].append("Build capabilities for radical operational transformation")
            implications["risks"].append("Prepare for potential obsolescence of current business models")
            implications["opportunities"].append("Identify potential to lead industry transformation")
        
        elif "crisis" in scenario_name or "downturn" in scenario_name:
            implications["strategic"].append("Develop crisis response and recovery strategies")
            implications["operational"].append("Strengthen business continuity planning")
            implications["risks"].append("Assess financial resilience and liquidity needs")
            implications["opportunities"].append("Identify potential for counter-cyclical investments")
        
        return implications
    
    def _identify_common_elements(self, scenarios: List[Dict], implications: Dict) -> Dict:
        """
        Identify common elements across scenarios.
        
        Args:
            scenarios: List of scenarios
            implications: Dictionary of scenario implications
            
        Returns:
            Dictionary of common elements
        """
        logger.info("Identifying common elements across scenarios")
        
        # Initialize common elements
        common_elements = {
            "strategic_imperatives": [],
            "operational_needs": [],
            "consistent_risks": [],
            "persistent_opportunities": []
        }
        
        # Extract all implications
        all_strategic = []
        all_operational = []
        all_risks = []
        all_opportunities = []
        
        for scenario_name, scenario_implications in implications.items():
            all_strategic.extend(scenario_implications.get("strategic", []))
            all_operational.extend(scenario_implications.get("operational", []))
            all_risks.extend(scenario_implications.get("risks", []))
            all_opportunities.extend(scenario_implications.get("opportunities", []))
        
        # Identify common elements using simple text similarity
        common_elements["strategic_imperatives"] = self._find_common_items(all_strategic)
        common_elements["operational_needs"] = self._find_common_items(all_operational)
        common_elements["consistent_risks"] = self._find_common_items(all_risks)
        common_elements["persistent_opportunities"] = self._find_common_items(all_opportunities)
        
        return common_elements
    
    def _find_common_items(self, items: List[str], similarity_threshold: float = 0.6) -> List[str]:
        """
        Find common items in a list based on text similarity.
        
        Args:
            items: List of text items
            similarity_threshold: Threshold for considering items similar
            
        Returns:
            List of common items
        """
        if not items:
            return []
        
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
        for item in items:
            found_group = False
            for group in groups:
                for group_item in group:
                    if similarity(item, group_item) > similarity_threshold:
                        group.append(item)
                        found_group = True
                        break
                if found_group:
                    break
            
            if not found_group:
                groups.append([item])
        
        # Select common groups (appearing in multiple items)
        common_groups = [group for group in groups if len(group) > 1]
        
        # Select representative item from each common group
        common_items = []
        for group in common_groups:
            # Use the shortest item as representative
            representative = min(group, key=len)
            common_items.append(representative)
        
        return common_items
    
    def _determine_robust_strategies(self, question: str, scenarios: List[Dict], implications: Dict, parameters: Dict) -> List[Dict]:
        """
        Determine robust strategies that work across scenarios.
        
        Args:
            question: The analytical question
            scenarios: List of scenarios
            implications: Dictionary of scenario implications
            parameters: Technique parameters
            
        Returns:
            List of robust strategies
        """
        logger.info("Determining robust strategies")
        
        # Determine robust strategies using Llama4ScoutMCP
        llama4_scout = self.get_mcp("llama4_scout")
        if llama4_scout:
            # Create scenario summaries
            scenario_summaries = ""
            for scenario in scenarios:
                scenario_name = scenario.get("name")
                scenario_summaries += f"\n## {scenario_name}\n"
                scenario_summaries += f"Narrative: {scenario.get('narrative', '')}\n"
                
                # Add implications
                if scenario_name in implications:
                    scenario_impl = implications[scenario_name]
                    
                    strategic = scenario_impl.get("strategic", [])
                    if strategic:
                        scenario_summaries += "Strategic Implications: " + "; ".join(strategic[:3]) + "\n"
                    
                    risks = scenario_impl.get("risks", [])
                    if risks:
                        scenario_summaries += "Key Risks: " + "; ".join(risks[:3]) + "\n"
                    
                    opportunities = scenario_impl.get("opportunities", [])
                    if opportunities:
                        scenario_summaries += "Key Opportunities: " + "; ".join(opportunities[:3]) + "\n"
            
            # Create prompt for robust strategies
            prompt = f"""
            Based on the following question and scenario analysis, identify 3-5 robust strategies that would be effective across multiple scenarios.
            
            Question: {question}
            
            Scenarios:
            {scenario_summaries}
            
            For each robust strategy:
            1. Provide a clear name and description
            2. Explain why it's robust across different scenarios
            3. Identify key implementation considerations
            4. Assess the potential impact and feasibility
            
            Focus on strategies that provide value in multiple scenarios and help manage key uncertainties.
            """
            # Include relevant data in prompt
            if relevant_data:
                prompt += "\n\nRelevant Economic and Geopolitical Data:\n"
                for data_source, data_items in relevant_data.items():
                    prompt += f"{data_source}:\n" + "\n".join(data_items) + "\n\n"

            
            # Call Llama4ScoutMCP
            llama_response = llama4_scout.process({
                "question": question,
                "analysis_type": "strategic",
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
        
        # Fallback: Generate generic robust strategies
        return self._generate_generic_robust_strategies()
    
    def _parse_strategies_from_text(self, text: str) -> List[Dict]:
        """
        Parse strategies from text.
        
        Args:
            text: Text containing strategy descriptions
            
        Returns:
            List of parsed strategies
        """
        strategies = []
        
        # Simple parsing based on patterns
        import re
        
        # Look for strategy sections
        strategy_pattern = r'(?:^|\n)(?:Strategy|#)\s*(\d+|[A-Z])(?::|\.|\))\s*([^\n]+)(?:\n|$)(.*?)(?=(?:\n(?:Strategy|#)\s*(?:\d+|[A-Z])(?::|\.|\))|$))'
        strategy_matches = re.findall(strategy_pattern, text, re.DOTALL)
        
        for match in strategy_matches:
            strategy_num = match[0].strip()
            strategy_name = match[1].strip()
            strategy_content = match[2].strip()
            
            # Extract description
            description = strategy_content
            
            # Extract rationale
            rationale = ""
            rationale_pattern = r'(?:Why it\'s robust|Rationale|Robustness).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            rationale_match = re.search(rationale_pattern, strategy_content, re.IGNORECASE | re.DOTALL)
            if rationale_match:
                rationale = rationale_match.group(1).strip()
            
            # Extract implementation
            implementation = ""
            implementation_pattern = r'(?:Implementation|How to implement|Key considerations).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            implementation_match = re.search(implementation_pattern, strategy_content, re.IGNORECASE | re.DOTALL)
            if implementation_match:
                implementation = implementation_match.group(1).strip()
            
            # Extract impact
            impact = ""
            impact_pattern = r'(?:Impact|Potential impact|Feasibility).*?(?::|$)(.*?)(?=(?:\n\n|\Z))'
            impact_match = re.search(impact_pattern, strategy_content, re.IGNORECASE | re.DOTALL)
            if impact_match:
                impact = impact_match.group(1).strip()
            
            strategies.append({
                "name": strategy_name,
                "description": description,
                "rationale": rationale,
                "implementation": implementation,
                "impact": impact
            })
        
        return strategies
    
    def _generate_generic_robust_strategies(self) -> List[Dict]:
        """
        Generate generic robust strategies when other methods fail.
        
        Returns:
            List of generic robust strategies
        """
        return [
            {
                "name": "Adaptive Flexibility",
                "description": "Build organizational capabilities for rapid adaptation to changing conditions",
                "rationale": "Flexibility provides value across multiple scenarios by enabling quick pivots as conditions change",
                "implementation": "Develop modular systems, cross-functional teams, and rapid decision processes",
                "impact": "High impact with medium implementation difficulty"
            },
            {
                "name": "No-Regrets Investments",
                "description": "Focus on investments that deliver value across all plausible scenarios",
                "rationale": "No-regrets moves reduce downside risk while maintaining upside potential",
                "implementation": "Identify core capabilities needed regardless of which scenario unfolds",
                "impact": "Medium impact with low implementation difficulty"
            },
            {
                "name": "Strategic Hedging",
                "description": "Develop a portfolio of initiatives that collectively address multiple scenarios",
                "rationale": "Hedging provides protection against downside risks while positioning for opportunities",
                "implementation": "Allocate resources across initiatives aligned with different scenarios",
                "impact": "High impact with high implementation difficulty"
            },
            {
                "name": "Enhanced Sensing Capabilities",
                "description": "Develop robust early warning systems to detect which scenarios are emerging",
                "rationale": "Early detection allows for more timely and effective responses to emerging conditions",
                "implementation": "Establish key indicators to monitor and regular review processes",
                "impact": "Medium impact with medium implementation difficulty"
            }
        ]
    
    def _extract_findings(self, scenarios: List[Dict], implications: Dict, robust_strategies: List[Dict]) -> List[Dict]:
        """
        Extract key findings from scenario analysis.
        
        Args:
            scenarios: List of scenarios
            implications: Dictionary of scenario implications
            robust_strategies: List of robust strategies
            
        Returns:
            List of key findings
        """
        findings = []
        
        # Add finding about scenario diversity
        if scenarios:
            findings.append({
                "finding": f"Analysis identified {len(scenarios)} distinct scenarios with different implications",
                "confidence": "high",
                "source": "scenario_triangulation"
            })
        
        # Add finding about most probable scenario
        most_probable = max(scenarios, key=lambda x: x.get("probability", 0)) if scenarios else None
        if most_probable:
            findings.append({
                "finding": f"The '{most_probable.get('name')}' scenario appears most probable based on current trends",
                "confidence": "medium",
                "source": "scenario_triangulation"
            })
        
        # Add finding about robust strategies
        if robust_strategies:
            strategy_names = [s.get("name") for s in robust_strategies]
            findings.append({
                "finding": f"Key robust strategies identified: {', '.join(strategy_names)}",
                "confidence": "medium",
                "source": "scenario_triangulation"
            })
        
        # Add finding about common implications
        common_risks = []
        for scenario_name, scenario_impl in implications.items():
            risks = scenario_impl.get("risks", [])
            if risks:
                common_risks.extend(risks[:1])  # Take first risk from each scenario
        
        if common_risks:
            findings.append({
                "finding": f"Common risk across scenarios: {common_risks[0]}",
                "confidence": "medium",
                "source": "scenario_triangulation"
            })
        
        return findings
    
    def _extract_assumptions(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Extract assumptions from scenario analysis.
        
        Args:
            scenarios: List of scenarios
            
        Returns:
            List of assumptions
        """
        assumptions = []
        
        # Add assumption about scenario timeframe
        assumptions.append({
            "assumption": "The analysis assumes that scenarios represent plausible futures within the specified time horizon",
            "criticality": "high",
            "source": "scenario_triangulation"
        })
        
        # Add assumption about scenario probabilities
        assumptions.append({
            "assumption": "The analysis assumes that scenario probabilities are indicative rather than precise",
            "criticality": "medium",
            "source": "scenario_triangulation"
        })
        
        # Add assumption about key drivers
        assumptions.append({
            "assumption": "The analysis assumes that identified uncertainties are the primary drivers of future outcomes",
            "criticality": "high",
            "source": "scenario_triangulation"
        })
        
        return assumptions
    
    def _extract_scenario_uncertainties(self, scenarios: List[Dict], uncertainties: List[Dict]) -> List[Dict]:
        """
        Extract uncertainties from scenario analysis.
        
        Args:
            scenarios: List of scenarios
            uncertainties: List of key uncertainties
            
        Returns:
            List of uncertainties
        """
        result_uncertainties = []
        
        # Add uncertainty about scenario probabilities
        result_uncertainties.append({
            "uncertainty": "The relative probabilities of different scenarios are highly uncertain",
            "impact": "high",
            "source": "scenario_triangulation"
        })
        
        # Add uncertainties from key uncertainties
        for uncertainty in uncertainties:
            factor = uncertainty.get("factor", "")
            description = uncertainty.get("description", "")
            
            result_uncertainties.append({
                "uncertainty": f"Uncertainty in {factor}: {description}",
                "impact": "medium",
                "source": "scenario_triangulation"
            })
        
        # Add uncertainty about timing
        result_uncertainties.append({
            "uncertainty": "The timing and pace of transitions between current state and scenario outcomes is uncertain",
            "impact": "medium",
            "source": "scenario_triangulation"
        })
        
        return result_uncertainties
