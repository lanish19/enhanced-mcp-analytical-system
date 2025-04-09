"""
Cross-Impact Analysis Technique implementation.
This module provides the CrossImpactAnalysisTechnique class for analyzing interactions between factors.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional

from .analytical_technique import AnalyticalTechnique
from utils.llm_integration import call_llm, extract_content, parse_json_response, MODEL_CONFIG
from src.mcps.economics_mcp import EconomicsMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossImpactAnalysisTechnique(AnalyticalTechnique):
    """
    Analyzes interactions and mutual influences between different factors.
    
    This technique systematically assesses how different factors influence each other,
    identifying reinforcing and balancing relationships, and revealing non-obvious
    second-order effects.
    """
    
    def execute(self, context, parameters):
        """
        Execute the cross-impact analysis technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing cross-impact analysis results
        """
        logger.info(f"Executing CrossImpactAnalysisTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        source_technique = parameters.get("source_technique", None)
        
        # Get relevant data from Domain MCPs
        economic_data, geopolitical_data = self._fetch_domain_data(context)
        max_factors = parameters.get("max_factors", 8)
        
        # Step 1: Identify key factors
        factors = self._identify_factors(context, source_technique, max_factors)
        
        # Step 2: Create cross-impact matrix, including domain data in the prompt
        impact_matrix = self._create_impact_matrix(context.question, factors)
        
        # Step 3: Analyze matrix for insights
        matrix_analysis = self._analyze_matrix(impact_matrix)
        
        # Step 4: Identify second-order effects
        second_order_effects = self._identify_second_order_effects(impact_matrix)
        
        # Step 5: Generate final synthesis, including domain data in the prompt
        synthesis = self._generate_synthesis(context.question, factors, impact_matrix, matrix_analysis, second_order_effects)
        
        return {
            "technique": "Cross-Impact Analysis",
            "status": "Completed",
            "factors": factors,
            "impact_matrix": impact_matrix,
            "matrix_analysis": matrix_analysis,
            "second_order_effects": second_order_effects,
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
        return ["factor_extraction_mcp", "impact_analysis_mcp"]

    def _fetch_domain_data(self, context):
        """
        Fetch relevant economic and geopolitical data based on the question.

        Args:
            context: The analysis context

        Returns:
            Tuple: economic_data, geopolitical_data
        """
        economic_data = {}
        geopolitical_data = {}
        try:
            economics_mcp: EconomicsMCP = self.mcp_registry.get_mcp("economics_mcp")
            if economics_mcp:
                economic_data = economics_mcp.fetch_fred_data("GDP", "2020-01-01", "2023-12-31")
            geopolitics_mcp = self.mcp_registry.get_mcp("geopolitics_mcp")
            if geopolitics_mcp:
                geopolitical_data = geopolitics_mcp.fetch_gdelt_data("world", "2023-01-01", "2023-12-31")
        except Exception as e:
            logger.error(f"Error fetching domain data: {e}")
            context.log_error(f"Error fetching domain data: {e}")

        return economic_data, geopolitical_data
    
    def _identify_factors(self, context, source_technique, max_factors):
        """
        Identify key factors for cross-impact analysis.
        
        Args:
            context: The analysis context
            source_technique: Name of technique to source factors from
            max_factors: Maximum number of factors to include
            
        Returns:
            List of factor dictionaries
        """
        logger.info(f"Identifying key factors (max: {max_factors})...")
        
        # Try to get factors from source technique if specified
        if source_technique and source_technique in context.results:
            source_results = context.results[source_technique]
            
            # Look for factors in common fields
            potential_factors = []
            
            # Check for entities in causal network analysis
            if "entities" in source_results:
                potential_factors.extend(source_results["entities"])
            
            # Check for key drivers
            if "key_drivers" in source_results:
                for driver in source_results["key_drivers"]:
                    potential_factors.append({"name": driver, "category": "Driver"})
            
            # Check for scenarios in scenario triangulation
            if "scenarios" in source_results:
                for scenario in source_results["scenarios"]:
                    if "key_factors" in scenario:
                        for factor in scenario["key_factors"]:
                            potential_factors.append({"name": factor, "category": "Scenario Factor"})
            
            # If we found potential factors, process them
            if potential_factors:
                logger.info(f"Found {len(potential_factors)} potential factors from {source_technique}")
                return self._process_factors(potential_factors, max_factors)
        
        # Use factor extraction MCP if available
        factor_mcp = self.mcp_registry.get_mcp("factor_extraction_mcp")
        
        if factor_mcp:
            try:
                logger.info("Using factor extraction MCP")
                factors = factor_mcp.extract_factors(context.question, max_factors)
                return factors
            except Exception as e:
                logger.error(f"Error using factor extraction MCP: {e}")
                # Fall through to LLM-based extraction
        
        # Use LLM to identify factors
        prompt = f"""
        Identify the key factors that are relevant to the following analytical question:
        
        "{context.question}"
        
        A factor is a variable, trend, force, or element that influences the situation or outcome.
        
        For this analysis:
        1. Identify {max_factors} distinct factors that are most relevant to the question
        2. Include a mix of different types of factors (e.g., economic, political, technological, social)
        3. Focus on factors that are likely to interact with each other
        4. Provide a clear name and description for each factor
        
        Return your response as a JSON object with the following structure:
        {{
            "factors": [
                {{
                    "name": "Name of the factor",
                    "description": "Brief description of the factor",
                    "category": "Category of the factor (e.g., Economic, Political, Technological, Social)"
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
                logger.error(f"Error identifying factors: {parsed_response.get('error')}")
                return self._generate_fallback_factors(context.question, max_factors)
            
            factors = parsed_response.get("factors", [])
            
            if not factors:
                logger.warning("No factors identified")
                return self._generate_fallback_factors(context.question, max_factors)
            
            return self._process_factors(factors, max_factors)
        
        except Exception as e:
            logger.error(f"Error parsing factors: {e}")
            return self._generate_fallback_factors(context.question, max_factors)
    
    def _process_factors(self, factors, max_factors):
        """
        Process and normalize factors.
        
        Args:
            factors: List of raw factor dictionaries
            max_factors: Maximum number of factors to include
            
        Returns:
            List of processed factor dictionaries
        """
        processed_factors = []
        seen_names = set()
        
        for factor in factors:
            # Extract name and ensure it exists
            name = factor.get("name", "")
            if not name:
                continue
            
            # Skip duplicates
            if name.lower() in seen_names:
                continue
            
            seen_names.add(name.lower())
            
            # Ensure all required fields exist
            processed_factor = {
                "name": name,
                "description": factor.get("description", f"Description of {name}"),
                "category": factor.get("category", "General")
            }
            
            processed_factors.append(processed_factor)
            
            # Stop if we've reached the maximum
            if len(processed_factors) >= max_factors:
                break
        
        return processed_factors[:max_factors]
    
    def _generate_fallback_factors(self, question, max_factors):
        """
        Generate fallback factors when identification fails.
        
        Args:
            question: The analytical question
            max_factors: Maximum number of factors to generate
            
        Returns:
            List of fallback factor dictionaries
        """
        fallback_factors = [
            {
                "name": "Economic Conditions",
                "description": "Overall economic environment including growth, inflation, and employment",
                "category": "Economic"
            },
            {
                "name": "Technological Innovation",
                "description": "Rate and direction of technological change and adoption",
                "category": "Technological"
            },
            {
                "name": "Regulatory Environment",
                "description": "Laws, regulations, and policies affecting the domain",
                "category": "Political"
            },
            {
                "name": "Social Attitudes",
                "description": "Prevailing social values, beliefs, and behaviors",
                "category": "Social"
            },
            {
                "name": "Competitive Dynamics",
                "description": "Interactions between competing entities in the domain",
                "category": "Market"
            },
            {
                "name": "Resource Availability",
                "description": "Access to key resources including capital, talent, and materials",
                "category": "Resource"
            },
            {
                "name": "Global Events",
                "description": "Major international developments affecting multiple domains",
                "category": "External"
            },
            {
                "name": "Demographic Trends",
                "description": "Changes in population size, composition, and distribution",
                "category": "Demographic"
            }
        ]
        
        return fallback_factors[:max_factors]
    
    def _create_impact_matrix(self, question, factors, economic_data=None, geopolitical_data=None):
        """
        Create a cross-impact matrix assessing how factors influence each other.
        
        Args:
            economic_data (dict): Economic data from the EconomicsMCP.
            geopolitical_data (dict): Geopolitical data from the GeopoliticsMCP.
        Args:
            question: The analytical question
            factors: List of factor dictionaries
            
        Returns:
            Dictionary containing the impact matrix
        """
        logger.info(f"Creating cross-impact matrix for {len(factors)} factors...")
        
        # Use impact analysis MCP if available
        impact_mcp = self.mcp_registry.get_mcp("impact_analysis_mcp")
        
        if impact_mcp:
            try:
                logger.info("Using impact analysis MCP")
                impact_matrix = impact_mcp.create_impact_matrix(question, factors)
                return impact_matrix
            except Exception as e:
                logger.error(f"Error using impact analysis MCP: {e}")
                # Fall through to LLM-based creation
        
        # Initialize matrix structure
        factor_names = [factor["name"] for factor in factors]
        matrix = {
            "factors": factor_names,
            "impacts": []
        }
        
        # For each factor, assess its impact on all other factors
        for i, source_factor in enumerate(factors):
            source_name = source_factor["name"]
            source_desc = source_factor["description"]
            
            # Include economic and geopolitical data in the prompt, if available
            domain_data_context = ""
            if economic_data:
                domain_data_context += f"\nEconomic Data:\n{json.dumps(economic_data, indent=2)}"
            if geopolitical_data:
                domain_data_context += f"\nGeopolitical Data:\n{json.dumps(geopolitical_data, indent=2)}"
            
            # Create a row of impacts
            impact_row = []
            
            for j, target_factor in enumerate(factors):
                target_name = target_factor["name"]
                target_desc = target_factor["description"]
                
                # Skip self-impact
                if i == j:
                    impact_row.append({
                        "source": source_name,
                        "target": target_name,
                        "impact": 0,
                        "description": "Self-impact not assessed"
                    })
                    continue
                
                # Use LLM to assess impact
                prompt = f"""
                Assess the impact of one factor on another in the context of this question:
                
                Question: "{question}"
                
                Source Factor: {source_name}
                Source Description: {source_desc}
                
                Target Factor: {target_name}
                Target Description: {target_desc}
                
                Assess:
                1. How strongly does the source factor influence the target factor?
                2. What is the nature of this influence?
                
                Rate the impact on a scale from -3 to +3:
                -3: Strong negative impact (source strongly decreases target)
                -2: Moderate negative impact
                -1: Weak negative impact
                0: No significant impact
                +1: Weak positive impact
                +2: Moderate positive impact
                +3: Strong positive impact (source strongly increases target)
                
                Return your assessment as a JSON object with the following structure:
                {{
                    "impact": Impact score (-3 to +3)
                    
                    {domain_data_context}
                    "description": "Brief explanation of the impact relationship"
                }}
                """
                
                model_config = MODEL_CONFIG["llama4"]
                response = call_llm(prompt, model_config)
                content = extract_content(response)
                
                try:
                    parsed_response = parse_json_response(content)
                    
                    if parsed_response.get("fallback_generated"):
                        logger.error(f"Error assessing impact: {parsed_response.get('error')}")
                        impact = 0
                        description = f"Error assessing impact: {parsed_response.get('error', 'Unknown error')}"
                    else:
                        impact = parsed_response.get("impact", 0)
                        description = parsed_response.get("description", "No description provided")
                        
                        # Validate impact score
                        try:
                            impact = int(impact)
                            if impact < -3:
                                impact = -3
                            elif impact > 3:
                                impact = 3
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid impact value: {impact}, defaulting to 0")
                            impact = 0
                    
                    impact_row.append({
                        "source": source_name,
                        "target": target_name,
                        "impact": impact,
                        "description": description
                    })
                
                except Exception as e:
                    logger.error(f"Error parsing impact assessment: {e}")
                    impact_row.append({
                        "source": source_name,
                        "target": target_name,
                        "impact": 0,
                        "description": f"Error assessing impact: {str(e)}"
                    })
            
            matrix["impacts"].append(impact_row)
        
        return matrix
    
    def _analyze_matrix(self, impact_matrix):
        """
        Analyze the cross-impact matrix for insights.
        
        Args:
            impact_matrix: Dictionary containing the impact matrix
            
        Returns:
            Dictionary containing matrix analysis
        """
        logger.info("Analyzing cross-impact matrix...")
        
        factors = impact_matrix["factors"]
        impacts = impact_matrix["impacts"]
        
        # Calculate activity and passivity scores
        activity_scores = []
        passivity_scores = []
        
        for i, factor in enumerate(factors):
            # Activity: sum of absolute values in row (how much this factor affects others)
            activity = sum(abs(impact["impact"]) for impact in impacts[i] if impact["source"] == factor)
            
            # Passivity: sum of absolute values in column (how much this factor is affected by others)
            passivity = sum(abs(impacts[j][i]["impact"]) for j in range(len(factors)) if j != i)
            
            activity_scores.append({
                "factor": factor,
                "activity": activity
            })
            
            passivity_scores.append({
                "factor": factor,
                "passivity": passivity
            })
        
        # Sort by scores
        activity_scores.sort(key=lambda x: x["activity"], reverse=True)
        passivity_scores.sort(key=lambda x: x["passivity"], reverse=True)
        
        # Identify factor roles
        factor_roles = []
        
        for i, factor in enumerate(factors):
            activity = next(item["activity"] for item in activity_scores if item["factor"] == factor)
            passivity = next(item["passivity"] for item in passivity_scores if item["factor"] == factor)
            
            # Determine role based on activity and passivity
            if activity > passivity and activity > 0:
                role = "Driver"
            elif passivity > activity and passivity > 0:
                role = "Outcome"
            elif activity > 0 and passivity > 0:
                role = "Linkage"
            else:
                role = "Autonomous"
            
            factor_roles.append({
                "factor": factor,
                "activity": activity,
                "passivity": passivity,
                "role": role
            })
        
        # Identify strongest relationships
        relationships = []
        
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j:
                    impact = impacts[i][j]["impact"]
                    if abs(impact) >= 2:  # Only include moderate to strong impacts
                        relationships.append({
                            "source": factors[i],
                            "target": factors[j],
                            "impact": impact,
                            "description": impacts[i][j]["description"]
                        })
        
        # Sort by absolute impact
        relationships.sort(key=lambda x: abs(x["impact"]), reverse=True)
        
        return {
            "activity_scores": activity_scores,
            "passivity_scores": passivity_scores,
            "factor_roles": factor_roles,
            "strongest_relationships": relationships
        }
    
    def _identify_second_order_effects(self, impact_matrix):
        """
        Identify second-order effects in the cross-impact matrix.
        
        Args:
            impact_matrix: Dictionary containing the impact matrix
            
        Returns:
            List of second-order effect dictionaries
        """
        logger.info("Identifying second-order effects...")
        
        factors = impact_matrix["factors"]
        impacts = impact_matrix["impacts"]
        
        # Identify potential chains of influence
        chains = []
        
        for i in range(len(factors)):
            for j in range(len(factors)):
                if i != j and abs(impacts[i][j]["impact"]) >= 2:  # First link is strong
                    for k in range(len(factors)):
                        if k != i and k != j and abs(impacts[j][k]["impact"]) >= 2:  # Second link is strong
                            # Found a potential chain: i -> j -> k
                            chains.append({
                                "path": [factors[i], factors[j], factors[k]],
                                "impacts": [impacts[i][j]["impact"], impacts[j][k]["impact"]],
                                "net_effect": impacts[i][j]["impact"] * impacts[j][k]["impact"]  # Multiply to get net effect
                            })
        
        # Sort by absolute net effect
        chains.sort(key=lambda x: abs(x["net_effect"]), reverse=True)
        
        # Use LLM to describe the most significant chains
        second_order_effects = []
        
        for chain in chains[:5]:  # Process top 5 chains
            path = chain["path"]
            impacts = chain["impacts"]
            net_effect = chain["net_effect"]
            
            prompt = f"""
            Describe the following chain of influence:
            
            Path: {path[0]} -> {path[1]} -> {path[2]}
            First Impact: {impacts[0]} (on a scale from -3 to +3)
            Second Impact: {impacts[1]} (on a scale from -3 to +3)
            Net Effect: {net_effect}
            
            Provide:
            1. A clear description of how this chain of influence works
            2. An assessment of whether this creates a reinforcing or balancing effect
            3. The potential implications of this second-order effect
            
            Return your description as a JSON object with the following structure:
            {{
                "description": "Clear description of how this chain works",
                "effect_type": "Reinforcing/Balancing",
                "implications": ["Implication 1", "Implication 2", ...]
            }}
            """
            
            model_config = MODEL_CONFIG["llama4"]
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            
            try:
                parsed_response = parse_json_response(content)
                
                if parsed_response.get("fallback_generated"):
                    logger.error(f"Error describing second-order effect: {parsed_response.get('error')}")
                    description = {
                        "description": f"Error describing effect: {parsed_response.get('error', 'Unknown error')}",
                        "effect_type": "Unknown",
                        "implications": ["Unable to determine implications due to error"]
                    }
                else:
                    description = parsed_response
                
                second_order_effect = {
                    "path": path,
                    "impacts": impacts,
                    "net_effect": net_effect,
                    "description": description.get("description", "No description provided"),
                    "effect_type": description.get("effect_type", "Unknown"),
                    "implications": description.get("implications", [])
                }
                
                second_order_effects.append(second_order_effect)
            
            except Exception as e:
                logger.error(f"Error parsing second-order effect description: {e}")
                second_order_effect = {
                    "path": path,
                    "impacts": impacts,
                    "net_effect": net_effect,
                    "description": f"Error describing effect: {str(e)}",
                    "effect_type": "Unknown",
                    "implications": ["Unable to determine implications due to error"]
                }
                
                second_order_effects.append(second_order_effect)
        
        return second_order_effects
    
    def _generate_synthesis(self, question, factors, impact_matrix, matrix_analysis, second_order_effects):
        """
        Generate a synthesis of the cross-impact analysis.
        
        Args:
            question: The analytical question
            factors: List of factor dictionaries
            impact_matrix: Dictionary containing the impact matrix
            matrix_analysis: Dictionary containing matrix analysis
            second_order_effects: List of second-order effect dictionaries
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of cross-impact analysis...")
        
        prompt = f"""
        Synthesize the following cross-impact analysis for the question:
        
        "{question}"
        
        Factors:
        {json.dumps([f["name"] for f in factors], indent=2)}
        
        Factor Roles:
        {json.dumps(matrix_analysis.get("factor_roles", []), indent=2)}
        
        Strongest Relationships:
        {json.dumps(matrix_analysis.get("strongest_relationships", []), indent=2)}
        
        Second-Order Effects:
        {json.dumps([{
            "path": e.get("path", []),
            "description": e.get("description", ""),
            "effect_type": e.get("effect_type", ""),
            "implications": e.get("implications", [])
        } for e in second_order_effects], indent=2)}
        
        Based on this cross-impact analysis:
        1. What are the key drivers in the system?
        2. What are the most important relationships between factors?
        3. What are the most significant second-order effects?
        4. What insights does this provide for the original question?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the cross-impact analysis
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "key_drivers": ["Driver 1", "Driver 2", ...],
            "key_relationships": ["Relationship 1", "Relationship 2", ...],
            "key_second_order_effects": ["Effect 1", "Effect 2", ...],
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
                    "key_drivers": ["Error in synthesis generation"],
                    "key_relationships": ["Error in synthesis generation"],
                    "key_second_order_effects": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_drivers": ["Error in synthesis generation"],
                "key_relationships": ["Error in synthesis generation"],
                "key_second_order_effects": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
