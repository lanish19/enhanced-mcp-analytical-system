"""
System Dynamics Modeling Technique implementation.
This module provides the SystemDynamicsModelingTechnique class for modeling complex system behaviors.
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

class SystemDynamicsModelingTechnique(AnalyticalTechnique):
    """
    Models complex system behaviors with feedback loops and time delays.
    
    This technique creates qualitative and semi-quantitative models of system behavior,
    identifying stocks, flows, feedback loops, and time delays to understand
    how a system evolves over time.
    """
    
    def execute(self, context, parameters):
        """
        Execute the system dynamics modeling technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing system dynamics modeling results
        """
        logger.info(f"Executing SystemDynamicsModelingTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        source_technique = parameters.get("source_technique", None)
        time_horizon = parameters.get("time_horizon", "5 years")
        
        # Step 1: Define system boundary and key variables
        system_definition = self._define_system(context, source_technique)
        
        # Step 2: Identify stocks and flows
        stocks_flows = self._identify_stocks_flows(context.question, system_definition)
        
        # Step 3: Identify feedback loops
        feedback_loops = self._identify_feedback_loops(stocks_flows)
        
        # Step 4: Analyze system behavior
        system_behavior = self._analyze_system_behavior(context.question, stocks_flows, feedback_loops, time_horizon)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, system_definition, stocks_flows, feedback_loops, system_behavior)
        
        return {
            "technique": "System Dynamics Modeling",
            "status": "Completed",
            "system_definition": system_definition,
            "stocks_flows": stocks_flows,
            "feedback_loops": feedback_loops,
            "system_behavior": system_behavior,
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
        return ["systems_thinking_mcp", "causal_modeling_mcp"]
    
    def _define_system(self, context, source_technique):
        """
        Define the system boundary and key variables.
        
        Args:
            context: The analysis context
            source_technique: Name of technique to source variables from
            
        Returns:
            Dictionary containing system definition
        """
        logger.info("Defining system boundary and key variables...")
        
        # Try to get variables from source technique if specified
        if source_technique and source_technique in context.results:
            source_results = context.results[source_technique]
            
            # Look for variables in common fields
            potential_variables = []
            
            # Check for factors in cross-impact analysis
            if "factors" in source_results:
                potential_variables.extend(source_results["factors"])
            
            # Check for entities in causal network analysis
            if "entities" in source_results:
                potential_variables.extend(source_results["entities"])
            
            # Check for key drivers
            if "key_drivers" in source_results:
                for driver in source_results["key_drivers"]:
                    potential_variables.append({"name": driver, "type": "Driver"})
            
            # If we found potential variables, process them
            if potential_variables:
                logger.info(f"Found {len(potential_variables)} potential variables from {source_technique}")
                
                # Use systems thinking MCP if available
                systems_mcp = self.mcp_registry.get_mcp("systems_thinking_mcp")
                
                if systems_mcp:
                    try:
                        logger.info("Using systems thinking MCP to process variables")
                        system_definition = systems_mcp.define_system(context.question, potential_variables)
                        return system_definition
                    except Exception as e:
                        logger.error(f"Error using systems thinking MCP: {e}")
                        # Fall through to LLM-based definition
        
        # Use LLM to define the system
        prompt = f"""
        Define a system dynamics model for the following analytical question:
        
        "{context.question}"
        
        For this system definition:
        1. Identify the system boundary (what's included and excluded)
        2. Identify 5-8 key variables that are most important to the system
        3. Provide a clear name and description for each variable
        4. Classify each variable as endogenous (internal to the system) or exogenous (external to the system)
        5. Provide a brief overview of how these variables interact at a high level
        
        Return your system definition as a JSON object with the following structure:
        {{
            "system_name": "Name of the system being modeled",
            "system_boundary": "Description of what's included and excluded from the model",
            "key_variables": [
                {{
                    "name": "Name of the variable",
                    "description": "Brief description of the variable",
                    "type": "Endogenous/Exogenous"
                }},
                ...
            ],
            "system_overview": "Brief overview of how these variables interact at a high level"
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error defining system: {parsed_response.get('error')}")
                return self._generate_fallback_system_definition(context.question)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing system definition: {e}")
            return self._generate_fallback_system_definition(context.question)
    
    def _generate_fallback_system_definition(self, question):
        """
        Generate fallback system definition when normal definition fails.
        
        Args:
            question: The analytical question
            
        Returns:
            Dictionary containing fallback system definition
        """
        return {
            "system_name": f"System model for: {question[:50]}...",
            "system_boundary": "This model includes key economic, social, and technological factors relevant to the question, excluding detailed implementation specifics.",
            "key_variables": [
                {
                    "name": "Economic Conditions",
                    "description": "Overall economic environment including growth, inflation, and employment",
                    "type": "Exogenous"
                },
                {
                    "name": "Technological Innovation",
                    "description": "Rate and direction of technological change and adoption",
                    "type": "Endogenous"
                },
                {
                    "name": "Regulatory Environment",
                    "description": "Laws, regulations, and policies affecting the domain",
                    "type": "Exogenous"
                },
                {
                    "name": "Market Adoption",
                    "description": "Rate at which market participants adopt new solutions",
                    "type": "Endogenous"
                },
                {
                    "name": "Resource Allocation",
                    "description": "How resources are distributed and utilized within the system",
                    "type": "Endogenous"
                }
            ],
            "system_overview": "This system is characterized by interactions between economic conditions, technological innovation, regulatory constraints, market adoption patterns, and resource allocation decisions. These variables form feedback loops that drive system behavior over time."
        }
    
    def _identify_stocks_flows(self, question, system_definition):
        """
        Identify stocks and flows in the system.
        
        Args:
            question: The analytical question
            system_definition: Dictionary containing system definition
            
        Returns:
            Dictionary containing stocks and flows
        """
        logger.info("Identifying stocks and flows...")
        
        # Use causal modeling MCP if available
        causal_mcp = self.mcp_registry.get_mcp("causal_modeling_mcp")
        
        if causal_mcp:
            try:
                logger.info("Using causal modeling MCP")
                stocks_flows = causal_mcp.identify_stocks_flows(question, system_definition)
                return stocks_flows
            except Exception as e:
                logger.error(f"Error using causal modeling MCP: {e}")
                # Fall through to LLM-based identification
        
        # Extract key variables from system definition
        key_variables = system_definition.get("key_variables", [])
        variable_names = [var.get("name", "") for var in key_variables]
        
        # Use LLM to identify stocks and flows
        prompt = f"""
        Identify stocks and flows for a system dynamics model addressing this question:
        
        "{question}"
        
        System Definition:
        System Name: {system_definition.get("system_name", "")}
        System Boundary: {system_definition.get("system_boundary", "")}
        Key Variables: {", ".join(variable_names)}
        
        For this analysis:
        1. Identify 3-5 key stocks (accumulations) in the system
        2. Identify the inflows and outflows for each stock
        3. Describe the key variables that influence each flow
        
        A stock is a quantity that accumulates over time (e.g., population, capital, knowledge).
        A flow is a rate that changes a stock (e.g., births/deaths, investment/depreciation, learning/forgetting).
        
        Return your analysis as a JSON object with the following structure:
        {{
            "stocks": [
                {{
                    "name": "Name of the stock",
                    "description": "Brief description of what this stock represents",
                    "initial_condition": "Qualitative description of the initial condition (High/Medium/Low)",
                    "inflows": [
                        {{
                            "name": "Name of the inflow",
                            "description": "Description of this inflow",
                            "influencing_variables": ["Variable 1", "Variable 2", ...]
                        }},
                        ...
                    ],
                    "outflows": [
                        {{
                            "name": "Name of the outflow",
                            "description": "Description of this outflow",
                            "influencing_variables": ["Variable 1", "Variable 2", ...]
                        }},
                        ...
                    ]
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
                logger.error(f"Error identifying stocks and flows: {parsed_response.get('error')}")
                return self._generate_fallback_stocks_flows(system_definition)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing stocks and flows: {e}")
            return self._generate_fallback_stocks_flows(system_definition)
    
    def _generate_fallback_stocks_flows(self, system_definition):
        """
        Generate fallback stocks and flows when normal identification fails.
        
        Args:
            system_definition: Dictionary containing system definition
            
        Returns:
            Dictionary containing fallback stocks and flows
        """
        # Extract variable names from system definition
        key_variables = system_definition.get("key_variables", [])
        variable_names = [var.get("name", "") for var in key_variables]
        
        # Use a subset of variables or default names if none available
        if not variable_names:
            variable_names = ["Economic Conditions", "Technological Innovation", "Market Adoption"]
        
        return {
            "stocks": [
                {
                    "name": "Knowledge Base",
                    "description": "Accumulated knowledge and expertise in the domain",
                    "initial_condition": "Medium",
                    "inflows": [
                        {
                            "name": "Knowledge Acquisition",
                            "description": "Rate at which new knowledge is acquired",
                            "influencing_variables": [variable_names[1] if len(variable_names) > 1 else "Innovation"]
                        }
                    ],
                    "outflows": [
                        {
                            "name": "Knowledge Obsolescence",
                            "description": "Rate at which knowledge becomes outdated",
                            "influencing_variables": [variable_names[1] if len(variable_names) > 1 else "Innovation"]
                        }
                    ]
                },
                {
                    "name": "Market Penetration",
                    "description": "Level of adoption in the target market",
                    "initial_condition": "Low",
                    "inflows": [
                        {
                            "name": "Adoption Rate",
                            "description": "Rate at which new users adopt the solution",
                            "influencing_variables": [variable_names[0] if len(variable_names) > 0 else "Economic Conditions", 
                                                     variable_names[2] if len(variable_names) > 2 else "Market Adoption"]
                        }
                    ],
                    "outflows": [
                        {
                            "name": "Abandonment Rate",
                            "description": "Rate at which users abandon the solution",
                            "influencing_variables": [variable_names[1] if len(variable_names) > 1 else "Innovation"]
                        }
                    ]
                },
                {
                    "name": "Resource Pool",
                    "description": "Available resources for development and operations",
                    "initial_condition": "Medium",
                    "inflows": [
                        {
                            "name": "Resource Acquisition",
                            "description": "Rate at which new resources are acquired",
                            "influencing_variables": [variable_names[0] if len(variable_names) > 0 else "Economic Conditions"]
                        }
                    ],
                    "outflows": [
                        {
                            "name": "Resource Consumption",
                            "description": "Rate at which resources are consumed",
                            "influencing_variables": [variable_names[2] if len(variable_names) > 2 else "Market Adoption"]
                        }
                    ]
                }
            ]
        }
    
    def _identify_feedback_loops(self, stocks_flows):
        """
        Identify feedback loops in the system.
        
        Args:
            stocks_flows: Dictionary containing stocks and flows
            
        Returns:
            Dictionary containing feedback loops
        """
        logger.info("Identifying feedback loops...")
        
        # Use causal modeling MCP if available
        causal_mcp = self.mcp_registry.get_mcp("causal_modeling_mcp")
        
        if causal_mcp:
            try:
                logger.info("Using causal modeling MCP")
                feedback_loops = causal_mcp.identify_feedback_loops(stocks_flows)
                return feedback_loops
            except Exception as e:
                logger.error(f"Error using causal modeling MCP: {e}")
                # Fall through to LLM-based identification
        
        # Extract stocks from stocks_flows
        stocks = stocks_flows.get("stocks", [])
        
        # Use LLM to identify feedback loops
        prompt = f"""
        Identify feedback loops in the following system dynamics model:
        
        Stocks and Flows:
        {json.dumps(stocks, indent=2)}
        
        For this analysis:
        1. Identify 3-5 key feedback loops in the system
        2. Classify each loop as reinforcing (positive) or balancing (negative)
        3. Describe the causal chain that forms each loop
        4. Explain the behavior generated by each loop
        
        A reinforcing loop amplifies change (e.g., compound interest, viral growth).
        A balancing loop counteracts change (e.g., predator-prey, market equilibrium).
        
        Return your analysis as a JSON object with the following structure:
        {{
            "feedback_loops": [
                {{
                    "name": "Name of the feedback loop",
                    "type": "Reinforcing/Balancing",
                    "causal_chain": ["Variable 1", "Variable 2", "...", "Variable 1"],
                    "description": "Description of how this loop functions",
                    "behavior": "Description of the behavior generated by this loop",
                    "time_delays": ["Description of any significant time delays in this loop"]
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
                logger.error(f"Error identifying feedback loops: {parsed_response.get('error')}")
                return self._generate_fallback_feedback_loops(stocks)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing feedback loops: {e}")
            return self._generate_fallback_feedback_loops(stocks)
    
    def _generate_fallback_feedback_loops(self, stocks):
        """
        Generate fallback feedback loops when normal identification fails.
        
        Args:
            stocks: List of stock dictionaries
            
        Returns:
            Dictionary containing fallback feedback loops
        """
        # Extract stock names
        stock_names = [stock.get("name", f"Stock {i+1}") for i, stock in enumerate(stocks)]
        
        # Ensure we have at least two stock names for loops
        if len(stock_names) < 2:
            stock_names = ["Knowledge Base", "Market Penetration"]
        
        return {
            "feedback_loops": [
                {
                    "name": "Growth Reinforcing Loop",
                    "type": "Reinforcing",
                    "causal_chain": [stock_names[0], "Knowledge Acquisition", stock_names[1], "Adoption Rate", stock_names[0]],
                    "description": f"As {stock_names[0]} increases, it enhances Knowledge Acquisition, which increases {stock_names[1]}, leading to higher Adoption Rate, which further increases {stock_names[0]}.",
                    "behavior": "Exponential growth pattern until limited by external constraints",
                    "time_delays": ["Delay between knowledge acquisition and market adoption"]
                },
                {
                    "name": "Resource Constraint Balancing Loop",
                    "type": "Balancing",
                    "causal_chain": [stock_names[1], "Resource Consumption", "Resource Pool", "Resource Availability", stock_names[1]],
                    "description": f"As {stock_names[1]} increases, Resource Consumption increases, which decreases Resource Pool, leading to lower Resource Availability, which constrains further growth in {stock_names[1]}.",
                    "behavior": "Growth followed by stabilization as resources become constrained",
                    "time_delays": ["Delay between resource consumption and resource constraints becoming apparent"]
                },
                {
                    "name": "Innovation Cycle",
                    "type": "Reinforcing",
                    "causal_chain": ["Technological Innovation", stock_names[0], "Knowledge Obsolescence", "Innovation Pressure", "Technological Innovation"],
                    "description": "Technological Innovation increases the Knowledge Base but also accelerates Knowledge Obsolescence, creating pressure for more Innovation.",
                    "behavior": "Cyclical patterns of innovation and obsolescence",
                    "time_delays": ["Significant delay between innovation pressure and new technological breakthroughs"]
                }
            ]
        }
    
    def _analyze_system_behavior(self, question, stocks_flows, feedback_loops, time_horizon):
        """
        Analyze system behavior over time.
        
        Args:
            question: The analytical question
            stocks_flows: Dictionary containing stocks and flows
            feedback_loops: Dictionary containing feedback loops
            time_horizon: Time horizon for the analysis
            
        Returns:
            Dictionary containing system behavior analysis
        """
        logger.info(f"Analyzing system behavior over {time_horizon}...")
        
        # Use systems thinking MCP if available
        systems_mcp = self.mcp_registry.get_mcp("systems_thinking_mcp")
        
        if systems_mcp:
            try:
                logger.info("Using systems thinking MCP")
                system_behavior = systems_mcp.analyze_behavior(question, stocks_flows, feedback_loops, time_horizon)
                return system_behavior
            except Exception as e:
                logger.error(f"Error using systems thinking MCP: {e}")
                # Fall through to LLM-based analysis
        
        # Extract key components for the prompt
        stocks = stocks_flows.get("stocks", [])
        loops = feedback_loops.get("feedback_loops", [])
        
        # Use LLM to analyze system behavior
        prompt = f"""
        Analyze the behavior of the following system dynamics model over {time_horizon}:
        
        Question: "{question}"
        
        Stocks:
        {json.dumps([{
            "name": stock.get("name", ""),
            "initial_condition": stock.get("initial_condition", "")
        } for stock in stocks], indent=2)}
        
        Feedback Loops:
        {json.dumps([{
            "name": loop.get("name", ""),
            "type": loop.get("type", ""),
            "behavior": loop.get("behavior", "")
        } for loop in loops], indent=2)}
        
        For this analysis:
        1. Describe how each key stock is likely to change over the time horizon
        2. Identify which feedback loops will dominate at different points in time
        3. Describe any potential tipping points or regime shifts
        4. Identify key uncertainties that could significantly alter system behavior
        
        Return your analysis as a JSON object with the following structure:
        {{
            "stock_trajectories": [
                {{
                    "stock": "Name of the stock",
                    "short_term_trajectory": "Description of likely behavior in the short term",
                    "medium_term_trajectory": "Description of likely behavior in the medium term",
                    "long_term_trajectory": "Description of likely behavior in the long term"
                }},
                ...
            ],
            "dominant_loops": [
                {{
                    "time_period": "Description of the time period",
                    "dominant_loop": "Name of the dominant loop during this period",
                    "rationale": "Explanation for why this loop dominates during this period"
                }},
                ...
            ],
            "tipping_points": [
                {{
                    "description": "Description of the potential tipping point",
                    "conditions": ["Condition 1", "Condition 2", ...],
                    "consequences": ["Consequence 1", "Consequence 2", ...]
                }},
                ...
            ],
            "key_uncertainties": [
                {{
                    "uncertainty": "Description of the uncertainty",
                    "potential_impact": "Description of how this could alter system behavior"
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
                logger.error(f"Error analyzing system behavior: {parsed_response.get('error')}")
                return self._generate_fallback_system_behavior(stocks, loops, time_horizon)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing system behavior: {e}")
            return self._generate_fallback_system_behavior(stocks, loops, time_horizon)
    
    def _generate_fallback_system_behavior(self, stocks, loops, time_horizon):
        """
        Generate fallback system behavior when normal analysis fails.
        
        Args:
            stocks: List of stock dictionaries
            loops: List of feedback loop dictionaries
            time_horizon: Time horizon for the analysis
            
        Returns:
            Dictionary containing fallback system behavior
        """
        # Extract stock names
        stock_names = [stock.get("name", f"Stock {i+1}") for i, stock in enumerate(stocks)]
        
        # Ensure we have at least two stock names
        if len(stock_names) < 2:
            stock_names = ["Knowledge Base", "Market Penetration", "Resource Pool"]
        
        # Extract loop names
        loop_names = [loop.get("name", f"Loop {i+1}") for i, loop in enumerate(loops)]
        
        # Ensure we have at least two loop names
        if len(loop_names) < 2:
            loop_names = ["Growth Reinforcing Loop", "Resource Constraint Balancing Loop"]
        
        return {
            "stock_trajectories": [
                {
                    "stock": stock_names[0],
                    "short_term_trajectory": "Gradual increase as initial investments build the knowledge base",
                    "medium_term_trajectory": "Accelerating growth as reinforcing loops take effect",
                    "long_term_trajectory": "Slowing growth as diminishing returns and obsolescence balance new knowledge acquisition"
                },
                {
                    "stock": stock_names[1],
                    "short_term_trajectory": "Slow initial growth during early adoption phase",
                    "medium_term_trajectory": "Rapid growth during mainstream adoption phase",
                    "long_term_trajectory": "Saturation and stabilization as market becomes mature"
                }
            ],
            "dominant_loops": [
                {
                    "time_period": "Early phase (first 20% of time horizon)",
                    "dominant_loop": loop_names[0],
                    "rationale": "Initial growth dynamics are dominated by reinforcing mechanisms as the system builds momentum"
                },
                {
                    "time_period": "Later phase (last 50% of time horizon)",
                    "dominant_loop": loop_names[1] if len(loop_names) > 1 else "Balancing Loop",
                    "rationale": "As the system grows, constraints become more significant and balancing loops begin to dominate"
                }
            ],
            "tipping_points": [
                {
                    "description": "Critical mass threshold",
                    "conditions": ["Sufficient knowledge accumulation", "Positive user experiences", "Network effects reaching critical threshold"],
                    "consequences": ["Accelerated adoption", "Self-sustaining growth", "Reduced marketing requirements"]
                },
                {
                    "description": "Resource constraint threshold",
                    "conditions": ["Resource consumption exceeding replenishment", "Increasing competition for limited resources", "Rising costs"],
                    "consequences": ["Growth slowdown", "Increased efficiency pressure", "Potential system restructuring"]
                }
            ],
            "key_uncertainties": [
                {
                    "uncertainty": "External disruption",
                    "potential_impact": "A major technological or regulatory disruption could fundamentally alter system dynamics, potentially rendering existing knowledge obsolete"
                },
                {
                    "uncertainty": "Feedback loop strength",
                    "potential_impact": "The relative strength of reinforcing versus balancing loops could significantly alter the timing and magnitude of system behavior"
                }
            ]
        }
    
    def _generate_synthesis(self, question, system_definition, stocks_flows, feedback_loops, system_behavior):
        """
        Generate a synthesis of the system dynamics modeling.
        
        Args:
            question: The analytical question
            system_definition: Dictionary containing system definition
            stocks_flows: Dictionary containing stocks and flows
            feedback_loops: Dictionary containing feedback loops
            system_behavior: Dictionary containing system behavior analysis
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of system dynamics modeling...")
        
        prompt = f"""
        Synthesize the following system dynamics modeling for the question:
        
        "{question}"
        
        System Definition:
        {json.dumps({
            "system_name": system_definition.get("system_name", ""),
            "system_boundary": system_definition.get("system_boundary", "")
        }, indent=2)}
        
        Key Stocks:
        {json.dumps([stock.get("name", "") for stock in stocks_flows.get("stocks", [])], indent=2)}
        
        Key Feedback Loops:
        {json.dumps([{
            "name": loop.get("name", ""),
            "type": loop.get("type", "")
        } for loop in feedback_loops.get("feedback_loops", [])], indent=2)}
        
        System Behavior:
        {json.dumps({
            "tipping_points": system_behavior.get("tipping_points", []),
            "key_uncertainties": system_behavior.get("key_uncertainties", [])
        }, indent=2)}
        
        Based on this system dynamics modeling:
        1. What are the key insights about system behavior?
        2. What leverage points exist for influencing the system?
        3. What policy recommendations emerge from this analysis?
        4. How does this address the original question?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the system dynamics modeling
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "key_insights": ["Insight 1", "Insight 2", ...],
            "leverage_points": ["Leverage point 1", "Leverage point 2", ...],
            "policy_recommendations": ["Recommendation 1", "Recommendation 2", ...],
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
                    "key_insights": ["Error in synthesis generation"],
                    "leverage_points": ["Error in synthesis generation"],
                    "policy_recommendations": ["Error in synthesis generation"],
                    "confidence_level": "Low",
                    "potential_biases": ["Technical error bias"]
                }
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing synthesis: {e}")
            return {
                "final_judgment": f"Error generating synthesis: {str(e)}",
                "judgment_rationale": "Error in synthesis generation",
                "key_insights": ["Error in synthesis generation"],
                "leverage_points": ["Error in synthesis generation"],
                "policy_recommendations": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
