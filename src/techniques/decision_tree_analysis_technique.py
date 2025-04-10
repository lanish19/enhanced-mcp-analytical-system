"""
Decision Tree Analysis Technique implementation.
This module provides the DecisionTreeAnalysisTechnique class for structured decision analysis.
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

    
    def visualize_tree(self, tree_data):
        """
        Generates a text-based visualization of the decision tree.
        Args:
            tree_data (Dict[str, Any]): The decision tree data in JSON format.
        """
        def print_node(node, indent=0):
            node_type = node.get("type", "unknown")
            label = node.get("label", "Node")
            if node_type == "decision":
                print("  " * indent + f"Decision: {label}")
                for option in node.get("options", []):
                    print("  " * (indent + 1) + f"Option: {option['label']}")
                    if "node" in option:
                        print_node(option["node"], indent + 2)
            elif node_type == "chance":
                print("  " * indent + f"Chance: {label}")
                for outcome in node.get("probabilities", []):
                    print("  " * (indent + 1) + f"Outcome: {outcome['label']} (Probability: {outcome['probability']})")
                    if "node" in outcome:
                        print_node(outcome["node"], indent + 2)
            elif node_type == "outcome":
                print("  " * indent + f"Outcome: {label} (Value: {node.get('value', 'N/A')})")
            else:
                print("  " * indent + f"Unknown Node Type: {label}")
        if "tree" in tree_data and "root" in tree_data["tree"]:
            print_node(tree_data["tree"]["root"])
        else:
            print("Invalid tree data format for visualization.")






class DecisionTreeAnalysisTechnique(AnalyticalTechnique):
    """
    Structures complex decisions into a tree of options and outcomes.
    
    This technique breaks down complex decisions into a series of smaller choices,
    mapping potential outcomes and their probabilities to enable systematic
    evaluation of different decision paths.
    """
    
    def execute(self, context, parameters):
        """
        Execute the decision tree analysis technique.
        
        Args:
            context: The analysis context
            parameters: Parameters for the technique
            
        Returns:
            Dictionary containing decision tree analysis results
        """
        try:
            # Extract necessary information from context and parameters
            decision_problem = context.get("decision_problem")
            if not decision_problem:
                raise ValueError("Decision problem not found in context")
            # Set up the prompt for the LLM
            prompt = f"""
            Analyze the following decision problem using a decision tree approach:
            {decision_problem}
            Create a decision tree that includes:
            1. Clear decision nodes with options.
            2. Chance nodes with probabilities for different outcomes.
            3. Outcome values or utilities for each final outcome.
            Provide the decision tree in a JSON format that can be easily parsed and visualized.
            The JSON should have the following structure:
            {{
                "decision_problem": "...",
                "tree": {{
                    "root": {{
                        "type": "decision",
                        "label": "Decision 1",
                        "options": [
                            {{
                                "label": "Option A",
                                "node": {{
                                    "type": "chance",
                                    "label": "Outcome of Option A",
                                    "probabilities": [
                                        {{
                                            "label": "Outcome A1",
                                            "probability": 0.6,
                                            "node": {{
                                                "type": "outcome",
                                                "label": "Outcome A1 Result",
                                                "value": 100
                                            }}
                                        }},
                                        {{
                                            "label": "Outcome A2",
                                            "probability": 0.4,
                                            "node": {{
                                                "type": "outcome",
                                                "label": "Outcome A2 Result",
                                                "value": 20
                                            }}
                                        }}
                                    ]
                                }}
                            }},
                            {{
                                "label": "Option B",
                                "node": {{
                                    "type": "outcome",
                                    "label": "Outcome of Option B",
                                    "value": 50
                                }}
                            }}
                        ]
                    }}
                }}
            }}
            """
            # Call the LLM to generate the decision tree
            start_time = time.time()
            llm_response = call_llm(prompt, model_config=MODEL_CONFIG)
            end_time = time.time()
            # Extract and parse the decision tree from the LLM response
            tree_json = extract_content(llm_response)
            if not tree_json:
                raise ValueError("No decision tree found in LLM response")
            decision_tree = json.loads(tree_json)
            # Basic validation of the parsed JSON (can be expanded)
            if "decision_problem" not in decision_tree or "tree" not in decision_tree:
                raise ValueError("Invalid decision tree JSON format")
            # Log the execution time and other relevant information
            self.visualize_tree(decision_tree)
            self.visualize_tree(decision_tree)
            logger.info(f"Decision tree analysis completed in {end_time - start_time:.4f} seconds")
            logger.debug(f"Decision tree: {json.dumps(decision_tree, indent=2)}")
            return {
                "decision_tree_analysis": decision_tree,
                "execution_time": end_time - start_time
        except Exception as e:
            logger.error(f"Error during decision tree analysis: {e}", exc_info=True)
            return {
                "error": str(e)
            }

        logger.info(f"Executing DecisionTreeAnalysisTechnique for question: {context.question[:50]}...")
        
        # Get parameters or use defaults
        decision_focus = parameters.get("decision_focus", None)
        max_depth = parameters.get("max_depth", 3)
        include_probabilities = parameters.get("include_probabilities", True)
        include_utilities = parameters.get("include_utilities", True)
        
        # Step 1: Identify key decision
        key_decision = self._identify_key_decision(context, decision_focus)
        
        # Step 2: Generate decision tree
        decision_tree = self._generate_decision_tree(context.question, key_decision, max_depth, 
                                                    include_probabilities, include_utilities)
        
        # Step 3: Analyze decision paths
        path_analysis = self._analyze_decision_paths(decision_tree, include_probabilities, include_utilities)
        
        # Step 4: Perform sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(decision_tree, path_analysis)
        
        # Step 5: Generate final synthesis
        synthesis = self._generate_synthesis(context.question, key_decision, decision_tree, 
                                            path_analysis, sensitivity_analysis)
        
        return {
            "technique": "Decision Tree Analysis",
            "status": "Completed",
            "key_decision": key_decision,
            "decision_tree": decision_tree,
            "path_analysis": path_analysis,
            "sensitivity_analysis": sensitivity_analysis,
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
        return ["decision_analysis_mcp", "probability_estimation_mcp"]
    
    def _identify_key_decision(self, context, decision_focus):
        """
        Identify the key decision to analyze.
        
        Args:
            context: The analysis context
            decision_focus: Optional specific decision focus
            
        Returns:
            Dictionary containing key decision information
        """
        logger.info("Identifying key decision...")
        
        # If decision focus is provided, use it directly
        if decision_focus:
            logger.info(f"Using provided decision focus: {decision_focus}")
            return {
                "decision": decision_focus,
                "decision_maker": "Specified decision maker",
                "timeframe": "Specified timeframe",
                "context": "Provided decision focus"
            }
        
        # Use decision analysis MCP if available
        decision_mcp = self.mcp_registry.get_mcp("decision_analysis_mcp")
        
        if decision_mcp:
            try:
                logger.info("Using decision analysis MCP")
                key_decision = decision_mcp.identify_key_decision(context.question)
                return key_decision
            except Exception as e:
                logger.error(f"Error using decision analysis MCP: {e}")
                # Fall through to LLM-based identification
        
        # Use LLM to identify key decision
        prompt = f"""        
        You are an expert analyst. Your task is to identify the key decision implied by the following analytical question:

        "{context.question}"

        Your analysis should be comprehensive and structured. Consider the following aspects:

        1.  **Decision:** Identify the central decision that needs to be made. Frame the decision as a clear choice between distinct options or courses of action.
        2.  **Decision Maker:** Determine who the primary decision maker is or which entity is responsible for making this decision.
        3.  **Timeframe:** Specify the timeframe within which this decision needs to be made (e.g., immediately, short-term, mid-term, long-term).
        4.  **Context:** Provide relevant context that informs or influences this decision. What are the background or surrounding circumstances that are important to understand the decision?

        Ensure that your response is in the format of a single JSON object. Here's an example of what I am looking for:

        {{
            "decision": "Whether to expand into the European market or consolidate in North America",
            "decision_maker": "Senior management team",
            "timeframe": "Short-term (within the next year)",
            "context": "The company has seen strong growth in North America but faces increasing competition. Expanding into Europe could unlock new growth potential but carries higher risks."
        }}

        Here are the required fields:
        -   **decision:** A clear, concise statement of the decision to be made. This should be framed as a choice between options.
        -   **decision_maker:** Identifies who or which group needs to make the decision.
        -   **timeframe:** Specifies when the decision needs to be made.
        -   **context:** Provides the background or relevant circumstances affecting the decision.

        Remember:
        - Your output must be a single, valid JSON object.
        - Be as specific as possible, based on the information given in the question.
        - Use the provided example structure.
        - Do not return any text outside of the JSON output.

        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error identifying key decision: {parsed_response.get('error')}")
                return self._generate_fallback_key_decision(context.question)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing key decision: {e}")
            return self._generate_fallback_key_decision(context.question)
    
    def _generate_fallback_key_decision(self, question):
        """
        Generate fallback key decision when identification fails.
        
        Args:
            question: The analytical question
            
        Returns:
            Dictionary containing fallback key decision
        """
        return {
            "decision": "Whether to proceed with the current approach or pursue alternatives",
            "decision_maker": "Primary stakeholder",
            "timeframe": "Near-term (within 3-6 months)",
            "context": f"Based on the analytical question: {question}"
        }
    
    def _generate_decision_tree(self, question, key_decision, max_depth, include_probabilities, include_utilities):
        """
        Generate a decision tree for the key decision.
        
        Args:
            question: The analytical question
            key_decision: Dictionary containing key decision information
            max_depth: Maximum depth of the decision tree
            include_probabilities: Whether to include outcome probabilities
            include_utilities: Whether to include outcome utilities
            
        Returns:
            Dictionary containing the decision tree
        """
        logger.info(f"Generating decision tree (max depth: {max_depth})...")
        
        # Use decision analysis MCP if available
        decision_mcp = self.mcp_registry.get_mcp("decision_analysis_mcp")
        
        if decision_mcp:
            try:
                logger.info("Using decision analysis MCP")
                decision_tree = decision_mcp.generate_decision_tree(question, key_decision, max_depth, 
                                                                  include_probabilities, include_utilities)
                return decision_tree
            except Exception as e:
                logger.error(f"Error using decision analysis MCP: {e}")
                # Fall through to LLM-based generation
        
        # Use LLM to generate decision tree
        prompt = f"""
        Generate a decision tree for the following decision:
        
        Question: "{question}"
        
        Decision: "{key_decision.get('decision', '')}"
        Decision Maker: {key_decision.get('decision_maker', '')}
        Timeframe: {key_decision.get('timeframe', '')}
        Context: {key_decision.get('context', '')}
        
        For this decision tree:
        1. Identify the main decision options (branches)
        2. For each option, identify key uncertain events that could occur
        3. For each uncertain event, identify possible outcomes
        4. Continue this process up to a maximum depth of {max_depth} levels
        5. {"Estimate probabilities for each uncertain outcome" if include_probabilities else "Do not include probabilities"}
        6. {"Estimate utilities (value or desirability) for each end state" if include_utilities else "Do not include utilities"}
        
        Return your decision tree as a JSON object with the following structure:
        {{
            "decision_node": {{
                "name": "The main decision",
                "type": "decision",
                "options": [
                    {{
                        "name": "Option 1",
                        "description": "Description of option 1",
                        "child_node": {{
                            "name": "Uncertain event after option 1",
                            "type": "chance",
                            "outcomes": [
                                {{
                                    "name": "Outcome 1",
                                    "description": "Description of outcome 1",
                                    "probability": 0.X,  # Only if include_probabilities is true
                                    "child_node": {{...}} or null,
                                    "utility": Y  # Only if include_utilities is true and this is an end state
                                }},
                                ...
                            ]
                        }} or null
                    }},
                    ...
                ]
            }}
        }}
        """
        
        model_config = MODEL_CONFIG["sonar_deep"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error generating decision tree: {parsed_response.get('error')}")
                return self._generate_fallback_decision_tree(key_decision, include_probabilities, include_utilities)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing decision tree: {e}")
            return self._generate_fallback_decision_tree(key_decision, include_probabilities, include_utilities)
    
    def _generate_fallback_decision_tree(self, key_decision, include_probabilities, include_utilities):
        """
        Generate fallback decision tree when generation fails.
        
        Args:
            key_decision: Dictionary containing key decision information
            include_probabilities: Whether to include outcome probabilities
            include_utilities: Whether to include outcome utilities
            
        Returns:
            Dictionary containing fallback decision tree
        """
        decision = key_decision.get("decision", "Whether to proceed with the current approach or pursue alternatives")
        
        # Create a basic decision tree with two options
        tree = {
            "decision_node": {
                "name": decision,
                "type": "decision",
                "options": [
                    {
                        "name": "Option A: Proceed with current approach",
                        "description": "Continue with the existing strategy and implementation plan",
                        "child_node": {
                            "name": "Market conditions",
                            "type": "chance",
                            "outcomes": [
                                {
                                    "name": "Favorable conditions",
                                    "description": "Market conditions support the current approach",
                                    "child_node": None
                                },
                                {
                                    "name": "Unfavorable conditions",
                                    "description": "Market conditions create challenges for the current approach",
                                    "child_node": None
                                }
                            ]
                        }
                    },
                    {
                        "name": "Option B: Pursue alternatives",
                        "description": "Explore and implement alternative strategies",
                        "child_node": {
                            "name": "Implementation success",
                            "type": "chance",
                            "outcomes": [
                                {
                                    "name": "Successful implementation",
                                    "description": "Alternative is implemented successfully",
                                    "child_node": None
                                },
                                {
                                    "name": "Implementation challenges",
                                    "description": "Significant challenges in implementing the alternative",
                                    "child_node": None
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # Add probabilities if requested
        if include_probabilities:
            tree["decision_node"]["options"][0]["child_node"]["outcomes"][0]["probability"] = 0.6
            tree["decision_node"]["options"][0]["child_node"]["outcomes"][1]["probability"] = 0.4
            tree["decision_node"]["options"][1]["child_node"]["outcomes"][0]["probability"] = 0.5
            tree["decision_node"]["options"][1]["child_node"]["outcomes"][1]["probability"] = 0.5
        
        # Add utilities if requested
        if include_utilities:
            tree["decision_node"]["options"][0]["child_node"]["outcomes"][0]["utility"] = 80
            tree["decision_node"]["options"][0]["child_node"]["outcomes"][1]["utility"] = 30
            tree["decision_node"]["options"][1]["child_node"]["outcomes"][0]["utility"] = 100
            tree["decision_node"]["options"][1]["child_node"]["outcomes"][1]["utility"] = 20
        
        return tree
    
    def _analyze_decision_paths(self, decision_tree, include_probabilities, include_utilities):
        """
        Analyze the decision paths in the tree.
        
        Args:
            decision_tree: Dictionary containing the decision tree
            include_probabilities: Whether probabilities are included
            include_utilities: Whether utilities are included
            
        Returns:
            Dictionary containing path analysis
        """
        logger.info("Analyzing decision paths...")
        
        # Use decision analysis MCP if available
        decision_mcp = self.mcp_registry.get_mcp("decision_analysis_mcp")
        
        if decision_mcp:
            try:
                logger.info("Using decision analysis MCP")
                path_analysis = decision_mcp.analyze_decision_paths(decision_tree, include_probabilities, include_utilities)
                return path_analysis
            except Exception as e:
                logger.error(f"Error using decision analysis MCP: {e}")
                # Fall through to LLM-based analysis
        
        # Extract all paths from the decision tree
        paths = self._extract_paths(decision_tree)
        
        # Use LLM to analyze paths
        prompt = f"""
        Analyze the following decision paths extracted from a decision tree:
        
        Paths:
        {json.dumps(paths, indent=2)}
        
        For this analysis:
        1. Identify all complete paths from the initial decision to end states
        2. {"Calculate the probability of each complete path" if include_probabilities else "Do not calculate path probabilities"}
        3. {"Calculate the expected value of each option based on probabilities and utilities" if include_probabilities and include_utilities else "Do not calculate expected values"}
        4. Identify the optimal decision path based on {"expected value" if include_probabilities and include_utilities else "qualitative assessment"}
        5. Identify key decision points and their implications
        
        Return your analysis as a JSON object with the following structure:
        {{
            "complete_paths": [
                {{
                    "path_description": "Description of the complete path",
                    "path_nodes": ["Node 1", "Node 2", ...],
                    "probability": X.XX,  # Only if include_probabilities is true
                    "utility": Y,  # Only if include_utilities is true
                    "expected_value": Z.ZZ  # Only if both include_probabilities and include_utilities are true
                }},
                ...
            ],
            "option_analysis": [
                {{
                    "option": "Name of the option",
                    "expected_value": X.XX,  # Only if both include_probabilities and include_utilities are true
                    "qualitative_assessment": "Qualitative assessment of the option"
                }},
                ...
            ],
            "optimal_path": {{
                "path_description": "Description of the optimal path",
                "rationale": "Rationale for why this path is optimal"
            }},
            "key_decision_points": [
                {{
                    "decision_point": "Description of the decision point",
                    "implications": "Implications of this decision point"
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
                logger.error(f"Error analyzing decision paths: {parsed_response.get('error')}")
                return self._generate_fallback_path_analysis(paths, include_probabilities, include_utilities)
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing path analysis: {e}")
            return self._generate_fallback_path_analysis(paths, include_probabilities, include_utilities)
    
    def _extract_paths(self, decision_tree):
        """
        Extract all paths from the decision tree.
        
        Args:
            decision_tree: Dictionary containing the decision tree
            
        Returns:
            List of paths in the decision tree
        """
        paths = []
        
        def traverse(node, current_path, path_type):
            if not node:
                paths.append({
                    "path": current_path.copy(),
                    "type": path_type
                })
                return
            
            if "type" in node and node["type"] == "decision":
                current_path.append(node["name"])
                for option in node.get("options", []):
                    option_path = current_path.copy()
                    option_path.append(option["name"])
                    traverse(option.get("child_node"), option_path, "decision")
            
            elif "type" in node and node["type"] == "chance":
                current_path.append(node["name"])
                for outcome in node.get("outcomes", []):
                    outcome_path = current_path.copy()
                    outcome_path.append(outcome["name"])
                    
                    # Check if this is an end state
                    if not outcome.get("child_node"):
                        paths.append({
                            "path": outcome_path.copy(),
                            "type": "complete",
                            "probability": outcome.get("probability") if "probability" in outcome else None,
                            "utility": outcome.get("utility") if "utility" in outcome else None
                        })
                    else:
                        traverse(outcome.get("child_node"), outcome_path, "chance")
        
        # Start traversal from the root decision node
        root = decision_tree.get("decision_node")
        if root:
            traverse(root, [], "root")
        
        return paths
    
    def _generate_fallback_path_analysis(self, paths, include_probabilities, include_utilities):
        """
        Generate fallback path analysis when analysis fails.
        
        Args:
            paths: List of paths in the decision tree
            include_probabilities: Whether probabilities are included
            include_utilities: Whether utilities are included
            
        Returns:
            Dictionary containing fallback path analysis
        """
        # Extract complete paths
        complete_paths = [p for p in paths if p.get("type") == "complete"]
        
        # Create basic path analysis
        path_analysis = {
            "complete_paths": [
                {
                    "path_description": " → ".join(path.get("path", [])),
                    "path_nodes": path.get("path", [])
                } for path in complete_paths[:4]  # Limit to first 4 paths
            ],
            "option_analysis": [
                {
                    "option": "Option A: Proceed with current approach",
                    "qualitative_assessment": "This option provides stability but may limit potential upside"
                },
                {
                    "option": "Option B: Pursue alternatives",
                    "qualitative_assessment": "This option offers higher potential upside but with increased risk"
                }
            ],
            "optimal_path": {
                "path_description": "Option B → Successful implementation",
                "rationale": "This path offers the highest potential value, though it comes with implementation risks"
            },
            "key_decision_points": [
                {
                    "decision_point": "Initial approach selection",
                    "implications": "Sets the overall strategic direction and risk profile"
                },
                {
                    "decision_point": "Response to implementation challenges",
                    "implications": "Determines whether challenges can be overcome or will derail the approach"
                }
            ]
        }
        
        # Add probabilities and expected values if requested
        if include_probabilities and include_utilities:
            for i, path in enumerate(path_analysis["complete_paths"]):
                path["probability"] = 0.3 if i == 0 else 0.2 if i == 1 else 0.25
                path["utility"] = 80 if i == 0 else 30 if i == 1 else 100 if i == 2 else 20
                path["expected_value"] = path["probability"] * path["utility"]
            
            path_analysis["option_analysis"][0]["expected_value"] = 60
            path_analysis["option_analysis"][1]["expected_value"] = 65
        
        return path_analysis
    
    def _perform_sensitivity_analysis(self, decision_tree, path_analysis):
        """
        Perform sensitivity analysis on the decision tree.
        
        Args:
            decision_tree: Dictionary containing the decision tree
            path_analysis: Dictionary containing path analysis
            
        Returns:
            Dictionary containing sensitivity analysis
        """
        logger.info("Performing sensitivity analysis...")
        
        # Use probability estimation MCP if available
        probability_mcp = self.mcp_registry.get_mcp("probability_estimation_mcp")
        
        if probability_mcp:
            try:
                logger.info("Using probability estimation MCP")
                sensitivity_analysis = probability_mcp.perform_sensitivity_analysis(decision_tree, path_analysis)
                return sensitivity_analysis
            except Exception as e:
                logger.error(f"Error using probability estimation MCP: {e}")
                # Fall through to LLM-based analysis
        
        # Use LLM to perform sensitivity analysis
        prompt = f"""
        Perform sensitivity analysis on the following decision tree and path analysis:
        
        Path Analysis:
        {json.dumps(path_analysis, indent=2)}
        
        For this sensitivity analysis:
        1. Identify the key uncertainties that most affect the decision
        2. Determine how changes in probabilities would affect the optimal decision
        3. Identify threshold values where the optimal decision would change
        4. Assess the robustness of the current recommendation
        
        Return your analysis as a JSON object with the following structure:
        {{
            "key_uncertainties": [
                {{
                    "uncertainty": "Description of the uncertainty",
                    "current_estimate": "Current probability or value estimate",
                    "impact": "How this uncertainty impacts the decision"
                }},
                ...
            ],
            "sensitivity_tests": [
                {{
                    "parameter": "Parameter being varied",
                    "base_value": "Base value of the parameter",
                    "threshold_value": "Value at which the optimal decision changes",
                    "alternative_decision": "The decision that becomes optimal at the threshold"
                }},
                ...
            ],
            "robustness_assessment": {{
                "robustness_level": "High/Medium/Low",
                "explanation": "Explanation of the robustness assessment"
            }},
            "key_insights": ["Insight 1", "Insight 2", ...]
        }}
        """
        
        model_config = MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        try:
            parsed_response = parse_json_response(content)
            
            if parsed_response.get("fallback_generated"):
                logger.error(f"Error performing sensitivity analysis: {parsed_response.get('error')}")
                return self._generate_fallback_sensitivity_analysis()
            
            return parsed_response
        
        except Exception as e:
            logger.error(f"Error parsing sensitivity analysis: {e}")
            return self._generate_fallback_sensitivity_analysis()
    
    def _generate_fallback_sensitivity_analysis(self):
        """
        Generate fallback sensitivity analysis when analysis fails.
        
        Returns:
            Dictionary containing fallback sensitivity analysis
        """
        return {
            "key_uncertainties": [
                {
                    "uncertainty": "Market conditions",
                    "current_estimate": "60% favorable, 40% unfavorable",
                    "impact": "Directly affects the success of the current approach"
                },
                {
                    "uncertainty": "Implementation success",
                    "current_estimate": "50% successful, 50% challenging",
                    "impact": "Determines whether alternative approaches can be effectively executed"
                }
            ],
            "sensitivity_tests": [
                {
                    "parameter": "Probability of favorable market conditions",
                    "base_value": "0.6",
                    "threshold_value": "0.7",
                    "alternative_decision": "Option A becomes optimal above this threshold"
                },
                {
                    "parameter": "Probability of successful implementation",
                    "base_value": "0.5",
                    "threshold_value": "0.4",
                    "alternative_decision": "Option A becomes optimal below this threshold"
                }
            ],
            "robustness_assessment": {
                "robustness_level": "Medium",
                "explanation": "The optimal decision is moderately sensitive to changes in key probabilities, but remains stable within reasonable ranges of uncertainty."
            },
            "key_insights": [
                "The decision is most sensitive to estimates of implementation success probability",
                "Market condition estimates have less impact on the optimal choice",
                "The relative utility difference between successful and unsuccessful outcomes is a critical factor",
                "Additional information about implementation challenges would be valuable before making a final decision"
            ]
        }
    
    def _generate_synthesis(self, question, key_decision, decision_tree, path_analysis, sensitivity_analysis):
        """
        Generate a synthesis of the decision tree analysis.
        
        Args:
            question: The analytical question
            key_decision: Dictionary containing key decision information
            decision_tree: Dictionary containing the decision tree
            path_analysis: Dictionary containing path analysis
            sensitivity_analysis: Dictionary containing sensitivity analysis
            
        Returns:
            Dictionary containing the synthesis
        """
        logger.info("Generating synthesis of decision tree analysis...")
        
        prompt = f"""
        Synthesize the following decision tree analysis for the question:
        
        "{question}"
        
        Key Decision:
        {json.dumps(key_decision, indent=2)}
        
        Optimal Path:
        {json.dumps(path_analysis.get("optimal_path", {}), indent=2)}
        
        Robustness Assessment:
        {json.dumps(sensitivity_analysis.get("robustness_assessment", {}), indent=2)}
        
        Key Insights:
        {json.dumps(sensitivity_analysis.get("key_insights", []), indent=2)}
        
        Based on this decision tree analysis:
        1. What is the recommended decision?
        2. What are the key factors that drive this recommendation?
        3. What uncertainties should be monitored or resolved?
        4. What contingency plans should be considered?
        
        Provide:
        1. A final judgment that addresses the original question
        2. A rationale for this judgment based on the decision tree analysis
        3. A confidence level in this judgment (High/Medium/Low)
        4. Potential biases that might be affecting this analysis
        
        Return as JSON:
        {{
            "final_judgment": "Your assessment addressing the original question",
            "judgment_rationale": "Explanation for your judgment",
            "key_drivers": ["Driver 1", "Driver 2", ...],
            "critical_uncertainties": ["Uncertainty 1", "Uncertainty 2", ...],
            "contingency_recommendations": ["Recommendation 1", "Recommendation 2", ...],
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
                    "critical_uncertainties": ["Error in synthesis generation"],
                    "contingency_recommendations": ["Error in synthesis generation"],
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
                "critical_uncertainties": ["Error in synthesis generation"],
                "contingency_recommendations": ["Error in synthesis generation"],
                "confidence_level": "Low",
                "potential_biases": ["Technical error bias"]
            }
