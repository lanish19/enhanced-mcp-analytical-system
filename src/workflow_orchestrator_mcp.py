"""
WorkflowOrchestratorMCP for orchestrating analytical workflows.
This module provides the central orchestrator for dynamic analytical workflows.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple

from .base_mcp import BaseMCP
from .mcp_registry import MCPRegistry
from .analysis_context import AnalysisContext
from .analysis_strategy import AnalysisStrategy
from .technique_mcp_integrator import TechniqueMCPIntegrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowOrchestratorMCP(BaseMCP):
    """
    Central orchestrator that dynamically constructs and manages analytical workflows.
    
    This MCP is responsible for:
    - Analyzing question characteristics to determine optimal workflow sequence
    - Selecting techniques based on question type (predictive, causal, evaluative)
    - Adapting workflow dynamically based on interim findings
    - Managing technique dependencies and complementary pairs
    """

    MAX_RETRIES = 3  # Maximum number of retries for a technique
    
    def __init__(self, mcp_registry=None):
        """
        Initialize the workflow orchestrator.
        
        Args:
            mcp_registry: Optional MCP registry instance
        """
        super().__init__("workflow_orchestrator", "Orchestrates analytical workflows")
        self.mcp_registry = mcp_registry or MCPRegistry.get_instance()
        self.technique_integrator = self.mcp_registry.get_technique_integrator()
        if not self.technique_integrator:
            logger.warning("TechniqueMCPIntegrator not found in registry, creating new instance")
            self.technique_integrator = TechniqueMCPIntegrator(self.mcp_registry)
        
        self.technique_registry = self.technique_integrator.get_all_techniques()
        logger.info(f"Initialized WorkflowOrchestratorMCP with {len(self.technique_registry)} techniques")
    
    def get_capabilities(self) -> List[str]:
        """
        Get a list of capabilities provided by this MCP.
        
        Returns:
            List of capability names
        """
        return ["workflow_orchestration", "strategy_generation", "adaptive_execution"]
    
    def analyze_question(self, question: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dynamically analyze a question using an adaptive workflow.
        
        Args:
            question: The analytical question to analyze
            parameters: Optional parameters for the analysis
            
        Returns:
            Dictionary containing analysis results
        """
        # Create analysis context
        context = AnalysisContext()
        context.add("question", question)
        if parameters:
            context.add("parameters", parameters)
        
        logger.info(f"Starting analysis for question: {question[:50]}...")
        
        try:
            # Phase 1: Preliminary research using Perplexity Sonar
            self._run_preliminary_research(context)
            
            # Phase 2: Question analysis
            self._analyze_question_characteristics(context)
            
            # Phase 3: Strategy generation
            strategy = self._generate_analysis_strategy(context)
            context.add("strategy", strategy)
            
            # Phase 4: Execute dynamic workflow
            self._execute_dynamic_workflow(context)
            
            # Phase 5: Synthesis and integration
            result = self._generate_final_synthesis(context)
            
            logger.info(f"Analysis completed for question: {question[:50]}...")
            return result
        
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            context.add_event("error", "Analysis failed", {"error": str(e)})
            return {
                "error": str(e),
                "context": context,
                "status": "failed"
            }
    
    def _run_preliminary_research(self, context: AnalysisContext) -> None: 
        """
        Perform preliminary research using Perplexity Sonar.
        
        Args:
            context: The analysis context
        """
        logger.info("Running preliminary research...")
        context.add_event("info", "Starting preliminary research phase")
        
        try:
            # Get the Perplexity Sonar MCP if available
            sonar_mcp = self.mcp_registry.get_mcp("perplexity_sonar")
            if not sonar_mcp:
                logger.warning("Perplexity Sonar MCP not available, skipping preliminary research")
                context.add_event("warning", "Preliminary research skipped", 
                                 {"reason": "Perplexity Sonar MCP not available"})
                return
            
            # Prepare the input dictionary for Perplexity Sonar
            sonar_input = {
                "operation": "research",
                "question": context.question
            }
        
            # Call the Perplexity Sonar MCP to perform the research
            sonar_response = sonar_mcp.process(sonar_input)
            
            # Check if the response has the expected format and extract the research results
            if not isinstance(sonar_response, dict) or "result" not in sonar_response:
                raise ValueError(f"Unexpected response format from Perplexity Sonar: {sonar_response}")
            
            research_results = sonar_response["result"]

            # Extract specific information from the research results
            insights = research_results.get("insights", [])  
            hypotheses = research_results.get("hypotheses", []) 
            recommendations = research_results.get("recommendations", []) 
            
            # Store the extracted information in context
            context.add("preliminary_research_insights", insights)
            context.add("preliminary_research_hypotheses", hypotheses)
            context.add("preliminary_research_recommendations", recommendations)
            context.add_mcp_result("perplexity_sonar", research_results)
            context.add_event("info", "Preliminary research completed successfully")
        
        except Exception as e:
            logger.error(f"Error during preliminary research: {e}", exc_info=True)
            context.add_event("error", "Preliminary research failed", {"error": str(e)})
            # Create a simple fallback research result
            fallback_research = {
                "insights": ["Unable to perform preliminary research due to an error."],
                "hypotheses": [],
                "recommendations": ["Proceed with analysis using available information."]
            }
            context.add("preliminary_research_insights", fallback_research["insights"])
            context.add("preliminary_research_hypotheses", fallback_research["hypotheses"])
            context.add("preliminary_research_recommendations", fallback_research["recommendations"])
            context.add_mcp_result("perplexity_sonar_fallback", fallback_research)
    
    def _analyze_question_characteristics(self, context: AnalysisContext) -> None:
        """
        Analyze the question characteristics using an LLM.

        Args:
            context: The analysis context containing the question
        """
        logger.info("Analyzing question characteristics...")
        context.add_event("info", "Starting question analysis phase")

        try:
            llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
            if not llama4_scout:
                logger.warning("llama4_scout MCP not available, using default question analysis")
                analysis_result = self._get_default_question_analysis(context.question)
                context.add("question_analysis", analysis_result)
                context.add_event("warning", "Using default question analysis", 
                                 {"reason": "llama4_scout MCP not available"})
                return
            
            # Incorporate preliminary research insights if available
            insights = context.get("preliminary_research_insights", [])
            insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "No preliminary insights available."
            
            prompt = f"""
            Analyze the following question and extract its key characteristics. Provide the analysis in JSON format.

            Question: {context.question} 

            Preliminary Research Insights:
            {insights_text}

            Characteristics to extract:
            - type: The type of question (e.g., predictive, causal, evaluative, descriptive).
            - domains: The relevant domains or subject areas (e.g., economics, geopolitics, technology).
            - complexity: The complexity of the question (e.g., low, medium, high) based on the number of factors, relationships, and potential uncertainties involved.
            - uncertainty: The level of uncertainty associated with answering the question (e.g., low, medium, high) considering data availability, potential biases, and the predictability of the subject matter.
            - time_horizon: The relevant time horizon for the question (e.g., short-term, medium-term, long-term), if applicable.
            - potential_biases: Any potential biases to consider when addressing the question (e.g., confirmation bias, selection bias, availability bias).

            Return the analysis in the following JSON format:

            {{
              "type": "question_type",
              "domains": ["domain1", "domain2"],
              "complexity": "complexity_level",
              "uncertainty": "uncertainty_level",
              "time_horizon": "time_horizon",
              "potential_biases": ["bias1", "bias2"]
            }}
            """
            
            # Process the prompt using llama4_scout
            response = llama4_scout.process({"prompt": prompt})

            # Extract and parse the analysis result from the LLM response
            if isinstance(response, dict) and "result" in response:
                result_text = response["result"]
                # Try to extract JSON from the response if it's not already JSON
                if isinstance(result_text, str):
                    # Find JSON content between curly braces
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        analysis_result = json.loads(json_str)
                    else:
                        raise ValueError(f"Could not extract JSON from response: {result_text}")
                else:
                    analysis_result = result_text
            else:
                raise ValueError(f"Unexpected response format from llama4_scout: {response}")

            # Validate that all expected keys are present
            expected_keys = ["type", "domains", "complexity", "uncertainty", "time_horizon", "potential_biases"]
            if not all(key in analysis_result for key in expected_keys):
                missing_keys = [key for key in expected_keys if key not in analysis_result]
                logger.warning(f"Missing expected keys in analysis result: {missing_keys}")
                # Add missing keys with default values
                for key in missing_keys:
                    if key == "domains" or key == "potential_biases":
                        analysis_result[key] = []
                    else:
                        analysis_result[key] = "unknown"

            # Store the analysis result in the analysis context and log completion
            context.add("question_analysis", analysis_result)
            context.add_mcp_result("question_analysis", analysis_result)
            context.add_event("info", "Question analysis completed", {"analysis": analysis_result})

        except Exception as e:
            logger.error(f"Error during question analysis: {e}", exc_info=True)
            context.add_event("error", "Question analysis failed", {"error": str(e)})
            analysis_result = self._get_default_question_analysis(context.question)
            context.add("question_analysis", analysis_result)
            context.add_event("warning", "Using default question analysis due to error", {"error": str(e)})
    
    def _get_default_question_analysis(self, question: str) -> Dict[str, Any]:
        """
        Generate a default question analysis when LLM analysis fails.
        
        Args:
            question: The question to analyze
            
        Returns:
            Default question analysis
        """
        # Simple keyword-based analysis
        domains = []
        if any(kw in question.lower() for kw in ["economy", "market", "financial", "trade", "economic"]):
            domains.append("economics")
        if any(kw in question.lower() for kw in ["politic", "government", "policy", "election"]):
            domains.append("politics")
        if any(kw in question.lower() for kw in ["tech", "technology", "digital", "software", "hardware"]):
            domains.append("technology")
        if any(kw in question.lower() for kw in ["science", "scientific", "research", "experiment"]):
            domains.append("science")
        if not domains:
            domains.append("general")
            
        # Determine question type
        if any(kw in question.lower() for kw in ["will", "future", "predict", "forecast"]):
            q_type = "predictive"
        elif any(kw in question.lower() for kw in ["why", "cause", "reason", "because"]):
            q_type = "causal"
        elif any(kw in question.lower() for kw in ["evaluate", "assess", "compare", "better"]):
            q_type = "evaluative"
        else:
            q_type = "descriptive"
            
        return {
            "type": q_type,
            "domains": domains,
            "complexity": "medium",
            "uncertainty": "medium",
            "time_horizon": "medium-term",
            "potential_biases": ["confirmation bias", "recency bias"]
        }

    def _generate_analysis_strategy(self, context: AnalysisContext) -> AnalysisStrategy:
        """
        Generate a strategy dynamically based on question characteristics.
        
        Args:
            context: The analysis context
            
        Returns:
            The generated analysis strategy
        """
        question_analysis = context.get("question_analysis")
        if not question_analysis:
            logger.warning("No question analysis available, using default strategy")
            return self._generate_default_strategy(context)

        logger.info("Generating analysis strategy...")
        context.add_event("info", "Starting strategy generation phase")

        # Rule-based strategy generation based on question characteristics
        strategy_data = {
            "name": f"Dynamic Strategy for {question_analysis.get('type', 'unknown')} Question",
            "description": f"Strategy tailored for {', '.join(question_analysis.get('domains', ['general']))} domains",
            "adaptive": True,
            "steps": []
        }
        
        # Always start with research
        strategy_data["steps"].append({
            "technique": "research_to_hypothesis",
            "purpose": "Conduct initial research and generate hypotheses",
            "parameters": {},
            "adaptive_criteria": ["conflicting_evidence_found", "overall_uncertainty > 0.7"]
        })
        
        # Add domain-specific techniques based on the identified domains
        domains = question_analysis.get("domains", [])
        
        # Add techniques based on question type
        q_type = question_analysis.get("type", "unknown")
        
        if q_type == "predictive":
            # For predictive questions, add scenario generation and uncertainty mapping
            if self._is_technique_available("scenario_triangulation"):
                strategy_data["steps"].append({
                    "technique": "scenario_triangulation",
                    "purpose": "Generate multiple plausible futures",
                    "parameters": {"num_scenarios": 3},
                    "adaptive_criteria": []
                })
            
            if self._is_technique_available("uncertainty_mapping"):
                strategy_data["steps"].append({
                    "technique": "uncertainty_mapping",
                    "purpose": "Map and quantify areas of uncertainty",
                    "parameters": {},
                    "adaptive_criteria": []
                })
        
        elif q_type == "causal":
            # For causal questions, add causal analysis techniques
            if self._is_technique_available("causal_network_analysis"):
                strategy_data["steps"].append({
                    "technique": "causal_network_analysis",
                    "purpose": "Identify causal relationships",
                    "parameters": {},
                    "adaptive_criteria": []
                })
            
            if self._is_technique_available("key_assumptions_check"):
                strategy_data["steps"].append({
                    "technique": "key_assumptions_check",
                    "purpose": "Identify and challenge key assumptions",
                    "parameters": {},
                    "adaptive_criteria": []
                })
        
        elif q_type == "evaluative":
            # For evaluative questions, add comparison and assessment techniques
            if self._is_technique_available("analysis_of_competing_hypotheses"):
                strategy_data["steps"].append({
                    "technique": "analysis_of_competing_hypotheses",
                    "purpose": "Systematically evaluate competing hypotheses",
                    "parameters": {},
                    "adaptive_criteria": []
                })
            
            if self._is_technique_available("argument_mapping"):
                strategy_data["steps"].append({
                    "technique": "argument_mapping",
                    "purpose": "Map arguments for and against",
                    "parameters": {},
                    "adaptive_criteria": []
                })
        
        # Add techniques based on complexity
        complexity = question_analysis.get("complexity", "medium")
        
        if complexity == "high":
            # For high complexity questions, add techniques for breaking down the problem
            if self._is_technique_available("system_dynamics_modeling"):
                strategy_data["steps"].append({
                    "technique": "system_dynamics_modeling",
                    "purpose": "Model system dynamics",
                    "parameters": {},
                    "adaptive_criteria": []
                })
        
        # Add techniques based on uncertainty
        uncertainty = question_analysis.get("uncertainty", "medium")
        
        if uncertainty == "high":
            # For high uncertainty questions, add techniques for handling uncertainty
            if self._is_technique_available("uncertainty_mapping"):
                strategy_data["steps"].append({
                    "technique": "uncertainty_mapping",
                    "purpose": "Map and quantify areas of uncertainty",
                    "parameters": {},
                    "adaptive_criteria": []
                })
        
        # Add bias detection if potential biases are identified
        if question_analysis.get("potential_biases"):
            if self._is_technique_available("bias_detection"):
                strategy_data["steps"].append({
                    "technique": "bias_detection",
                    "purpose": "Identify and mitigate potential biases",
                    "parameters": {"target_biases": question_analysis.get("potential_biases")},
                    "adaptive_criteria": ["overall_bias_level == 'High'"]
                })
        
        # Always end with synthesis
        if self._is_technique_available("synthesis_generation"):
            strategy_data["steps"].append({
                "technique": "synthesis_generation",
                "purpose": "Generate final synthesis of analysis",
                "parameters": {"include_confidence": True},
                "adaptive_criteria": []
            })
        
        # Ensure we have at least two techniques
        if len(strategy_data["steps"]) < 2:
            logger.warning("Strategy has fewer than 2 steps, adding default techniques")
            if len(strategy_data["steps"]) == 0 or strategy_data["steps"][0]["technique"] != "research_to_hypothesis":
                strategy_data["steps"].insert(0, {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research and generate hypotheses",
                    "parameters": {},
                    "adaptive_criteria": []
                })
            
            if len(strategy_data["steps"]) == 1 or strategy_data["steps"][-1]["technique"] != "synthesis_generation":
                strategy_data["steps"].append({
                    "technique": "synthesis_generation",
                    "purpose": "Generate final synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                })
        
        # Validate all techniques exist
        valid_steps = []
        for step in strategy_data["steps"]:
            if self._is_technique_available(step["technique"]):
                valid_steps.append(step)
            else:
                logger.warning(f"Technique {step['technique']} not available, removing from strategy")
        
        strategy_data["steps"] = valid_steps
        
        strategy = AnalysisStrategy(strategy_data)
        context.add_event("info", "Strategy generation completed", {"strategy_name": strategy.name})
        return strategy
    
    def _generate_default_strategy(self, context: AnalysisContext) -> AnalysisStrategy:
        """
        Generate a default strategy when no question analysis is available.
        
        Args:
            context: The analysis context
            
        Returns:
            The default analysis strategy
        """
        strategy_data = {
            "name": "Default Analysis Strategy",
            "description": "Basic analytical strategy with essential techniques",
            "adaptive": False,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research and generate hypotheses",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "scenario_triangulation",
                    "purpose": "Generate multiple plausible futures",
                    "parameters": {"num_scenarios": 3},
                    "adaptive_criteria": []
                },
                {
                    "technique": "consensus_challenge",
                    "purpose": "Challenge prevailing consensus views",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate final synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                }
            ]
        }
        
        # Validate all techniques exist
        valid_steps = []
        for step in strategy_data["steps"]:
            if self._is_technique_available(step["technique"]):
                valid_steps.append(step)
            else:
                logger.warning(f"Technique {step['technique']} not available, removing from default strategy")
        
        strategy_data["steps"] = valid_steps
        return AnalysisStrategy(strategy_data)
    
    def _execute_dynamic_workflow(self, context: AnalysisContext) -> None:
        """
        Execute the dynamic workflow according to the generated strategy.
        
        Args:
            context: The analysis context
        """
        logger.info("Executing dynamic workflow...")
        context.add_event("info", "Starting workflow execution phase")
        
        strategy = context.get("strategy")
        if not strategy:
            logger.warning("No strategy available, skipping workflow execution")
            context.add_event("error", "Workflow execution skipped", {"reason": "No strategy available"})
            return
        
        step_idx = 0
        while step_idx < len(strategy.steps):
            step = strategy.steps[step_idx]
            technique_name = step.get("technique")
            parameters = step.get("parameters", {})
            purpose = step.get("purpose", "No purpose specified")
            
            logger.info(f"Executing step {step_idx + 1}/{len(strategy.steps)}: {technique_name} - {purpose}")
            context.add_event("info", f"Executing technique: {technique_name}", {"purpose": purpose})
            
            # Update context with current step information
            context.add("current_step", {
                "index": step_idx,
                "technique": technique_name,
                "purpose": purpose,
                "parameters": parameters
            })
            
            try:
                # Execute the technique using the TechniqueMCPIntegrator
                result = self.technique_integrator.execute_step(
                    technique_name, 
                    context,
                    parameters
                )
                
                # Add result to context
                context.add_mcp_result(technique_name, result)
                logger.info(f"Technique {technique_name} executed successfully")
                context.add_event("success", f"Technique {technique_name} executed successfully")
                
                # Check if adaptation is needed
                needs_adaptation, trigger_type = self._check_adaptation_criteria(context, step_idx)
                if needs_adaptation:
                    self._adapt_strategy(context, step_idx, trigger_type)
                    # Get the potentially updated strategy
                    strategy = context.get("strategy")
                
            except ValueError as ve:
                logger.warning(f"Expected error executing technique {technique_name}: {ve}", exc_info=True)
                context.add_event("warning", f"Expected error executing technique {technique_name}", {"error": str(ve)})
                
                # Add error result to context
                error_result = {
                    "technique": technique_name,
                    "status": "failed",
                    "error": str(ve),
                    "error_type": "expected",
                    "timestamp": time.time()
                }
                
                context.add_mcp_result(technique_name, error_result)
                
                # Handle error with fallback or alternative techniques
                self._handle_execution_error(context, technique_name, ve)
            
            except Exception as e:
                logger.error(f"Error executing technique {technique_name}: {e}", exc_info=True)
                context.add_event("error", f"Error executing technique {technique_name}", {"error": str(e)})
                
                # Add error result to context
                error_result = {
                    "technique": technique_name,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
                
                context.add_mcp_result(technique_name, error_result)
                
                # Handle error with fallback or alternative techniques
                self._handle_execution_error(context, technique_name, e)
            
            # Move to the next step
            step_idx += 1
            
        context.add_event("info", "Workflow execution completed")
    
    def _check_adaptation_criteria(self, context: AnalysisContext, current_step_idx: int) -> Tuple[bool, Optional[str]]:
        """
        Check if adaptation criteria are met for the current step.
        
        Args:
            context: The analysis context
            current_step_idx: Index of the current step
            
        Returns:
            Tuple containing (needs_adaptation, trigger_type)
        """    
        logger.info(f"Checking if strategy adaptation is needed for step {current_step_idx + 1}...")

        strategy = context.get("strategy")
        if not strategy:
            logger.warning("No strategy available, skipping adaptation check")
            return False, None

        if current_step_idx >= len(strategy.steps):
            logger.warning(f"Step index {current_step_idx} out of range for strategy with {len(strategy.steps)} steps")
            return False, None

        current_step = strategy.steps[current_step_idx]
        technique_name = current_step.get("technique")
        adaptation_criteria = current_step.get("adaptive_criteria", [])
        
        if not adaptation_criteria:
            logger.info(f"No adaptation criteria defined for step {current_step_idx + 1}, skipping adaptation")
            return False, None

        # Get the result for the current technique
        result = context.get_mcp_result(technique_name)
        if not result:
            logger.info(f"No results available for technique {technique_name}, skipping adaptation")
            return False, None
        
        # Check if the result indicates a failure
        if isinstance(result, dict) and result.get("status") == "failed":
            logger.info(f"Technique {technique_name} failed, triggering adaptation")
            return True, "TechniqueFailed"
        
        # Check each adaptation criteria
        for criteria in adaptation_criteria:
            if isinstance(criteria, str):
                # Handle string criteria
                if criteria == "conflicting_evidence_found" and result.get("conflicting_evidence_found", False):
                    logger.info(f"Adaptation criteria 'conflicting_evidence_found' met")
                    return True, "ConflictingEvidenceDetected"
                
                elif criteria == "overall_bias_level == 'High'" and result.get("overall_bias_level") == "High":
                    logger.info(f"Adaptation criteria 'overall_bias_level == High' met")
                    return True, "HighBiasDetected"
                
                elif criteria.startswith("overall_uncertainty >"):
                    try:
                        threshold = float(criteria.split(">")[1].strip())
                        uncertainty = result.get("overall_uncertainty")
                        if uncertainty is not None and float(uncertainty) > threshold:
                            logger.info(f"Adaptation criteria '{criteria}' met")
                            return True, "HighUncertaintyDetected"
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse uncertainty threshold from criteria: {criteria}")
                
                elif criteria.startswith("score_difference <"):
                    try:
                        threshold = float(criteria.split("<")[1].strip())
                        score_diff = result.get("score_difference")
                        if score_diff is not None and float(score_diff) < threshold:
                            logger.info(f"Adaptation criteria '{criteria}' met")
                            return True, "LowScoreDifferenceDetected"
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse score difference threshold from criteria: {criteria}")
            
            elif isinstance(criteria, dict):
                # Handle dictionary criteria
                for field, condition in criteria.items():
                    if field in result:
                        if isinstance(condition, str) and result[field] == condition:
                            logger.info(f"Adaptation criteria '{field} == {condition}' met")
                            return True, f"{field}_{condition}"
                        elif isinstance(condition, dict) and "value" in condition and result[field] == condition["value"]:
                            logger.info(f"Adaptation criteria '{field} == {condition['value']}' met")
                            return True, f"{field}_{condition['value']}"
        
        logger.info(f"No adaptation criteria met for step {current_step_idx + 1}")
        return False, None

    def _handle_execution_error(self, context: AnalysisContext, technique_name: str, error: Exception) -> None:
        """
        Handle execution errors with fallback or alternative techniques.
        
        Args:
            context: The analysis context
            technique_name: Name of the technique that failed
            error: The error that occurred
        """
        logger.error(f"Error executing technique {technique_name}: {error}")
        context.add_event("error", f"Error executing technique {technique_name}", {"error": str(error)})
        
        # Get the current retry count from context
        retry_count = context.get(f"{technique_name}_retry_count", 0)
        
        # Check if we should retry
        if retry_count < self.MAX_RETRIES:
            retry_count += 1
            context.add(f"{technique_name}_retry_count", retry_count)
            logger.info(f"Retrying technique {technique_name}, attempt {retry_count}/{self.MAX_RETRIES}")
            context.add_event("info", f"Retrying technique {technique_name}", {"attempt": retry_count})
            
            try:
                # Get the current step parameters
                current_step = context.get("current_step", {})
                parameters = current_step.get("parameters", {})
                
                # Execute the technique again
                result = self.technique_integrator.execute_step(
                    technique_name, 
                    context,
                    parameters
                )
                
                # Add result to context
                context.add_mcp_result(f"{technique_name}_retry_{retry_count}", result)
                logger.info(f"Retry {retry_count} of technique {technique_name} succeeded")
                context.add_event("success", f"Retry of technique {technique_name} succeeded", {"attempt": retry_count})
                
                # Update the original result
                context.add_mcp_result(technique_name, result)
                
                return
            
            except Exception as retry_error:
                logger.error(f"Retry {retry_count} of technique {technique_name} failed: {retry_error}", exc_info=True)
                context.add_event("error", f"Retry of technique {technique_name} failed", {"attempt": retry_count, "error": str(retry_error)})
        
        # If we've exhausted retries or the retry failed, try a fallback technique
        logger.info(f"Using fallback for technique {technique_name}")
        context.add_event("info", f"Using fallback for technique {technique_name}")
        
        # Define fallback techniques for common techniques
        fallbacks = {
            "research_to_hypothesis": "basic_research",
            "scenario_triangulation": "simple_scenario_generation",
            "synthesis_generation": "basic_synthesis",
            "uncertainty_mapping": "simple_uncertainty_assessment",
            "bias_detection": "simple_bias_check"
        }
        
        fallback_technique = fallbacks.get(technique_name)
        if fallback_technique and self._is_technique_available(fallback_technique):
            try:
                # Get the current step parameters
                current_step = context.get("current_step", {})
                parameters = current_step.get("parameters", {})
                
                # Execute the fallback technique
                result = self.technique_integrator.execute_step(
                    fallback_technique, 
                    context,
                    parameters
                )
                
                # Add result to context
                context.add_mcp_result(f"{technique_name}_fallback", result)
                logger.info(f"Fallback technique {fallback_technique} for {technique_name} succeeded")
                context.add_event("success", f"Fallback technique for {technique_name} succeeded", {"fallback": fallback_technique})
                
                # Update the original result with a note about the fallback
                result["fallback_used"] = True
                result["original_technique"] = technique_name
                result["fallback_technique"] = fallback_technique
                context.add_mcp_result(technique_name, result)
                
                return
            
            except Exception as fallback_error:
                logger.error(f"Fallback technique {fallback_technique} for {technique_name} failed: {fallback_error}", exc_info=True)
                context.add_event("error", f"Fallback technique for {technique_name} failed", {"fallback": fallback_technique, "error": str(fallback_error)})
        
        # If no fallback is available or the fallback failed, create a minimal result
        minimal_result = {
            "status": "failed",
            "error": str(error),
            "fallback_used": False,
            "retry_attempted": retry_count > 0,
            "technique": technique_name,
            "timestamp": time.time()
        }
        
        context.add_mcp_result(technique_name, minimal_result)
        logger.warning(f"Using minimal result for failed technique {technique_name}")
        context.add_event("warning", f"Using minimal result for failed technique {technique_name}")

    def _adapt_strategy(self, context: AnalysisContext, current_step_idx: int, trigger_type: str) -> None:
        """
        Adapt the strategy based on the trigger type.
        
        Args:
            context: The analysis context
            current_step_idx: Index of the current step
            trigger_type: The type of trigger that initiated the adaptation
        """
        logger.info(f"Adapting strategy due to trigger: {trigger_type}")
        context.add_event("info", "Adapting strategy", {"trigger": trigger_type})

        strategy = context.get("strategy")
        if not strategy:
            logger.warning("No strategy available, cannot adapt")
            return
        
        if current_step_idx >= len(strategy.steps):
            logger.warning(f"Step index {current_step_idx} out of range for strategy with {len(strategy.steps)} steps")
            return
        
        # Get the remaining steps
        completed_steps = strategy.steps[:current_step_idx + 1]
        remaining_steps = strategy.steps[current_step_idx + 1:]
        
        # Apply rule-based adaptations based on trigger type
        if trigger_type == "TechniqueFailed":
            # Skip the failed technique and continue with the next one
            logger.info("Technique failed, continuing with remaining steps")
            # No changes to remaining_steps needed
        
        elif trigger_type == "HighUncertaintyDetected" and self._is_technique_available("uncertainty_mapping"):
            # Add uncertainty mapping technique
            logger.info("Adding uncertainty_mapping technique due to high uncertainty")
            new_step = {
                "technique": "uncertainty_mapping",
                "purpose": "Map and quantify areas of uncertainty",
                "parameters": {},
                "adaptive_criteria": []
            }
            remaining_steps.insert(0, new_step)
        
        elif trigger_type == "ConflictingEvidenceDetected" and self._is_technique_available("analysis_of_competing_hypotheses"):
            # Add competing hypotheses analysis
            logger.info("Adding analysis_of_competing_hypotheses technique due to conflicting evidence")
            new_step = {
                "technique": "analysis_of_competing_hypotheses",
                "purpose": "Systematically evaluate competing hypotheses",
                "parameters": {},
                "adaptive_criteria": []
            }
            remaining_steps.insert(0, new_step)
        
        elif trigger_type == "HighBiasDetected" and self._is_technique_available("bias_detection"):
            # Add bias detection technique
            logger.info("Adding bias_detection technique due to high bias")
            new_step = {
                "technique": "bias_detection",
                "purpose": "Identify and mitigate potential biases",
                "parameters": {},
                "adaptive_criteria": []
            }
            remaining_steps.insert(0, new_step)
        
        elif trigger_type == "LowScoreDifferenceDetected" and self._is_technique_available("key_assumptions_check"):
            # Add key assumptions check
            logger.info("Adding key_assumptions_check technique due to low score difference")
            new_step = {
                "technique": "key_assumptions_check",
                "purpose": "Identify and challenge key assumptions",
                "parameters": {},
                "adaptive_criteria": []
            }
            remaining_steps.insert(0, new_step)
        
        # Update the strategy with adapted steps
        adapted_strategy_data = {
            "name": f"{strategy.name} (Adapted)",
            "description": f"{strategy.description} - Adapted due to {trigger_type}",
            "adaptive": strategy.adaptive,
            "steps": completed_steps + remaining_steps
        }
        
        adapted_strategy = AnalysisStrategy(adapted_strategy_data)
        context.add("strategy", adapted_strategy)
        context.add_event("info", "Strategy adapted", {
            "trigger": trigger_type,
            "original_steps": len(strategy.steps),
            "adapted_steps": len(adapted_strategy.steps)
        })
        
        logger.info(f"Strategy adapted: {len(strategy.steps)} steps -> {len(adapted_strategy.steps)} steps")

    def _generate_final_synthesis(self, context: AnalysisContext) -> Dict[str, Any]:    
        """
        Generate final synthesis of all analysis results.
        
        Args:
            context: The analysis context
            
        Returns:
            Dictionary containing the final synthesis
        """
        logger.info("Generating final synthesis...")
        context.add_event("info", "Starting synthesis generation phase")
        
        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.error("llama4_scout MCP not available, cannot generate synthesis")
            return {
                "error": "llama4_scout MCP not available", 
                "status": "failed",
                "context": context
            }

        # Prepare a simplified context for the prompt to avoid token limits
        simplified_context = {
            "question": context.question,
            "question_analysis": context.get("question_analysis", {}),
            "preliminary_research": {
                "insights": context.get("preliminary_research_insights", []),
                "hypotheses": context.get("preliminary_research_hypotheses", [])
            },
            "workflow_summary": []
        }
        
        # Add a summary of each technique's results
        for technique_name, result in context.get_mcp_results().items():
            if isinstance(result, dict):
                # Skip internal results like retries and fallbacks
                if "_retry_" in technique_name or "_fallback" in technique_name:
                    continue
                
                # Extract key information from the result
                result_summary = {
                    "technique": technique_name,
                    "status": result.get("status", "completed")
                }
                
                # Add key outputs based on technique type
                if "key_findings" in result:
                    result_summary["key_findings"] = result["key_findings"]
                if "hypotheses" in result:
                    result_summary["hypotheses"] = result["hypotheses"]
                if "scenarios" in result:
                    result_summary["scenarios"] = result["scenarios"]
                if "confidence" in result:
                    result_summary["confidence"] = result["confidence"]
                
                simplified_context["workflow_summary"].append(result_summary)

        prompt = f"""
        Generate a comprehensive synthesis of the analysis results for the following question: '{context.question}'.

        Here is the context of the analysis, including the question characteristics, and key results:
        {json.dumps(simplified_context, indent=2)}

        Your synthesis should include the following sections:
        - Integrated Assessment: A concise summary of the overall findings and their implications.
        - Key Judgments: The most important conclusions or judgments derived from the analysis, and the supporting evidence for each.
        - Confidence Level: An assessment of the confidence in the analysis and its findings (e.g., low, medium, high), with a rationale for this level.
        - Alternative Perspectives: A discussion of alternative perspectives or interpretations of the analysis results, including potential limitations or uncertainties.
        - Indicators to Monitor: Key indicators that should be monitored to track the evolution of the situation or to validate the analysis.

        Provide your synthesis in the following JSON format:
        {{
            "integrated_assessment": "...",
            "key_judgments": [
                {{
                    "judgment": "...",
                    "supporting_evidence": "...",
                    "confidence": "..."
                }}
            ],
            "confidence_level": "...",
            "confidence_explanation": "...",
            "alternative_perspectives": ["...", "..."],
            "indicators_to_monitor": ["...", "..."]
        }}
        """
        
        try:
            response = llama4_scout.process({"prompt": prompt})
            
            # Extract and parse the synthesis result
            if isinstance(response, dict) and "result" in response:
                result_text = response["result"]
                # Try to extract JSON from the response if it's not already JSON
                if isinstance(result_text, str):
                    # Find JSON content between curly braces
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        synthesis_data = json.loads(json_str)
                    else:
                        raise ValueError(f"Could not extract JSON from response: {result_text}")
                else:
                    synthesis_data = result_text
            else:
                raise ValueError(f"Unexpected response format from llama4_scout: {response}")

            # Validate the synthesis format
            expected_keys = ["integrated_assessment", "key_judgments", "confidence_level", 
                            "confidence_explanation", "alternative_perspectives", "indicators_to_monitor"]
            
            missing_keys = [key for key in expected_keys if key not in synthesis_data]
            if missing_keys:
                logger.warning(f"Missing keys in synthesis: {missing_keys}")
                # Add missing keys with default values
                for key in missing_keys:
                    if key in ["alternative_perspectives", "indicators_to_monitor"]:
                        synthesis_data[key] = []
                    elif key == "key_judgments":
                        synthesis_data[key] = [{"judgment": "No specific judgments could be made.", 
                                               "supporting_evidence": "Insufficient data.", 
                                               "confidence": "low"}]
                    else:
                        synthesis_data[key] = "Not available."

            # Prepare the final result
            result = {
                "question": context.question,
                "synthesis": synthesis_data,
                "workflow_events": context.get_events(),
                "status": "completed",
                "context": context
            }

            context.add_event("info", "Synthesis generation completed")
            return result

        except Exception as e:
            logger.error(f"Error generating synthesis: {e}", exc_info=True)
            context.add_event("error", "Synthesis generation failed", {"error": str(e)})
            
            # Return error result with context
            return {
                "question": context.question,
                "error": str(e),
                "workflow_events": context.get_events(),
                "status": "failed",
                "context": context
            }
        
    def _is_technique_available(self, technique_name: str) -> bool:
        """
        Check if a technique is available in the technique registry.
        
        Args:
            technique_name: The name of the technique to check
            
        Returns:
            True if the technique is available, False otherwise
        """
        return technique_name in self.technique_registry
