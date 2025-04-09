
"""
WorkflowOrchestratorMCP for orchestrating analytical workflows.
This module provides the central orchestrator for dynamic analytical workflows.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional

from .base_mcp import BaseMCP
from .mcp_registry import MCPRegistry
from .analysis_context import AnalysisContext
from .analysis_strategy import AnalysisStrategy
from .analytical_technique import AnalyticalTechnique
from .technique_registry import TechniqueRegistry

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
        self.technique_registry: Dict[str, AnalyticalTechnique] = {}
        self.register_techniques()
        logger.info("Initialized WorkflowOrchestratorMCP")

        
        
    
    def register_techniques(self):
        """
        Register available techniques using the TechniqueMCPIntegrator.
        """
        logger.info("Registering techniques...")
        integrator = self.mcp_registry.get_technique_integrator()
        if integrator:
            self.technique_registry = integrator.get_all_techniques()
            logger.info(f"Registered {len(self.technique_registry)} techniques.")
            logger.debug(f"Techniques registered: {list(self.technique_registry.keys())}")
    
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
        context = AnalysisContext(question, parameters)
        logger.info(f"Starting analysis for question: {question[:50]}...")
        
        try:
            # Phase 1: Preliminary research using Perplexity Sonar
            self._run_preliminary_research(context)
            
            # Phase 2: Question analysis
            self._analyze_question_characteristics(context)
            
            # Phase 3: Strategy generation
            strategy = self._generate_analysis_strategy(context)
            context.set_strategy(strategy)
            
            # Phase 4: Execute dynamic workflow
            self._execute_dynamic_workflow(context)
            
            # Phase 5: Synthesis and integration
            result = self._generate_final_synthesis(context)
            
            logger.info(f"Analysis completed for question: {question[:50]}...")
            return result
        
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            context.log_error(f"Analysis failed: {str(e)}")
            return {
                "error": str(e),
                "context": context.to_dict(),
                "status": "failed"
            }
    
    def _run_preliminary_research(self, context: AnalysisContext) -> None: 
        """
        Perform preliminary research using Perplexity Sonar.
        
        Args:
            context: The analysis context
        """
        logger.info("Running preliminary research...")
        context.log_info("Starting preliminary research phase")
        
        try:

            # Get the Perplexity Sonar MCP if available
            sonar_mcp = self.mcp_registry.get_mcp("perplexity_sonar")
            if not sonar_mcp:
                logger.warning("Perplexity Sonar MCP not available, skipping preliminary research")
                context.log_info("Preliminary research skipped: Perplexity Sonar MCP not available")
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
            context.add_metadata("preliminary_research_insights", insights)
            context.add_metadata("preliminary_research_hypotheses", hypotheses)
            context.add_metadata("preliminary_research_recommendations", recommendations)
            context.log_info("Preliminary research completed successfully")
        
        except Exception as e:
            logger.error(f"Error during preliminary research: {e}, response: {sonar_response}", exc_info=True)
            context.log_error(f"Preliminary research failed: {str(e)}")
    
    def _analyze_question_characteristics(self, context: AnalysisContext) -> None:
        """
        Analyze the question characteristics using an LLM.

        Args:
            context: The analysis context containing the question
            
        Returns:
            None
        """
        logger.info("Analyzing question characteristics...")
        context.log_info("Starting question analysis phase")

        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.error("llama4_scout MCP not available, using default question analysis")
            analysis_result = self._get_default_question_analysis(context.question)
        analysis_result = {}
        prompt = f"""
            Analyze the following question and extract its key characteristics. Provide the analysis in JSON format.

            Question: {context.question} 

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
        try:
            
            
            # Process the prompt using llama4_scout
            response = llama4_scout.process(prompt)

            # Extract and parse the analysis result from the LLM response
            analysis_result: Dict = json.loads(response["result"])

            # Validate that all expected keys are present
            expected_keys = ["type", "domains", "complexity", "uncertainty", "time_horizon", "potential_biases"]
            if not all(key in analysis_result for key in expected_keys):
                raise ValueError(f"Missing expected keys in analysis result: {analysis_result}")

            # Store the analysis result in the analysis context and log completion
            context.question_analysis = analysis_result
            context.log_info(f"Question analysis completed: {analysis_result}")
            

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error during question analysis: {e}, using default analysis", exc_info=True)
            context.log_warning(f"Using default question analysis due to error: {e}")
            analysis_result = self._get_default_question_analysis(context.question)
        context.question_analysis = analysis_result
    

    def _generate_analysis_strategy(self, context: AnalysisContext) -> AnalysisStrategy:
        """
        Generate a strategy dynamically using the llama4_scout MCP.
        
        Args:
            context: The analysis context
            
        Raises:
            ValueError: If the generated strategy is invalid or if there's an error during dynamic strategy generation.
        
        Returns:
            AnalysisStrategy: The generated analysis strategy.
        """
        question_analysis = context.question_analysis
        if not question_analysis:
            logger.warning("No question analysis available, using default strategy")
            return self._generate_default_strategy(context)

        logger.info("Generating analysis strategy...")
        context.log_info("Starting strategy generation phase")

        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.error("llama4_scout MCP not available, using default strategy")
            return self._generate_default_strategy(context)

        prompt = f"""
        You are an expert analytical strategist, tasked with designing an analysis strategy to address a complex question.
        
        Here's the analysis of the question, the result of the question analysis phase:
        {json.dumps(question_analysis, indent=2)}
        The question is: '{context.question}'
        
        The available analytical techniques (techniques registry) are:
        {json.dumps(list(self.technique_registry.keys()), indent=2)}

        Your strategy should be a sequence of analytical techniques, chosen and ordered to effectively address the question. Each step in the strategy should have a clear purpose and a set of parameters to guide its execution.

        Provide your strategy in the following JSON format:
        {{
            "name": "Strategy Name",
            "description": "A brief description of the strategy",
            "adaptive": true,
            "steps": [
                {{
                    "technique": "technique_name",
                    "purpose": "A concise description of the technique's purpose in this strategy",
                    "parameters": {{  // Parameters specific to the technique
                        // e.g., "num_hypotheses": 3, "sources": ["source1", "source2"], "n_personas": 3
                    }},
                    "adaptive_criteria": [  // Criteria for adapting the strategy after this step
                        // e.g., "overall_uncertainty > 0.8", "overall_bias_level == 'High'", "conflicting_evidence_found", "score_difference < 0.2"
                        // If any of these criteria are met, the strategy should be adapted.
                        // Use clear and concise criteria that can be easily evaluated based on the technique's results.
                    ]
                }}
                // Additional steps
            ]
        }}

        Consider the following principles when designing the strategy:
        - **Relevance:** Choose techniques that are most relevant to the question's type, domains, complexity, and uncertainty.
        - **Efficiency:** Aim for a strategy that addresses the question effectively with a minimal number of steps.
        - **Depth:** Select techniques that allow for a thorough and in-depth analysis of the question.
        - **Adaptability:** Design the strategy to be adaptive, with clear criteria for modifying the remaining steps based on the results of each step.
        - **Data Grounding:** Prioritize techniques that leverage data and evidence to support their analysis and conclusions.
        - **Transparency:** Ensure the strategy is transparent and the rationale behind each step is clear.

        Your strategy should include at least two techniques and consider how the choice of the second technique might depend on the outcome of the first. Provide clear adaptive criteria for each step, if applicable.
        """
        try:
            response = llama4_scout.process(prompt)
            strategy_data = json.loads(response["result"])

            if not isinstance(strategy_data, dict) or "steps" not in strategy_data or not isinstance(strategy_data["steps"], list) or len(strategy_data["steps"]) < 2 :
                raise ValueError(f"Invalid strategy format or insufficient number of steps received from llama4_scout: {response['result']}")


            
            for step in strategy_data["steps"]:
                if "technique" not in step or step.get("technique") not in self.technique_registry:
                    raise ValueError(f"Invalid technique name: {step.get('technique', 'unknown')}")

            strategy = AnalysisStrategy(strategy_data)  
            context.log_info(f"Strategy generation completed: {strategy.name}")
            return strategy

        except (json.JSONDecodeError, ValueError) as e:            
            logger.error(f"Error during dynamic strategy generation: {e}, using fallback strategy", exc_info=True)            
            return self._generate_default_strategy(context)
    
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
        return AnalysisStrategy(strategy_data)
    
    def _execute_dynamic_workflow(self, context: AnalysisContext) -> None:
        """
        Execute the dynamic workflow according to the generated strategy.
        
        Args:
            context: The analysis context
        """
        logger.info("Executing dynamic workflow...")
        context.log_info("Starting workflow execution phase")
        
        strategy = context.strategy
        if not strategy:
            logger.warning("No strategy available, skipping workflow execution")
            context.log_error("Workflow execution skipped: No strategy available")
            return
        
        step_idx = 0
        while step_idx < len(strategy.steps):
            step = strategy.steps[step_idx]
            technique_name = step.get("technique")
            parameters = step.get("parameters", {})
            purpose = step.get("purpose", "No purpose specified")
            
            logger.info(f"Executing step {step_idx + 1}/{len(strategy.steps)}: {technique_name} - {purpose}")
            context.log_info(f"Executing technique: {technique_name} - {purpose}")
            
            # Check if technique exists
            technique = self.technique_registry.get(technique_name)
            if not technique:
                logger.warning(f"Technique not found: {technique_name}")
                context.log_error(f"Technique not found: {technique_name}")
                step_idx += 1
                continue
            
            try:
                # Execute the technique
                result = technique.process(question=context.question, **parameters)
                
                # Add result to context
                context.add_result(technique_name, result)
                logger.info(f"Technique {technique_name} executed successfully. Result: {result}")
                

            except ValueError as ve:
                logger.warning(f"Expected error executing technique {technique_name}: {ve}", exc_info=True)
                context.log_warning(f"Expected error executing technique {technique_name}: {str(ve)}")
                
                # Add error result to context
                error_result = {
                    "technique": technique_name,
                    "status": "failed",
                    "error": str(ve),
                    "error_type": "expected",
                    "timestamp": time.time()
                }
                
                context.add_result(technique_name, error_result)
                
                # Handle error with fallback or alternative techniques
                self._handle_execution_error(context, technique_name, ve)
            
            except KeyError as ke:                
                logger.error(f"Unexpected error executing technique {technique_name}: {ke}", exc_info=True)
                context.log_error(f"Unexpected error executing technique {technique_name}: {str(ke)}")
                
                # Add error result to context
                error_result = {
                    "technique": technique_name,
                    "status": "failed",
                    "error": str(ke),
                    "error_type": "unexpected",
                    "timestamp": time.time()
                }
                
                context.add_result(technique_name, error_result)
                
            except Exception as e:
                logger.error(f"Error executing technique {technique_name}: {e}", exc_info=True)
                context.log_error(f"Unexpected error executing technique {technique_name}: {str(e)}")
                
                # Add error result to context
                error_result = {
                    "technique": technique_name,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                }

                
                context.add_result(technique_name, error_result)
                
                # Handle error with fallback or alternative techniques
                self._handle_execution_error(context, technique_name, e)
            
            # Check if adaptation is needed (after potential fallback)
            if strategy.adaptive and step.get("adaptive_criteria"):
              if step_idx < len(strategy.steps):
                  if adapted_strategy:
                    self._adapt_strategy(context, step_idx, trigger_type)
                    strategy = context.strategy

            step_idx += 1

            
        context.log_info("Workflow execution completed")
    
    def _check_adaptation_criteria(self, context: AnalysisContext, current_step_idx: int) -> tuple[bool, str]:
        """
        Check if adaptation criteria are met using the strategy_adaptation MCP.
        
        Args:
            context: The analysis context
            current_step_idx: Index of the current step
            
        Returns:
            tuple: (bool,str) True if adaptation is needed, False otherwise, and the trigger type.
        """    
        logger.info(f"Checking if strategy adaptation is needed for step {current_step_idx + 1}...")

        strategy = context.strategy
        if not strategy:
            logger.warning("No strategy available, skipping adaptation check, return False")
            return False, None

        current_step = strategy.steps[current_step_idx]

        adaptation_criteria = current_step.get("adaptive_criteria", [])        
        technique_name = current_step.get("technique")
        results = context.results.get(technique_name, {})

        if not adaptation_criteria or adaptation_criteria == []:
            logger.info(f"No adaptation criteria defined for step {current_step_idx + 1}, skipping adaptation.")
            return False, None

        if not results or results == {}:
            logger.info(f"No results available for step {current_step_idx + 1}, skipping adaptation.")
            return False, None
        
        trigger_type = None
        
        for criteria in adaptation_criteria:
            

            if isinstance(criteria, dict):
                for field, conditions in criteria.items():
                    if field in results and results[field] is not None:
                        if isinstance(conditions, str) and results[field] == conditions:
                            trigger_type = f"{field}_eq_{conditions}"
                            logger.info(f"Adaptation criteria '{field} == {conditions}' met, triggering adaptation.")
                            break
                        elif isinstance(conditions, dict) and "value" in conditions and results[field] == conditions["value"]:
                            trigger_type = f"{field}_eq_{conditions['value']}"

                            logger.info(f"Adaptation criteria '{field} == {conditions}' met, triggering adaptation.")
                            break
                    
            elif isinstance(criteria, str):
                if criteria.startswith("overall_uncertainty >"):
                    threshold = float(criteria.split(">")[1].strip())
                    if results.get("overall_uncertainty") is not None and float(results["overall_uncertainty"]) > threshold:
                        trigger_type = "HighUncertaintyDetected"
                elif criteria == "overall_bias_level == 'High'" and results.get("overall_bias_level") == "High":
                     trigger_type = "HighBiasDetected"
                elif criteria == "conflicting_evidence_found" and results.get("conflicting_evidence_found", False):
                    trigger_type = "ConflictingEvidenceDetected"
                elif criteria.startswith("score_difference <") and "ach_score_difference" in results:
                    threshold = float(criteria.split("<")[1].strip())
                    if float(results["ach_score_difference"]) < threshold:
                        trigger_type = "ACHScoreDifferenceBelowThreshold"
            if trigger_type:
                 logger.info(f"Adaptation criteria '{criteria}' met, triggering adaptation.")
                 break

        return bool(trigger_type) , trigger_type

    
    def _handle_execution_error(self, context: AnalysisContext, technique_name: str, error: Exception, retry_count: int = 0) -> None:
        """
        Handle execution errors with fallback or alternative techniques.
        
        Args:
            context: The analysis context
            technique_name: Name of the technique that failed
            error: The error that occurred
        """
        logger.error(f"Error executing technique {technique_name}: {error}")        

        if retry_count > 0:
            logger.info(f"Retrying technique {technique_name}, attempt {retry_count}...")
            context.log_info(f"Retrying technique {technique_name}, attempt {retry_count}...")
        context.log_error(f"Error executing technique {technique_name}: {str(error)}")        

        # Check if the technique has a fallback
        technique = self.technique_registry.get(technique_name)
        if technique and callable(getattr(technique, "fallback", None)):
            logger.info(f"Attempting fallback for technique: {technique_name}")
            context.log_info(f"Attempting fallback for technique: {technique_name}")
            
            try:
                fallback_result = technique.fallback(context)
                if fallback_result:
                    logger.info(f"Fallback successful for technique: {technique_name}")
                    context.log_info(f"Fallback successful for technique: {technique_name}")
                    context.add_result(technique_name + "_fallback", fallback_result)
                    return
                else:
                    logger.warning(f"Fallback did not return a result for technique: {technique_name}")
                    context.log_warning(f"Fallback did not return a result for technique: {technique_name}")

            except Exception as fb_e:
                logger.error(f"Error during fallback execution for {technique_name}: {fb_e}")
                context.log_error(f"Error during fallback execution for {technique_name}: {str(fb_e)}")
                context.add_result(technique_name + "_fallback", {"status": "failed", "error": str(fb_e)})


        # Check for specific error types and decide on action
        if isinstance(error, KeyError):
            logger.warning(f"KeyError encountered for technique {technique_name}, skipping step.")
            context.log_warning(f"KeyError encountered for technique {technique_name}, skipping step.")
            # Can also try to recover by executing a fallback
            return
        
        # For other errors, or if no fallback is available, halt the workflow (or implement retry)
        if technique_name == "synthesis_generation" :
            logger.error(f"Critical error in {technique_name}, halting workflow.")
            context.log_error(f"Critical error in {technique_name}, halting workflow.")
            # Halt workflow or raise exception
            raise error    
        
        # Implement retry logic with a retry counter
        if retry_count < self.MAX_RETRIES:
            logger.info(f"Retrying technique {technique_name} after error.")
            context.log_info(f"Retrying technique {technique_name} after error.")
            # Retry by calling the technique.process method again
            self._execute_dynamic_workflow(context,technique_name, retry_count + 1 )
            return
        
        else:
            logger.warning(f"Max retries exceeded for technique {technique_name}, skipping.")
            context.log_warning(f"Max retries exceeded for technique {technique_name}, skipping.")
            return
    

            


    def _adapt_strategy(self, context: AnalysisContext, current_step_idx: int, trigger_type: str) -> None:        
        """
        Adapt the analysis strategy based on the specified trigger.
        
        Args:
            context: The analysis context
            current_step_idx: Index of the current step
            trigger_type: The type of trigger that initiated the adaptation
        """
        logger.info(f"Adapting strategy due to trigger: {trigger_type}")
        context.log_info(f"Adapting strategy due to trigger: {trigger_type}")

        strategy = context.strategy
        if not strategy:
            logger.warning("No strategy available, cannot adapt")
            return
        

        remaining_steps = strategy.steps[current_step_idx + 1:]
        
        # Hybrid adaptation approach
        if trigger_type == "HighUncertaintyDetected" and self._is_technique_available("UncertaintyReductionTechnique"):
            # Common adaptation: Add an uncertainty reduction technique
            new_step = {
                "technique": "UncertaintyReductionTechnique",
                "purpose": "Reduce uncertainty identified in the previous step",
                "parameters": {}
            }
            remaining_steps.insert(0, new_step)
            logger.info("Added UncertaintyReductionTechnique to the strategy.")

        elif trigger_type == "ConflictingEvidenceDetected" and self._is_technique_available("EvidenceReconciliationTechnique"):
            # Common adaptation: Add an evidence reconciliation technique
            new_step = {
                "technique": "EvidenceReconciliationTechnique",
                "purpose": "Reconcile conflicting evidence identified in the previous step",
                "parameters": {}
            }
            remaining_steps.insert(0, new_step)
            logger.info("Added EvidenceReconciliationTechnique to the strategy.")

        else:
            # Complex or unhandled trigger: Use LLM-based adaptation
            logger.info("Using LLM-based adaptation for this trigger.")
            llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
            if not llama4_scout:
                logger.error("llama4_scout MCP not available, cannot adapt strategy")
                return

            prompt:str = f"""
            The analysis strategy needs to be adapted due to the following trigger: {trigger_type}.

            Here's the context of the analysis:
            {json.dumps(context.to_dict(), indent=2)}

            The remaining steps in the strategy are:
            {json.dumps(remaining_steps, indent=2)}

            The available analytical techniques are:
            {json.dumps(list(self.technique_registry.keys()), indent=2)}

            Provide a new list of remaining steps that addresses the trigger and adapts the strategy accordingly. Maintain the overall goal of the analysis and ensure the adapted strategy is coherent and effective. Use the same JSON format as the 'steps' in the original strategy.
            """

            try:
                response = llama4_scout.process(prompt)
                new_steps = json.loads(response["result"])
                if not isinstance(new_steps, list):
                    raise ValueError("Invalid format for adapted steps received from llama4_scout")

                # Validate technique names in the new steps
                for step in new_steps:
                    if step["technique"] not in self.technique_registry:
                        raise ValueError(f"Invalid technique name: {step['technique']}")

                remaining_steps = new_steps  # Replace remaining steps with the adapted ones
                logger.info("Successfully adapted strategy using LLM.")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error during LLM-based strategy adaptation: {e}", exc_info=True)
                logger.warning("Could not adapt strategy, keeping original steps.")
                return  # Keep the original steps if adaptation fails

        # Update the strategy with the adapted steps
        strategy.steps = strategy.steps[:current_step_idx+1] + remaining_steps
        context.log_info(f"Adapted strategy: {strategy.steps}")

    def _generate_final_synthesis(self, context: AnalysisContext) -> Dict[str, Any]:    
        """
        Generate final synthesis of all analysis results.
        
        Args:
            context: The analysis context
            
        Returns:
            Dictionary containing the final synthesis
        """
        logger.info("Generating final synthesis...")
        context.log_info("Starting synthesis generation phase")
        
        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.error("llama4_scout MCP not available, cannot generate synthesis")
            return {"error": "llama4_scout MCP not available", "status": "failed"}

        prompt = f"""
        Generate a comprehensive synthesis of the analysis results for the following question: '{context.question}'.

        Here is the context of the analysis, including the question characteristics, strategy, steps, and results:
        {json.dumps(context.to_dict(), indent=2)}

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
            response = llama4_scout.process(prompt)
            synthesis_data = json.loads(response["result"])

            # Validate the synthesis format
            if not isinstance(synthesis_data, dict) or any(key not in synthesis_data for key in ["integrated_assessment", "key_judgments", "confidence_level", "confidence_explanation", "alternative_perspectives", "indicators_to_monitor"]):
                raise ValueError("Invalid synthesis format received from llama4_scout")

            # Check if key_judgments is a list
            if not isinstance(synthesis_data["key_judgments"], list):
                raise ValueError("Invalid key_judgments format in synthesis")

            # Validate each key_judgment format
            for judgment in synthesis_data["key_judgments"]:
                if not isinstance(judgment, dict) or any(key not in judgment for key in ["judgment", "supporting_evidence", "confidence"]):
                    raise ValueError("Invalid key_judgment format in synthesis")

            # Prepare the final result
            result = {
                "question": context.question,
                "context": context.to_dict(),
                "synthesis": synthesis_data,
                "status": "completed"
            }

            context.log_info("Synthesis generation completed")
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error generating synthesis: {e}", exc_info=True)
            context.log_error(f"Synthesis generation failed: {str(e)}")
            
            # Return error result
            return {
                "question": context.question,
                "context": context.to_dict(),
                "error": str(e),
                "status": "failed"
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
