"""
Workflow Orchestrator MCP for dynamic analysis workflow management.
This module provides the WorkflowOrchestratorMCP class for orchestrating analytical workflows.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import json
import re

from src.base_mcp import BaseMCP
from src.mcp_registry import MCPRegistry
from src.analysis_context import AnalysisContext
from src.analysis_strategy import AnalysisStrategy
from src.technique_mcp_integrator import TechniqueMCPIntegrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowOrchestratorMCP(BaseMCP):
    """
    Workflow Orchestrator MCP for dynamic analysis workflow management.
    
    This MCP provides capabilities for:
    1. Analyzing question characteristics to determine optimal workflow sequence
    2. Selecting techniques based on question type (predictive, causal, evaluative)
    3. Adapting workflow dynamically based on interim findings
    4. Managing technique dependencies and complementary pairs
    5. Orchestrating the execution of analytical techniques
    """
    
    def __init__(self, mcp_registry: MCPRegistry, techniques_dir: str = "src/techniques"):
        """
        Initialize the WorkflowOrchestratorMCP.
        
        Args:
            mcp_registry: Registry of available MCPs
            techniques_dir: Directory containing technique modules
        """
        super().__init__(
            name="workflow_orchestrator",
            description="Orchestrates dynamic analytical workflows based on question characteristics",
            version="1.0.0"
        )
        
        self.mcp_registry = mcp_registry
        self.integrator = TechniqueMCPIntegrator(mcp_registry, techniques_dir)
        self.active_workflows = {}  # workflow_id -> workflow configuration
        self.technique_registry = {}  # name -> technique instance
        
        # Register available techniques
        self.register_techniques()
        
        logger.info("Initialized WorkflowOrchestratorMCP")
    
    def register_techniques(self):
        """Register available analytical techniques."""
        logger.info("Registering analytical techniques")
        
        # Get all available techniques from the integrator
        available_techniques = self.integrator.get_all_techniques()
        if not available_techniques:
            logger.warning("No techniques available from integrator, using default techniques")
            self._register_default_techniques()
            return
        
        # Register techniques
        for name, technique in available_techniques.items():
            self.technique_registry[name] = technique
            logger.info(f"Registered technique: {name}")
        
        logger.info(f"Registered {len(self.technique_registry)} techniques")
    
    def _register_default_techniques(self):
        """Register default techniques if none are available from the integrator."""
        logger.info("Registering default techniques")
        
        # This is a fallback method that would be implemented with basic techniques
        # For now, we'll just log a warning
        logger.warning("Default technique registration not implemented")
    
    def _get_available_techniques(self):
        """Get names of available techniques."""
        return list(self.technique_registry.keys())
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in WorkflowOrchestratorMCP")
        
        # Validate input
        if not isinstance(input_data, dict):
            error_msg = "Input must be a dictionary"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get operation type
        operation = input_data.get("operation", "analyze_question")
        
        # Process based on operation type
        if operation == "analyze_question":
            return self._analyze_question(input_data)
        elif operation == "create_workflow":
            return self._create_workflow(input_data)
        elif operation == "execute_workflow":
            return self._execute_workflow(input_data)
        elif operation == "update_workflow":
            return self._update_workflow(input_data)
        elif operation == "get_workflow_status":
            return self._get_workflow_status(input_data)
        elif operation == "get_available_techniques":
            return self._get_available_techniques_info(input_data)
        elif operation == "execute_analysis":
            return self._execute_analysis(input_data)
        else:
            error_msg = f"Unknown operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _analyze_question(self, input_data: Dict) -> Dict:
        """
        Analyze a question to determine its characteristics and optimal workflow.
        
        Args:
            input_data: Input data dictionary containing the question
            
        Returns:
            Question analysis results
        """
        logger.info("Analyzing question")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context
        context = input_data.get("context")
        if not isinstance(context, AnalysisContext):
            # Create new context if not provided
            context = AnalysisContext()
            context.add("question", question)
        
        # Run preliminary research if requested
        run_preliminary_research = input_data.get("run_preliminary_research", True)
        if run_preliminary_research:
            self._run_preliminary_research(context)
        
        # Analyze question characteristics
        self._analyze_question_characteristics(context)
        
        # Generate analysis strategy
        strategy = self._generate_strategy_based_on_characteristics(context)
        
        # Store strategy in context
        context.add("strategy", strategy)
        
        # Get question analysis from context
        question_analysis = context.get("question_analysis", {})
        
        # Compile results
        results = {
            "question": question,
            "question_analysis": question_analysis,
            "strategy": strategy.to_dict() if strategy else {},
            "preliminary_research": context.get("preliminary_research", {})
        }
        
        return results
    
    def _run_preliminary_research(self, context):
        """Run preliminary research using Perplexity Sonar."""
        logger.info("Running preliminary research")
        
        # Get question from context
        question = context.get("question", "")
        if not question:
            logger.warning("No question found in context, skipping preliminary research")
            return False
        
        # Get Perplexity Sonar MCP
        perplexity_sonar = self.mcp_registry.get_mcp("perplexity_sonar")
        if not perplexity_sonar:
            logger.warning("PerplexitySonarMCP not available, skipping preliminary research")
            return False
        
        # Prepare input for research
        input_data = {
            "operation": "research",
            "question": question,
            "context": context
        }
        
        # Execute research using PerplexitySonarMCP
        try:
            research_result = perplexity_sonar.process(input_data)
            
            # Validate research result
            if "error" in research_result:
                logger.error(f"Error running preliminary research: {research_result['error']}")
                return False
            
            # Store research results in context
            if "research_data" in research_result:
                context.add("research_data", research_result["research_data"])
            
            if "key_insights" in research_result:
                context.add("key_insights", research_result["key_insights"])
            
            if "initial_hypotheses" in research_result:
                context.add("initial_hypotheses", research_result["initial_hypotheses"])
            
            if "recommended_workflow" in research_result:
                context.add("recommended_workflow", research_result["recommended_workflow"])
            
            # Store complete research result
            context.add("preliminary_research", research_result)
            
            logger.info("Preliminary research completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running preliminary research: {str(e)}")
            return False
    
    def _analyze_question_characteristics(self, context):
        """Analyze question characteristics using LLM."""
        logger.info("Analyzing question characteristics")
        
        # Get question from context
        question = context.get("question", "")
        if not question:
            logger.warning("No question found in context, skipping question analysis")
            return False
        
        # Get Llama4ScoutMCP
        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.warning("Llama4ScoutMCP not available, using default question analysis")
            return self._default_question_analysis(context)
        
        # Prepare input for question analysis
        input_data = {
            "operation": "analyze_question",
            "question": question,
            "research_data": context.get("research_data", {}),
            "key_insights": context.get("key_insights", [])
        }
        
        # Analyze question using Llama4ScoutMCP
        try:
            analysis_result = llama4_scout.process(input_data)
            
            # Validate analysis result
            if "error" in analysis_result:
                logger.error(f"Error analyzing question: {analysis_result['error']}")
                return self._default_question_analysis(context)
            
            # Store analysis in context
            context.add("question_analysis", analysis_result)
            
            logger.info("Question analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            return self._default_question_analysis(context)
    
    def _default_question_analysis(self, context):
        """Generate default question analysis when LLM analysis fails."""
        logger.info("Generating default question analysis")
        
        # Get question from context
        question = context.get("question", "")
        
        # Determine question type using simple heuristics
        question_type = self._determine_question_type(question)
        
        # Determine domain using simple heuristics
        domain = self._determine_domain(question)
        
        # Determine complexity using simple heuristics
        complexity = self._determine_complexity(question)
        
        # Create default analysis
        analysis = {
            "question_type": question_type,
            "domains": [domain],
            "complexity": complexity,
            "uncertainty": "medium",
            "time_horizon": "medium",
            "potential_biases": ["recency", "availability"],
            "key_entities": [],
            "key_concepts": []
        }
        
        # Store analysis in context
        context.add("question_analysis", analysis)
        
        logger.info("Default question analysis generated")
        return True
    
    def _generate_strategy_based_on_characteristics(self, context):
        """Generate an analysis strategy based on question characteristics."""
        logger.info("Generating analysis strategy based on question characteristics")
        
        # Get question analysis from context
        question_analysis = context.get("question_analysis", {})
        if not question_analysis:
            logger.warning("No question analysis found in context, using default strategy")
            return self._generate_default_strategy(context)
        
        # Get Llama4ScoutMCP for strategy generation
        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.warning("Llama4ScoutMCP not available, using default strategy")
            return self._generate_default_strategy(context)
        
        # Prepare input for strategy generation
        input_data = {
            "operation": "generate_strategy",
            "question": context.get("question", ""),
            "question_analysis": question_analysis,
            "available_techniques": self._get_available_techniques(),
            "research_data": context.get("research_data", {}),
            "key_insights": context.get("key_insights", []),
            "initial_hypotheses": context.get("initial_hypotheses", []),
            "recommended_workflow": context.get("recommended_workflow", {})
        }
        
        # Generate strategy using Llama4ScoutMCP
        try:
            strategy_result = llama4_scout.process(input_data)
            
            # Validate strategy result
            if "error" in strategy_result:
                logger.error(f"Error generating strategy: {strategy_result['error']}")
                return self._generate_default_strategy(context)
            
            # Create AnalysisStrategy from result
            strategy = AnalysisStrategy()
            
            # Add steps from strategy result
            for step_data in strategy_result.get("steps", []):
                technique_name = step_data.get("technique")
                parameters = step_data.get("parameters", {})
                dependencies = step_data.get("dependencies", [])
                optional = step_data.get("optional", False)
                
                # Validate technique exists
                if technique_name not in self.technique_registry:
                    logger.warning(f"Technique {technique_name} not found in registry, skipping")
                    continue
                
                # Add step to strategy
                strategy.add_step(technique_name, parameters, dependencies, optional)
            
            # Add metadata to strategy
            strategy.add_metadata("generation_method", "llm")
            strategy.add_metadata("question_type", question_analysis.get("question_type"))
            strategy.add_metadata("domains", question_analysis.get("domains", []))
            strategy.add_metadata("complexity", question_analysis.get("complexity"))
            strategy.add_metadata("timestamp", time.time())
            
            logger.info(f"Generated strategy with {len(strategy.steps)} steps")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return self._generate_default_strategy(context)
    
    def _generate_default_strategy(self, context):
        """Generate a default analysis strategy when dynamic generation fails."""
        logger.info("Generating default analysis strategy")
        
        # Create default strategy
        strategy = AnalysisStrategy()
        
        # Get question type from context or determine it
        question_analysis = context.get("question_analysis", {})
        question_type = question_analysis.get("question_type")
        
        if not question_type:
            question = context.get("question", "")
            question_type = self._determine_question_type(question)
        
        # Add steps based on question type
        if question_type == "predictive":
            # Predictive question strategy
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("scenario_triangulation", {"num_scenarios": 3}, [0])
            strategy.add_step("key_assumptions_check", {}, [1])
            strategy.add_step("uncertainty_mapping", {}, [2])
            strategy.add_step("synthesis_generation", {}, [3])
            
        elif question_type == "causal":
            # Causal question strategy
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("causal_network_analysis", {}, [0])
            strategy.add_step("backward_reasoning", {}, [1])
            strategy.add_step("key_assumptions_check", {}, [2])
            strategy.add_step("synthesis_generation", {}, [3])
            
        elif question_type == "evaluative":
            # Evaluative question strategy
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("multi_persona", {"num_personas": 3}, [0])
            strategy.add_step("consensus_challenge", {}, [1])
            strategy.add_step("red_teaming", {}, [2])
            strategy.add_step("synthesis_generation", {}, [3])
            
        elif question_type == "decision":
            # Decision question strategy
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("decision_tree_analysis", {}, [0])
            strategy.add_step("premortem_analysis", {}, [1])
            strategy.add_step("key_assumptions_check", {}, [2])
            strategy.add_step("synthesis_generation", {}, [3])
            
        else:
            # Default/descriptive question strategy
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("analysis_of_competing_hypotheses", {}, [0])
            strategy.add_step("uncertainty_mapping", {}, [1])
            strategy.add_step("synthesis_generation", {}, [2])
        
        # Add metadata to strategy
        strategy.add_metadata("generation_method", "default")
        strategy.add_metadata("question_type", question_type)
        strategy.add_metadata("timestamp", time.time())
        
        logger.info(f"Generated default strategy with {len(strategy.steps)} steps")
        return strategy
    
    def _execute_analysis(self, input_data: Dict) -> Dict:
        """
        Execute a complete analysis for a question.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Analysis results
        """
        logger.info("Executing analysis")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context
        context = input_data.get("context")
        if not isinstance(context, AnalysisContext):
            # Create new context if not provided
            context = AnalysisContext()
            context.add("question", question)
        
        # Analyze question to get strategy
        analysis_result = self._analyze_question({
            "question": question,
            "context": context,
            "run_preliminary_research": True
        })
        
        # Check for errors
        if "error" in analysis_result:
            logger.error(f"Error analyzing question: {analysis_result['error']}")
            return analysis_result
        
        # Get strategy from context
        strategy = context.get("strategy")
        if not strategy:
            error_msg = "No strategy generated for analysis"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Execute strategy
        execution_result = self._execute_strategy(context, strategy)
        
        # Compile results
        results = {
            "question": question,
            "question_analysis": context.get("question_analysis", {}),
            "strategy": strategy.to_dict(),
            "execution_result": execution_result,
            "final_synthesis": context.get("final_synthesis", {}),
            "completed": execution_result.get("completed", False)
        }
        
        return results
    
    def _execute_strategy(self, context, strategy):
        """
        Execute an analysis strategy.
        
        Args:
            context: Analysis context
            strategy: Analysis strategy
            
        Returns:
            Execution results
        """
        logger.info(f"Executing strategy with {len(strategy.steps)} steps")
        
        # Initialize execution state
        completed_steps = []
        step_results = {}
        execution_metadata = {
            "start_time": time.time(),
            "step_times": {},
            "adaptations": []
        }
        
        # Execute steps
        current_step_index = 0
        while current_step_index < len(strategy.steps):
            # Get current step
            step = strategy.steps[current_step_index]
            
            # Check if step can be executed (dependencies satisfied)
            dependencies_satisfied = all(dep in completed_steps for dep in step.dependencies)
            if not dependencies_satisfied:
                logger.warning(f"Dependencies not satisfied for step {current_step_index}, skipping")
                current_step_index += 1
                continue
            
            # Execute step
            logger.info(f"Executing step {current_step_index}: {step.technique}")
            step_start_time = time.time()
            
            try:
                # Get technique
                technique = self.technique_registry.get(step.technique)
                if not technique:
                    raise ValueError(f"Technique {step.technique} not found in registry")
                
                # Execute technique
                step_result = technique.execute(context, step.parameters)
                
                # Store result
                step_results[step.technique] = step_result
                context.add(f"result_{step.technique}", step_result)
                
                # Update execution metadata
                step_time = time.time() - step_start_time
                execution_metadata["step_times"][step.technique] = step_time
                
                logger.info(f"Step {current_step_index} completed in {step_time:.2f} seconds")
                
                # Add to completed steps
                completed_steps.append(current_step_index)
                
                # Check if adaptation is needed
                adaptation_needed = self._check_adaptation_criteria(context, step_result)
                if adaptation_needed:
                    logger.info(f"Adaptation needed after step {current_step_index}")
                    
                    # Adapt strategy
                    adapted = self._adapt_strategy(context, current_step_index)
                    if adapted:
                        # Get updated strategy
                        strategy = context.get("strategy")
                        
                        # Record adaptation
                        adaptation_info = {
                            "step_index": current_step_index,
                            "step_technique": step.technique,
                            "time": time.time(),
                            "new_strategy_length": len(strategy.steps)
                        }
                        execution_metadata["adaptations"].append(adaptation_info)
                        
                        logger.info(f"Strategy adapted, now has {len(strategy.steps)} steps")
                
            except Exception as e:
                logger.error(f"Error executing step {current_step_index}: {str(e)}")
                
                # Handle execution error
                recovery_action = self._handle_execution_error(context, step, e)
                
                if recovery_action == "retry":
                    # Retry the step
                    logger.info(f"Retrying step {current_step_index}")
                    continue
                    
                elif recovery_action == "continue":
                    # Continue with next step
                    current_step_index += 1
                    continue
                    
                elif recovery_action == "halt":
                    # Halt execution
                    logger.info("Halting strategy execution due to critical error")
                    break
            
            # Move to next step
            current_step_index += 1
        
        # Generate final synthesis
        final_synthesis = self._generate_final_synthesis(context, step_results)
        context.add("final_synthesis", final_synthesis)
        
        # Update execution metadata
        execution_metadata["end_time"] = time.time()
        execution_metadata["total_time"] = execution_metadata["end_time"] - execution_metadata["start_time"]
        execution_metadata["completed_steps"] = len(completed_steps)
        execution_metadata["total_steps"] = len(strategy.steps)
        
        # Determine if execution is complete
        is_complete = strategy.is_complete(completed_steps)
        
        # Compile execution results
        execution_result = {
            "completed": is_complete,
            "completed_steps": completed_steps,
            "step_results": step_results,
            "metadata": execution_metadata,
            "final_synthesis": final_synthesis
        }
        
        logger.info(f"Strategy execution completed: {is_complete}")
        return execution_result
    
    def _check_adaptation_criteria(self, context, current_step_result):
        """Check if the strategy needs to be adapted based on current step result."""
        logger.info("Checking adaptation criteria")
        
        # Initialize adaptation triggers
        adaptation_triggers = {
            "high_uncertainty": False,
            "conflicting_hypotheses": False,
            "low_confidence": False,
            "new_evidence": False,
            "bias_detected": False,
            "additional_analysis_recommended": False
        }
        
        # Check for high uncertainty
        if "uncertainty_assessment" in current_step_result:
            uncertainty = current_step_result["uncertainty_assessment"].get("overall_uncertainty", 0)
            if uncertainty > 0.7:  # High uncertainty threshold
                adaptation_triggers["high_uncertainty"] = True
                logger.info("High uncertainty detected, adaptation may be needed")
        
        # Check for conflicting hypotheses
        if "hypotheses" in current_step_result:
            hypotheses = current_step_result.get("hypotheses", [])
            high_confidence_hypotheses = [h for h in hypotheses if h.get("confidence", 0) > 0.7]
            if len(high_confidence_hypotheses) >= 2:
                # Check if hypotheses are conflicting (simplified check)
                for i, h1 in enumerate(high_confidence_hypotheses[:-1]):
                    for h2 in high_confidence_hypotheses[i+1:]:
                        if h1.get("contradicts", []) and h2.get("id") in h1["contradicts"]:
                            adaptation_triggers["conflicting_hypotheses"] = True
                            logger.info("Conflicting hypotheses detected, adaptation may be needed")
                            break
        
        # Check for low confidence
        if "confidence_assessment" in current_step_result:
            confidence = current_step_result["confidence_assessment"].get("overall_confidence", 0)
            if confidence < 0.3:  # Low confidence threshold
                adaptation_triggers["low_confidence"] = True
                logger.info("Low confidence detected, adaptation may be needed")
        
        # Check for new evidence
        if "new_evidence" in current_step_result and current_step_result["new_evidence"]:
            adaptation_triggers["new_evidence"] = True
            logger.info("New evidence detected, adaptation may be needed")
        
        # Check for bias detection
        if "biases" in current_step_result:
            biases = current_step_result.get("biases", [])
            if any(b.get("severity", 0) > 0.7 for b in biases):  # High bias severity threshold
                adaptation_triggers["bias_detected"] = True
                logger.info("Significant bias detected, adaptation may be needed")
        
        # Check for explicit recommendations
        if "recommendations" in current_step_result:
            recommendations = current_step_result.get("recommendations", [])
            for rec in recommendations:
                if "additional analysis" in rec.get("recommendation", "").lower():
                    adaptation_triggers["additional_analysis_recommended"] = True
                    logger.info("Additional analysis recommended, adaptation may be needed")
                    break
        
        # Determine if adaptation is needed
        adaptation_needed = any(adaptation_triggers.values())
        
        # Store adaptation triggers in context for use in adaptation
        context.add("adaptation_triggers", adaptation_triggers)
        
        return adaptation_needed
    
    def _adapt_strategy(self, context, current_step_index):
        """Adapt the strategy based on interim results."""
        logger.info(f"Adapting strategy after step {current_step_index}")
        
        # Get current strategy
        strategy = context.get("strategy")
        if not strategy:
            logger.warning("No strategy found in context, cannot adapt")
            return False
        
        # Get adaptation triggers
        adaptation_triggers = context.get("adaptation_triggers", {})
        if not adaptation_triggers:
            logger.warning("No adaptation triggers found in context, using default adaptation")
            return self._default_adaptation(context, current_step_index)
        
        # Get Llama4ScoutMCP for adaptation
        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.warning("Llama4ScoutMCP not available, using default adaptation")
            return self._default_adaptation(context, current_step_index)
        
        # Get completed steps and their results
        completed_steps = strategy.steps[:current_step_index + 1]
        completed_results = {}
        for i, step in enumerate(completed_steps):
            result_key = f"result_{step.technique}"
            if context.has(result_key):
                completed_results[step.technique] = context.get(result_key)
        
        # Get remaining steps
        remaining_steps = strategy.steps[current_step_index + 1:]
        
        # Prepare input for adaptation
        input_data = {
            "operation": "adapt_strategy",
            "question": context.get("question", ""),
            "question_analysis": context.get("question_analysis", {}),
            "adaptation_triggers": adaptation_triggers,
            "completed_steps": [{"technique": step.technique, "parameters": step.parameters} for step in completed_steps],
            "completed_results": completed_results,
            "remaining_steps": [{"technique": step.technique, "parameters": step.parameters} for step in remaining_steps],
            "available_techniques": self._get_available_techniques(),
            "research_data": context.get("research_data", {})
        }
        
        # Generate adaptation using Llama4ScoutMCP
        try:
            adaptation_result = llama4_scout.process(input_data)
            
            # Validate adaptation result
            if "error" in adaptation_result:
                logger.error(f"Error generating adaptation: {adaptation_result['error']}")
                return self._default_adaptation(context, current_step_index)
            
            # Create new strategy with completed steps
            new_strategy = AnalysisStrategy()
            for step in completed_steps:
                new_strategy.add_step(step.technique, step.parameters, step.dependencies, step.optional)
            
            # Add adapted steps from adaptation result
            for step_data in adaptation_result.get("adapted_steps", []):
                technique_name = step_data.get("technique")
                parameters = step_data.get("parameters", {})
                dependencies = step_data.get("dependencies", [])
                optional = step_data.get("optional", False)
                
                # Validate technique exists
                if technique_name not in self.technique_registry:
                    logger.warning(f"Technique {technique_name} not found in registry, skipping")
                    continue
                
                # Add step to strategy
                new_strategy.add_step(technique_name, parameters, dependencies, optional)
            
            # Add adaptation metadata
            new_strategy.add_metadata("adapted", True)
            new_strategy.add_metadata("adaptation_triggers", adaptation_triggers)
            new_strategy.add_metadata("adaptation_time", time.time())
            new_strategy.add_metadata("original_strategy_length", len(strategy.steps))
            
            # Copy original metadata
            for key, value in strategy.metadata.items():
                if key not in new_strategy.metadata:
                    new_strategy.add_metadata(key, value)
            
            # Update strategy in context
            context.add("strategy", new_strategy)
            context.add("strategy_adapted", True)
            
            logger.info(f"Strategy adapted with {len(new_strategy.steps) - len(completed_steps)} new steps")
            return True
            
        except Exception as e:
            logger.error(f"Error adapting strategy: {str(e)}")
            return self._default_adaptation(context, current_step_index)
    
    def _default_adaptation(self, context, current_step_index):
        """Default strategy adaptation when dynamic adaptation fails."""
        logger.info(f"Performing default adaptation after step {current_step_index}")
        
        # Get current strategy
        strategy = context.get("strategy")
        if not strategy:
            logger.warning("No strategy found in context, cannot adapt")
            return False
        
        # Get adaptation triggers
        adaptation_triggers = context.get("adaptation_triggers", {})
        
        # Create new strategy with completed steps
        new_strategy = AnalysisStrategy()
        for i, step in enumerate(strategy.steps[:current_step_index + 1]):
            new_strategy.add_step(step.technique, step.parameters, step.dependencies, step.optional)
        
        # Add additional steps based on triggers
        if adaptation_triggers.get("high_uncertainty", False):
            # Add uncertainty mapping if not already in strategy
            if not any(s.technique == "uncertainty_mapping" for s in strategy.steps):
                new_strategy.add_step("uncertainty_mapping", {}, [current_step_index])
                logger.info("Added uncertainty_mapping technique due to high uncertainty")
        
        if adaptation_triggers.get("conflicting_hypotheses", False):
            # Add ACH if not already in strategy
            if not any(s.technique == "analysis_of_competing_hypotheses" for s in strategy.steps):
                new_strategy.add_step("analysis_of_competing_hypotheses", {}, [current_step_index])
                logger.info("Added analysis_of_competing_hypotheses technique due to conflicting hypotheses")
        
        if adaptation_triggers.get("bias_detected", False):
            # Add red teaming if not already in strategy
            if not any(s.technique == "red_teaming" for s in strategy.steps):
                new_strategy.add_step("red_teaming", {}, [current_step_index])
                logger.info("Added red_teaming technique due to bias detection")
        
        # Add remaining steps from original strategy
        for step in strategy.steps[current_step_index + 1:]:
            # Update dependencies to account for new steps
            updated_dependencies = [dep if dep <= current_step_index else dep + len(new_strategy.steps) - (current_step_index + 1) for dep in step.dependencies]
            new_strategy.add_step(step.technique, step.parameters, updated_dependencies, step.optional)
        
        # Add adaptation metadata
        new_strategy.add_metadata("adapted", True)
        new_strategy.add_metadata("adaptation_method", "default")
        new_strategy.add_metadata("adaptation_triggers", adaptation_triggers)
        new_strategy.add_metadata("adaptation_time", time.time())
        new_strategy.add_metadata("original_strategy_length", len(strategy.steps))
        
        # Copy original metadata
        for key, value in strategy.metadata.items():
            if key not in new_strategy.metadata:
                new_strategy.add_metadata(key, value)
        
        # Update strategy in context
        context.add("strategy", new_strategy)
        context.add("strategy_adapted", True)
        
        logger.info(f"Strategy adapted with {len(new_strategy.steps) - len(strategy.steps)} new steps")
        return True
    
    def _handle_execution_error(self, context, step, error):
        """Handle execution error for a step."""
        logger.error(f"Error executing step {step.technique}: {str(error)}")
        
        # Store error in context
        error_data = {
            "step": step.technique,
            "error": str(error),
            "time": time.time()
        }
        
        errors = context.get("errors", [])
        errors.append(error_data)
        context.add("errors", errors)
        
        # Determine recovery strategy based on technique
        recovery_strategy = self._determine_recovery_strategy(step.technique)
        
        # Execute recovery strategy
        if recovery_strategy == "skip":
            logger.info(f"Skipping failed step {step.technique} and continuing workflow")
            return "continue"
        
        elif recovery_strategy == "retry":
            logger.info(f"Retrying failed step {step.technique}")
            return "retry"
        
        elif recovery_strategy == "fallback":
            logger.info(f"Using fallback for failed step {step.technique}")
            fallback_result = self._execute_fallback(context, step)
            
            # Store fallback result in context
            context.add(f"result_{step.technique}", fallback_result)
            
            return "continue"
        
        else:  # halt
            logger.info(f"Halting workflow due to critical error in step {step.technique}")
            return "halt"
    
    def _determine_recovery_strategy(self, technique_name):
        """Determine recovery strategy based on technique."""
        # Critical techniques that should halt the workflow on failure
        critical_techniques = [
            "research_to_hypothesis",
            "synthesis_generation"
        ]
        
        # Techniques that should be retried on failure
        retry_techniques = [
            "causal_network_analysis",
            "key_assumptions_check"
        ]
        
        # Techniques that should use fallback on failure
        fallback_techniques = [
            "uncertainty_mapping",
            "red_teaming",
            "bias_detection"
        ]
        
        if technique_name in critical_techniques:
            return "halt"
        elif technique_name in retry_techniques:
            return "retry"
        elif technique_name in fallback_techniques:
            return "fallback"
        else:
            return "skip"
    
    def _execute_fallback(self, context, step):
        """Execute fallback for a failed step."""
        technique_name = step.technique
        
        # Get technique instance
        technique = self.technique_registry.get(technique_name)
        if not technique:
            logger.warning(f"Technique {technique_name} not found in registry, cannot execute fallback")
            return {"error": f"Technique {technique_name} not found"}
        
        # Check if technique has fallback method
        if hasattr(technique, "fallback") and callable(technique.fallback):
            try:
                fallback_result = technique.fallback(context, step.parameters)
                fallback_result["fallback"] = True
                return fallback_result
            except Exception as e:
                logger.error(f"Error executing fallback for {technique_name}: {str(e)}")
                return {"error": str(e), "fallback_failed": True}
        
        # Default fallback
        return {
            "fallback": True,
            "message": f"Fallback result for {technique_name}",
            "findings": [],
            "confidence_assessment": {"overall_confidence": "low"}
        }
    
    def _generate_final_synthesis(self, context, step_results):
        """Generate final synthesis of analysis results."""
        logger.info("Generating final synthesis")
        
        # Get Llama4ScoutMCP for synthesis
        llama4_scout = self.mcp_registry.get_mcp("llama4_scout")
        if not llama4_scout:
            logger.warning("Llama4ScoutMCP not available, using default synthesis")
            return self._generate_default_synthesis(context, step_results)
        
        # Check if synthesis_generation technique was executed
        if "synthesis_generation" in step_results:
            synthesis_result = step_results["synthesis_generation"]
            
            # If synthesis result is valid, use it as the basis for final synthesis
            if "error" not in synthesis_result and "synthesis" in synthesis_result:
                logger.info("Using synthesis_generation result as basis for final synthesis")
                
                # Prepare input for final synthesis
                input_data = {
                    "operation": "final_synthesis",
                    "question": context.get("question", ""),
                    "question_analysis": context.get("question_analysis", {}),
                    "synthesis_result": synthesis_result,
                    "step_results": step_results,
                    "research_data": context.get("research_data", {})
                }
                
                # Generate final synthesis using Llama4ScoutMCP
                try:
                    final_synthesis = llama4_scout.process(input_data)
                    
                    # Validate final synthesis
                    if "error" in final_synthesis:
                        logger.error(f"Error generating final synthesis: {final_synthesis['error']}")
                        return self._generate_default_synthesis(context, step_results)
                    
                    logger.info("Final synthesis generated successfully")
                    return final_synthesis
                    
                except Exception as e:
                    logger.error(f"Error generating final synthesis: {str(e)}")
                    return self._generate_default_synthesis(context, step_results)
        
        # If synthesis_generation was not executed or failed, generate synthesis from all results
        logger.info("Generating synthesis from all results")
        
        # Prepare input for synthesis
        input_data = {
            "operation": "synthesize_results",
            "question": context.get("question", ""),
            "question_analysis": context.get("question_analysis", {}),
            "step_results": step_results,
            "research_data": context.get("research_data", {})
        }
        
        # Generate synthesis using Llama4ScoutMCP
        try:
            synthesis = llama4_scout.process(input_data)
            
            # Validate synthesis
            if "error" in synthesis:
                logger.error(f"Error generating synthesis: {synthesis['error']}")
                return self._generate_default_synthesis(context, step_results)
            
            logger.info("Synthesis generated successfully")
            return synthesis
            
        except Exception as e:
            logger.error(f"Error generating synthesis: {str(e)}")
            return self._generate_default_synthesis(context, step_results)
    
    def _generate_default_synthesis(self, context, step_results):
        """Generate default synthesis when dynamic synthesis fails."""
        logger.info("Generating default synthesis")
        
        # Collect findings from all steps
        all_findings = []
        for technique, result in step_results.items():
            if "findings" in result:
                for finding in result["findings"]:
                    all_findings.append({
                        "technique": technique,
                        "finding": finding.get("finding", ""),
                        "confidence": finding.get("confidence", 0.5),
                        "evidence": finding.get("evidence", [])
                    })
        
        # Collect hypotheses from all steps
        all_hypotheses = []
        for technique, result in step_results.items():
            if "hypotheses" in result:
                for hypothesis in result["hypotheses"]:
                    all_hypotheses.append({
                        "technique": technique,
                        "hypothesis": hypothesis.get("hypothesis", ""),
                        "confidence": hypothesis.get("confidence", 0.5),
                        "evidence": hypothesis.get("evidence", [])
                    })
        
        # Create default synthesis
        synthesis = {
            "question": context.get("question", ""),
            "summary": "Default synthesis generated due to synthesis generation failure.",
            "findings": all_findings,
            "hypotheses": all_hypotheses,
            "confidence_assessment": {
                "overall_confidence": "medium",
                "explanation": "Default confidence assessment."
            },
            "uncertainties": [
                {
                    "uncertainty": "Default uncertainty due to synthesis generation failure.",
                    "impact": "medium"
                }
            ],
            "default_synthesis": True
        }
        
        logger.info("Default synthesis generated")
        return synthesis
    
    def _create_workflow(self, input_data: Dict) -> Dict:
        """
        Create a workflow based on input specifications.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Workflow creation results
        """
        logger.info("Creating workflow")
        
        # Get techniques
        techniques = input_data.get("techniques", [])
        if not techniques:
            error_msg = "No techniques provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context
        context = input_data.get("context")
        if not isinstance(context, AnalysisContext):
            # Create new context if not provided
            context = AnalysisContext()
            
            # Add question if provided
            question = input_data.get("question", "")
            if question:
                context.add("question", question)
        
        # Create workflow
        workflow = self.integrator.create_technique_workflow(techniques, context)
        
        # Store workflow if valid
        if "error" not in workflow:
            workflow_id = str(int(time.time()))
            self.active_workflows[workflow_id] = workflow
            workflow["id"] = workflow_id
        
        return {"workflow": workflow}
    
    def _execute_workflow(self, input_data: Dict) -> Dict:
        """
        Execute a workflow.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Workflow execution results
        """
        logger.info("Executing workflow")
        
        # Get workflow ID
        workflow_id = input_data.get("workflow_id", "")
        
        # Get workflow
        workflow = None
        if workflow_id:
            workflow = self.active_workflows.get(workflow_id)
        
        if not workflow:
            # Check if workflow is provided directly
            workflow = input_data.get("workflow")
        
        if not workflow:
            error_msg = "No valid workflow provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context
        context = input_data.get("context")
        if not isinstance(context, AnalysisContext):
            error_msg = "No valid context provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get parameters
        parameters = input_data.get("parameters", {})
        
        # Execute workflow
        results = self.integrator.execute_technique_workflow(workflow, context, parameters)
        
        # Update workflow with results
        if workflow_id and workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["results"] = results
            self.active_workflows[workflow_id]["status"] = "completed" if results.get("completed", False) else "partial"
            self.active_workflows[workflow_id]["last_updated"] = time.time()
        
        return {"results": results}
    
    def _update_workflow(self, input_data: Dict) -> Dict:
        """
        Update a workflow based on interim results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Workflow update results
        """
        logger.info("Updating workflow")
        
        # Get workflow ID
        workflow_id = input_data.get("workflow_id", "")
        
        # Get workflow
        workflow = None
        if workflow_id:
            workflow = self.active_workflows.get(workflow_id)
        
        if not workflow:
            # Check if workflow is provided directly
            workflow = input_data.get("workflow")
        
        if not workflow:
            error_msg = "No valid workflow provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context
        context = input_data.get("context")
        if not isinstance(context, AnalysisContext):
            error_msg = "No valid context provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get interim results
        interim_results = input_data.get("interim_results", {})
        if not interim_results:
            # Check if results are in the workflow
            interim_results = workflow.get("results", {})
        
        # Update workflow
        updated_workflow = self.integrator.update_adaptive_workflow(workflow, context, interim_results)
        
        # Store updated workflow
        if workflow_id and workflow_id in self.active_workflows:
            self.active_workflows[workflow_id] = updated_workflow
            updated_workflow["id"] = workflow_id
            updated_workflow["last_updated"] = time.time()
        
        return {"updated_workflow": updated_workflow}
    
    def _get_workflow_status(self, input_data: Dict) -> Dict:
        """
        Get the status of a workflow.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Workflow status
        """
        logger.info("Getting workflow status")
        
        # Get workflow ID
        workflow_id = input_data.get("workflow_id", "")
        if not workflow_id:
            error_msg = "No workflow ID provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get workflow
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            error_msg = f"Workflow not found: {workflow_id}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get status
        status = workflow.get("status", "pending")
        
        # Get progress
        progress = 0
        if "results" in workflow:
            executed_techniques = list(workflow["results"].get("results", {}).keys())
            total_techniques = len(workflow.get("techniques", []))
            if total_techniques > 0:
                progress = len(executed_techniques) / total_techniques
        
        return {
            "workflow_id": workflow_id,
            "status": status,
            "progress": progress,
            "last_updated": workflow.get("last_updated", 0)
        }
    
    def _get_available_techniques_info(self, input_data: Dict) -> Dict:
        """
        Get available analytical techniques.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Available techniques
        """
        logger.info("Getting available techniques")
        
        # Get all techniques
        all_techniques = self.technique_registry
        
        # Get technique names
        technique_names = list(all_techniques.keys())
        
        # Get technique categories
        categories = self.integrator.get_technique_categories()
        
        return {
            "techniques": technique_names,
            "categories": categories,
            "count": len(technique_names)
        }
    
    def _determine_question_type(self, question: str) -> str:
        """
        Determine the type of question.
        
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
        
        # Check for decision questions
        decision_terms = ["decide", "choice", "option", "alternative", "select", "choose", "decision"]
        if any(term in question_lower for term in decision_terms):
            return "decision"
        
        # Default to descriptive
        return "descriptive"
    
    def _determine_domain(self, question: str) -> str:
        """
        Determine the domain of the question.
        
        Args:
            question: The analytical question
            
        Returns:
            Domain name
        """
        question_lower = question.lower()
        
        domains = {
            "economic": ["economic", "economy", "market", "financial", "growth", "recession", "inflation", "investment"],
            "political": ["political", "government", "policy", "regulation", "election", "democratic", "republican"],
            "technological": ["technology", "innovation", "digital", "ai", "automation", "tech", "software", "hardware"],
            "social": ["social", "cultural", "demographic", "education", "healthcare", "society", "community"],
            "environmental": ["environmental", "climate", "ecological", "sustainability", "resource", "energy"],
            "security": ["security", "defense", "military", "conflict", "threat", "risk", "war", "terrorism"]
        }
        
        # Count domain keywords
        domain_counts = {domain: sum(1 for term in terms if term in question_lower) for domain, terms in domains.items()}
        
        # Get domain with highest count
        if any(domain_counts.values()):
            primary_domain = max(domain_counts.items(), key=lambda x: x[1])[0]
            return primary_domain
        
        # Default domain
        return "general"
    
    def _determine_complexity(self, question: str) -> str:
        """
        Determine the complexity of the question.
        
        Args:
            question: The analytical question
            
        Returns:
            Complexity level
        """
        # Simple heuristics for complexity
        word_count = len(question.split())
        clause_count = len(re.findall(r'[,;]', question)) + 1
        
        # Check for complex terms
        complex_terms = ["interdependencies", "systemic", "multifaceted", "interrelated", "complex", "dynamics", "feedback", "non-linear"]
        complex_term_count = sum(1 for term in complex_terms if term in question.lower())
        
        # Calculate complexity score
        complexity_score = word_count / 10 + clause_count + complex_term_count * 2
        
        # Determine complexity level
        if complexity_score > 10:
            return "high"
        elif complexity_score > 5:
            return "medium"
        else:
            return "low"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this MCP.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "operations": [
                "analyze_question",
                "create_workflow",
                "execute_workflow",
                "update_workflow",
                "get_workflow_status",
                "get_available_techniques",
                "execute_analysis"
            ],
            "technique_count": len(self.technique_registry)
        }
