# WorkflowOrchestratorMCP Enhancement Design

This document outlines the design for enhancing the WorkflowOrchestratorMCP to implement dynamic strategy generation and workflow adaptation capabilities.

## 1. Dynamic Strategy Generation

### Current Limitations
The current implementation uses hardcoded logic in `_generate_strategy_based_on_characteristics` instead of dynamic, LLM-driven strategy generation. This limits the system's ability to adapt to different question types and contexts.

### Enhancement Design
We will replace the hardcoded logic with a dynamic, LLM-driven approach that:
1. Takes `context.question_analysis` as input
2. Uses Llama4ScoutMCP to generate an optimal sequence of analytical techniques
3. Determines parameters for each technique based on context
4. Identifies dependencies between techniques
5. Populates an `AnalysisStrategy` object with these dynamic steps and parameters

### Implementation Details
```python
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
        "context_metadata": context.get_metadata()
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
            
            # Validate technique exists
            if technique_name not in self.technique_registry:
                logger.warning(f"Technique {technique_name} not found in registry, skipping")
                continue
            
            # Add step to strategy
            strategy.add_step(technique_name, parameters)
        
        # Add metadata to strategy
        strategy.add_metadata("generation_method", "llm")
        strategy.add_metadata("question_type", question_analysis.get("question_type"))
        strategy.add_metadata("domains", question_analysis.get("domains", []))
        strategy.add_metadata("complexity", question_analysis.get("complexity"))
        
        logger.info(f"Generated strategy with {len(strategy.steps)} steps")
        return strategy
        
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}")
        return self._generate_default_strategy(context)
```

## 2. Workflow Adaptation

### Current Limitations
The current implementation lacks logic for adapting workflows based on interim results. This prevents the system from responding to new information or changing analysis needs during execution.

### Enhancement Design
We will implement the logic within `_adapt_strategy` and `_check_adaptation_criteria` to:
1. Analyze results from completed steps to identify adaptation triggers
2. Modify the remaining steps in the strategy based on these triggers
3. Add, remove, or reorder techniques as needed
4. Update parameters for remaining techniques

### Implementation Details
```python
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
    completed_results = context.get("results", {})
    
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
        "context_metadata": context.get_metadata()
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
            new_strategy.add_step(step.technique, step.parameters)
        
        # Add adapted steps from adaptation result
        for step_data in adaptation_result.get("adapted_steps", []):
            technique_name = step_data.get("technique")
            parameters = step_data.get("parameters", {})
            
            # Validate technique exists
            if technique_name not in self.technique_registry:
                logger.warning(f"Technique {technique_name} not found in registry, skipping")
                continue
            
            # Add step to strategy
            new_strategy.add_step(technique_name, parameters)
        
        # Add adaptation metadata
        new_strategy.add_metadata("adapted", True)
        new_strategy.add_metadata("adaptation_triggers", adaptation_triggers)
        new_strategy.add_metadata("adaptation_time", time.time())
        
        # Update strategy in context
        context.add("strategy", new_strategy)
        context.add("strategy_adapted", True)
        
        logger.info(f"Strategy adapted with {len(new_strategy.steps) - len(completed_steps)} new steps")
        return True
        
    except Exception as e:
        logger.error(f"Error adapting strategy: {str(e)}")
        return self._default_adaptation(context, current_step_index)
```

## 3. Preliminary Research Call

### Current Limitations
The current implementation has a placeholder comment in `_run_preliminary_research` instead of an actual call to the PerplexitySonarMCP.

### Enhancement Design
We will replace the placeholder with an actual call to the PerplexitySonarMCP to:
1. Execute preliminary research on the question
2. Extract key insights from the research
3. Generate initial hypotheses
4. Store results in the AnalysisContext

### Implementation Details
```python
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
        
        logger.info("Preliminary research completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running preliminary research: {str(e)}")
        return False
```

## 4. Question Analysis Call

### Current Limitations
The current implementation uses mock data in `_analyze_question_characteristics` instead of an actual LLM call.

### Enhancement Design
We will replace the placeholder with an actual call to the Llama4ScoutMCP to:
1. Analyze the question to determine its characteristics
2. Extract question type, domains, complexity, uncertainty, etc.
3. Store the analysis in the AnalysisContext

### Implementation Details
```python
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
        "question": question
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
```

## 5. Technique Registration

### Current Limitations
The current implementation lacks a proper `register_techniques` method to populate the technique registry.

### Enhancement Design
We will implement the `register_techniques` method to:
1. Discover available techniques using the TechniqueMCPIntegrator
2. Register them in the technique registry
3. Handle technique dependencies and compatibility

### Implementation Details
```python
def register_techniques(self):
    """Register available analytical techniques."""
    logger.info("Registering analytical techniques")
    
    # Get technique integrator
    technique_integrator = self.mcp_registry.get_technique_integrator()
    if not technique_integrator:
        logger.warning("TechniqueMCPIntegrator not available, using default techniques")
        self._register_default_techniques()
        return
    
    # Get all available techniques
    available_techniques = technique_integrator.get_all_techniques()
    if not available_techniques:
        logger.warning("No techniques available from integrator, using default techniques")
        self._register_default_techniques()
        return
    
    # Register techniques
    for name, technique in available_techniques.items():
        self.technique_registry[name] = technique
        logger.info(f"Registered technique: {name}")
    
    logger.info(f"Registered {len(self.technique_registry)} techniques")
```

## 6. Error Handling

### Current Limitations
The current implementation has a placeholder comment in `_handle_execution_error` instead of actual error handling logic.

### Enhancement Design
We will implement robust error handling to:
1. Log detailed error information
2. Implement recovery strategies based on error type
3. Provide fallback options when techniques fail
4. Ensure workflow can continue when possible

### Implementation Details
```python
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
        results = context.get("results", {})
        results[step.technique] = fallback_result
        context.add("results", results)
        
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
```

## Implementation Plan

1. **Update WorkflowOrchestratorMCP Class**:
   - Implement dynamic strategy generation
   - Implement workflow adaptation
   - Implement preliminary research call
   - Implement question analysis call
   - Implement technique registration
   - Implement error handling

2. **Update Llama4ScoutMCP**:
   - Add operations for strategy generation
   - Add operations for workflow adaptation
   - Add operations for question analysis

3. **Update PerplexitySonarMCP**:
   - Ensure research operation is properly implemented
   - Add methods for extracting key insights
   - Add methods for generating initial hypotheses

4. **Testing**:
   - Create unit tests for each new method
   - Create integration tests for the enhanced workflow orchestration
   - Test error handling and recovery strategies

This enhancement will provide the core intelligence of the system, enabling dynamic workflow orchestration and adaptation based on question characteristics and interim findings.
