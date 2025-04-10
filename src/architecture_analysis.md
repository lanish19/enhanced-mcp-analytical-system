# Current System Architecture Analysis

## Overview
The current analytical system is built as a Streamlit application that performs autonomous analysis of complex questions using multiple analytical techniques. The system employs a relatively linear workflow orchestration with some flexibility, but lacks the dynamic adaptability and modularity that an MCP (Modular Cognitive Processor) architecture would provide.

## Core Components

### 1. AutonomousAnalysis Class
The central orchestrator of the analytical process, found in `components/analysis.py`. This class:
- Manages the overall analysis workflow
- Sequentially executes a predefined set of analytical techniques
- Collects results from each technique
- Generates a final synthesis
- Has limited ability to adapt workflows based on interim findings

Key limitations:
- Uses a fixed sequence of analytical methods
- Does not dynamically select techniques based on question characteristics
- Limited adaptation capabilities during execution
- Tightly coupled with specific implementation details

### 2. Analytical Techniques
Currently implemented as methods within the AutonomousAnalysis class or as separate classes:

- **Scenario Triangulation**: Generates multiple plausible scenarios
- **Consensus Challenge**: Identifies and challenges consensus views
- **Multi-Persona Swarm**: Analyzes questions from multiple expert perspectives
- **Backward Reasoning**: Works backward from potential end states
- **Research-to-Hypothesis**: Conducts research and generates/tests hypotheses

These techniques are currently:
- Relatively monolithic
- Tightly coupled to specific LLM implementations
- Not easily composable or reusable
- Not designed to leverage specialized MCPs

### 3. Supporting Components

- **HypothesisManager**: Manages hypothesis generation and evaluation
- **BiasAnalyzer**: Analyzes and mitigates biases
- **UncertaintyNetworkAnalyzer**: Analyzes uncertainties and their relationships
- **StructuredAnalyticTechniques**: Implements various analytical techniques
- **ScenarioGenerator**: Generates and analyzes different types of scenarios

### 4. Utilities

- **LLM Integration**: Handles interactions with LLM APIs (Groq, Perplexity)
- **Error Handler**: Manages error handling and recovery
- **Research Integration**: Placeholder for online research capabilities

## Current Workflow

1. User submits a question via the Streamlit interface
2. AutonomousAnalysis analyzes the question for biases
3. Initial hypotheses are generated
4. A fixed sequence of analytical techniques is executed:
   - Scenario Triangulation
   - Consensus Challenge
   - Multi-Persona Swarm
   - Backward Reasoning
   - Research-to-Hypothesis
5. Biases in the workflows are analyzed
6. A synthesis is generated
7. Red teaming and premortem analysis are performed
8. An uncertainty network is generated
9. Results are presented to the user

## LLM Integration

The system currently uses three main LLM configurations:
- **Llama 4 Scout (17B)** via Groq: For simpler analytical tasks
- **Sonar** via Perplexity: For tasks requiring web search
- **Sonar Deep Research** via Perplexity: For complex research tasks

## Limitations of Current Architecture

1. **Limited Adaptability**: The system follows a relatively fixed workflow regardless of question type.
2. **Lack of Modularity**: Components are tightly coupled, making it difficult to add, remove, or modify techniques.
3. **Inefficient Resource Usage**: All techniques are executed for all questions, even when some may not be relevant.
4. **Limited Specialization**: Does not leverage specialized processors for domain-specific knowledge.
5. **Minimal Dynamic Adaptation**: Limited ability to adapt the workflow based on interim findings.
6. **Research Capabilities**: Uses placeholder online researcher without proper integration.
7. **Context Management**: No centralized context store for efficient information sharing between components.

## Transformation Opportunities

1. **Workflow Orchestration**: Replace fixed workflow with dynamic orchestration based on question characteristics.
2. **Modular Techniques**: Refactor analytical techniques into modular, composable components.
3. **Specialized MCPs**: Implement domain-specific MCPs for economics, geopolitics, etc.
4. **Research Integration**: Replace placeholder with proper Research MCPs.
5. **Context Management**: Implement Redis Context Store for efficient information sharing.
6. **Preliminary Research**: Add Perplexity Sonar phase before workflow selection.
7. **Model Integration**: Standardize on Llama 4 Scout via Groq for LLM components.

This analysis provides the foundation for designing the new MCP architecture that will address these limitations and opportunities.
