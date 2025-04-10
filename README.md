# MCP Analytical System

## Architecture Overview

The MCP Analytical System is a sophisticated AI-powered framework designed for complex analysis with dynamic workflow orchestration. It leverages multiple specialized components (Modular Cognitive Processes or MCPs) to provide data-grounded, cognitively diverse analysis with integrated challenge mechanisms and uncertainty handling.

## Key Features

- **Dynamic Workflow Orchestration**: Analyzes question characteristics and adapts analytical workflows in real-time based on intermediate findings
- **Data-Grounded Analysis**: Integrates with research services and domain-specific data sources
- **Cognitive Diversity**: Employs multiple thinking personas to approach problems from different perspectives
- **Challenge Mechanisms**: Implements bias detection, assumption challenges, and red team analysis
- **Uncertainty Handling**: Identifies, quantifies, and communicates uncertainty throughout the analysis
- **Transparent Reasoning**: Provides traceable reasoning with source references and confidence assessments

## System Components

### Core Components

1. **WorkflowOrchestratorMCP**: The central "brain" of the system that:
   - Analyzes question characteristics
   - Generates tailored analysis strategies
   - Dynamically adapts workflows based on intermediate findings
   - Manages error handling and recovery

2. **AnalysisContext**: Shared state management that:
   - Stores all analysis data, events, and results
   - Tracks execution history and adaptation events
   - Provides structured access to intermediate findings

3. **AnalysisStrategy**: Represents the analysis plan with:
   - Sequence of analytical techniques to execute
   - Adaptation criteria for dynamic workflow modification
   - Parameters for each technique execution

4. **TechniqueMCPIntegrator**: Manages analytical techniques by:
   - Loading and executing technique implementations
   - Handling technique dependencies and prerequisites
   - Managing technique execution errors

### Specialized MCPs

1. **Research MCPs**:
   - **PerplexitySonarMCP**: Conducts preliminary research using Perplexity API
   - **Llama4ScoutMCP**: Analyzes question characteristics and domains
   - **ResearchMCP**: Performs comprehensive research with source tracking

2. **Domain MCPs**:
   - Specialized analysis for different domains (e.g., PhysicalSciencesMCP, LifeSciencesMCP)
   - Domain-specific data integration and analytical methods
   - Structured outputs with confidence assessments

3. **Cognitive Diversity MCPs**:
   - **ThinkingPersonaMCP**: Implements different thinking styles (analytical, creative, strategic, etc.)
   - Provides diverse perspectives on the same analysis
   - Synthesizes insights across different cognitive approaches

4. **Challenge MCPs**:
   - **ChallengeMechanismMCP**: Implements bias detection and assumption challenges
   - Performs red team analysis and premortem analysis
   - Generates alternative hypotheses and counterarguments

5. **Uncertainty MCPs**:
   - **UncertaintyHandlingMCP**: Identifies and quantifies uncertainty
   - Maps uncertainty across analysis components
   - Provides structured uncertainty communication

### Analytical Techniques

The system implements various analytical techniques as modular components:

1. **Research to Hypothesis**: Conducts research and generates hypotheses with confidence assessments
2. **Synthesis Generation**: Integrates findings into comprehensive synthesis with key judgments
3. **Scenario Triangulation**: Develops multiple scenarios with probability assessments
4. **Analysis of Competing Hypotheses**: Evaluates evidence against multiple hypotheses
5. **Uncertainty Mapping**: Identifies and quantifies uncertainties in the analysis

## Data Flow

1. **Input Processing**:
   - Question is analyzed for characteristics (type, domains, complexity)
   - Preliminary research is conducted to ground the analysis

2. **Strategy Generation**:
   - Analysis strategy is generated based on question characteristics
   - Appropriate techniques and MCPs are selected

3. **Dynamic Execution**:
   - Techniques are executed according to the strategy
   - Results are stored in the AnalysisContext
   - Strategy is adapted based on intermediate findings

4. **Synthesis and Output**:
   - Final synthesis integrates all analysis components
   - Confidence assessments and uncertainty are communicated
   - Traceable reasoning with source references is provided

## Configuration

The system uses a Pydantic-based configuration system with:

- Environment variable loading with validation
- Hierarchical settings for different components
- Default values and type checking

## Error Handling

Robust error handling includes:

- Retry mechanisms for transient failures
- Fallback techniques when primary techniques fail
- Graceful degradation with partial results
- Comprehensive error logging and tracking

## Testing

The system includes comprehensive testing:

- Unit tests for individual components
- Integration tests for end-to-end functionality
- Mock implementations for external dependencies

## Usage

```python
from src.workflow_orchestrator_mcp import WorkflowOrchestratorMCP
from src.analysis_context import AnalysisContext
from src.mcp_registry import MCPRegistry

# Initialize components
registry = MCPRegistry()
orchestrator = WorkflowOrchestratorMCP(registry)

# Create analysis context
context = AnalysisContext()
context.add("question", "What are the potential economic impacts of quantum computing over the next decade?")

# Process the question
result = orchestrator.process({"context": context})

# Access results
synthesis = context.get("synthesis_integrated_assessment")
key_judgments = context.get("synthesis_key_judgments")
```

## Environment Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and configure API keys
4. Run tests: `python -m unittest discover tests`

## Configuration Options

The system can be configured through environment variables:

```
# LLM Configuration
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
DEFAULT_LLM_PROVIDER=groq
DEFAULT_LLM_MODEL=llama4-scout-17b-16e

# Research Configuration
PERPLEXITY_API_KEY=your_perplexity_key
DEFAULT_RESEARCH_DEPTH=standard

# Workflow Configuration
ENABLE_ADAPTIVE_WORKFLOW=true
MAX_TECHNIQUE_RETRIES=3

# System Configuration
ENVIRONMENT=development
DEBUG_MODE=false
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request
