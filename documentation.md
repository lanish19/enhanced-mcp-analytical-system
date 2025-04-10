# MCP Analytical System Documentation

## Overview

The MCP (Modular Cognitive Processor) Analytical System is a dynamic architecture designed for sophisticated analytical reasoning. It transforms the original AutonomousAnalysis class into a modular, adaptive system that can handle complex analytical questions across various domains.

The system uses a WorkflowOrchestratorMCP that analyzes question characteristics, selects appropriate analytical techniques, and dynamically adapts workflows based on interim findings. It integrates domain expertise from specialized MCPs and leverages advanced LLM capabilities through Llama 4 Scout and Perplexity Sonar.

## Core Architecture

### MCP (Modular Cognitive Processor)

MCPs are the fundamental building blocks of the system. Each MCP is a specialized module that provides specific capabilities:

- **Base MCP**: The foundation class that all MCPs inherit from, providing standard interfaces and capabilities.
- **Domain MCPs**: Specialized processors with expertise in specific domains (Economics, Geopolitics).
- **Infrastructure MCPs**: Provide core system functionality (Redis Context Store, Workflow Orchestrator).
- **Research MCPs**: Handle information gathering and processing (Brave/Academic search, Perplexity Sonar).
- **LLM MCPs**: Provide advanced reasoning capabilities (Llama 4 Scout).

### Analytical Techniques

The system implements 20+ modular analytical techniques, each designed for specific analytical purposes:

- **Scenario Triangulation**: Generates and analyzes multiple future scenarios.
- **Consensus Challenge**: Identifies and challenges consensus views.
- **Multi-Persona Analysis**: Analyzes questions from multiple perspectives.
- **Backward Reasoning**: Works backward from potential outcomes to identify causal paths.
- **Research to Hypothesis**: Converts research findings into testable hypotheses.
- **Causal Network Analysis**: Maps causal relationships between factors.
- **Key Assumptions Check**: Identifies and tests critical assumptions.
- **Analysis of Competing Hypotheses**: Evaluates evidence against multiple hypotheses.
- **Uncertainty Mapping**: Identifies and quantifies sources of uncertainty.
- **Red Teaming**: Challenges analysis from an adversarial perspective.
- **Premortem Analysis**: Imagines failure and works backward to identify risks.
- **Synthesis Generation**: Integrates findings into coherent conclusions.
- **Cross-Impact Analysis**: Analyzes how factors influence each other.
- **System Dynamics Modeling**: Models complex system behavior over time.
- **Indicators Development**: Identifies observable indicators for monitoring.
- **Argument Mapping**: Structures logical arguments and counterarguments.
- **Bias Detection**: Identifies and mitigates cognitive biases.
- **Decision Tree Analysis**: Maps decision points and potential outcomes.
- **Delphistic Forecasting**: Generates forecasts through structured expert elicitation.
- **Historical Analogies**: Identifies and applies relevant historical precedents.

### Core Components

#### WorkflowOrchestratorMCP

The central intelligence of the system that:
- Analyzes question characteristics to determine optimal workflow sequence
- Selects techniques based on question type (predictive, causal, evaluative)
- Adapts workflow dynamically based on interim findings
- Manages technique dependencies and complementary pairs

#### TechniqueMCPIntegrator

Connects analytical techniques with MCPs:
- Maintains mappings between techniques and required MCPs
- Tracks technique dependencies and complementary pairs
- Facilitates dynamic workflow adaptation
- Evaluates technique effectiveness

#### MCPSystemIntegrator

The central integration point for all components:
- Initializes and manages all MCPs
- Provides unified interface for system interaction
- Coordinates analysis workflow execution
- Manages domain expertise integration

#### AnalysisContext

Maintains the context for analysis:
- Stores question and metadata
- Tracks hypotheses and findings
- Manages uncertainty levels
- Preserves analysis history

#### AnalysisStrategy

Determines the strategy for analysis:
- Identifies question type and characteristics
- Selects initial techniques
- Sets analysis parameters
- Guides workflow orchestration

## Domain Expertise Integration

The system integrates domain expertise through specialized MCPs:

### EconomicsMCP

Provides economic domain expertise:
- Economic data retrieval and analysis
- Economic forecasting and scenario generation
- Policy impact assessment
- Market trend analysis
- Macroeconomic modeling

### GeopoliticsMCP

Provides geopolitical domain expertise:
- Geopolitical risk assessment
- Regional stability analysis
- Conflict potential evaluation
- International relations modeling
- Political trend analysis

## Research Capabilities

The system includes robust research capabilities:

### ResearchMCP

Handles information gathering:
- Brave search integration for web content
- Academic search for scholarly sources
- Content extraction and processing
- Source credibility assessment
- Information synthesis

### PerplexitySonarMCP

Provides preliminary research:
- Comprehensive research capabilities
- Key insight extraction
- Initial hypothesis formulation
- Workflow recommendation

## LLM Integration

The system leverages advanced LLM capabilities:

### Llama4ScoutMCP

Provides advanced reasoning:
- Chain-of-thought reasoning
- Uncertainty quantification
- Bias detection
- Multi-step analysis

## Dynamic Workflow Adaptation

The system can dynamically adapt workflows based on:

- Interim findings from executed techniques
- Uncertainty levels in the analysis
- Conflicting hypotheses
- Domain-specific findings
- Time horizon considerations

## System Flow

1. **Question Analysis**: The system analyzes the question to determine its type, domain, and characteristics.
2. **Preliminary Research**: PerplexitySonarMCP performs initial research to gather context.
3. **Strategy Determination**: The system determines the optimal analysis strategy.
4. **Workflow Orchestration**: WorkflowOrchestratorMCP selects and sequences techniques.
5. **Technique Execution**: Selected techniques are executed with MCP support.
6. **Dynamic Adaptation**: The workflow adapts based on interim findings.
7. **Domain Integration**: Domain MCPs provide specialized expertise.
8. **Synthesis**: Results are synthesized into comprehensive analysis.
9. **Delivery**: Final analysis is delivered with confidence levels and uncertainty quantification.

## Usage

### Basic Usage

```python
from src.mcp_system_integrator import MCPSystemIntegrator

# Initialize system
config = {
    "perplexity_api_key": "your_perplexity_api_key",
    "groq_api_key": "your_groq_api_key",
    # Other API keys...
}
system = MCPSystemIntegrator(config)

# Analyze a question
result = system.analyze_question("What are the economic implications of rising interest rates?")

# Print result
print(result)
```

### Advanced Usage

```python
from src.mcp_system_integrator import MCPSystemIntegrator
from src.analysis_context import AnalysisContext

# Initialize system
system = MCPSystemIntegrator(config)

# Create custom context
context = AnalysisContext(
    question="What are the geopolitical implications of renewable energy adoption?",
    question_type="causal",
    time_horizon="long",
    uncertainty_level=0.7
)

# Execute specific technique
result = system.execute_technique("scenario_triangulation", context.to_dict(), {
    "num_scenarios": 5,
    "focus_domains": ["geopolitical", "economic"]
})

# Get domain expertise
geopolitics_result = system.get_domain_expertise("geopolitics", "assess_geopolitical_risk", {
    "countries": ["USA", "China", "Russia"],
    "time_horizon": "medium"
})
```

## API Reference

### MCPSystemIntegrator

- `analyze_question(question, additional_context=None)`: Analyze a question
- `execute_technique(technique_name, context, parameters=None)`: Execute a specific technique
- `adapt_workflow(context, interim_findings)`: Adapt workflow based on findings
- `get_domain_expertise(domain, operation, parameters)`: Get domain expertise
- `perform_research(query, depth="standard")`: Perform research
- `get_mcp_capabilities()`: Get capabilities of all MCPs
- `get_system_status()`: Get system status

### WorkflowOrchestratorMCP

- `analyze_question(question)`: Analyze question characteristics
- `determine_workflow(context, strategy)`: Determine optimal workflow
- `execute_workflow(context, strategy)`: Execute complete workflow
- `execute_technique(technique, context, parameters)`: Execute specific technique
- `adapt_workflow(context, interim_findings)`: Adapt workflow based on findings

### TechniqueMCPIntegrator

- `execute_technique(technique, context, parameters)`: Execute technique with MCP support
- `suggest_next_techniques(context, current_technique)`: Suggest next techniques
- `adapt_workflow(context, interim_findings)`: Adapt workflow based on findings
- `get_technique_dependencies(technique_name)`: Get technique dependencies
- `get_complementary_techniques(technique_name)`: Get complementary techniques

## Deployment

The system can be deployed in several ways:

### Streamlit Web Application

The system includes a Streamlit web application for easy interaction:

```bash
streamlit run src/app.py
```

### Docker Deployment

The system can be deployed using Docker:

```bash
docker build -t mcp-analytical-system .
docker run -p 8501:8501 mcp-analytical-system
```

### Cloud Deployment

The system can be deployed to cloud platforms:

- **Heroku**: Follow instructions in deployment_instructions.md
- **AWS**: Follow instructions in deployment_instructions.md
- **GCP**: Follow instructions in deployment_instructions.md

## Configuration

The system is configured through a configuration dictionary:

```python
config = {
    # API Keys
    "perplexity_api_key": "your_perplexity_api_key",
    "groq_api_key": "your_groq_api_key",
    "brave_api_key": "your_brave_api_key",
    "academic_api_key": "your_academic_api_key",
    "fred_api_key": "your_fred_api_key",
    "world_bank_api_key": "your_world_bank_api_key",
    "imf_api_key": "your_imf_api_key",
    "gdelt_api_key": "your_gdelt_api_key",
    "acled_api_key": "your_acled_api_key",
    
    # Infrastructure
    "redis_url": "redis://localhost:6379/0",
    
    # System Settings
    "log_level": "INFO",
    "cache_enabled": True,
    "max_techniques_per_workflow": 10
}
```

## Extending the System

### Adding New MCPs

To add a new MCP:

1. Create a new class that inherits from BaseMCP
2. Implement the required methods
3. Register the MCP with the MCPRegistry

```python
from src.base_mcp import BaseMCP

class MyNewMCP(BaseMCP):
    def __init__(self):
        super().__init__(
            name="my_new_mcp",
            description="My new MCP",
            version="1.0.0"
        )
        
    def process(self, input_data):
        # Process input data
        return {"result": "processed data"}
        
    def get_capabilities(self):
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "operations": ["operation1", "operation2"]
        }
```

### Adding New Techniques

To add a new analytical technique:

1. Create a new class that inherits from AnalyticalTechnique
2. Implement the required methods
3. Update the TechniqueMCPIntegrator mappings

```python
from src.analytical_technique import AnalyticalTechnique

class MyNewTechnique(AnalyticalTechnique):
    def __init__(self):
        super().__init__(
            name="my_new_technique",
            description="My new analytical technique",
            version="1.0.0"
        )
        
    def execute(self, execution_context):
        # Execute technique
        return {"result": "technique output"}
        
    def get_capabilities(self):
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": ["param1", "param2"]
        }
```

## Troubleshooting

### Common Issues

- **API Key Issues**: Ensure all required API keys are provided in the configuration.
- **Redis Connection**: Verify Redis is running and accessible at the specified URL.
- **Technique Execution Failures**: Check the logs for specific error messages.
- **MCP Initialization Failures**: Ensure all required dependencies are installed.

### Logging

The system uses Python's logging module. To enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing

The system includes comprehensive tests:

```bash
# Run integration tests
python src/integration_test.py

# Run system test
python src/test_mcp_system.py
```

## Conclusion

The MCP Analytical System provides a powerful, flexible architecture for sophisticated analytical reasoning. Its modular design, dynamic workflow adaptation, and domain expertise integration make it capable of handling complex analytical questions across various domains.

The system's core intelligence lies in its dynamic workflow orchestration and adaptation capabilities, which allow it to select the most appropriate techniques and adapt its approach based on interim findings. Combined with its grounding in real data through domain MCPs, this makes the system a robust and adaptive analytical tool.
