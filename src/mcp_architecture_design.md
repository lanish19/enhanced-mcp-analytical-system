# MCP Architecture Design

## Overview
This document outlines the design for transforming the current analytical system into a dynamic MCP (Modular Cognitive Processor) architecture. The new architecture will provide greater adaptability, modularity, and efficiency while maintaining analytical depth and robustness.

## Core Architectural Principles

1. **Modularity**: All components are designed as independent, interchangeable modules
2. **Adaptability**: Workflows are dynamically constructed based on question characteristics
3. **Specialization**: Domain-specific MCPs provide targeted expertise
4. **Efficiency**: Resources are allocated based on question requirements
5. **Extensibility**: New techniques and MCPs can be easily added
6. **Robustness**: Error handling and fallback mechanisms ensure reliability

## MCP Architecture Components

### 1. WorkflowOrchestratorMCP

The central orchestrator that replaces the current AutonomousAnalysis class. This MCP:

- Analyzes question characteristics to determine optimal workflow sequence
- Selects techniques based on question type (predictive, causal, evaluative)
- Adapts workflow dynamically based on interim findings
- Manages technique dependencies and complementary pairs
- Monitors execution and handles errors

```python
class WorkflowOrchestratorMCP(BaseMCP):
    """
    Central orchestrator that dynamically constructs and manages analytical workflows.
    """
    
    def __init__(self, mcp_registry):
        super().__init__("workflow_orchestrator", "Orchestrates analytical workflows")
        self.mcp_registry = mcp_registry
        self.technique_registry = {}
        self.register_techniques()
        
    def register_techniques(self):
        """Register all available analytical techniques."""
        # Register core techniques
        self.technique_registry = {
            "scenario_triangulation": ScenarioTriangulationTechnique(),
            "consensus_challenge": ConsensusChallengeTechnique(),
            "multi_persona": MultiPersonaTechnique(),
            "backward_reasoning": BackwardReasoningTechnique(),
            "research_to_hypothesis": ResearchToHypothesisTechnique(),
            "causal_network_analysis": CausalNetworkAnalysisTechnique(),
            "key_assumptions_check": KeyAssumptionsCheckTechnique(),
            "analysis_of_competing_hypotheses": ACHTechnique(),
            "uncertainty_mapping": UncertaintyMappingTechnique(),
            "red_teaming": RedTeamingTechnique(),
            "premortem_analysis": PremortermAnalysisTechnique(),
            "synthesis_generation": SynthesisGenerationTechnique()
        }
    
    def analyze_question(self, question, parameters=None):
        """
        Dynamically analyze a question using an adaptive workflow.
        """
        # Create analysis context
        context = AnalysisContext(question, parameters)
        
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
        
        return result
```

### 2. AnalyticalTechnique Classes

Refactored from the current workflows into modular, independent classes:

```python
class AnalyticalTechnique:
    """Base class for all analytical techniques."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.mcp_registry = MCPRegistry.get_instance()
    
    def execute(self, context, parameters):
        """Execute this technique with given context and parameters."""
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_required_mcps(self):
        """Return list of MCPs required by this technique."""
        return []
    
    def get_technique_metadata(self):
        """Return metadata about this technique."""
        return {
            "name": self.name,
            "description": self.__doc__,
            "required_mcps": self.get_required_mcps()
        }
```

Example implementation:

```python
class ScenarioTriangulationTechnique(AnalyticalTechnique):
    """Generates multiple plausible scenarios related to the question."""
    
    def execute(self, context, parameters):
        # Get parameters or use defaults
        num_scenarios = parameters.get("num_scenarios", 4)
        scenario_types = parameters.get("scenario_types", ["base", "best", "worst", "wildcard"])
        detail_level = parameters.get("detail_level", "medium")
        
        # Use MCPs to enhance scenario generation
        economic_data_mcp = self.mcp_registry.get("economic_data")
        trend_analysis_mcp = self.mcp_registry.get("trend_analysis")
        
        # Build prompt with enhanced context
        prompt_context = context.question
        
        if economic_data_mcp and "economic" in context.get_metadata("domains", []):
            economic_context = economic_data_mcp.get_relevant_indicators(context.question)
            prompt_context += f"\n\nEconomic Context:\n{economic_context}"
        
        if trend_analysis_mcp:
            trend_data = trend_analysis_mcp.analyze_trends(context.question)
            prompt_context += f"\n\nTrend Analysis:\n{trend_data}"
        
        # Create tailored prompt based on detail level
        prompt = self._create_scenario_prompt(prompt_context, num_scenarios, scenario_types, detail_level)
        
        # Call appropriate model
        model_config = MODEL_CONFIG["sonar_deep"] if detail_level == "high" else MODEL_CONFIG["sonar"]
        response = call_llm(prompt, model_config)
        content = extract_content(response)
        
        # Process and return results
        # ...
```

### 3. Core MCPs

#### Research MCPs

```python
class BraveSearchMCP(BaseMCP):
    """MCP for performing web searches using Brave Search API."""
    
    def search(self, query, num_results=5):
        """Perform a web search and return results."""
        # Implementation using Brave Search API
        pass
    
    def extract_content(self, url):
        """Extract and process content from a URL."""
        # Implementation for content extraction
        pass

class AcademicSearchMCP(BaseMCP):
    """MCP for searching academic papers and research."""
    
    def search(self, query, num_results=5):
        """Search academic papers and return results."""
        # Implementation using academic search APIs
        pass
    
    def extract_paper(self, paper_id):
        """Extract and summarize a paper."""
        # Implementation for paper extraction
        pass
```

#### Domain MCPs

```python
class EconomicsMCP(BaseMCP):
    """MCP for economic analysis and data."""
    
    def get_relevant_indicators(self, question):
        """Identify and retrieve relevant economic indicators."""
        pass
    
    def analyze_economic_trends(self, data):
        """Analyze economic trends in the provided data."""
        pass

class GeopoliticsMCP(BaseMCP):
    """MCP for geopolitical analysis."""
    
    def identify_relevant_actors(self, question):
        """Identify relevant geopolitical actors."""
        pass
    
    def analyze_regional_dynamics(self, region):
        """Analyze geopolitical dynamics in a region."""
        pass
```

#### Infrastructure MCPs

```python
class RedisContextStoreMCP(BaseMCP):
    """MCP for storing and retrieving analysis context using Redis."""
    
    def __init__(self):
        super().__init__("redis_context_store", "Stores and retrieves analysis context")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def store_context(self, session_id, context_data):
        """Store context data in Redis."""
        self.redis_client.set(f"context:{session_id}", json.dumps(context_data))
    
    def retrieve_context(self, session_id):
        """Retrieve context data from Redis."""
        context_json = self.redis_client.get(f"context:{session_id}")
        return json.loads(context_json) if context_json else None
    
    def update_context(self, session_id, key, value):
        """Update a specific key in the context."""
        context = self.retrieve_context(session_id)
        if context:
            context[key] = value
            self.store_context(session_id, context)
```

### 4. AnalysisContext

Central object for managing the state of an analysis:

```python
class AnalysisContext:
    """Manages the state and context of an ongoing analysis."""
    
    def __init__(self, question, parameters=None):
        self.session_id = str(uuid.uuid4())
        self.question = question
        self.parameters = parameters or {}
        self.metadata = {}
        self.results = {}
        self.events = []
        self.strategy = None
        self.question_analysis = None
        self.error_log = []
        self.start_time = time.time()
    
    def add_metadata(self, key, value):
        """Add metadata to the context."""
        self.metadata[key] = value
    
    def get_metadata(self, key, default=None):
        """Get metadata from the context."""
        return self.metadata.get(key, default)
    
    def set_question_analysis(self, analysis):
        """Set question analysis result."""
        self.question_analysis = analysis
    
    def set_strategy(self, strategy):
        """Set the analysis strategy."""
        self.strategy = strategy
    
    def add_result(self, technique_name, result):
        """Add result from an analytical technique."""
        self.results[technique_name] = result
        self.events.append({
            "timestamp": time.time(),
            "type": "technique_result",
            "technique": technique_name,
            "success": "error" not in result
        })
```

### 5. MCPRegistry

Central registry for managing and accessing MCPs:

```python
class MCPRegistry:
    """Singleton registry for managing MCPs."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MCPRegistry()
        return cls._instance
    
    def __init__(self):
        self.mcps = {}
    
    def register_mcp(self, mcp):
        """Register an MCP with the registry."""
        self.mcps[mcp.name] = mcp
    
    def get_mcp(self, name):
        """Get an MCP by name."""
        return self.mcps.get(name)
    
    def get_all_mcps(self):
        """Get all registered MCPs."""
        return self.mcps
```

### 6. AnalysisStrategy

Model for representing dynamic analysis strategies:

```python
class AnalysisStrategy:
    """Represents a dynamic analytical strategy."""
    
    def __init__(self, strategy_data):
        self.name = strategy_data.get("name", "Unnamed Strategy")
        self.description = strategy_data.get("description", "")
        self.adaptive = strategy_data.get("adaptive", True)
        self.steps = strategy_data.get("steps", [])
    
    def to_dict(self):
        """Convert strategy to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "adaptive": self.adaptive,
            "steps": self.steps
        }
```

## Workflow Orchestration Process

1. **Preliminary Research Phase**
   - Use Perplexity Sonar to gather initial information
   - Extract key entities, concepts, and relationships
   - Identify relevant domains and time frames

2. **Question Analysis Phase**
   - Determine question type (predictive, causal, evaluative)
   - Identify required domains of expertise
   - Assess complexity and uncertainty level
   - Detect potential biases

3. **Strategy Generation Phase**
   - Select appropriate techniques based on question characteristics
   - Determine optimal sequence of techniques
   - Identify technique dependencies and complementary pairs
   - Set parameters for each technique

4. **Dynamic Execution Phase**
   - Execute techniques in sequence
   - Monitor results and adapt strategy as needed
   - Handle errors and fallbacks
   - Collect and integrate results

5. **Synthesis Phase**
   - Integrate results from all techniques
   - Generate comprehensive synthesis
   - Assess confidence and uncertainty
   - Provide final judgments and recommendations

## Integration with Llama 4 Scout

The system will standardize on Llama 4 Scout (17Bx16E) via Groq for LLM components due to its speed/cost advantage:

```python
MODEL_CONFIG = {
    "llama4": {
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 4096,
        "temperature": 0.7,
        "description": "Used for simpler analytical tasks and basic reasoning"
    },
    "sonar": {
        "provider": "perplexity",
        "model": "sonar",
        "max_tokens": 4096,
        "temperature": 0.7,
        "description": "Used for tasks requiring web search and factual information"
    },
    "sonar_deep": {
        "provider": "perplexity",
        "model": "sonar-deep-research",
        "max_tokens": 4096,
        "temperature": 0.7,
        "description": "Used for complex research tasks requiring extensive analysis"
    }
}
```

## Deployment Architecture

The system will be deployed as a permanent website using Next.js:

1. **Frontend**: Next.js application with Streamlit-like UI components
2. **Backend**: Python FastAPI service for MCP orchestration
3. **Storage**: Redis for context storage and caching
4. **Deployment**: Deployed to a permanent web hosting solution

## Implementation Roadmap

1. Implement BaseMCP and MCPRegistry
2. Implement WorkflowOrchestratorMCP
3. Refactor existing workflows into AnalyticalTechnique classes
4. Implement core Research MCPs
5. Implement Domain MCPs
6. Implement Infrastructure MCPs
7. Integrate Perplexity Sonar for preliminary research
8. Standardize on Llama 4 Scout via Groq
9. Develop Next.js frontend
10. Deploy to permanent web hosting solution
