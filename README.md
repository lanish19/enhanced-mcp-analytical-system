README.md Outline

Project Title: Enhanced MCP Analytical System v2

Introduction / Purpose:

Describe the system as an advanced AI agent designed to perform in-depth, autonomous analysis of complex questions.
Explain the core concept: leveraging a dynamic workflow composed of modular analytical techniques and specialized Modular Cognitive Processors (MCPs).
Highlight the goal of moving beyond rigid, predefined analytical pipelines to a system that adapts its methodology based on the specific question and interim findings.
Desired Final Product / Aims:

State the ambition: To create a "world-class" AI assessment capability.
Detail the target characteristics:
Robustness & Objectivity: Grounded in verifiable evidence, minimizing cognitive and LLM biases.
Adaptability: Dynamically tailoring analytical workflows (techniques, parameters, sequence) to each unique question.
Depth & Breadth: Capable of tackling diverse topics ("art to rocket science") by integrating specialized knowledge.
Uncertainty Management: Explicitly identifying, assessing, and communicating analytical uncertainty and confidence levels.
Transparency: Providing traceable reasoning from question to conclusion.
Architecture Overview:

Briefly explain the MCP (Modular Cognitive Processor) architecture principle: specialized components for different tasks/domains.
Core Components:
MCPSystemIntegrator: The main entry point managing system initialization and high-level analysis execution.
MCPRegistry: Central registry holding instances of all active MCPs.
BaseMCP: The common interface for all MCP modules.
WorkflowOrchestratorMCP: (Key Component - Note Current Status) Intended to be the system's "brain," responsible for analyzing questions, dynamically generating AnalysisStrategy objects (workflow plans), and adapting these strategies mid-execution based on findings. Mention that the core dynamic logic is currently planned but not fully implemented (see todo.md).
AnalysisContext: A state object holding all information for a single analysis run (question, results, metadata, strategy, events).
AnalysisStrategy: Represents the sequence of analytical steps planned by the orchestrator.
AnalyticalTechnique: Base class for modular analytical methods (e.g., ScenarioTriangulationTechnique, ACHTechnique, BiasDetectionTechnique, SynthesisGenerationTechnique). List key examples.
TechniqueMCPIntegrator: Responsible for loading and managing AnalyticalTechnique modules. Note potential architectural overlap with the orchestrator regarding workflow logic that needs clarification.
Specialized MCPs:
ResearchMCP: Performs web/academic searches and extracts content (using real APIs/libraries).
PerplexitySonarMCP / Llama4ScoutMCP: Interface layers for interacting with specific LLM APIs.
RedisContextStoreMCP: Handles persistence of analysis context/results.
Domain MCPs (EconomicsMCP, GeopoliticsMCP): Provide domain-specific analysis. Mention current reliance on LLM calls and the goal/need to integrate real data sources/APIs.
(Optional: Mention planned/missing enhancing MCPs like CognitiveBiasMCP, EvidenceExtractionMCP).
High-Level Workflow:
Describe the intended flow:
User Input (app.py / MCPSystemIntegrator).
(Optional/Integrated) Preliminary Research (PerplexitySonarMCP).
Question Analysis (WorkflowOrchestratorMCP using LLM).
Dynamic Strategy Generation (WorkflowOrchestratorMCP using LLM).
Iterative Workflow Execution (WorkflowOrchestratorMCP calling TechniqueMCPIntegrator or techniques directly):
Execute technique.
Check for adaptation triggers.
Adapt strategy if needed.
Final Synthesis (SynthesisGenerationTechnique, potentially refined by orchestrator/LLM).
Emphasize the intended dynamic and adaptive nature driven by the orchestrator's (currently unimplemented) core logic.
Current Status & Next Steps:

Summarize the current state: Foundational architecture is in place, core techniques exist, ResearchMCP is functional.
Clearly state the critical next steps: Implementing the dynamic strategy generation and adaptation logic within WorkflowOrchestratorMCP, and integrating real data sources into Domain MCPs, as detailed in todo.md.
Mention other key areas from todo.md (unit testing, enhancing MCPs, refining synthesis/uncertainty, etc.).
Setup & Usage (Optional but Recommended):

Instructions for setting up the environment (e.g., pip install -r requirements.txt).
List required environment variables (API keys for Groq, Perplexity, Brave Search, etc.).
Basic example of how to run an analysis (e.g., using src/app.py if it's the intended interface, or example Python script using MCPSystemIntegrator).
