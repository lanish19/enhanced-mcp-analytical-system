# MCP Analytical System - Detailed TODO Checklist (v2 Review)

This checklist details outstanding tasks based on the review of the v2 codebase against previous recommendations. Focus is on implementing core dynamic logic and grounding analysis in data.

## Section 1: Core Orchestration (`src/workflow_orchestrator_mcp.py`)

**Objective:** Implement the critical dynamic and adaptive intelligence of the system.

*   `[ ]` **Implement Dynamic Strategy Generation (`_generate_analysis_strategy`)**
    *   **What:** Replace hardcoded/mock strategy logic with dynamic generation.
    *   **How:**
        1.  Ensure `_analyze_question_characteristics` (see below) populates `context.question_analysis` correctly.
        2.  Get the `llama4_scout` MCP instance from `self.mcp_registry`.
        3.  Construct a detailed prompt (as outlined in `dynamic_adaptation_ideas.md` and `workflow_orchestrator_enhancement_design.md`) including `question_analysis`, available techniques (from `self.technique_registry`), and desired JSON output structure (`AnalysisStrategy` format).
        4.  Call `llama4_scout.process(...)` with the prompt.
        5.  Parse and **validate** the returned JSON (using Pydantic or similar). Check if technique names are valid and parameters are reasonable.
        6.  Instantiate and return the `AnalysisStrategy` object.
        7.  Implement robust error handling and fallback to `_generate_default_strategy` if the LLM call or validation fails.
    *   **Why:** This is essential for tailoring the analysis to the specific question, improving relevance, efficiency, and depth. Avoids rigid, one-size-fits-all workflows. (Addresses Principle 1).

*   `[ ]` **Implement Question Analysis (`_analyze_question_characteristics`)**
    *   **What:** Replace mock analysis data with an actual LLM call.
    *   **How:**
        1.  Get the `llama4_scout` MCP instance.
        2.  Construct a prompt asking the LLM to analyze `context.question` and extract characteristics (type, domains, complexity, uncertainty, time_horizon, potential_biases) into a specific JSON format.
        3.  Call `llama4_scout.process(...)`.
        4.  Parse and validate the response.
        5.  Store the validated analysis result in `context.question_analysis`.
        6.  Implement fallback to default/mock analysis if the call fails.
    *   **Why:** Provides the necessary input for dynamic strategy generation. Ensures the workflow is based on an understanding of the question's nature. (Supports Principle 1).

*   `[ ]` **Implement Workflow Adaptation (`_check_adaptation_criteria`, `_adapt_strategy`)**
    *   **What:** Replace mock adaptation logic with real checks and actions.
    *   **How:**
        1.  **`_check_adaptation_criteria`:** Implement programmatic checks on the `last_result` from `context.results`. Define specific thresholds and conditions based on technique output fields (e.g., `overall_uncertainty > 0.7`, `overall_bias_level == 'High'`, score difference in ACH < threshold, presence of contradictions in synthesis). Return a specific trigger type string/enum or `None`.
        2.  **`_adapt_strategy`:** Implement the hybrid approach:
            *   Use `if/elif` based on the `trigger_type` for common adaptations (e.g., `if trigger_type == 'HighBiasDetected': insert_step('RedTeamingTechnique', ...)`). Check technique availability first.
            *   For complex/unhandled triggers, construct a prompt for `llama4_scout` including the trigger, context, remaining steps, and available techniques. Ask it to output a *new list* of remaining steps (JSON).
            *   Parse, validate, and replace the remaining steps in `strategy.steps` using `strategy.replace_steps(...)`.
            *   Log the reason for adaptation.
    *   **Why:** Allows the system to react intelligently to findings, address issues like high uncertainty or bias, and improve the analysis iteratively. Core to the "dynamic" aspect. (Addresses Principle 1).

*   `[ ]` **Implement Preliminary Research Call (`_run_preliminary_research`)**
    *   **What:** Replace placeholder with actual call to Perplexity Sonar MCP.
    *   **How:**
        1.  Get the `perplexity_sonar` MCP instance from `self.mcp_registry`.
        2.  Prepare the input dictionary for its `process` method (e.g., `{"operation": "research", "question": context.question}`).
        3.  Call `perplexity_sonar.process(...)`.
        4.  Store the relevant parts of the result (e.g., insights, hypotheses, recommendations) in `context.metadata` or dedicated context fields. Handle potential errors from the MCP call.
    *   **Why:** Provides initial context and grounding for the subsequent analysis and strategy generation. (Supports Principle 2).

*   `[ ]` **Implement Technique Registration (`register_techniques`)**
    *   **What:** Replace placeholder with logic to populate `self.technique_registry`.
    *   **How:**
        1.  Decide on the source of truth: Should the orchestrator load techniques directly, or rely on `TechniqueMCPIntegrator`? (See Section 6 Architectural Clarity).
        2.  *If relying on Integrator:* Get the integrator instance (`self.mcp_registry.get_technique_integrator()`), call its `get_all_techniques()` method, and populate `self.technique_registry` with the returned dictionary.
        3.  *If loading directly:* Implement logic similar to `TechniqueMCPIntegrator._load_techniques` to discover and instantiate techniques from the `src/techniques/` directory.
    *   **Why:** The orchestrator needs access to technique instances and their metadata to generate strategies and execute steps.

*   `[ ]` **Implement Error Handling (`_handle_execution_error`)**
    *   **What:** Replace placeholder with concrete error recovery logic.
    *   **How:** Implement the logic described in the enhancement design doc: based on the `technique_name` and potentially the `error` type, decide whether to `skip`, `retry` (implement retry counter), `fallback` (call `_execute_fallback` which needs to check for and call `technique.fallback()`), or `halt` the workflow. Log actions clearly.
    *   **Why:** Makes the workflow more robust and resilient to failures in individual techniques or external dependencies.

## Section 2: Data Grounding & Domain Expertise

**Objective:** Ensure analysis is based on real data, not just LLM knowledge.

*   `[ ]` **Integrate Data Sources in Domain MCPs (`EconomicsMCP`, `GeopoliticsMCP`, etc.)** **(CRITICAL - NOT ADDRESSED)**
    *   **What:** Modify Domain MCPs to use external data APIs.
    *   **How:**
        1.  For `EconomicsMCP`: Integrate FRED API (e.g., using `requests` or a Python client library) to fetch indicators like GDP, CPI, unemployment rates based on context.
        2.  For `GeopoliticsMCP`: Integrate GDELT API or ACLED API to fetch recent event data relevant to the question/region.
        3.  In methods like `analyze_economic_trends`, first fetch data using the API client, then include this data *in the prompt* for the LLM analysis call.
        4.  Add API key management to the configuration system.
        5.  Implement caching for API calls.
    *   **Why:** Grounds domain-specific analysis in verifiable, real-world data, drastically improving accuracy, relevance, and robustness. Avoids relying on potentially outdated/biased LLM knowledge for factual domain information. (Addresses Principles 2, 5).

*   `[ ]` **Ground Technique LLM Calls**
    *   **What:** Ensure techniques use data from Research/Domain MCPs in their prompts.
    *   **How:** Modify techniques (e.g., `ACHTechnique`, `HypothesisTesting`) to retrieve relevant data/evidence snippets from `context.results` (populated by `ResearchMCP` or Domain MCPs). Inject these snippets into the LLM prompts, explicitly instructing the LLM to reason based *on the provided data*.
    *   **Why:** Connects analytical reasoning steps directly to the gathered evidence, improving grounding and traceability. (Addresses Principle 2).

## Section 3: Enhancing MCPs & Techniques

**Objective:** Implement missing specialized capabilities and refine existing components.

*   `[ ]` **Implement Enhancing MCPs** **(NOT ADDRESSED)**
    *   **What:** Create implementations for optional MCPs mentioned in techniques.
    *   **How:** Review techniques (ACH, Bias Detection, Synthesis, Uncertainty Mapping) for mentions (e.g., `evidence_extraction_mcp`, `cognitive_bias_mcp`, `synthesis_mcp`, `uncertainty_detection_mcp`, `meta_analysis_mcp`). Decide whether to build these as separate MCP classes in `src/mcps/` (likely involving specialized LLM prompts or potentially non-LLM logic/libraries) or integrate the logic directly into the techniques. Update techniques to use them.
    *   **Why:** Provides specialized capabilities that can significantly improve the quality of evidence handling, bias detection, synthesis, and uncertainty analysis.
    *   **Difficulty:** Medium to Hard, depending on the complexity of the desired capability.

*   `[ ]` **Add New Domain MCPs** **(NOT ADDRESSED)**
    *   **What:** Implement MCPs for additional domains (e.g., Technology, Social/Cultural, Scientific Literature).
    *   **How:** Create new classes inheriting from `BaseMCP`. **Prioritize integrating relevant data sources/APIs** for each new domain from the start. Design methods and LLM prompts specific to the domain's analytical needs.
    *   **Why:** Expands the system's applicability to a wider range of topics ("art to rocket science"). (Addresses Principle 5).
    *   **Difficulty:** Hard per domain.

*   `[ ]` **Refine Technique Logic & Prompts**
    *   **What:** Review and improve the core logic and LLM prompts within each technique in `src/techniques/`.
    *   **How:** Perform a detailed code review. Ensure prompts are clear, specific, and aligned with the technique's purpose. Check JSON parsing logic. Refine prompts for challenge techniques (`RedTeaming`, `ConsensusChallenge`) to demand specific, evidence-based critiques. Refine `BiasDetectionTechnique` prompts to check for LLM-specific biases.
    *   **Why:** Improves the quality, reliability, and effectiveness of each individual analytical step. (Addresses Principles 4, 6).
    *   **Difficulty:** Medium. Requires careful prompt engineering and understanding of each technique.

## Section 4: Uncertainty & Synthesis

**Objective:** Ensure robust handling of uncertainty and coherent final output.

*   `[ ]` **Connect Uncertainty Mapping to Synthesis**
    *   **What:** Ensure `SynthesisGenerationTechnique` uses the output of `UncertaintyMappingTechnique`.
    *   **How:** Modify `SynthesisGenerationTechnique` methods (`_generate_integrated_synthesis`, `_generate_final_assessment`) to check `context.results` for `UncertaintyMappingTechnique` output. If present, include key uncertainties and impact assessments in the prompts. Instruct the LLM to explicitly justify the final confidence level based on these uncertainties.
    *   **Why:** Ensures the final assessment accurately reflects the identified analytical limitations and avoids false certainty. (Addresses Principle 3).
    *   **Difficulty:** Medium.

*   `[ ]` **Refine Synthesis Technique Prompts**
    *   **What:** Improve prompts in `SynthesisGenerationTechnique`.
    *   **How:** Ensure prompts clearly instruct the LLM to integrate diverse inputs, resolve conflicts where possible, cite source techniques/evidence for key judgments, and clearly state remaining uncertainties.
    *   **Why:** Improves the coherence, traceability, and overall quality of the final analytical product. (Addresses Principles 5, 6).
    *   **Difficulty:** Medium.

*   `[ ]` **Clarify Final Analysis Step (`app.py`)**
    *   **What:** Evaluate the final `llama4_scout.analyze_with_cot` call in `app.py`.
    *   **How:** Determine its specific role relative to `SynthesisGenerationTechnique`. Is it redundant? Should it only do final formatting? Consider moving this logic into the orchestrator's `_generate_final_synthesis` or making `SynthesisGenerationTechnique` the definitive final step.
    *   **Why:** Ensures a clean, non-redundant final step in the workflow.
    *   **Difficulty:** Medium. Requires architectural decision and potential refactoring.

## Section 5: General System Improvements

**Objective:** Improve code quality, maintainability, and robustness.

*   `[ ]` **Implement Unit Tests** **(NOT ADDRESSED)**
    *   **What:** Add unit tests for all major components.
    *   **How:** Create test files (e.g., `tests/test_technique_xyz.py`, `tests/test_mcp_abc.py`). Use `pytest` and `pytest-mock` to test individual methods in isolation, mocking LLM calls, API requests, and other dependencies.
    *   **Why:** Catches bugs early, ensures components work as expected, facilitates refactoring. Essential for a complex system.
    *   **Difficulty:** Medium to Hard. Requires setting up test infrastructure and writing comprehensive tests.

*   `[ ]` **Enhance Configuration Management** **(NOT ADDRESSED)**
    *   **What:** Implement a better configuration system.
    *   **How:** Use Pydantic for config validation. Load from YAML/TOML files. Allow environment variable overrides. Create a central config object.
    *   **Why:** Improves maintainability, reduces errors from incorrect config, makes deployment easier.
    *   **Difficulty:** Medium.

*   `[ ]` **Refine LLM Integration Utilities (`utils/llm_integration.py`)**
    *   **What:** Improve robustness of LLM interaction code.
    *   **How:** Add better error handling (timeouts, API errors), implement retry logic, explore structured output parsing (e.g., Pydantic integration with `instructor` library if compatible, or robust JSON parsing with validation).
    *   **Why:** Makes LLM calls more reliable and easier to work with.
    *   **Difficulty:** Medium.

*   `[ ]` **Improve Logging**
    *   **What:** Add more detailed and structured logs.
    *   **How:** Log context IDs, technique parameters, orchestrator decisions (with reasons), and timing. Consider using structured logging libraries.
    *   **Why:** Improves debuggability and traceability. (Addresses Principle 6).
    *   **Difficulty:** Medium.

*   `[ ]` **Clarify Architectural Roles (`WorkflowOrchestratorMCP` vs. `TechniqueMCPIntegrator`)**
    *   **What:** Resolve overlapping responsibilities.
    *   **How:** Decide which class owns workflow creation/adaptation/execution logic (likely the Orchestrator). Refactor `TechniqueMCPIntegrator` to focus solely on loading, registering, and providing access to techniques and their metadata/dependencies, removing workflow logic from it. Update `MCPSystemIntegrator` to use the clarified roles.
    *   **Why:** Improves code clarity, maintainability, and reduces potential confusion or bugs from duplicated logic.
    *   **Difficulty:** Medium. Requires careful refactoring.

*   `[ ]` **Update Documentation**
    *   **What:** Update docstrings and design documents.
    *   **How:** Ensure docstrings are accurate. **Crucially,** update `src/architecture_analysis.md` and `src/mcp_architecture_design.md` (and potentially delete/archive `workflow_orchestrator_enhancement_design.md` once implemented) to reflect the *actual* implemented architecture. Update `README.md`.
    *   **Why:** Keeps documentation aligned with the code, essential for maintainability.
    *   **Difficulty:** Medium. Requires reviewing code and updating text.
