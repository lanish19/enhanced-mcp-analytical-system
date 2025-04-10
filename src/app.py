"""
Main application file for the MCP-based analytical system.
This module integrates all components and provides the Streamlit interface.
"""

import streamlit as st
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional

# Import MCP components
from src.mcps.perplexity_sonar_mcp import PerplexitySonarMCP
from src.mcps.llama4_scout_mcp import Llama4ScoutMCP
from src.mcps.workflow_orchestrator_mcp import WorkflowOrchestratorMCP
from src.mcps.redis_context_store_mcp import RedisContextStoreMCP
from src.mcps.research_mcp import ResearchMCP
from src.mcps.economics_mcp import EconomicsMCP
from src.mcps.geopolitics_mcp import GeopoliticsMCP

# Import MCP registry
from src.mcp_registry import MCPRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPAnalyticalSystem:
    """
    Main class for the MCP-based analytical system.
    
    This class integrates all MCP components and provides the main interface
    for conducting analytical workflows.
    """
    
    def __init__(self):
        """Initialize the MCP-based analytical system."""
        logger.info("Initializing MCP Analytical System...")
        
        # Initialize MCP registry
        self.mcp_registry = MCPRegistry()
        
        # Initialize core MCPs
        self._initialize_mcps()
        
        # Register analytical techniques
        self._register_techniques()
        
        logger.info("MCP Analytical System initialized successfully")
    
    def _initialize_mcps(self):
        """Initialize all MCP components."""
        # Load configuration
        config = self._load_config()
        
        # Initialize infrastructure MCPs
        self.context_store = RedisContextStoreMCP(config.get("redis_context_store", {}))
        self.workflow_orchestrator = WorkflowOrchestratorMCP(config.get("workflow_orchestrator", {}))
        
        # Initialize research MCPs
        self.perplexity_sonar = PerplexitySonarMCP(config.get("perplexity_sonar", {}))
        self.research_mcp = ResearchMCP(config.get("research", {}))
        
        # Initialize domain MCPs
        self.economics_mcp = EconomicsMCP(config.get("economics", {}))
        self.geopolitics_mcp = GeopoliticsMCP(config.get("geopolitics", {}))
        
        # Initialize LLM MCP
        self.llama4_scout = Llama4ScoutMCP(config.get("llama4_scout", {}))
        
        # Register MCPs with registry
        self.mcp_registry.register_mcp("context_store", self.context_store)
        self.mcp_registry.register_mcp("workflow_orchestrator", self.workflow_orchestrator)
        self.mcp_registry.register_mcp("perplexity_sonar", self.perplexity_sonar)
        self.mcp_registry.register_mcp("research", self.research_mcp)
        self.mcp_registry.register_mcp("economics", self.economics_mcp)
        self.mcp_registry.register_mcp("geopolitics", self.geopolitics_mcp)
        self.mcp_registry.register_mcp("llama4_scout", self.llama4_scout)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or environment variables.
        
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Try to load from config file
        config_path = os.environ.get("MCP_CONFIG_PATH", "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {e}")
        
        # Load API keys from environment variables
        config.setdefault("perplexity_sonar", {})["perplexity_api_key"] = os.environ.get("PERPLEXITY_API_KEY")
        config.setdefault("llama4_scout", {})["groq_api_key"] = os.environ.get("GROQ_API_KEY")
        
        # Set default Redis configuration
        if "redis_context_store" not in config:
            config["redis_context_store"] = {
                "redis": {
                    "host": os.environ.get("REDIS_HOST", "localhost"),
                    "port": int(os.environ.get("REDIS_PORT", "6379")),
                    "db": int(os.environ.get("REDIS_DB", "0")),
                    "password": os.environ.get("REDIS_PASSWORD")
                }
            }
        
        return config
    
    def _register_techniques(self):
        """Register all analytical techniques with the MCP registry."""
        # Import all technique classes
        from src.techniques.scenario_triangulation_technique import ScenarioTriangulationTechnique
        from src.techniques.consensus_challenge_technique import ConsensusChallengeTechnique
        from src.techniques.multi_persona_technique import MultiPersonaTechnique
        from src.techniques.backward_reasoning_technique import BackwardReasoningTechnique
        from src.techniques.research_to_hypothesis_technique import ResearchToHypothesisTechnique
        from src.techniques.causal_network_analysis_technique import CausalNetworkAnalysisTechnique
        from src.techniques.key_assumptions_check_technique import KeyAssumptionsCheckTechnique
        from src.techniques.analysis_of_competing_hypotheses_technique import ACHTechnique
        from src.techniques.uncertainty_mapping_technique import UncertaintyMappingTechnique
        from src.techniques.red_teaming_technique import RedTeamingTechnique
        from src.techniques.premortem_analysis_technique import PremortemAnalysisTechnique
        from src.techniques.synthesis_generation_technique import SynthesisGenerationTechnique
        from src.techniques.cross_impact_analysis_technique import CrossImpactAnalysisTechnique
        from src.techniques.system_dynamics_modeling_technique import SystemDynamicsModelingTechnique
        from src.techniques.indicators_development_technique import IndicatorsDevelopmentTechnique
        from src.techniques.argument_mapping_technique import ArgumentMappingTechnique
        from src.techniques.bias_detection_technique import BiasDetectionTechnique
        from src.techniques.decision_tree_analysis_technique import DecisionTreeAnalysisTechnique
        from src.techniques.delphistic_forecasting_technique import DelphisticForecastingTechnique
        from src.techniques.historical_analogies_technique import HistoricalAnalogiesTechnique
        
        # Register all techniques
        self.mcp_registry.register_technique("scenario_triangulation", ScenarioTriangulationTechnique())
        self.mcp_registry.register_technique("consensus_challenge", ConsensusChallengeTechnique())
        self.mcp_registry.register_technique("multi_persona", MultiPersonaTechnique())
        self.mcp_registry.register_technique("backward_reasoning", BackwardReasoningTechnique())
        self.mcp_registry.register_technique("research_to_hypothesis", ResearchToHypothesisTechnique())
        self.mcp_registry.register_technique("causal_network_analysis", CausalNetworkAnalysisTechnique())
        self.mcp_registry.register_technique("key_assumptions_check", KeyAssumptionsCheckTechnique())
        self.mcp_registry.register_technique("analysis_of_competing_hypotheses", ACHTechnique())
        self.mcp_registry.register_technique("uncertainty_mapping", UncertaintyMappingTechnique())
        self.mcp_registry.register_technique("red_teaming", RedTeamingTechnique())
        self.mcp_registry.register_technique("premortem_analysis", PremortemAnalysisTechnique())
        self.mcp_registry.register_technique("synthesis_generation", SynthesisGenerationTechnique())
        self.mcp_registry.register_technique("cross_impact_analysis", CrossImpactAnalysisTechnique())
        self.mcp_registry.register_technique("system_dynamics_modeling", SystemDynamicsModelingTechnique())
        self.mcp_registry.register_technique("indicators_development", IndicatorsDevelopmentTechnique())
        self.mcp_registry.register_technique("argument_mapping", ArgumentMappingTechnique())
        self.mcp_registry.register_technique("bias_detection", BiasDetectionTechnique())
        self.mcp_registry.register_technique("decision_tree_analysis", DecisionTreeAnalysisTechnique())
        self.mcp_registry.register_technique("delphistic_forecasting", DelphisticForecastingTechnique())
        self.mcp_registry.register_technique("historical_analogies", HistoricalAnalogiesTechnique())
        
        logger.info(f"Registered {len(self.mcp_registry.techniques)} analytical techniques")
    
    def analyze_question(self, question: str, research_depth: str = "standard") -> Dict[str, Any]:
        """
        Analyze a question using the MCP-based analytical system.
        
        Args:
            question: The analytical question
            research_depth: Depth of preliminary research ('standard' or 'deep')
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing question: {question[:100]}...")
        
        # Step 1: Conduct preliminary research using Perplexity Sonar
        logger.info("Step 1: Conducting preliminary research...")
        research_results = self.perplexity_sonar.conduct_research_phase(question, research_depth)
        
        # Step 2: Extract insights and formulate initial hypotheses
        logger.info("Step 2: Extracting insights and formulating hypotheses...")
        insights = research_results.get("key_insights", [])
        hypotheses = research_results.get("initial_hypotheses", [])
        
        # Step 3: Determine optimal workflow
        logger.info("Step 3: Determining optimal workflow...")
        workflow_recommendations = research_results.get("workflow_recommendations", {})
        
        # Step 4: Create analysis context
        context = {
            "research_results": research_results,
            "insights": insights,
            "hypotheses": hypotheses,
            "workflow_recommendations": workflow_recommendations
        }
        
        # Step 5: Create and execute workflow
        logger.info("Step 5: Creating and executing workflow...")
        workflow = self.workflow_orchestrator.create_workflow(question, context)
        workflow_results = self.workflow_orchestrator.execute_workflow(workflow, self.context_store)
        
        # Step 6: Perform final analysis with Llama 4 Scout
        logger.info("Step 6: Performing final analysis...")
        analysis_type = self._determine_analysis_type(workflow_recommendations)
        final_analysis = self.llama4_scout.analyze_with_cot(
            question, 
            {"workflow_results": workflow_results, "research_results": research_results},
            analysis_type
        )
        
        # Step 7: Quantify uncertainty and detect bias
        logger.info("Step 7: Quantifying uncertainty and detecting bias...")
        uncertainty = self.llama4_scout.quantify_uncertainty(final_analysis)
        bias = self.llama4_scout.detect_bias(final_analysis)
        
        # Combine all results
        analysis_results = {
            "question": question,
            "timestamp": time.time(),
            "research_phase": research_results,
            "workflow_phase": workflow_results,
            "final_analysis": final_analysis,
            "uncertainty_assessment": uncertainty,
            "bias_assessment": bias
        }
        
        # Store complete analysis in context store
        analysis_id = self.context_store.create_analysis_session(question)
        self.context_store.update_context("analysis_session", analysis_id, analysis_results)
        
        logger.info(f"Analysis completed with ID: {analysis_id}")
        return analysis_results
    
    def _determine_analysis_type(self, workflow_recommendations: Dict[str, Any]) -> str:
        """
        Determine the appropriate analysis type based on workflow recommendations.
        
        Args:
            workflow_recommendations: Workflow recommendations from research phase
            
        Returns:
            Analysis type for Llama 4 Scout
        """
        question_types = workflow_recommendations.get("question_types", [])
        
        if "predictive" in question_types:
            return "predictive_analysis"
        elif "causal" in question_types:
            return "causal_analysis"
        elif "evaluative" in question_types:
            return "evaluative_analysis"
        elif "strategic" in question_types:
            return "strategic_analysis"
        else:
            return "general_analysis"
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of previous analyses.
        
        Args:
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis summaries
        """
        sessions = self.context_store.get_all_sessions()
        
        # Sort by timestamp (newest first) and limit
        sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return sessions[:limit]
    
    def get_analysis_by_id(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get a specific analysis by ID.
        
        Args:
            analysis_id: Analysis session ID
            
        Returns:
            Complete analysis results
        """
        return self.context_store.get_context("analysis_session", analysis_id)


# Streamlit application
def main():
    st.set_page_config(
        page_title="MCP Analytical System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize system
    if "system" not in st.session_state:
        st.session_state.system = MCPAnalyticalSystem()
        st.session_state.analysis_results = None
        st.session_state.analysis_history = []
        st.session_state.selected_analysis_id = None
    
    # Sidebar
    st.sidebar.title("MCP Analytical System")
    st.sidebar.markdown("---")
    
    # Analysis history in sidebar
    st.sidebar.subheader("Analysis History")
    if st.sidebar.button("Refresh History"):
        st.session_state.analysis_history = st.session_state.system.get_analysis_history()
    
    if not st.session_state.analysis_history:
        st.session_state.analysis_history = st.session_state.system.get_analysis_history()
    
    for session in st.session_state.analysis_history:
        if st.sidebar.button(f"{session.get('question', 'Unknown')[:50]}...", key=f"history_{session.get('session_id')}"):
            st.session_state.selected_analysis_id = session.get("session_id")
            st.session_state.analysis_results = st.session_state.system.get_analysis_by_id(session.get("session_id"))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("MCP Analytical System uses modular cognitive processors to analyze complex questions.")
    
    # Main content
    st.title("MCP Analytical System")
    st.markdown("### Ask a complex analytical question")
    
    # Input form
    with st.form("analysis_form"):
        question = st.text_area("Enter your question:", height=100)
        research_depth = st.selectbox("Research depth:", ["standard", "deep"])
        submitted = st.form_submit_button("Analyze")
    
    # Process form submission
    if submitted and question:
        with st.spinner("Analyzing your question... This may take a few minutes."):
            st.session_state.analysis_results = st.session_state.system.analyze_question(question, research_depth)
            st.session_state.selected_analysis_id = None  # Clear selected analysis
            st.session_state.analysis_history = st.session_state.system.get_analysis_history()  # Refresh history
    
    # Display selected analysis from history
    if st.session_state.selected_analysis_id and st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)
    
    # Display new analysis results
    elif st.session_state.analysis_results and not st.session_state.selected_analysis_id:
        display_analysis_results(st.session_state.analysis_results)


def display_analysis_results(results: Dict[str, Any]):
    """Display analysis results in the Streamlit interface."""
    st.markdown("## Analysis Results")
    st.markdown(f"### Question: {results.get('question', 'Unknown')}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary", "Research Phase", "Workflow Phase", "Final Analysis", "Uncertainty & Bias"
    ])
    
    # Tab 1: Summary
    with tab1:
        st.markdown("### Analysis Summary")
        
        # Get final analysis
        final_analysis = results.get("final_analysis", {})
        
        # Display comprehensive answer
        st.markdown("#### Comprehensive Answer")
        st.markdown(final_analysis.get("comprehensive_answer", 
                   final_analysis.get("conclusion", "No comprehensive answer available.")))
        
        # Display key findings
        st.markdown("#### Key Findings")
        st.markdown(final_analysis.get("key_findings", "No key findings available."))
        
        # Display uncertainty summary
        uncertainty = results.get("uncertainty_assessment", {})
        st.markdown("#### Uncertainty Assessment")
        st.markdown(f"**Overall Uncertainty:** {uncertainty.get('overall_uncertainty', 'Unknown')}")
        st.markdown(f"**Confidence Score:** {uncertainty.get('confidence_score', 'Unknown')}")
        
        # Display next steps
        st.markdown("#### Recommended Next Steps")
        st.markdown(final_analysis.get("next_steps", "No next steps available."))
    
    # Tab 2: Research Phase
    with tab2:
        st.markdown("### Research Phase")
        
        research_phase = results.get("research_phase", {})
        research_results = research_phase.get("research_results", {})
        
        # Display key findings from research
        st.markdown("#### Key Findings")
        st.markdown(research_results.get("key_findings", "No key findings available."))
        
        # Display background context
        st.markdown("#### Background Context")
        st.markdown(research_results.get("background_context", "No background context available."))
        
        # Display key insights
        st.markdown("#### Key Insights")
        insights = research_phase.get("key_insights", [])
        for i, insight in enumerate(insights):
            with st.expander(f"Insight {i+1}: {insight.get('insight', '')[:100]}..."):
                st.markdown(f"**Confidence:** {insight.get('confidence', 'Unknown')}")
                st.markdown(f"**Supporting Evidence:** {insight.get('supporting_evidence', 'None')}")
                st.markdown(f"**Contradictory Evidence:** {insight.get('contradictory_evidence', 'None')}")
        
        # Display initial hypotheses
        st.markdown("#### Initial Hypotheses")
        hypotheses = research_phase.get("initial_hypotheses", [])
        for i, hypothesis in enumerate(hypotheses):
            with st.expander(f"Hypothesis {i+1}: {hypothesis.get('hypothesis', '')[:100]}..."):
                st.markdown(f"**Confidence:** {hypothesis.get('confidence', 'Unknown')}")
                st.markdown(f"**Supporting Evidence:** {hypothesis.get('supporting_evidence', 'None')}")
                st.markdown(f"**Evidence Gaps:** {hypothesis.get('evidence_gaps', 'None')}")
        
        # Display sources
        st.markdown("#### Sources")
        sources = research_results.get("sources", [])
        for source in sources:
            st.markdown(f"- {source.get('title', 'Unknown source')}")
    
    # Tab 3: Workflow Phase
    with tab3:
        st.markdown("### Workflow Phase")
        
        workflow_phase = results.get("workflow_phase", {})
        
        # Display workflow strategy
        st.markdown("#### Workflow Strategy")
        workflow_recommendations = research_phase.get("workflow_recommendations", {})
        st.markdown(f"**Question Types:** {', '.join(workflow_recommendations.get('question_types', ['Unknown']))}")
        st.markdown(f"**Domains:** {', '.join(workflow_recommendations.get('domains', ['Unknown']))}")
        st.markdown(f"**Complexity:** {workflow_recommendations.get('complexity', 'Unknown')}")
        st.markdown(f"**Uncertainty Level:** {workflow_recommendations.get('uncertainty_level', 'Unknown')}")
        st.markdown(f"**Workflow Approach:** {workflow_recommendations.get('workflow_approach', 'Unknown')}")
        
        # Display techniques used
        st.markdown("#### Techniques Used")
        techniques = workflow_phase.get("techniques_used", [])
        for technique in techniques:
            st.markdown(f"- {technique}")
        
        # Display workflow steps
        st.markdown("#### Workflow Steps")
        steps = workflow_phase.get("steps", [])
        for i, step in enumerate(steps):
            with st.expander(f"Step {i+1}: {step.get('name', 'Unknown step')}"):
                st.markdown(f"**Technique:** {step.get('technique', 'Unknown')}")
                st.markdown(f"**Status:** {step.get('status', 'Unknown')}")
                if step.get("outputs"):
                    st.markdown("**Outputs:**")
                    st.json(step.get("outputs"))
    
    # Tab 4: Final Analysis
    with tab4:
        st.markdown("### Final Analysis")
        
        final_analysis = results.get("final_analysis", {})
        
        # Display initial assessment
        st.markdown("#### Initial Assessment")
        st.markdown(final_analysis.get("initial_assessment", "No initial assessment available."))
        
        # Display detailed analysis
        st.markdown("#### Detailed Analysis")
        detailed_analysis = final_analysis.get("detailed_analysis", {})
        for key, value in detailed_analysis.items():
            with st.expander(f"{key.replace('_', ' ').title()}"):
                st.markdown(value)
        
        # Display conclusion
        st.markdown("#### Conclusion")
        st.markdown(final_analysis.get("conclusion", "No conclusion available."))
    
    # Tab 5: Uncertainty & Bias
    with tab5:
        st.markdown("### Uncertainty & Bias Assessment")
        
        # Display uncertainty assessment
        st.markdown("#### Uncertainty Assessment")
        uncertainty = results.get("uncertainty_assessment", {})
        st.markdown(f"**Overall Uncertainty:** {uncertainty.get('overall_uncertainty', 'Unknown')}")
        st.markdown(f"**Confidence Score:** {uncertainty.get('confidence_score', 'Unknown')}")
        
        st.markdown("**Uncertainty Factors:**")
        for factor in uncertainty.get("uncertainty_factors", []):
            st.markdown(f"- {factor}")
        
        st.markdown("**Component Reliability:**")
        component_reliability = uncertainty.get("component_reliability", {})
        for component, reliability in component_reliability.items():
            with st.expander(f"{component.replace('_', ' ').title()}"):
                st.markdown(f"**Reliability:** {reliability.get('reliability', 'Unknown')}")
                st.markdown(f"**Confidence Score:** {reliability.get('confidence_score', 'Unknown')}")
                st.markdown("**Key Uncertainties:**")
                for uncertainty_item in reliability.get("key_uncertainties", []):
                    st.markdown(f"- {uncertainty_item}")
        
        # Display bias assessment
        st.markdown("#### Bias Assessment")
        bias = results.get("bias_assessment", {})
        st.markdown(f"**Bias Detected:** {'Yes' if bias.get('bias_detected', False) else 'No'}")
        st.markdown(f"**Bias Assessment:** {bias.get('bias_assessment', 'No bias assessment available.')}")
        
        st.markdown("**Potential Biases:**")
        potential_biases = bias.get("potential_biases", [])
        for bias_item in potential_biases:
            with st.expander(f"{bias_item.get('bias_type', 'Unknown bias')}"):
                st.markdown(f"**Description:** {bias_item.get('description', 'No description available.')}")
                st.markdown(f"**Evidence:** {bias_item.get('evidence', 'No evidence available.')}")
                st.markdown(f"**Severity:** {bias_item.get('severity', 'Unknown')}")
                st.markdown(f"**Mitigation Suggestion:** {bias_item.get('mitigation_suggestion', 'No suggestion available.')}")


if __name__ == "__main__":
    main()
