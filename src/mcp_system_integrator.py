"""
MCP System Integrator for connecting all components of the MCP architecture.

This module provides the MCPSystemIntegrator class that serves as the central
integration point for all components of the MCP architecture, including the
WorkflowOrchestratorMCP, TechniqueMCPIntegrator, and all other MCPs.
"""

import logging
from typing import Dict, List, Any, Optional
import json
import time

from src.base_mcp import BaseMCP
from src.mcp_registry import MCPRegistry
from src.analysis_context import AnalysisContext
from src.analysis_strategy import AnalysisStrategy
from src.technique_mcp_integrator import TechniqueMCPIntegrator
from src.mcps.workflow_orchestrator_mcp import WorkflowOrchestratorMCP
from src.mcps.perplexity_sonar_mcp import PerplexitySonarMCP
from src.mcps.llama4_scout_mcp import Llama4ScoutMCP
from src.mcps.redis_context_store_mcp import RedisContextStoreMCP
from src.mcps.research_mcp import ResearchMCP
from src.mcps.economics_mcp import EconomicsMCP
from src.mcps.geopolitics_mcp import GeopoliticsMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPSystemIntegrator:
    """
    System integrator class that connects all components of the MCP architecture.
    
    This class serves as the central integration point for all components of the MCP
    architecture, providing a unified interface for interacting with the system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the MCPSystemIntegrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.mcp_registry = MCPRegistry()
        self.initialize_mcps()
        self.technique_integrator = TechniqueMCPIntegrator(self.mcp_registry)
        self.workflow_orchestrator = self.mcp_registry.get_mcp("workflow_orchestrator")
        
        logger.info("Initialized MCPSystemIntegrator")
    
    def initialize_mcps(self):
        """
        Initialize all MCPs and register them with the MCP registry.
        """
        logger.info("Initializing MCPs")
        
        # Initialize core MCPs
        perplexity_sonar = PerplexitySonarMCP(
            api_key=self.config.get("perplexity_api_key")
        )
        self.mcp_registry.register_mcp(perplexity_sonar)
        
        llama4_scout = Llama4ScoutMCP(
            api_key=self.config.get("groq_api_key")
        )
        self.mcp_registry.register_mcp(llama4_scout)
        
        redis_context_store = RedisContextStoreMCP(
            redis_url=self.config.get("redis_url")
        )
        self.mcp_registry.register_mcp(redis_context_store)
        
        research_mcp = ResearchMCP(
            brave_api_key=self.config.get("brave_api_key"),
            academic_api_key=self.config.get("academic_api_key")
        )
        self.mcp_registry.register_mcp(research_mcp)
        
        # Initialize domain MCPs
        economics_mcp = EconomicsMCP(
            api_keys={
                "fred_api_key": self.config.get("fred_api_key"),
                "world_bank_api_key": self.config.get("world_bank_api_key"),
                "imf_api_key": self.config.get("imf_api_key")
            }
        )
        self.mcp_registry.register_mcp(economics_mcp)
        
        geopolitics_mcp = GeopoliticsMCP(
            api_keys={
                "gdelt_api_key": self.config.get("gdelt_api_key"),
                "acled_api_key": self.config.get("acled_api_key")
            }
        )
        self.mcp_registry.register_mcp(geopolitics_mcp)
        
        # Initialize workflow orchestrator MCP (must be last as it depends on other MCPs)
        workflow_orchestrator = WorkflowOrchestratorMCP(
            mcp_registry=self.mcp_registry
        )
        self.mcp_registry.register_mcp(workflow_orchestrator)
        
        logger.info(f"Initialized {len(self.mcp_registry.get_all_mcps())} MCPs")
    
    def analyze_question(self, question: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a question using the MCP architecture.
        
        Args:
            question: The question to analyze
            additional_context: Optional additional context for the analysis
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing question: {question}")
        
        # Create analysis context
        context = AnalysisContext(
            question=question,
            additional_context=additional_context
        )
        
        # Perform preliminary research using Perplexity Sonar
        logger.info("Performing preliminary research with Perplexity Sonar")
        perplexity_sonar = self.mcp_registry.get_mcp("perplexity_sonar")
        preliminary_research = perplexity_sonar.process({
            "operation": "research",
            "query": question,
            "depth": "standard"
        })
        
        # Update context with preliminary research
        context.update_from_preliminary_research(preliminary_research)
        
        # Determine analysis strategy
        logger.info("Determining analysis strategy")
        strategy = AnalysisStrategy.from_context(context)
        
        # Execute workflow using WorkflowOrchestratorMCP
        logger.info("Executing workflow")
        workflow_result = self.workflow_orchestrator.process({
            "operation": "execute_workflow",
            "context": context.to_dict(),
            "strategy": strategy.to_dict()
        })
        
        # Compile final results
        results = {
            "question": question,
            "preliminary_research": preliminary_research.get("summary"),
            "workflow_result": workflow_result,
            "context": context.to_dict(),
            "strategy": strategy.to_dict(),
            "timestamp": time.time()
        }
        
        return results
    
    def get_mcp_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of all registered MCPs.
        
        Returns:
            Dictionary of MCP capabilities
        """
        capabilities = {}
        
        for mcp_name, mcp in self.mcp_registry.get_all_mcps().items():
            capabilities[mcp_name] = mcp.get_capabilities()
        
        return capabilities
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the MCP system.
        
        Returns:
            System status information
        """
        # Get MCP statuses
        mcp_statuses = {}
        for mcp_name, mcp in self.mcp_registry.get_all_mcps().items():
            try:
                # Basic connectivity check
                if hasattr(mcp, "check_status"):
                    status = mcp.check_status()
                else:
                    status = {"status": "unknown"}
                
                mcp_statuses[mcp_name] = status
            except Exception as e:
                mcp_statuses[mcp_name] = {"status": "error", "error": str(e)}
        
        # Get technique integrator status
        technique_integrator_status = {
            "execution_history_count": len(self.technique_integrator.get_execution_history()),
            "technique_mcp_mappings_count": len(self.technique_integrator.technique_mcp_mappings),
            "technique_dependencies_count": len(self.technique_integrator.technique_dependencies),
            "technique_complementary_pairs_count": len(self.technique_integrator.technique_complementary_pairs)
        }
        
        # Compile system status
        system_status = {
            "mcps": mcp_statuses,
            "technique_integrator": technique_integrator_status,
            "mcp_count": len(self.mcp_registry.get_all_mcps()),
            "timestamp": time.time()
        }
        
        return system_status
    
    def execute_technique(self, technique_name: str, context: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a specific analytical technique.
        
        Args:
            technique_name: Name of the technique to execute
            context: Analysis context
            parameters: Optional parameters for technique execution
            
        Returns:
            Technique execution results
        """
        logger.info(f"Executing technique: {technique_name}")
        
        # Convert context dict to AnalysisContext object if needed
        if isinstance(context, dict):
            context_obj = AnalysisContext.from_dict(context)
        else:
            context_obj = context
        
        # Execute technique through workflow orchestrator
        result = self.workflow_orchestrator.process({
            "operation": "execute_technique",
            "technique": technique_name,
            "context": context_obj.to_dict(),
            "parameters": parameters or {}
        })
        
        return result
    
    def adapt_workflow(self, context: Dict[str, Any], interim_findings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt the workflow based on interim findings.
        
        Args:
            context: Analysis context
            interim_findings: Interim findings from executed techniques
            
        Returns:
            Adaptation recommendations
        """
        logger.info("Adapting workflow based on interim findings")
        
        # Convert context dict to AnalysisContext object if needed
        if isinstance(context, dict):
            context_obj = AnalysisContext.from_dict(context)
        else:
            context_obj = context
        
        # Adapt workflow through workflow orchestrator
        result = self.workflow_orchestrator.process({
            "operation": "adapt_workflow",
            "context": context_obj.to_dict(),
            "interim_findings": interim_findings
        })
        
        return result
    
    def get_domain_expertise(self, domain: str, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get domain expertise from a domain MCP.
        
        Args:
            domain: Domain name (e.g., "economics", "geopolitics")
            operation: Operation to perform
            parameters: Operation parameters
            
        Returns:
            Domain expertise results
        """
        logger.info(f"Getting domain expertise: {domain}.{operation}")
        
        # Check if domain MCP exists
        if not self.mcp_registry.has_mcp(domain):
            error_msg = f"Domain MCP not found: {domain}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get domain MCP
        domain_mcp = self.mcp_registry.get_mcp(domain)
        
        # Process operation
        result = domain_mcp.process({
            "operation": operation,
            **parameters
        })
        
        return result
    
    def perform_research(self, query: str, depth: str = "standard") -> Dict[str, Any]:
        """
        Perform research using the research MCP.
        
        Args:
            query: Research query
            depth: Research depth ("quick", "standard", or "deep")
            
        Returns:
            Research results
        """
        logger.info(f"Performing research: {query}")
        
        # Get research MCP
        research_mcp = self.mcp_registry.get_mcp("research")
        
        # Process research operation
        result = research_mcp.process({
            "operation": "search",
            "query": query,
            "depth": depth
        })
        
        return result
    
    def store_context(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Store a value in the context store.
        
        Args:
            key: Context key
            value: Context value
            
        Returns:
            Storage result
        """
        logger.info(f"Storing context: {key}")
        
        # Get context store MCP
        context_store = self.mcp_registry.get_mcp("redis_context_store")
        
        # Process store operation
        result = context_store.process({
            "operation": "store",
            "key": key,
            "value": value
        })
        
        return result
    
    def retrieve_context(self, key: str) -> Dict[str, Any]:
        """
        Retrieve a value from the context store.
        
        Args:
            key: Context key
            
        Returns:
            Retrieval result
        """
        logger.info(f"Retrieving context: {key}")
        
        # Get context store MCP
        context_store = self.mcp_registry.get_mcp("redis_context_store")
        
        # Process retrieve operation
        result = context_store.process({
            "operation": "retrieve",
            "key": key
        })
        
        return result
