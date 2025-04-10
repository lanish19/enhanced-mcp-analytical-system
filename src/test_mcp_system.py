"""
Test script for the MCP system.

This module provides a test script for the MCP system, demonstrating its capabilities
and validating its functionality.
"""

import logging
import json
import time
import os
from typing import Dict, Any

from src.mcp_system_integrator import MCPSystemIntegrator
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mcp_system():
    """
    Test the MCP system with a sample question.
    """
    logger.info("Starting MCP system test")
    
    # Initialize system integrator with mock API keys
    # In a production environment, these would be real API keys
    config = {
        "perplexity_api_key": os.environ.get("PERPLEXITY_API_KEY", "mock_perplexity_api_key"),
        "groq_api_key": os.environ.get("GROQ_API_KEY", "mock_groq_api_key"),
        "brave_api_key": os.environ.get("BRAVE_API_KEY", "mock_brave_api_key"),
        "academic_api_key": os.environ.get("ACADEMIC_API_KEY", "mock_academic_api_key"),
        "fred_api_key": os.environ.get("FRED_API_KEY", "mock_fred_api_key"),
        "world_bank_api_key": os.environ.get("WORLD_BANK_API_KEY", "mock_world_bank_api_key"),
        "imf_api_key": os.environ.get("IMF_API_KEY", "mock_imf_api_key"),
        "gdelt_api_key": os.environ.get("GDELT_API_KEY", "mock_gdelt_api_key"),
        "acled_api_key": os.environ.get("ACLED_API_KEY", "mock_acled_api_key"),
        "redis_url": os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    }
    
    system_integrator = MCPSystemIntegrator(config)
    
    # Get system status
    logger.info("Getting system status")
    system_status = system_integrator.get_system_status()
    print("System Status:")
    print(json.dumps(system_status, indent=2))
    
    # Get MCP capabilities
    logger.info("Getting MCP capabilities")
    mcp_capabilities = system_integrator.get_mcp_capabilities()
    print("\nMCP Capabilities:")
    print(json.dumps(mcp_capabilities, indent=2))
    
    # Test question
    question = "What are the economic and geopolitical implications of rising energy prices on global inflation in the next 12 months?"
    
    # Analyze question
    logger.info(f"Analyzing question: {question}")
    result = system_integrator.analyze_question(question)
    
    # Print analysis result
    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2))
    
    # Test domain expertise
    logger.info("Testing domain expertise")
    
    # Economics domain
    economics_result = system_integrator.get_domain_expertise("economics", "analyze_economic_indicators", {
        "indicators": ["inflation", "energy_prices", "interest_rates"],
        "countries": ["USA", "EU", "China"],
        "time_period": "recent"
    })
    
    print("\nEconomics Domain Expertise:")
    print(json.dumps(economics_result, indent=2))
    
    # Geopolitics domain
    geopolitics_result = system_integrator.get_domain_expertise("geopolitics", "analyze_regional_stability", {
        "region": "middle_east",
        "time_period": "recent"
    })
    
    print("\nGeopolitics Domain Expertise:")
    print(json.dumps(geopolitics_result, indent=2))
    
    # Test workflow adaptation
    logger.info("Testing workflow adaptation")
    
    # Create context
    context = AnalysisContext(
        question=question,
        question_type="predictive",
        time_horizon="medium",
        uncertainty_level=0.6
    )
    
    # Create interim findings
    interim_findings = {
        "uncertainty_level": 0.8,
        "hypotheses": [
            {"statement": "Rising energy prices will significantly increase global inflation", "confidence": 0.7},
            {"statement": "Central bank interventions will mitigate inflation despite energy price increases", "confidence": 0.5}
        ],
        "domain_findings": {
            "economic": {
                "energy_price_trend": "increasing",
                "inflation_trend": "increasing",
                "interest_rate_trend": "increasing"
            },
            "geopolitical": {
                "energy_producing_regions": {
                    "middle_east": {
                        "stability": "fragile",
                        "production_trend": "stable"
                    },
                    "russia": {
                        "stability": "uncertain",
                        "production_trend": "decreasing"
                    }
                }
            }
        },
        "time_horizon": "medium"
    }
    
    # Adapt workflow
    adaptation_result = system_integrator.adapt_workflow(context.to_dict(), interim_findings)
    
    print("\nWorkflow Adaptation Result:")
    print(json.dumps(adaptation_result, indent=2))
    
    # Test technique execution
    logger.info("Testing technique execution")
    
    # Scenario triangulation
    scenario_result = system_integrator.execute_technique("scenario_triangulation", context.to_dict(), {
        "num_scenarios": 3,
        "focus_domains": ["economic", "geopolitical"]
    })
    
    print("\nScenario Triangulation Result:")
    print(json.dumps(scenario_result, indent=2))
    
    # Causal network analysis
    causal_result = system_integrator.execute_technique("causal_network_analysis", context.to_dict(), {
        "focus_domains": ["economic", "geopolitical"],
        "depth": "standard"
    })
    
    print("\nCausal Network Analysis Result:")
    print(json.dumps(causal_result, indent=2))
    
    logger.info("MCP system test completed successfully")
    return {
        "system_status": system_status,
        "mcp_capabilities": mcp_capabilities,
        "analysis_result": result,
        "domain_expertise": {
            "economics": economics_result,
            "geopolitics": geopolitics_result
        },
        "workflow_adaptation": adaptation_result,
        "technique_execution": {
            "scenario_triangulation": scenario_result,
            "causal_network_analysis": causal_result
        }
    }

if __name__ == "__main__":
    test_results = test_mcp_system()
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to test_results.json")
