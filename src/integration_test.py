"""
Integration test for the MCP architecture.

This module provides integration tests for the MCP architecture, testing the
interaction between all components including the WorkflowOrchestratorMCP,
TechniqueMCPIntegrator, and domain MCPs.
"""

import logging
import json
import time
from typing import Dict, Any

from src.mcp_system_integrator import MCPSystemIntegrator
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_end_to_end_analysis():
    """
    Test end-to-end analysis of a question using the MCP architecture.
    """
    logger.info("Starting end-to-end analysis test")
    
    # Initialize system integrator
    config = {
        # API keys would be provided in a real environment
        "perplexity_api_key": "mock_perplexity_api_key",
        "groq_api_key": "mock_groq_api_key",
        "brave_api_key": "mock_brave_api_key",
        "academic_api_key": "mock_academic_api_key",
        "fred_api_key": "mock_fred_api_key",
        "world_bank_api_key": "mock_world_bank_api_key",
        "imf_api_key": "mock_imf_api_key",
        "gdelt_api_key": "mock_gdelt_api_key",
        "acled_api_key": "mock_acled_api_key",
        "redis_url": "redis://localhost:6379/0"
    }
    
    system_integrator = MCPSystemIntegrator(config)
    
    # Test question
    question = "What are the economic implications of rising interest rates on housing markets in the next 12 months?"
    
    # Analyze question
    result = system_integrator.analyze_question(question)
    
    # Check result
    assert "workflow_result" in result, "Workflow result not found in analysis result"
    assert "preliminary_research" in result, "Preliminary research not found in analysis result"
    
    logger.info("End-to-end analysis test completed successfully")
    return result

def test_workflow_adaptation():
    """
    Test workflow adaptation based on interim findings.
    """
    logger.info("Starting workflow adaptation test")
    
    # Initialize system integrator
    config = {
        # API keys would be provided in a real environment
        "perplexity_api_key": "mock_perplexity_api_key",
        "groq_api_key": "mock_groq_api_key",
        "brave_api_key": "mock_brave_api_key",
        "academic_api_key": "mock_academic_api_key",
        "fred_api_key": "mock_fred_api_key",
        "world_bank_api_key": "mock_world_bank_api_key",
        "imf_api_key": "mock_imf_api_key",
        "gdelt_api_key": "mock_gdelt_api_key",
        "acled_api_key": "mock_acled_api_key",
        "redis_url": "redis://localhost:6379/0"
    }
    
    system_integrator = MCPSystemIntegrator(config)
    
    # Create context
    context = AnalysisContext(
        question="What are the economic implications of rising interest rates on housing markets in the next 12 months?",
        question_type="predictive",
        time_horizon="medium",
        uncertainty_level=0.6
    )
    
    # Create interim findings
    interim_findings = {
        "uncertainty_level": 0.8,
        "hypotheses": [
            {"statement": "Rising interest rates will significantly decrease housing demand", "confidence": 0.7},
            {"statement": "Rising interest rates will have minimal impact on housing demand due to supply constraints", "confidence": 0.6}
        ],
        "domain_findings": {
            "economic": {
                "interest_rate_trend": "increasing",
                "housing_supply": "constrained",
                "mortgage_applications": "decreasing"
            }
        }
    }
    
    # Adapt workflow
    adaptation_result = system_integrator.adapt_workflow(context.to_dict(), interim_findings)
    
    # Check result
    assert "add_techniques" in adaptation_result or "skip_techniques" in adaptation_result or "modify_parameters" in adaptation_result, "No adaptation recommendations found"
    
    logger.info("Workflow adaptation test completed successfully")
    return adaptation_result

def test_domain_expertise():
    """
    Test domain expertise from domain MCPs.
    """
    logger.info("Starting domain expertise test")
    
    # Initialize system integrator
    config = {
        # API keys would be provided in a real environment
        "perplexity_api_key": "mock_perplexity_api_key",
        "groq_api_key": "mock_groq_api_key",
        "brave_api_key": "mock_brave_api_key",
        "academic_api_key": "mock_academic_api_key",
        "fred_api_key": "mock_fred_api_key",
        "world_bank_api_key": "mock_world_bank_api_key",
        "imf_api_key": "mock_imf_api_key",
        "gdelt_api_key": "mock_gdelt_api_key",
        "acled_api_key": "mock_acled_api_key",
        "redis_url": "redis://localhost:6379/0"
    }
    
    system_integrator = MCPSystemIntegrator(config)
    
    # Test economics domain
    economics_result = system_integrator.get_domain_expertise("economics", "analyze_economic_indicators", {
        "indicators": ["gdp", "inflation", "interest_rates"],
        "countries": ["USA", "EU"],
        "time_period": "recent"
    })
    
    # Check economics result
    assert "error" not in economics_result, f"Error in economics domain expertise: {economics_result.get('error')}"
    
    # Test geopolitics domain
    geopolitics_result = system_integrator.get_domain_expertise("geopolitics", "assess_geopolitical_risk", {
        "countries": ["USA", "China", "Russia"],
        "time_horizon": "medium"
    })
    
    # Check geopolitics result
    assert "error" not in geopolitics_result, f"Error in geopolitics domain expertise: {geopolitics_result.get('error')}"
    
    logger.info("Domain expertise test completed successfully")
    return {
        "economics": economics_result,
        "geopolitics": geopolitics_result
    }

def test_technique_execution():
    """
    Test execution of specific analytical techniques.
    """
    logger.info("Starting technique execution test")
    
    # Initialize system integrator
    config = {
        # API keys would be provided in a real environment
        "perplexity_api_key": "mock_perplexity_api_key",
        "groq_api_key": "mock_groq_api_key",
        "brave_api_key": "mock_brave_api_key",
        "academic_api_key": "mock_academic_api_key",
        "fred_api_key": "mock_fred_api_key",
        "world_bank_api_key": "mock_world_bank_api_key",
        "imf_api_key": "mock_imf_api_key",
        "gdelt_api_key": "mock_gdelt_api_key",
        "acled_api_key": "mock_acled_api_key",
        "redis_url": "redis://localhost:6379/0"
    }
    
    system_integrator = MCPSystemIntegrator(config)
    
    # Create context
    context = AnalysisContext(
        question="What are the economic implications of rising interest rates on housing markets in the next 12 months?",
        question_type="predictive",
        time_horizon="medium",
        uncertainty_level=0.6
    )
    
    # Test scenario triangulation technique
    scenario_result = system_integrator.execute_technique("scenario_triangulation", context.to_dict(), {
        "num_scenarios": 3,
        "focus_domain": "economic"
    })
    
    # Check scenario result
    assert "error" not in scenario_result, f"Error in scenario triangulation technique: {scenario_result.get('error')}"
    assert "scenarios" in scenario_result, "No scenarios found in scenario triangulation result"
    
    # Test causal network analysis technique
    causal_result = system_integrator.execute_technique("causal_network_analysis", context.to_dict(), {
        "focus_domain": "economic",
        "depth": "standard"
    })
    
    # Check causal result
    assert "error" not in causal_result, f"Error in causal network analysis technique: {causal_result.get('error')}"
    assert "causal_relationships" in causal_result, "No causal relationships found in causal network analysis result"
    
    logger.info("Technique execution test completed successfully")
    return {
        "scenario_triangulation": scenario_result,
        "causal_network_analysis": causal_result
    }

def run_all_tests():
    """
    Run all integration tests.
    """
    logger.info("Running all integration tests")
    
    results = {}
    
    try:
        results["end_to_end_analysis"] = test_end_to_end_analysis()
        logger.info("End-to-end analysis test passed")
    except Exception as e:
        logger.error(f"End-to-end analysis test failed: {str(e)}")
        results["end_to_end_analysis"] = {"error": str(e)}
    
    try:
        results["workflow_adaptation"] = test_workflow_adaptation()
        logger.info("Workflow adaptation test passed")
    except Exception as e:
        logger.error(f"Workflow adaptation test failed: {str(e)}")
        results["workflow_adaptation"] = {"error": str(e)}
    
    try:
        results["domain_expertise"] = test_domain_expertise()
        logger.info("Domain expertise test passed")
    except Exception as e:
        logger.error(f"Domain expertise test failed: {str(e)}")
        results["domain_expertise"] = {"error": str(e)}
    
    try:
        results["technique_execution"] = test_technique_execution()
        logger.info("Technique execution test passed")
    except Exception as e:
        logger.error(f"Technique execution test failed: {str(e)}")
        results["technique_execution"] = {"error": str(e)}
    
    logger.info("All integration tests completed")
    return results

if __name__ == "__main__":
    results = run_all_tests()
    print(json.dumps(results, indent=2))
