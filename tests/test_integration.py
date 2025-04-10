"""
Integration tests for the MCP Analytical System.
"""

import unittest
import json
import time
from unittest.mock import MagicMock, patch

# Import components to test
from src.workflow_orchestrator_mcp import WorkflowOrchestratorMCP
from src.analysis_context import AnalysisContext
from src.analysis_strategy import AnalysisStrategy
from src.technique_mcp_integrator import TechniqueMCPIntegrator
from src.mcp_registry import MCPRegistry
from src.config import get_config, configure_logging

class TestEndToEndWorkflow(unittest.TestCase):
    """Test cases for end-to-end workflow execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging
        configure_logging()
        
        # Create real registry
        self.registry = MCPRegistry()
        
        # Create mock MCPs for testing
        self.mock_perplexity_sonar = MagicMock()
        self.mock_perplexity_sonar.process.return_value = {
            "result": {
                "insights": ["Insight 1", "Insight 2"],
                "hypotheses": ["Hypothesis 1", "Hypothesis 2"],
                "recommendations": ["Recommendation 1"]
            }
        }
        
        self.mock_llama4_scout = MagicMock()
        self.mock_llama4_scout.process.return_value = {
            "result": json.dumps({
                "type": "predictive",
                "domains": ["economics", "technology"],
                "complexity": "medium",
                "uncertainty": "medium",
                "time_horizon": "medium-term",
                "potential_biases": ["confirmation bias"]
            })
        }
        
        # Register mock MCPs
        self.registry.register_mcp("perplexity_sonar", self.mock_perplexity_sonar)
        self.registry.register_mcp("llama4_scout", self.mock_llama4_scout)
        
        # Create mock technique integrator
        self.mock_technique_integrator = MagicMock()
        self.mock_technique_integrator.get_all_techniques.return_value = {
            "research_to_hypothesis": MagicMock(),
            "synthesis_generation": MagicMock()
        }
        self.mock_technique_integrator.execute_step.return_value = {
            "status": "completed",
            "result": {"key": "value"}
        }
        
        # Register mock technique integrator
        self.registry.register_technique_integrator(self.mock_technique_integrator)
        
        # Create WorkflowOrchestratorMCP instance with registry
        self.orchestrator = WorkflowOrchestratorMCP(self.registry)
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow execution."""
        # Create context with question
        context = AnalysisContext()
        context.add("question", "What are the potential economic impacts of quantum computing over the next decade?")
        
        # Process the question
        result = self.orchestrator.process({"context": context})
        
        # Verify orchestrator returned success
        self.assertEqual(result["status"], "success")
        
        # Verify preliminary research was conducted
        self.mock_perplexity_sonar.process.assert_called_once()
        self.assertIsNotNone(context.get("preliminary_research_insights"))
        
        # Verify question analysis was performed
        self.mock_llama4_scout.process.assert_called_once()
        self.assertIsNotNone(context.get("question_analysis"))
        
        # Verify strategy was generated
        self.assertIsNotNone(context.get("strategy"))
        
        # Verify techniques were executed
        self.assertTrue(self.mock_technique_integrator.execute_step.called)
        
        # Verify events were recorded
        events = context.get_events()
        self.assertTrue(len(events) > 0)
        
        # Verify workflow completion event was recorded
        completion_events = [e for e in events if "Workflow completed" in str(e)]
        self.assertTrue(len(completion_events) > 0)
    
    def test_workflow_with_adaptation(self):
        """Test workflow execution with adaptation."""
        # Create context with question
        context = AnalysisContext()
        context.add("question", "What are the potential economic impacts of quantum computing over the next decade?")
        
        # Set up mock technique integrator to trigger adaptation
        original_execute_step = self.mock_technique_integrator.execute_step
        
        def execute_step_with_adaptation(step, context):
            # First call returns conflicting evidence to trigger adaptation
            if step["technique"] == "research_to_hypothesis":
                return {
                    "status": "completed",
                    "conflicting_evidence_found": True,
                    "result": {"key": "value"}
                }
            # Other calls use original mock
            return original_execute_step(step, context)
        
        self.mock_technique_integrator.execute_step.side_effect = execute_step_with_adaptation
        
        # Process the question
        result = self.orchestrator.process({"context": context})
        
        # Verify orchestrator returned success
        self.assertEqual(result["status"], "success")
        
        # Verify strategy was adapted
        adaptation_events = [e for e in context.get_events() if "Strategy adapted" in str(e)]
        self.assertTrue(len(adaptation_events) > 0)
        
        # Verify adapted strategy is different from original
        strategy = context.get("strategy")
        self.assertTrue("Adapted" in strategy.name)
    
    def test_workflow_with_error_handling(self):
        """Test workflow execution with error handling."""
        # Create context with question
        context = AnalysisContext()
        context.add("question", "What are the potential economic impacts of quantum computing over the next decade?")
        
        # Set up mock technique integrator to simulate errors and recovery
        call_count = 0
        
        def execute_step_with_errors(step, context):
            nonlocal call_count
            call_count += 1
            
            # First call to research_to_hypothesis fails
            if step["technique"] == "research_to_hypothesis" and call_count == 1:
                raise Exception("Simulated error")
            
            # Subsequent calls succeed
            return {
                "status": "completed",
                "result": {"key": "value"}
            }
        
        self.mock_technique_integrator.execute_step.side_effect = execute_step_with_errors
        
        # Process the question
        result = self.orchestrator.process({"context": context})
        
        # Verify orchestrator returned success despite error
        self.assertEqual(result["status"], "success")
        
        # Verify error was handled
        error_events = context.get_events("error")
        self.assertTrue(len(error_events) > 0)
        
        # Verify retry was attempted
        retry_events = [e for e in context.get_events() if "Retry" in str(e)]
        self.assertTrue(len(retry_events) > 0)
        
        # Verify workflow completed successfully after recovery
        completion_events = [e for e in context.get_events() if "Workflow completed" in str(e)]
        self.assertTrue(len(completion_events) > 0)


class TestMCPRegistry(unittest.TestCase):
    """Test cases for the MCPRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = MCPRegistry()
    
    def test_register_and_get_mcp(self):
        """Test registering and retrieving MCPs."""
        # Create mock MCP
        mock_mcp = MagicMock()
        mock_mcp.name = "test_mcp"
        
        # Register MCP
        self.registry.register_mcp("test_mcp", mock_mcp)
        
        # Get MCP
        retrieved_mcp = self.registry.get_mcp("test_mcp")
        
        # Verify MCP was retrieved
        self.assertEqual(retrieved_mcp, mock_mcp)
        
        # Get non-existent MCP
        with self.assertRaises(ValueError):
            self.registry.get_mcp("non_existent")
    
    def test_register_and_get_technique_integrator(self):
        """Test registering and retrieving technique integrator."""
        # Create mock technique integrator
        mock_integrator = MagicMock()
        
        # Register technique integrator
        self.registry.register_technique_integrator(mock_integrator)
        
        # Get technique integrator
        retrieved_integrator = self.registry.get_technique_integrator()
        
        # Verify technique integrator was retrieved
        self.assertEqual(retrieved_integrator, mock_integrator)
    
    def test_get_all_mcps(self):
        """Test getting all registered MCPs."""
        # Create mock MCPs
        mock_mcp1 = MagicMock()
        mock_mcp1.name = "test_mcp1"
        
        mock_mcp2 = MagicMock()
        mock_mcp2.name = "test_mcp2"
        
        # Register MCPs
        self.registry.register_mcp("test_mcp1", mock_mcp1)
        self.registry.register_mcp("test_mcp2", mock_mcp2)
        
        # Get all MCPs
        all_mcps = self.registry.get_all_mcps()
        
        # Verify all MCPs were retrieved
        self.assertEqual(len(all_mcps), 2)
        self.assertIn("test_mcp1", all_mcps)
        self.assertIn("test_mcp2", all_mcps)
    
    def test_mcp_capabilities(self):
        """Test getting MCP capabilities."""
        # Create mock MCP with capabilities
        mock_mcp = MagicMock()
        mock_mcp.name = "test_mcp"
        mock_mcp.get_capabilities.return_value = ["capability1", "capability2"]
        
        # Register MCP
        self.registry.register_mcp("test_mcp", mock_mcp)
        
        # Get MCPs with capability
        mcps_with_capability = self.registry.get_mcps_with_capability("capability1")
        
        # Verify MCP was retrieved
        self.assertEqual(len(mcps_with_capability), 1)
        self.assertEqual(mcps_with_capability[0], mock_mcp)
        
        # Get MCPs with non-existent capability
        mcps_with_capability = self.registry.get_mcps_with_capability("non_existent")
        
        # Verify no MCPs were retrieved
        self.assertEqual(len(mcps_with_capability), 0)


class TestTechniqueMCPIntegrator(unittest.TestCase):
    """Test cases for the TechniqueMCPIntegrator class."""
    
    @patch('src.technique_mcp_integrator.importlib.import_module')
    def test_load_technique(self, mock_import_module):
        """Test loading a technique."""
        # Create mock technique module
        mock_technique = MagicMock()
        mock_technique.AnalyticalTechnique.return_value = MagicMock()
        
        # Set up mock import_module to return mock technique
        mock_import_module.return_value = mock_technique
        
        # Create integrator
        integrator = TechniqueMCPIntegrator()
        
        # Load technique
        technique = integrator.load_technique("test_technique")
        
        # Verify technique was loaded
        self.assertIsNotNone(technique)
        
        # Verify import_module was called with correct path
        mock_import_module.assert_called_with("src.techniques.test_technique")
    
    @patch('src.technique_mcp_integrator.importlib.import_module')
    def test_execute_step(self, mock_import_module):
        """Test executing a technique step."""
        # Create mock technique
        mock_technique = MagicMock()
        mock_technique.execute.return_value = {"status": "completed", "result": "test_result"}
        
        # Create mock technique module
        mock_module = MagicMock()
        mock_module.AnalyticalTechnique.return_value = mock_technique
        
        # Set up mock import_module to return mock module
        mock_import_module.return_value = mock_module
        
        # Create integrator
        integrator = TechniqueMCPIntegrator()
        
        # Create step and context
        step = {
            "technique": "test_technique",
            "purpose": "Test purpose",
            "parameters": {"param1": "value1"}
        }
        context = AnalysisContext()
        context.add("question", "Test question")
        
        # Execute step
        result = integrator.execute_step(step, context)
        
        # Verify technique was executed
        mock_technique.execute.assert_called_once()
        
        # Verify result was returned
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"], "test_result")
        
        # Verify result was added to context
        self.assertEqual(context.get_mcp_result("test_technique"), {"status": "completed", "result": "test_result"})
    
    @patch('src.technique_mcp_integrator.importlib.import_module')
    def test_execute_step_with_error(self, mock_import_module):
        """Test executing a technique step with error."""
        # Create mock technique that raises an exception
        mock_technique = MagicMock()
        mock_technique.execute.side_effect = Exception("Test error")
        
        # Create mock technique module
        mock_module = MagicMock()
        mock_module.AnalyticalTechnique.return_value = mock_technique
        
        # Set up mock import_module to return mock module
        mock_import_module.return_value = mock_module
        
        # Create integrator
        integrator = TechniqueMCPIntegrator()
        
        # Create step and context
        step = {
            "technique": "test_technique",
            "purpose": "Test purpose",
            "parameters": {"param1": "value1"}
        }
        context = AnalysisContext()
        
        # Execute step and verify exception is raised
        with self.assertRaises(Exception):
            integrator.execute_step(step, context)
        
        # Verify error event was added to context
        error_events = context.get_events("error")
        self.assertTrue(len(error_events) > 0)
        self.assertTrue(any("Test error" in str(e) for e in error_events))


if __name__ == '__main__':
    unittest.main()
