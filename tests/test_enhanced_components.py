"""
Unit tests for enhanced components of the MCP Analytical System.
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
from src.config import SystemConfig, get_config

class TestWorkflowOrchestratorMCP(unittest.TestCase):
    """Test cases for the WorkflowOrchestratorMCP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock MCP registry
        self.mock_registry = MagicMock()
        
        # Create mock technique integrator
        self.mock_technique_integrator = MagicMock()
        self.mock_technique_integrator.get_all_techniques.return_value = {
            "research_to_hypothesis": MagicMock(),
            "synthesis_generation": MagicMock()
        }
        
        # Set up mock registry to return the mock technique integrator
        self.mock_registry.get_technique_integrator.return_value = self.mock_technique_integrator
        
        # Create mock MCPs
        self.mock_perplexity_sonar = MagicMock()
        self.mock_llama4_scout = MagicMock()
        
        # Set up mock registry to return mock MCPs
        self.mock_registry.get_mcp.side_effect = lambda name: {
            "perplexity_sonar": self.mock_perplexity_sonar,
            "llama4_scout": self.mock_llama4_scout
        }.get(name)
        
        # Create WorkflowOrchestratorMCP instance with mock registry
        self.orchestrator = WorkflowOrchestratorMCP(self.mock_registry)
    
    def test_initialization(self):
        """Test initialization of WorkflowOrchestratorMCP."""
        self.assertEqual(self.orchestrator.name, "workflow_orchestrator")
        self.assertEqual(self.orchestrator.mcp_registry, self.mock_registry)
        self.assertEqual(self.orchestrator.technique_integrator, self.mock_technique_integrator)
    
    def test_get_capabilities(self):
        """Test get_capabilities method."""
        capabilities = self.orchestrator.get_capabilities()
        self.assertIsInstance(capabilities, list)
        self.assertIn("workflow_orchestration", capabilities)
        self.assertIn("strategy_generation", capabilities)
        self.assertIn("adaptive_execution", capabilities)
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_run_preliminary_research_success(self, mock_logger):
        """Test _run_preliminary_research method with successful execution."""
        # Set up mock response from Perplexity Sonar
        self.mock_perplexity_sonar.process.return_value = {
            "result": {
                "insights": ["Insight 1", "Insight 2"],
                "hypotheses": ["Hypothesis 1", "Hypothesis 2"],
                "recommendations": ["Recommendation 1"]
            }
        }
        
        # Create context
        context = AnalysisContext()
        context.add("question", "Test question")
        
        # Run preliminary research
        self.orchestrator._run_preliminary_research(context)
        
        # Verify Perplexity Sonar was called
        self.mock_perplexity_sonar.process.assert_called_once()
        
        # Verify context was updated
        self.assertEqual(context.get("preliminary_research_insights"), ["Insight 1", "Insight 2"])
        self.assertEqual(context.get("preliminary_research_hypotheses"), ["Hypothesis 1", "Hypothesis 2"])
        self.assertEqual(context.get("preliminary_research_recommendations"), ["Recommendation 1"])
        
        # Verify event was added
        events = context.get_events("info")
        self.assertTrue(any("Preliminary research completed" in str(e) for e in events))
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_run_preliminary_research_error(self, mock_logger):
        """Test _run_preliminary_research method with error handling."""
        # Set up mock to raise an exception
        self.mock_perplexity_sonar.process.side_effect = Exception("Test error")
        
        # Create context
        context = AnalysisContext()
        context.add("question", "Test question")
        
        # Run preliminary research
        self.orchestrator._run_preliminary_research(context)
        
        # Verify error was logged
        mock_logger.error.assert_called()
        
        # Verify fallback research was added to context
        self.assertIsNotNone(context.get("preliminary_research_insights"))
        self.assertIsNotNone(context.get("preliminary_research_hypotheses"))
        self.assertIsNotNone(context.get("preliminary_research_recommendations"))
        
        # Verify error event was added
        events = context.get_events("error")
        self.assertTrue(any("Preliminary research failed" in str(e) for e in events))
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_analyze_question_characteristics_success(self, mock_logger):
        """Test _analyze_question_characteristics method with successful execution."""
        # Set up mock response from llama4_scout
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
        
        # Create context
        context = AnalysisContext()
        context.add("question", "Test question")
        
        # Analyze question
        self.orchestrator._analyze_question_characteristics(context)
        
        # Verify llama4_scout was called
        self.mock_llama4_scout.process.assert_called_once()
        
        # Verify context was updated
        question_analysis = context.get("question_analysis")
        self.assertIsNotNone(question_analysis)
        self.assertEqual(question_analysis["type"], "predictive")
        self.assertEqual(question_analysis["domains"], ["economics", "technology"])
        
        # Verify event was added
        events = context.get_events("info")
        self.assertTrue(any("Question analysis completed" in str(e) for e in events))
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_analyze_question_characteristics_error(self, mock_logger):
        """Test _analyze_question_characteristics method with error handling."""
        # Set up mock to raise an exception
        self.mock_llama4_scout.process.side_effect = Exception("Test error")
        
        # Create context
        context = AnalysisContext()
        context.add("question", "Test question")
        
        # Analyze question
        self.orchestrator._analyze_question_characteristics(context)
        
        # Verify error was logged
        mock_logger.error.assert_called()
        
        # Verify default analysis was added to context
        question_analysis = context.get("question_analysis")
        self.assertIsNotNone(question_analysis)
        
        # Verify error event was added
        events = context.get_events("error")
        self.assertTrue(any("Question analysis failed" in str(e) for e in events))
    
    def test_generate_analysis_strategy(self):
        """Test _generate_analysis_strategy method."""
        # Create context with question analysis
        context = AnalysisContext()
        context.add("question", "Test question")
        context.add("question_analysis", {
            "type": "predictive",
            "domains": ["economics", "technology"],
            "complexity": "medium",
            "uncertainty": "medium",
            "time_horizon": "medium-term",
            "potential_biases": ["confirmation bias"]
        })
        
        # Generate strategy
        strategy = self.orchestrator._generate_analysis_strategy(context)
        
        # Verify strategy was created
        self.assertIsInstance(strategy, AnalysisStrategy)
        self.assertTrue(len(strategy.steps) > 0)
        self.assertEqual(strategy.adaptive, True)
        
        # Verify strategy has appropriate techniques for predictive questions
        technique_names = [step["technique"] for step in strategy.steps]
        self.assertIn("research_to_hypothesis", technique_names)
    
    def test_generate_default_strategy(self):
        """Test _generate_default_strategy method."""
        # Create context
        context = AnalysisContext()
        
        # Generate default strategy
        strategy = self.orchestrator._generate_default_strategy(context)
        
        # Verify strategy was created
        self.assertIsInstance(strategy, AnalysisStrategy)
        self.assertTrue(len(strategy.steps) > 0)
        
        # Verify strategy has essential techniques
        technique_names = [step["technique"] for step in strategy.steps]
        self.assertIn("research_to_hypothesis", technique_names)
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_check_adaptation_criteria(self, mock_logger):
        """Test _check_adaptation_criteria method."""
        # Create context with strategy
        context = AnalysisContext()
        strategy = AnalysisStrategy({
            "name": "Test Strategy",
            "description": "Test description",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research",
                    "parameters": {},
                    "adaptive_criteria": ["conflicting_evidence_found"]
                }
            ]
        })
        context.add("strategy", strategy)
        
        # Add result with conflicting evidence
        context.add_mcp_result("research_to_hypothesis", {
            "conflicting_evidence_found": True
        })
        
        # Check adaptation criteria
        needs_adaptation, trigger_type = self.orchestrator._check_adaptation_criteria(context, 0)
        
        # Verify adaptation is needed
        self.assertTrue(needs_adaptation)
        self.assertEqual(trigger_type, "ConflictingEvidenceDetected")
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_adapt_strategy(self, mock_logger):
        """Test _adapt_strategy method."""
        # Create context with strategy
        context = AnalysisContext()
        strategy = AnalysisStrategy({
            "name": "Test Strategy",
            "description": "Test description",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research",
                    "parameters": {},
                    "adaptive_criteria": []
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate synthesis",
                    "parameters": {},
                    "adaptive_criteria": []
                }
            ]
        })
        context.add("strategy", strategy)
        
        # Adapt strategy
        self.orchestrator._adapt_strategy(context, 0, "ConflictingEvidenceDetected")
        
        # Verify strategy was adapted
        adapted_strategy = context.get("strategy")
        self.assertIsNotNone(adapted_strategy)
        self.assertNotEqual(adapted_strategy.name, strategy.name)
        self.assertTrue("Adapted" in adapted_strategy.name)
        
        # Verify event was added
        events = context.get_events("info")
        self.assertTrue(any("Strategy adapted" in str(e) for e in events))
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_handle_execution_error_retry(self, mock_logger):
        """Test _handle_execution_error method with retry."""
        # Create context
        context = AnalysisContext()
        context.add("current_step", {
            "technique": "research_to_hypothesis",
            "parameters": {}
        })
        
        # Set up mock technique integrator for retry
        self.mock_technique_integrator.execute_step.side_effect = [
            Exception("First attempt error"),  # First call fails
            {"status": "completed"}  # Second call succeeds
        ]
        
        # Handle error
        self.orchestrator._handle_execution_error(context, "research_to_hypothesis", Exception("Test error"))
        
        # Verify retry was attempted
        self.assertEqual(self.mock_technique_integrator.execute_step.call_count, 2)
        
        # Verify retry count was updated
        self.assertEqual(context.get("research_to_hypothesis_retry_count"), 1)
        
        # Verify result was added to context
        self.assertIsNotNone(context.get_mcp_result("research_to_hypothesis"))
        
        # Verify event was added
        events = context.get_events("success")
        self.assertTrue(any("Retry of technique" in str(e) for e in events))
    
    @patch('src.workflow_orchestrator_mcp.logger')
    def test_handle_execution_error_fallback(self, mock_logger):
        """Test _handle_execution_error method with fallback."""
        # Create context
        context = AnalysisContext()
        context.add("current_step", {
            "technique": "research_to_hypothesis",
            "parameters": {}
        })
        context.add("research_to_hypothesis_retry_count", self.orchestrator.MAX_RETRIES)
        
        # Set up mock technique integrator for fallback
        self.mock_technique_integrator.execute_step.return_value = {"status": "completed"}
        
        # Set up mock to check if technique is available
        self.orchestrator._is_technique_available = MagicMock(return_value=True)
        
        # Handle error
        self.orchestrator._handle_execution_error(context, "research_to_hypothesis", Exception("Test error"))
        
        # Verify fallback was attempted
        self.mock_technique_integrator.execute_step.assert_called_once()
        
        # Verify result was added to context
        self.assertIsNotNone(context.get_mcp_result("research_to_hypothesis_fallback"))
        
        # Verify event was added
        events = context.get_events("success")
        self.assertTrue(any("Fallback technique" in str(e) for e in events))


class TestAnalysisContext(unittest.TestCase):
    """Test cases for the AnalysisContext class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context = AnalysisContext()
    
    def test_initialization(self):
        """Test initialization of AnalysisContext."""
        self.assertIsNotNone(self.context._data)
        self.assertIsNotNone(self.context._events)
        self.assertIsNotNone(self.context._mcp_results)
        self.assertIn("created_at", self.context._data)
    
    def test_add_and_get(self):
        """Test add and get methods."""
        # Add a key-value pair
        self.context.add("test_key", "test_value")
        
        # Get the value
        value = self.context.get("test_key")
        
        # Verify value was retrieved
        self.assertEqual(value, "test_value")
        
        # Get a non-existent key with default
        value = self.context.get("non_existent", "default_value")
        
        # Verify default value was returned
        self.assertEqual(value, "default_value")
    
    def test_add_and_get_events(self):
        """Test add_event and get_events methods."""
        # Add events
        self.context.add_event("info", "Test info event")
        self.context.add_event("error", "Test error event")
        self.context.add_event("info", "Another info event")
        
        # Get all events
        all_events = self.context.get_events()
        
        # Verify all events were retrieved
        self.assertEqual(len(all_events), 3)
        
        # Get info events
        info_events = self.context.get_events("info")
        
        # Verify info events were retrieved
        self.assertEqual(len(info_events), 2)
        
        # Get error events
        error_events = self.context.get_events("error")
        
        # Verify error events were retrieved
        self.assertEqual(len(error_events), 1)
    
    def test_add_and_get_mcp_result(self):
        """Test add_mcp_result and get_mcp_result methods."""
        # Add MCP results
        self.context.add_mcp_result("test_mcp", {"result": "test_result"})
        
        # Get MCP result
        result = self.context.get_mcp_result("test_mcp")
        
        # Verify result was retrieved
        self.assertEqual(result, {"result": "test_result"})
        
        # Get non-existent MCP result
        result = self.context.get_mcp_result("non_existent")
        
        # Verify None was returned
        self.assertIsNone(result)
        
        # Get all MCP results
        all_results = self.context.get_mcp_results()
        
        # Verify all results were retrieved
        self.assertEqual(len(all_results), 1)
        self.assertIn("test_mcp", all_results)
    
    def test_to_dict(self):
        """Test to_dict method."""
        # Add data
        self.context.add("test_key", "test_value")
        self.context.add_event("info", "Test event")
        self.context.add_mcp_result("test_mcp", {"result": "test_result"})
        
        # Convert to dictionary
        context_dict = self.context.to_dict()
        
        # Verify dictionary structure
        self.assertIn("data", context_dict)
        self.assertIn("events", context_dict)
        self.assertIn("mcp_results", context_dict)
        
        # Verify data was included
        self.assertIn("test_key", context_dict["data"])
        self.assertEqual(context_dict["data"]["test_key"], "test_value")
        
        # Verify events were included
        self.assertEqual(len(context_dict["events"]), 2)  # Including the MCP result event
        
        # Verify MCP results were included
        self.assertIn("test_mcp", context_dict["mcp_results"])
        self.assertEqual(context_dict["mcp_results"]["test_mcp"], {"result": "test_result"})
    
    def test_properties(self):
        """Test convenience properties."""
        # Add data
        self.context.add("question", "Test question")
        self.context.add("question_analysis", {"type": "predictive"})
        self.context.add("preliminary_research", {"findings": ["Finding 1"]})
        self.context.add("strategy", {"name": "Test strategy"})
        
        # Verify properties
        self.assertEqual(self.context.question, "Test question")
        self.assertEqual(self.context.question_analysis, {"type": "predictive"})
        self.assertEqual(self.context.preliminary_research, {"findings": ["Finding 1"]})
        self.assertEqual(self.context.strategy, {"name": "Test strategy"})


class TestAnalysisStrategy(unittest.TestCase):
    """Test cases for the AnalysisStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy_data = {
            "name": "Test Strategy",
            "description": "Test description",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis",
                    "purpose": "Conduct research",
                    "parameters": {"depth": "standard"},
                    "adaptive_criteria": ["conflicting_evidence_found"]
                },
                {
                    "technique": "synthesis_generation",
                    "purpose": "Generate synthesis",
                    "parameters": {"include_confidence": True},
                    "adaptive_criteria": []
                }
            ]
        }
        self.strategy = AnalysisStrategy(self.strategy_data)
    
    def test_initialization(self):
        """Test initialization of AnalysisStrategy."""
        self.assertEqual(self.strategy.name, "Test Strategy")
        self.assertEqual(self.strategy.description, "Test description")
        self.assertEqual(self.strategy.adaptive, True)
        self.assertEqual(len(self.strategy.steps), 2)
    
    def test_initialization_with_missing_fields(self):
        """Test initialization with missing fields."""
        # Create strategy with missing fields in steps
        strategy_data = {
            "name": "Test Strategy",
            "description": "Test description",
            "adaptive": True,
            "steps": [
                {
                    "technique": "research_to_hypothesis"
                }
            ]
        }
        strategy = AnalysisStrategy(strategy_data)
        
        # Verify missing fields were added with defaults
        self.assertEqual(strategy.steps[0]["purpose"], "Execute research_to_hypothesis")
        self.assertEqual(strategy.steps[0]["parameters"], {})
        self.assertEqual(strategy.steps[0]["adaptive_criteria"], [])
    
    def test_to_dict(self):
        """Test to_dict method."""
        # Convert to dictionary
        strategy_dict = self.strategy.to_dict()
        
        # Verify dictionary structure
        self.assertEqual(strategy_dict["name"], "Test Strategy")
        self.assertEqual(strategy_dict["description"], "Test description")
        self.assertEqual(strategy_dict["adaptive"], True)
        self.assertEqual(len(strategy_dict["steps"]), 2)
    
    def test_from_dict(self):
        """Test from_dict class method."""
        # Create strategy from dictionary
        strategy = AnalysisStrategy.from_dict(self.strategy_data)
        
        # Verify strategy was created correctly
        self.assertEqual(strategy.name, "Test Strategy")
        self.assertEqual(strategy.description, "Test description")
        self.assertEqual(strategy.adaptive, True)
        self.assertEqual(len(strategy.steps), 2)
    
    def test_from_context(self):
        """Test from_context class method."""
        # Create context with question analysis
        context = AnalysisContext()
        context.add("question_analysis", {
            "type": "predictive",
            "domains": ["economics", "technology"]
        })
        
        # Create strategy from context
        strategy = AnalysisStrategy.from_context(context)
        
        # Verify strategy was created correctly
        self.assertIn("Predictive", strategy.name)
        self.assertTrue(strategy.adaptive)
        self.assertTrue(len(strategy.steps) > 0)
    
    def test_create_predictive_strategy(self):
        """Test create_predictive_strategy class method."""
        # Create predictive strategy
        strategy = AnalysisStrategy.create_predictive_strategy(["economics", "technology"])
        
        # Verify strategy was created correctly
        self.assertIn("Predictive", strategy.name)
        self.assertTrue(strategy.adaptive)
        self.assertTrue(len(strategy.steps) > 0)
        
        # Verify strategy has appropriate techniques
        technique_names = [step["technique"] for step in strategy.steps]
        self.assertIn("research_to_hypothesis", technique_names)
        self.assertIn("scenario_triangulation", technique_names)
    
    def test_add_step(self):
        """Test add_step method."""
        # Add a step
        self.strategy.add_step(
            "uncertainty_mapping",
            "Map uncertainties",
            {"detail_level": "high"},
            ["overall_uncertainty > 0.7"]
        )
        
        # Verify step was added
        self.assertEqual(len(self.strategy.steps), 3)
        self.assertEqual(self.strategy.steps[2]["technique"], "uncertainty_mapping")
        self.assertEqual(self.strategy.steps[2]["purpose"], "Map uncertainties")
        self.assertEqual(self.strategy.steps[2]["parameters"], {"detail_level": "high"})
        self.assertEqual(self.strategy.steps[2]["adaptive_criteria"], ["overall_uncertainty > 0.7"])
    
    def test_remove_step(self):
        """Test remove_step method."""
        # Remove a step
        self.strategy.remove_step(0)
        
        # Verify step was removed
        self.assertEqual(len(self.strategy.steps), 1)
        self.assertEqual(self.strategy.steps[0]["technique"], "synthesis_generation")
    
    def test_get_step(self):
        """Test get_step method."""
        # Get a step
        step = self.strategy.get_step(0)
        
        # Verify step was retrieved
        self.assertEqual(step["technique"], "research_to_hypothesis")
        
        # Get a non-existent step
        step = self.strategy.get_step(10)
        
        # Verify None was returned
        self.assertIsNone(step)
    
    def test_update_step(self):
        """Test update_step method."""
        # Update a step
        self.strategy.update_step(0, purpose="Updated purpose", parameters={"depth": "deep"})
        
        # Verify step was updated
        self.assertEqual(self.strategy.steps[0]["purpose"], "Updated purpose")
        self.assertEqual(self.strategy.steps[0]["parameters"], {"depth": "deep"})
        
        # Verify other fields were not changed
        self.assertEqual(self.strategy.steps[0]["technique"], "research_to_hypothesis")
        self.assertEqual(self.strategy.steps[0]["adaptive_criteria"], ["conflicting_evidence_found"])


class TestConfig(unittest.TestCase):
    """Test cases for the configuration system."""
    
    def test_system_config_initialization(self):
        """Test initialization of SystemConfig."""
        # Create config with defaults
        config = SystemConfig()
        
        # Verify default values
        self.assertEqual(config.environment, "development")
        self.assertEqual(config.debug_mode, False)
        self.assertIsNotNone(config.llm)
        self.assertIsNotNone(config.research)
        self.assertIsNotNone(config.workflow)
        self.assertIsNotNone(config.storage)
        self.assertIsNotNone(config.logging)
    
    def test_llm_config_validation(self):
        """Test validation in LLMConfig."""
        # Create config with invalid temperature
        with self.assertRaises(ValueError):
            config = SystemConfig(llm={"temperature": 2.0})
    
    def test_research_config_validation(self):
        """Test validation in ResearchConfig."""
        # Create config with invalid depth
        with self.assertRaises(ValueError):
            config = SystemConfig(research={"default_depth": "invalid"})
    
    @patch('src.config.os.environ')
    def test_load_config_from_env(self, mock_environ):
        """Test load_config_from_env function."""
        # Set up mock environment variables
        mock_environ.get.side_effect = lambda key, default=None: {
            "OPENAI_API_KEY": "test_openai_key",
            "GROQ_API_KEY": "test_groq_key",
            "DEFAULT_LLM_PROVIDER": "openai",
            "DEFAULT_RESEARCH_DEPTH": "deep",
            "ENABLE_ADAPTIVE_WORKFLOW": "true",
            "ENVIRONMENT": "production"
        }.get(key, default)
        
        # Load config from environment
        config = get_config()
        
        # Verify config values from environment
        self.assertEqual(config.llm.openai_api_key, "test_openai_key")
        self.assertEqual(config.llm.groq_api_key, "test_groq_key")
        self.assertEqual(config.llm.default_provider, "openai")
        self.assertEqual(config.research.default_depth, "deep")
        self.assertEqual(config.workflow.adaptive_workflow, True)
        self.assertEqual(config.environment, "production")


if __name__ == '__main__':
    unittest.main()
