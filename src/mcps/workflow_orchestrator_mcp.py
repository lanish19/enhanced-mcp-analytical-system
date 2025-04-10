"""
Workflow Orchestrator MCP for dynamic analysis workflow management.
This module provides the WorkflowOrchestratorMCP class for orchestrating analytical workflows.
"""

import logging
import time
import json
import re
from typing import Dict, List, Any, Optional

from src.base_mcp import BaseMCP
from src.mcp_registry import MCPRegistry
from src.analysis_context import AnalysisContext, PreliminaryResearchOutput, QuestionAnalysisOutput
from src.analysis_strategy import AnalysisStrategy
from src.technique_mcp_integrator import TechniqueMCPIntegrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkflowOrchestratorMCP(BaseMCP):
    """
    Workflow Orchestrator MCP for dynamic analysis workflow management.
    
    This MCP provides capabilities for:
    1. Analyzing question characteristics to determine optimal workflow sequence
    2. Selecting techniques based on question type (predictive, causal, evaluative)
    3. Adapting workflow dynamically based on interim findings
    4. Managing technique dependencies and complementary pairs
    5. Orchestrating the execution of analytical techniques
    """
    
    def __init__(self, mcp_registry: MCPRegistry, techniques_dir: str = "src/techniques"):
        """
        Initialize the WorkflowOrchestratorMCP.
        
        Args:
            mcp_registry: Registry of available MCPs
            techniques_dir: Directory containing technique modules
        """
        super().__init__(
            name="workflow_orchestrator",
            description="Orchestrates dynamic analytical workflows based on question characteristics",
            version="1.0.0"
        )
        
        self.mcp_registry = mcp_registry
        self.integrator = TechniqueMCPIntegrator(mcp_registry, techniques_dir)
        self.active_workflows = {}  # workflow_id -> workflow configuration
        self.technique_registry = {}  # name -> technique instance
        
        # Register available techniques
        self.register_techniques()
        
        logger.info("Initialized WorkflowOrchestratorMCP")
    
    def register_techniques(self):
        """Register available analytical techniques."""
        logger.info("Registering analytical techniques")
        
        # Get all available techniques from the integrator
        available_techniques = self.integrator.get_all_techniques()
        if not available_techniques:
            logger.warning("No techniques available from integrator, using default techniques")
            self._register_default_techniques()
            return
        
        # Register techniques
        for name, technique in available_techniques.items():
            self.technique_registry[name] = technique
            logger.info(f"Registered technique: {name}")
        
        logger.info(f"Registered {len(self.technique_registry)} techniques")
    
    def _register_default_techniques(self):
        """Register default techniques if none are available from the integrator."""
        logger.info("Registering default techniques")
        
        # This is a fallback method that would be implemented with basic techniques
        # For now, we'll just log a warning
        logger.warning("Default technique registration not implemented")
    
    def _get_available_techniques(self):
        """Get names of available techniques."""
        return list(self.technique_registry.keys())
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in WorkflowOrchestratorMCP")
        
        # Validate input
        if not isinstance(input_data, dict):
            error_msg = "Input must be a dictionary"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get operation type
        operation = input_data.get("operation", "analyze_question")
        
        # Process based on operation type
        if operation == "analyze_question":
            return self._analyze_question(input_data)
        elif operation == "create_workflow":
            return self._create_workflow(input_data)
        elif operation == "execute_workflow":
            return self._execute_workflow(input_data)
        elif operation == "update_workflow":
            return self._update_workflow(input_data)
        elif operation == "get_workflow_status":
            return self._get_workflow_status(input_data)
        elif operation == "get_available_techniques":
            return self._get_available_techniques_info(input_data)
        elif operation == "execute_analysis":
            return self._execute_analysis(input_data)
        else:
            error_msg = f"Unknown operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _analyze_question(self, input_data: Dict) -> Dict:
        """
        Analyze a question to determine its characteristics and optimal workflow.
        
        Args:
            input_data: Input data dictionary containing the question
            
        Returns:
            Question analysis results
        """
        logger.info("Analyzing question")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get context
        context = input_data.get("context")
        if not isinstance(context, AnalysisContext):
            # Create new context if not provided
            context = AnalysisContext()
            context.add("question", question)
        
        # Run preliminary research if requested
        run_preliminary_research = input_data.get("run_preliminary_research", True)
        if run_preliminary_research:
            self._run_preliminary_research(context)
        
        # Analyze question characteristics
        self._analyze_question_characteristics(context)
        
        # Generate analysis strategy
        strategy = self._generate_analysis_strategy(context)
        
        # Store strategy in context
        context.add("strategy", strategy)
        
        # Get question analysis from context
        question_analysis = context.get("question_analysis", {})
        
        # Compile results
        results = {
            "question": question,
            "question_analysis": question_analysis,
            "strategy": strategy.to_dict() if strategy else {},
            "preliminary_research": context.preliminary_research.dict() if context.preliminary_research else {}
        }
        
        return results
    
    def _run_preliminary_research(self, context: AnalysisContext) -> bool:
        """
        Run enhanced preliminary research using PerplexitySonarMCP.
        
        Args:
            context: Analysis context containing the question
            
        Returns:
            True if research was successful, False otherwise
        """
        logger.info("Running enhanced preliminary research")
        
        # Get question from context
        question = context.get("question", "")
        if not question:
            logger.warning("No question found in context, skipping preliminary research")
            return False
        
        # Get PerplexitySonarMCP
        try:
            research_mcp = self.mcp_registry.get_mcp("PerplexitySonarMCP")
            if not research_mcp:
                logger.warning("PerplexitySonarMCP not available, skipping preliminary research")
                return False
        except KeyError:
            logger.warning("PerplexitySonarMCP not found in registry, skipping preliminary research")
            return False
        
        # Craft enhanced prompt for comprehensive research
        enhanced_prompt = f"""
        Conduct comprehensive research on the following analytical question:
        
        QUESTION: {question}
        
        Your research should address three main goals:
        
        1. BASELINE UNDERSTANDING: Provide a concise summary of the current state of knowledge on this topic.
           Include key facts, relevant context, and established consensus views.
        
        2. UNKNOWNS AND SUBQUESTIONS: Identify specific unknowns, gaps in knowledge, and important 
           subquestions that need to be addressed to fully answer the main question.
        
        3. BRAINSTORMING: Generate a diverse set of initial possibilities, perspectives, or approaches 
           that could be relevant to answering this question.
        
        Format your response as a valid JSON object with the following structure:
        {{
            "baseline_summary": "Comprehensive summary of current knowledge...",
            "key_subquestions": ["Subquestion 1", "Subquestion 2", ...],
            "identified_unknowns": ["Unknown 1", "Unknown 2", ...],
            "brainstormed_possibilities": ["Possibility 1", "Possibility 2", ...],
            "key_sources": [
                {{"title": "Source 1", "url": "URL if available", "relevance": "Brief note on relevance"}},
                ...
            ]
        }}
        
        Ensure your response is a valid JSON object that can be parsed programmatically.
        """
        
        # Prepare input for research
        mcp_input = {
            "operation": "research",
            "question": question,
            "prompt": enhanced_prompt,
            "output_format": "json"
        }
        
        # Execute research using PerplexitySonarMCP
        try:
            raw_result = research_mcp.process(mcp_input)
            
            # Check for errors
            if isinstance(raw_result, dict) and "error" in raw_result:
                logger.error(f"Error running preliminary research: {raw_result['error']}")
                return False
            
            # Parse and validate the result
            try:
                # If raw_result is a string, try to parse it as JSON
                if isinstance(raw_result, str):
                    parsed_result = json.loads(raw_result)
                # If raw_result is already a dict, use it directly
                elif isinstance(raw_result, dict):
                    parsed_result = raw_result
                # If raw_result has a 'content' field (common API response format), try to parse that
                elif isinstance(raw_result, dict) and "content" in raw_result:
                    content = raw_result["content"]
                    # Try to extract JSON from the content if it's not already JSON
                    try:
                        parsed_result = json.loads(content)
                    except json.JSONDecodeError:
                        # Try to extract JSON from text using regex
                        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                        if json_match:
                            parsed_result = json.loads(json_match.group(1))
                        else:
                            # Try to find JSON-like content
                            json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                            if json_like:
                                parsed_result = json.loads(json_like.group(1))
                            else:
                                raise ValueError("Could not extract JSON from research result")
                else:
                    raise ValueError(f"Unexpected research result format: {type(raw_result)}")
                
                # Validate the parsed result using the Pydantic model
                validated_result = PreliminaryResearchOutput.model_validate(parsed_result)
                
                # Store the validated result in the context
                context.preliminary_research = validated_result
                
                # Also store in results for other components to access
                context.add_mcp_result("PerplexitySonarMCP", validated_result.dict())
                
                # Log success
                logger.info("Enhanced preliminary research completed successfully")
                context.add_event("research", "Enhanced preliminary research completed", 
                                 {"subquestions_count": len(validated_result.key_subquestions),
                                  "unknowns_count": len(validated_result.identified_unknowns)})
                return True
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing research result: {str(e)}")
                # Implement fallback with default values
                default_result = PreliminaryResearchOutput(
                    baseline_summary=f"Failed to parse research results for question: {question}",
                    key_subquestions=["What are the main factors affecting this question?"],
                    identified_unknowns=["Research parsing failed, unknowns could not be identified"],
                    brainstormed_possibilities=["Research parsing failed, possibilities could not be generated"],
                    key_sources=[]
                )
                context.preliminary_research = default_result
                context.add_event("error", "Research parsing failed", {"error": str(e)})
                return False
                
        except Exception as e:
            logger.error(f"Error running preliminary research: {str(e)}", exc_info=True)
            # Implement fallback with default values
            default_result = PreliminaryResearchOutput(
                baseline_summary=f"Research failed for question: {question}",
                key_subquestions=["What are the main factors affecting this question?"],
                identified_unknowns=["Research failed, unknowns could not be identified"],
                brainstormed_possibilities=["Research failed, possibilities could not be generated"],
                key_sources=[]
            )
            context.preliminary_research = default_result
            context.add_event("error", "Research execution failed", {"error": str(e)})
            return False
    
    def _analyze_question_characteristics(self, context: AnalysisContext) -> bool:
        """
        Analyze question characteristics using Llama4ScoutMCP, informed by preliminary research.
        
        Args:
            context: Analysis context containing the question and preliminary research
            
        Returns:
            True if analysis was successful, False otherwise
        """
        logger.info("Analyzing question characteristics")
        
        # Get question from context
        question = context.get("question", "")
        if not question:
            logger.warning("No question found in context, skipping question analysis")
            return False
        
        # Get Llama4ScoutMCP
        try:
            llm = self.mcp_registry.get_mcp("Llama4ScoutMCP")
            if not llm:
                logger.warning("Llama4ScoutMCP not available, using default question analysis")
                return self._default_question_analysis(context)
        except KeyError:
            logger.warning("Llama4ScoutMCP not found in registry, using default question analysis")
            return self._default_question_analysis(context)
        
        # Access preliminary research data
        research_data = context.preliminary_research
        if not research_data:
            logger.warning("No preliminary research data found, analysis may be limited")
        
        # Construct prompt for question analysis
        system_prompt = """You are an expert analytical assistant specializing in question analysis. 
        Your task is to analyze the given question and determine its characteristics, considering the 
        preliminary research provided. Provide your analysis in JSON format with the following fields:
        - question_type: The type of question (predictive, causal, evaluative, decision, descriptive)
        - complexity: Complexity level of the question (low, medium, high)
        - uncertainty_sources: List of specific sources of uncertainty relevant to this question
        - relevant_domains: List of knowledge domains most relevant to answering this question
        - suggested_personas: List of cognitive perspectives that would be valuable for analyzing this question
        - potential_biases: List of cognitive biases that might affect analysis of this question"""
        
        # Build prompt with research data if available
        prompt = f"QUESTION: {question}\n\n"
        
        if research_data:
            prompt += "PRELIMINARY RESEARCH:\n"
            prompt += f"Baseline Summary: {research_data.baseline_summary}\n\n"
            
            if research_data.key_subquestions:
                prompt += "Key Subquestions:\n"
                for i, subq in enumerate(research_data.key_subquestions):
                    prompt += f"{i+1}. {subq}\n"
                prompt += "\n"
            
            if research_data.identified_unknowns:
                prompt += "Identified Unknowns:\n"
                for i, unknown in enumerate(research_data.identified_unknowns):
                    prompt += f"{i+1}. {unknown}\n"
                prompt += "\n"
        
        prompt += "Please analyze this question and provide a detailed characterization in JSON format."
        
        # Call LLM
        try:
            llm_input = {
                "operation": "analyze_question",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "output_format": "json"
            }
            
            llm_response = llm.process(llm_input)
            
            # Check for errors
            if isinstance(llm_response, dict) and "error" in llm_response:
                logger.error(f"Error analyzing question: {llm_response['error']}")
                return self._default_question_analysis(context)
            
            # Parse and validate the result
            try:
                # Extract content based on response format
                if isinstance(llm_response, str):
                    content = llm_response
                elif isinstance(llm_response, dict) and "content" in llm_response:
                    content = llm_response["content"]
                else:
                    content = json.dumps(llm_response)
                
                # Try to extract JSON from the content
                try:
                    parsed_result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text using regex
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON-like content
                        json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_like:
                            parsed_result = json.loads(json_like.group(1))
                        else:
                            raise ValueError("Could not extract JSON from LLM response")
                
                # Validate the parsed result using the Pydantic model
                validated_result = QuestionAnalysisOutput.model_validate(parsed_result)
                
                # Store the validated result in the context
                context.question_analysis_output = validated_result
                
                # Also store in traditional format for backward compatibility
                context.question_analysis = validated_result.dict()
                
                # Store in results for other components to access
                context.add_mcp_result("Llama4ScoutMCP_QuestionAnalysis", validated_result.dict())
                
                # Log success
                logger.info("Question analysis completed successfully")
                context.add_event("analysis", "Question characteristics analysis completed", 
                                 {"question_type": validated_result.question_type,
                                  "complexity": validated_result.complexity})
                return True
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing question analysis: {str(e)}")
                return self._default_question_analysis(context)
                
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}", exc_info=True)
            return self._default_question_analysis(context)
    
    def _default_question_analysis(self, context: AnalysisContext) -> bool:
        """
        Generate default question analysis when LLM analysis fails.
        
        Args:
            context: Analysis context containing the question
            
        Returns:
            True if default analysis was generated, False otherwise
        """
        logger.info("Generating default question analysis")
        
        # Get question from context
        question = context.get("question", "")
        
        # Determine question type using simple heuristics
        question_type = self._determine_question_type(question)
        
        # Determine domain using simple heuristics
        domains = self._determine_domain(question)
        
        # Determine complexity using simple heuristics
        complexity = self._determine_complexity(question)
        
        # Create default analysis
        default_analysis = QuestionAnalysisOutput(
            question_type=question_type,
            complexity=complexity,
            uncertainty_sources=["data_limitations", "future_unpredictability"],
            relevant_domains=[domains],
            suggested_personas=["analytical", "critical"],
            potential_biases=["recency", "availability"]
        )
        
        # Store analysis in context
        context.question_analysis_output = default_analysis
        context.question_analysis = default_analysis.dict()
        
        # Store in results for other components to access
        context.add_mcp_result("DefaultQuestionAnalysis", default_analysis.dict())
        
        # Log event
        context.add_event("fallback", "Used default question analysis", 
                         {"question_type": question_type, "complexity": complexity})
        
        logger.info("Default question analysis generated")
        return True
    
    def _determine_question_type(self, question: str) -> str:
        """Determine question type using simple heuristics."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["will", "future", "predict", "forecast"]):
            return "predictive"
        elif any(word in question_lower for word in ["why", "cause", "reason", "lead to"]):
            return "causal"
        elif any(word in question_lower for word in ["evaluate", "assess", "compare", "better"]):
            return "evaluative"
        elif any(word in question_lower for word in ["should", "decision", "choose", "select"]):
            return "decision"
        else:
            return "descriptive"
    
    def _determine_domain(self, question: str) -> str:
        """Determine domain using simple heuristics."""
        question_lower = question.lower()
        
        domains = {
            "economic": ["economy", "market", "financial", "trade", "gdp", "inflation"],
            "political": ["government", "policy", "election", "political", "regime", "law"],
            "technological": ["technology", "innovation", "digital", "ai", "software", "hardware"],
            "social": ["society", "social", "cultural", "demographic", "population"],
            "environmental": ["environment", "climate", "pollution", "sustainability"],
            "security": ["security", "defense", "military", "threat", "conflict", "war"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in question_lower for keyword in keywords):
                return domain
        
        return "general"
    
    def _determine_complexity(self, question: str) -> str:
        """Determine complexity using simple heuristics."""
        # Count words as a simple proxy for complexity
        word_count = len(question.split())
        
        if word_count > 30:
            return "high"
        elif word_count > 15:
            return "medium"
        else:
            return "low"
    
    def _generate_analysis_strategy(self, context: AnalysisContext) -> AnalysisStrategy:
        """
        Generate an analysis strategy dynamically using the core LLM, informed by prior analysis.
        
        Args:
            context: Analysis context containing the question, preliminary research, and question analysis
            
        Returns:
            AnalysisStrategy object
        """
        logger.info("Generating dynamic analysis strategy")
        
        # Get inputs from context
        question = context.get("question", "")
        if not question:
            logger.warning("No question found in context, using default strategy")
            return self._generate_default_strategy(context)
        
        # Access preliminary research and question analysis
        preliminary_research = context.preliminary_research
        question_analysis = context.question_analysis_output
        
        # Handle cases where they might be None
        if not preliminary_research:
            logger.warning("No preliminary research found in context, strategy may be limited")
        
        if not question_analysis:
            logger.warning("No question analysis found in context, strategy may be limited")
        
        # Get available technique/MCP names and metadata
        available_techniques = list(self.technique_registry.keys())
        available_mcps = list(self.mcp_registry.get_all_mcps().keys())
        
        # Get Llama4ScoutMCP for strategy generation
        try:
            llm = self.mcp_registry.get_mcp("Llama4ScoutMCP")
            if not llm:
                logger.warning("Llama4ScoutMCP not available, using default strategy")
                return self._generate_default_strategy(context)
        except KeyError:
            logger.warning("Llama4ScoutMCP not found in registry, using default strategy")
            return self._generate_default_strategy(context)
        
        # Construct detailed prompt for strategy generation
        system_prompt = """You are an expert analytical strategist specializing in designing analytical workflows. 
        Your task is to generate an optimal sequence of analytical steps to answer the given question based on its 
        characteristics and preliminary research. Act as an expert analyst selecting the most appropriate techniques 
        and MCPs to address the identified unknowns and subquestions.
        
        Provide your strategy in JSON format with a 'steps' array, where each step has:
        - technique: Name of the analytical technique or MCP to use
        - purpose: Clear explanation of why this step is needed and what it will contribute
        - parameters: Parameters for the technique/MCP
        - dependencies: List of step indices that must be completed before this step (optional)
        - optional: Whether this step is optional (default: false)
        
        Select 3-5 steps that form a coherent analytical workflow addressing the key aspects of the question.
        """
        
        # Build prompt with all available context
        prompt = f"QUESTION: {question}\n\n"
        
        # Add preliminary research if available
        if preliminary_research:
            prompt += "PRELIMINARY RESEARCH:\n"
            prompt += f"Baseline Summary: {preliminary_research.baseline_summary}\n\n"
            
            if preliminary_research.key_subquestions:
                prompt += "Key Subquestions:\n"
                for i, subq in enumerate(preliminary_research.key_subquestions):
                    prompt += f"{i+1}. {subq}\n"
                prompt += "\n"
            
            if preliminary_research.identified_unknowns:
                prompt += "Identified Unknowns:\n"
                for i, unknown in enumerate(preliminary_research.identified_unknowns):
                    prompt += f"{i+1}. {unknown}\n"
                prompt += "\n"
        
        # Add question analysis if available
        if question_analysis:
            prompt += "QUESTION ANALYSIS:\n"
            prompt += f"Question Type: {question_analysis.question_type}\n"
            prompt += f"Complexity: {question_analysis.complexity}\n"
            
            if question_analysis.uncertainty_sources:
                prompt += "Uncertainty Sources:\n"
                for i, source in enumerate(question_analysis.uncertainty_sources):
                    prompt += f"{i+1}. {source}\n"
                prompt += "\n"
            
            if question_analysis.relevant_domains:
                prompt += "Relevant Domains:\n"
                for i, domain in enumerate(question_analysis.relevant_domains):
                    prompt += f"{i+1}. {domain}\n"
                prompt += "\n"
            
            if question_analysis.suggested_personas:
                prompt += "Suggested Personas:\n"
                for i, persona in enumerate(question_analysis.suggested_personas):
                    prompt += f"{i+1}. {persona}\n"
                prompt += "\n"
        
        # Add available components
        prompt += "AVAILABLE TECHNIQUES:\n"
        for technique in available_techniques:
            prompt += f"- {technique}\n"
        prompt += "\n"
        
        prompt += "AVAILABLE MCPs:\n"
        for mcp in available_mcps:
            prompt += f"- {mcp}\n"
        prompt += "\n"
        
        prompt += "Please generate an optimal analysis strategy with 3-5 steps to answer this question. Ensure each step has a clear purpose and appropriate parameters. Return your strategy as valid JSON."
        
        # Call LLM for strategy generation
        try:
            llm_input = {
                "operation": "generate_strategy",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "output_format": "json"
            }
            
            llm_response = llm.process(llm_input)
            
            # Check for errors
            if isinstance(llm_response, dict) and "error" in llm_response:
                logger.error(f"Error generating strategy: {llm_response['error']}")
                return self._generate_default_strategy(context)
            
            # Parse and validate the result
            try:
                # Extract content based on response format
                if isinstance(llm_response, str):
                    content = llm_response
                elif isinstance(llm_response, dict) and "content" in llm_response:
                    content = llm_response["content"]
                else:
                    content = json.dumps(llm_response)
                
                # Try to extract JSON from the content
                try:
                    parsed_result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text using regex
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON-like content
                        json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_like:
                            parsed_result = json.loads(json_like.group(1))
                        else:
                            raise ValueError("Could not extract JSON from LLM response")
                
                # Create AnalysisStrategy from parsed result
                strategy = AnalysisStrategy()
                
                # Validate steps exist
                if "steps" not in parsed_result or not isinstance(parsed_result["steps"], list):
                    logger.error("Invalid strategy format: 'steps' array missing or not a list")
                    return self._generate_default_strategy(context)
                
                # Add steps from strategy result
                for step_data in parsed_result["steps"]:
                    # Validate required fields
                    if "technique" not in step_data:
                        logger.warning("Step missing required 'technique' field, skipping")
                        continue
                    
                    technique_name = step_data["technique"]
                    purpose = step_data.get("purpose", "")
                    parameters = step_data.get("parameters", {})
                    dependencies = step_data.get("dependencies", [])
                    optional = step_data.get("optional", False)
                    
                    # Validate technique exists in registry
                    technique_exists = technique_name in self.technique_registry
                    mcp_exists = technique_name in available_mcps
                    
                    if not (technique_exists or mcp_exists):
                        logger.warning(f"Technique/MCP '{technique_name}' not found in registries, skipping")
                        continue
                    
                    # Add purpose to parameters if provided
                    if purpose:
                        parameters["purpose"] = purpose
                    
                    # Add step to strategy
                    strategy.add_step(technique_name, parameters, dependencies, optional)
                
                # If no valid steps were added, use default strategy
                if len(strategy.steps) == 0:
                    logger.warning("No valid steps in generated strategy, using default")
                    return self._generate_default_strategy(context)
                
                # Add metadata to strategy
                strategy.add_metadata("generation_method", "llm")
                if question_analysis:
                    strategy.add_metadata("question_type", question_analysis.question_type)
                    strategy.add_metadata("complexity", question_analysis.complexity)
                    strategy.add_metadata("domains", question_analysis.relevant_domains)
                strategy.add_metadata("timestamp", time.time())
                
                # Log success
                logger.info(f"Generated dynamic strategy with {len(strategy.steps)} steps")
                context.add_event("strategy", "Dynamic analysis strategy generated", 
                                 {"steps_count": len(strategy.steps)})
                
                return strategy
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing strategy: {str(e)}")
                return self._generate_default_strategy(context)
                
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}", exc_info=True)
            return self._generate_default_strategy(context)
    
    def _generate_default_strategy(self, context: AnalysisContext) -> AnalysisStrategy:
        """
        Generate a default analysis strategy when dynamic generation fails.
        
        Args:
            context: Analysis context
            
        Returns:
            Default AnalysisStrategy
        """
        logger.info("Generating default analysis strategy")
        
        # Create default strategy
        strategy = AnalysisStrategy()
        
        # Get question type from context or determine it
        question_analysis = context.question_analysis_output
        question_type = question_analysis.question_type if question_analysis else self._determine_question_type(context.question)
        
        # Add steps based on question type
        if question_type == "predictive":
            # For predictive questions, use scenario-based techniques
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("scenario_triangulation_technique", {"num_scenarios": 3}, [0])
            strategy.add_step("key_assumptions_check_technique", {}, [1])
            strategy.add_step("uncertainty_mapping_technique", {}, [2])
            strategy.add_step("synthesis_generation_technique", {}, [3])
        
        elif question_type == "causal":
            # For causal questions, use causal analysis techniques
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("causal_network_analysis_technique", {}, [0])
            strategy.add_step("key_assumptions_check_technique", {}, [1])
            strategy.add_step("red_teaming_technique", {}, [2])
            strategy.add_step("synthesis_generation_technique", {}, [3])
        
        elif question_type == "evaluative":
            # For evaluative questions, use comparative techniques
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("analysis_of_competing_hypotheses_technique", {}, [0])
            strategy.add_step("key_assumptions_check_technique", {}, [1])
            strategy.add_step("uncertainty_mapping_technique", {}, [2])
            strategy.add_step("synthesis_generation_technique", {}, [3])
        
        elif question_type == "decision":
            # For decision questions, use decision analysis techniques
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("decision_tree_analysis_technique", {}, [0])
            strategy.add_step("premortem_analysis_technique", {}, [1])
            strategy.add_step("key_assumptions_check_technique", {}, [2])
            strategy.add_step("synthesis_generation_technique", {}, [3])
        
        else:
            # For descriptive or other questions, use general techniques
            strategy.add_step("research_to_hypothesis", {"num_hypotheses": 3})
            strategy.add_step("key_assumptions_check_technique", {}, [0])
            strategy.add_step("uncertainty_mapping_technique", {}, [1])
            strategy.add_step("synthesis_generation_technique", {}, [2])
        
        # Add metadata
        strategy.add_metadata("generation_method", "default")
        strategy.add_metadata("question_type", question_type)
        strategy.add_metadata("timestamp", time.time())
        
        # Log event
        context.add_event("fallback", "Used default analysis strategy", 
                         {"question_type": question_type, "steps_count": len(strategy.steps)})
        
        logger.info(f"Generated default strategy with {len(strategy.steps)} steps")
        return strategy
    
    def _check_adaptation_criteria(self, context: AnalysisContext, last_result: Dict) -> Optional[str]:
        """
        Check if the workflow needs to be adapted based on the latest result.
        
        Args:
            context: Analysis context
            last_result: Result from the last executed technique/MCP
            
        Returns:
            Trigger type string if adaptation is needed, None otherwise
        """
        logger.info("Checking adaptation criteria")
        
        # Skip if no result
        if not last_result:
            return None
        
        # Get technique/MCP name
        technique = last_result.get('technique', 'Unknown')
        
        # Check for high uncertainty
        if 'output' in last_result and 'overall_uncertainty_level' in last_result['output']:
            uncertainty_level = last_result['output']['overall_uncertainty_level']
            if uncertainty_level == 'High':
                logger.info(f"High uncertainty detected in {technique}")
                return 'HighUncertaintyDetected'
        
        # Check for conflicting hypotheses
        if 'output' in last_result and 'hypotheses' in last_result['output']:
            hypotheses = last_result['output']['hypotheses']
            if isinstance(hypotheses, list) and len(hypotheses) >= 2:
                # Check if top hypotheses have similar scores
                if 'scores' in hypotheses[0] and 'scores' in hypotheses[1]:
                    score_diff = abs(hypotheses[0]['score'] - hypotheses[1]['score'])
                    if score_diff < 0.1:  # Small difference in scores
                        logger.info(f"Conflicting hypotheses detected in {technique}")
                        return 'ConflictingHypothesesDetected'
        
        # Check for bias detection
        if 'output' in last_result and 'biases_detected' in last_result['output']:
            biases = last_result['output']['biases_detected']
            if isinstance(biases, list) and len(biases) > 0:
                logger.info(f"Biases detected in {technique}: {biases}")
                return 'BiasesDetected'
        
        # Check for key assumptions
        if technique == 'key_assumptions_check_technique' and 'output' in last_result and 'assumptions' in last_result['output']:
            assumptions = last_result['output']['assumptions']
            if isinstance(assumptions, list) and len(assumptions) > 0:
                # Check for high-impact assumptions
                high_impact_assumptions = [a for a in assumptions if a.get('impact', 'Low') == 'High']
                if len(high_impact_assumptions) > 0:
                    logger.info(f"High-impact assumptions detected: {len(high_impact_assumptions)}")
                    return 'HighImpactAssumptionsDetected'
        
        # Check for new evidence
        if 'output' in last_result and 'new_evidence' in last_result['output'] and last_result['output']['new_evidence']:
            logger.info(f"New evidence detected in {technique}")
            return 'NewEvidenceDetected'
        
        # No adaptation needed
        return None
    
    def _adapt_strategy(self, context: AnalysisContext, strategy: AnalysisStrategy, trigger_type: str, current_step_idx: int) -> None:
        """
        Adapt the analysis strategy based on the trigger type.
        
        Args:
            context: Analysis context
            strategy: Current analysis strategy
            trigger_type: Type of adaptation trigger
            current_step_idx: Index of the current step
        """
        logger.info(f"Adapting strategy due to trigger: {trigger_type}")
        
        # Rule-based adaptation based on trigger type
        if trigger_type == 'HighUncertaintyDetected':
            # Insert uncertainty mapping technique if not already in remaining steps
            remaining_steps = strategy.get_remaining_steps([i for i in range(current_step_idx + 1)])
            remaining_techniques = [strategy.steps[i].technique for i in remaining_steps]
            
            if 'uncertainty_mapping_technique' not in remaining_techniques:
                # Insert uncertainty mapping after current step
                strategy.insert_step(
                    current_step_idx + 1,
                    'uncertainty_mapping_technique',
                    {'purpose': 'Map uncertainties detected in previous step'},
                    [current_step_idx]
                )
                logger.info("Inserted uncertainty_mapping_technique due to high uncertainty")
                context.add_event("adaptation", "Inserted uncertainty mapping due to high uncertainty", 
                                 {"trigger": trigger_type, "inserted_technique": "uncertainty_mapping_technique"})
        
        elif trigger_type == 'ConflictingHypothesesDetected':
            # Insert competing hypotheses analysis if not already in remaining steps
            remaining_steps = strategy.get_remaining_steps([i for i in range(current_step_idx + 1)])
            remaining_techniques = [strategy.steps[i].technique for i in remaining_steps]
            
            if 'analysis_of_competing_hypotheses_technique' not in remaining_techniques:
                # Insert ACH after current step
                strategy.insert_step(
                    current_step_idx + 1,
                    'analysis_of_competing_hypotheses_technique',
                    {'purpose': 'Systematically evaluate competing hypotheses'},
                    [current_step_idx]
                )
                logger.info("Inserted analysis_of_competing_hypotheses_technique due to conflicting hypotheses")
                context.add_event("adaptation", "Inserted competing hypotheses analysis due to conflicting hypotheses", 
                                 {"trigger": trigger_type, "inserted_technique": "analysis_of_competing_hypotheses_technique"})
        
        elif trigger_type == 'BiasesDetected':
            # Insert red teaming technique if not already in remaining steps
            remaining_steps = strategy.get_remaining_steps([i for i in range(current_step_idx + 1)])
            remaining_techniques = [strategy.steps[i].technique for i in remaining_steps]
            
            if 'red_teaming_technique' not in remaining_techniques:
                # Insert red teaming after current step
                strategy.insert_step(
                    current_step_idx + 1,
                    'red_teaming_technique',
                    {'purpose': 'Challenge analysis to mitigate detected biases'},
                    [current_step_idx]
                )
                logger.info("Inserted red_teaming_technique due to biases detected")
                context.add_event("adaptation", "Inserted red teaming due to biases detected", 
                                 {"trigger": trigger_type, "inserted_technique": "red_teaming_technique"})
        
        elif trigger_type == 'HighImpactAssumptionsDetected':
            # Insert premortem analysis if not already in remaining steps
            remaining_steps = strategy.get_remaining_steps([i for i in range(current_step_idx + 1)])
            remaining_techniques = [strategy.steps[i].technique for i in remaining_steps]
            
            if 'premortem_analysis_technique' not in remaining_techniques:
                # Insert premortem after current step
                strategy.insert_step(
                    current_step_idx + 1,
                    'premortem_analysis_technique',
                    {'purpose': 'Identify potential failure points in high-impact assumptions'},
                    [current_step_idx]
                )
                logger.info("Inserted premortem_analysis_technique due to high-impact assumptions")
                context.add_event("adaptation", "Inserted premortem analysis due to high-impact assumptions", 
                                 {"trigger": trigger_type, "inserted_technique": "premortem_analysis_technique"})
        
        elif trigger_type == 'NewEvidenceDetected':
            # Use LLM-based adaptation for complex triggers
            self._adapt_strategy_with_llm(context, strategy, trigger_type, current_step_idx)
        
        else:
            # Use LLM-based adaptation for unknown triggers
            self._adapt_strategy_with_llm(context, strategy, trigger_type, current_step_idx)
    
    def _adapt_strategy_with_llm(self, context: AnalysisContext, strategy: AnalysisStrategy, trigger_type: str, current_step_idx: int) -> None:
        """
        Adapt the analysis strategy using LLM for complex or unknown triggers.
        
        Args:
            context: Analysis context
            strategy: Current analysis strategy
            trigger_type: Type of adaptation trigger
            current_step_idx: Index of the current step
        """
        logger.info(f"Using LLM to adapt strategy for trigger: {trigger_type}")
        
        # Get Llama4ScoutMCP
        try:
            llm = self.mcp_registry.get_mcp("Llama4ScoutMCP")
            if not llm:
                logger.warning("Llama4ScoutMCP not available, skipping LLM-based adaptation")
                return
        except KeyError:
            logger.warning("Llama4ScoutMCP not found in registry, skipping LLM-based adaptation")
            return
        
        # Get remaining steps
        remaining_steps = strategy.get_remaining_steps([i for i in range(current_step_idx + 1)])
        remaining_steps_data = [strategy.steps[i].to_dict() for i in remaining_steps]
        
        # Get available techniques/MCPs
        available_techniques = list(self.technique_registry.keys())
        available_mcps = list(self.mcp_registry.get_all_mcps().keys())
        
        # Construct prompt for adaptation
        system_prompt = """You are an expert analytical strategist specializing in adapting analytical workflows. 
        Your task is to modify the remaining steps in an analysis strategy based on a trigger event. 
        Provide your adapted strategy as a list of steps in JSON format, where each step has:
        - technique: Name of the analytical technique or MCP to use
        - purpose: Clear explanation of why this step is needed
        - parameters: Parameters for the technique/MCP
        - dependencies: List of step indices that must be completed before this step
        
        Ensure your adaptation addresses the specific trigger while maintaining a coherent workflow.
        """
        
        # Build prompt with context summary and trigger information
        prompt = f"ADAPTATION TRIGGER: {trigger_type}\n\n"
        
        # Add question and analysis info
        prompt += f"QUESTION: {context.question}\n\n"
        
        if context.question_analysis_output:
            prompt += f"QUESTION TYPE: {context.question_analysis_output.question_type}\n"
            prompt += f"COMPLEXITY: {context.question_analysis_output.complexity}\n\n"
        
        # Add information about the current step that triggered adaptation
        current_step = strategy.steps[current_step_idx]
        prompt += f"CURRENT STEP: {current_step.technique}\n"
        prompt += f"CURRENT STEP PARAMETERS: {json.dumps(current_step.parameters)}\n\n"
        
        # Add remaining steps
        prompt += "REMAINING STEPS IN CURRENT STRATEGY:\n"
        for i, step_data in enumerate(remaining_steps_data):
            prompt += f"{i+1}. {step_data['technique']}"
            if 'purpose' in step_data.get('parameters', {}):
                prompt += f" - {step_data['parameters']['purpose']}"
            prompt += "\n"
        prompt += "\n"
        
        # Add available components
        prompt += "AVAILABLE TECHNIQUES:\n"
        for technique in available_techniques:
            prompt += f"- {technique}\n"
        prompt += "\n"
        
        prompt += "AVAILABLE MCPs:\n"
        for mcp in available_mcps:
            prompt += f"- {mcp}\n"
        prompt += "\n"
        
        prompt += f"Please provide an adapted list of steps to replace the remaining steps in the strategy. Your adaptation should address the '{trigger_type}' trigger while ensuring a coherent analytical workflow. Return your adapted steps as valid JSON."
        
        # Call LLM for adaptation
        try:
            llm_input = {
                "operation": "adapt_strategy",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "output_format": "json"
            }
            
            llm_response = llm.process(llm_input)
            
            # Check for errors
            if isinstance(llm_response, dict) and "error" in llm_response:
                logger.error(f"Error adapting strategy: {llm_response['error']}")
                return
            
            # Parse and validate the result
            try:
                # Extract content based on response format
                if isinstance(llm_response, str):
                    content = llm_response
                elif isinstance(llm_response, dict) and "content" in llm_response:
                    content = llm_response["content"]
                else:
                    content = json.dumps(llm_response)
                
                # Try to extract JSON from the content
                try:
                    parsed_result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text using regex
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON-like content
                        json_like = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_like:
                            parsed_result = json.loads(json_like.group(1))
                        else:
                            raise ValueError("Could not extract JSON from LLM response")
                
                # Get adapted steps
                adapted_steps = None
                if "steps" in parsed_result:
                    adapted_steps = parsed_result["steps"]
                elif "adapted_steps" in parsed_result:
                    adapted_steps = parsed_result["adapted_steps"]
                else:
                    # If the result is a list, assume it's the steps
                    if isinstance(parsed_result, list):
                        adapted_steps = parsed_result
                
                if not adapted_steps or not isinstance(adapted_steps, list):
                    logger.error("Invalid adaptation format: steps missing or not a list")
                    return
                
                # Validate steps
                valid_steps = []
                for step_data in adapted_steps:
                    # Validate required fields
                    if "technique" not in step_data:
                        logger.warning("Adapted step missing required 'technique' field, skipping")
                        continue
                    
                    technique_name = step_data["technique"]
                    
                    # Validate technique exists in registry
                    technique_exists = technique_name in self.technique_registry
                    mcp_exists = technique_name in available_mcps
                    
                    if not (technique_exists or mcp_exists):
                        logger.warning(f"Technique/MCP '{technique_name}' not found in registries, skipping")
                        continue
                    
                    valid_steps.append(step_data)
                
                # If no valid steps, return without adaptation
                if not valid_steps:
                    logger.warning("No valid steps in adaptation, skipping")
                    return
                
                # Replace remaining steps with adapted steps
                # First, remove all steps after current_step_idx
                for _ in range(len(strategy.steps) - 1, current_step_idx, -1):
                    strategy.remove_step(len(strategy.steps) - 1)
                
                # Then add the new steps
                for step_data in valid_steps:
                    technique_name = step_data["technique"]
                    purpose = step_data.get("purpose", "")
                    parameters = step_data.get("parameters", {})
                    dependencies = step_data.get("dependencies", [])
                    optional = step_data.get("optional", False)
                    
                    # Add purpose to parameters if provided
                    if purpose:
                        parameters["purpose"] = purpose
                    
                    # Adjust dependencies to be relative to current_step_idx
                    adjusted_dependencies = [dep + current_step_idx + 1 if isinstance(dep, int) else current_step_idx for dep in dependencies]
                    
                    # Add step to strategy
                    strategy.add_step(technique_name, parameters, adjusted_dependencies, optional)
                
                # Log success
                logger.info(f"Adapted strategy with {len(valid_steps)} new steps for trigger: {trigger_type}")
                context.add_event("adaptation", f"LLM-based strategy adaptation for {trigger_type}", 
                                 {"trigger": trigger_type, "new_steps_count": len(valid_steps)})
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing adaptation: {str(e)}")
                return
                
        except Exception as e:
            logger.error(f"Error adapting strategy: {str(e)}", exc_info=True)
            return
    
    def _handle_execution_error(self, context: AnalysisContext, strategy: AnalysisStrategy, failed_step_info: Dict, error: Exception) -> str:
        """
        Handle execution errors in the workflow.
        
        Args:
            context: Analysis context
            strategy: Analysis strategy
            failed_step_info: Information about the failed step
            error: The exception that occurred
            
        Returns:
            Action to take: 'skip', 'retry', 'halt', 'fallback_success', 'fallback_failure'
        """
        logger.error(f"Execution error in step: {failed_step_info.get('technique', 'Unknown')}", exc_info=True)
        
        # Get technique name
        technique_name = failed_step_info.get('technique', 'Unknown')
        
        # Get retry count from context
        retry_counts = context.get_parameter('retry_counts', {})
        current_retry_count = retry_counts.get(technique_name, 0)
        
        # Check if retry limit reached
        max_retries = 2  # Maximum number of retries
        if current_retry_count < max_retries:
            # Increment retry count
            retry_counts[technique_name] = current_retry_count + 1
            context.set_parameter('retry_counts', retry_counts)
            
            # Log retry
            logger.info(f"Retrying step: {technique_name} (Attempt {current_retry_count + 1}/{max_retries})")
            context.add_event("error", f"Retrying failed step: {technique_name}", 
                             {"error": str(error), "retry_count": current_retry_count + 1})
            
            return 'retry'
        
        # Check if technique has fallback method
        try:
            # Get technique instance
            technique_instance = self.technique_registry.get(technique_name)
            
            if technique_instance and hasattr(technique_instance, 'fallback') and callable(getattr(technique_instance, 'fallback')):
                # Call fallback method
                logger.info(f"Attempting fallback for step: {technique_name}")
                
                try:
                    fallback_result = technique_instance.fallback(context, error)
                    
                    # Store fallback result
                    context.add_technique_result(f"{technique_name}_fallback", fallback_result)
                    
                    # Log fallback success
                    logger.info(f"Fallback successful for step: {technique_name}")
                    context.add_event("recovery", f"Fallback successful for step: {technique_name}", 
                                     {"error": str(error), "fallback_result": "success"})
                    
                    return 'fallback_success'
                    
                except Exception as fallback_error:
                    # Log fallback failure
                    logger.error(f"Fallback failed for step: {technique_name}: {str(fallback_error)}")
                    context.add_event("error", f"Fallback failed for step: {technique_name}", 
                                     {"error": str(error), "fallback_error": str(fallback_error)})
                    
                    return 'fallback_failure'
        except Exception as e:
            logger.error(f"Error checking for fallback: {str(e)}")
        
        # Determine if step is critical
        is_critical = not failed_step_info.get('optional', False)
        
        if is_critical:
            # Log critical failure
            logger.error(f"Critical step failed: {technique_name}, halting workflow")
            context.add_event("error", f"Critical step failed: {technique_name}, halting workflow", 
                             {"error": str(error)})
            
            return 'halt'
        else:
            # Log skipping non-critical step
            logger.warning(f"Non-critical step failed: {technique_name}, skipping")
            context.add_event("error", f"Non-critical step failed: {technique_name}, skipping", 
                             {"error": str(error)})
            
            return 'skip'
    
    def _create_workflow(self, input_data: Dict) -> Dict:
        """Create a new workflow."""
        # Implementation omitted for brevity
        pass
    
    def _execute_workflow(self, input_data: Dict) -> Dict:
        """Execute a workflow."""
        # Implementation omitted for brevity
        pass
    
    def _update_workflow(self, input_data: Dict) -> Dict:
        """Update a workflow."""
        # Implementation omitted for brevity
        pass
    
    def _get_workflow_status(self, input_data: Dict) -> Dict:
        """Get workflow status."""
        # Implementation omitted for brevity
        pass
    
    def _get_available_techniques_info(self, input_data: Dict) -> Dict:
        """Get information about available techniques."""
        # Implementation omitted for brevity
        pass
    
    def _execute_analysis(self, input_data: Dict) -> Dict:
        """Execute an analysis."""
        # Implementation omitted for brevity
        pass
