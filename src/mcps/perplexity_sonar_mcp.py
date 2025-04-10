"""
Perplexity Sonar MCP for comprehensive research using Perplexity Sonar API.
This module provides the PerplexitySonarMCP class for advanced research capabilities.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
import requests

from src.base_mcp import BaseMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerplexitySonarMCP(BaseMCP):
    """
    Perplexity Sonar MCP for comprehensive research using Perplexity Sonar API.
    
    This MCP provides capabilities for:
    1. Comprehensive research on analytical questions
    2. Key insight extraction
    3. Initial hypothesis formulation
    4. Workflow recommendation
    5. Evidence gathering and evaluation
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the PerplexitySonarMCP.
        
        Args:
            api_key: Perplexity API key (if None, will try to get from environment variable)
        """
        super().__init__(
            name="perplexity_sonar",
            description="Comprehensive research using Perplexity Sonar API",
            version="1.0.0"
        )
        
        # Set API key
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            logger.warning("No Perplexity API key provided, research functionality will be limited")
        
        # Set model parameters
        self.model = "sonar"
        self.deep_research_model = "sonar-deep-research"
        
        # Operation handlers
        self.operation_handlers = {
            "research": self._research,
            "deep_research": self._deep_research,
            "extract_insights": self._extract_insights,
            "formulate_hypotheses": self._formulate_hypotheses,
            "recommend_workflow": self._recommend_workflow,
            "gather_evidence": self._gather_evidence,
            "evaluate_evidence": self._evaluate_evidence
        }
        
        logger.info("Initialized PerplexitySonarMCP")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processing results
        """
        logger.info("Processing input in PerplexitySonarMCP")
        
        # Validate input
        if not isinstance(input_data, dict):
            error_msg = "Input must be a dictionary"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get operation type
        operation = input_data.get("operation")
        if not operation:
            error_msg = "No operation specified"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Check if operation is supported
        if operation not in self.operation_handlers:
            error_msg = f"Unsupported operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Handle operation
        try:
            result = self.operation_handlers[operation](input_data)
            return result
        except Exception as e:
            error_msg = f"Error processing operation {operation}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _call_perplexity_api(self, query: str, model: str = None, follow_up: bool = False, previous_query_id: str = None) -> Dict:
        """
        Call the Perplexity API with a query.
        
        Args:
            query: Query string
            model: Model to use (if None, uses default)
            follow_up: Whether this is a follow-up query
            previous_query_id: ID of previous query (for follow-ups)
            
        Returns:
            API response
        """
        # Check if API key is available
        if not self.api_key:
            logger.warning("No API key available, using mock Perplexity response")
            return self._mock_perplexity_response(query)
        
        # Set default model if not provided
        if model is None:
            model = self.model
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "query": query
        }
        
        # Add follow-up parameters if needed
        if follow_up and previous_query_id:
            data["follow_up"] = True
            data["previous_query_id"] = previous_query_id
        
        # Call API
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data
            )
            
            # Check for errors
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "details": response.text}
            
            # Parse response
            result = response.json()
            
            # Extract content
            content = result["choices"][0]["message"]["content"]
            query_id = result.get("id")
            
            return {"content": content, "query_id": query_id, "model": model}
            
        except Exception as e:
            logger.error(f"Error calling Perplexity API: {str(e)}")
            return {"error": f"Error calling Perplexity API: {str(e)}"}
    
    def _mock_perplexity_response(self, query: str) -> Dict:
        """
        Generate a mock Perplexity response for testing without API access.
        
        Args:
            query: Query string
            
        Returns:
            Mock API response
        """
        logger.info("Generating mock Perplexity response")
        
        # Generate different responses based on query keywords
        if "research" in query.lower():
            return {
                "content": """Based on my research, here are the key findings:

1. The global economy is expected to grow at 3.2% in 2025, with emerging markets outpacing developed economies.
2. Inflation concerns remain, but central banks are projected to maintain stable interest rates through Q2 2025.
3. Technological innovation in AI and renewable energy continues to drive productivity gains across sectors.
4. Geopolitical tensions in Eastern Europe and the South China Sea present ongoing risks to global supply chains.
5. Climate change adaptation measures are becoming increasingly important for long-term economic planning.

These findings are based on recent reports from the IMF, World Bank, and leading economic research institutions. The data suggests a cautiously optimistic outlook, though significant uncertainties remain regarding geopolitical developments and potential new variants of concern in the public health sphere.""",
                "query_id": "mock-query-id-12345",
                "model": "mock-sonar"
            }
        
        elif "evidence" in query.lower():
            return {
                "content": """I've gathered the following evidence related to your query:

1. According to the latest IMF World Economic Outlook (April 2025), global growth projections have been revised upward by 0.3 percentage points.
2. The Federal Reserve's March 2025 meeting minutes indicate a consensus view that inflation has stabilized within the target range.
3. A recent McKinsey Global Institute report quantifies AI-driven productivity gains at 1.2-3.5% annually across industries.
4. The UN Security Council issued three resolutions in Q1 2025 addressing territorial disputes in contested maritime zones.
5. The IPCC's 2025 special report documents accelerating climate impacts on agricultural productivity in tropical and subtropical regions.

These sources are highly credible and represent the most current available data. The evidence presents a complex picture with both positive economic indicators and concerning risk factors that should be considered in any comprehensive analysis.""",
                "query_id": "mock-query-id-67890",
                "model": "mock-sonar"
            }
        
        elif "hypotheses" in query.lower():
            return {
                "content": """Based on the available evidence, I've formulated the following hypotheses:

1. H1: The combination of stable monetary policy and technological productivity gains will sustain economic growth despite geopolitical tensions.
   - Supporting evidence: IMF growth projections, Fed stability, documented AI productivity gains
   - Confidence: Medium-high (70-80%)

2. H2: Supply chain disruptions due to geopolitical conflicts will offset technological gains, leading to stagflation.
   - Supporting evidence: UN Security Council resolutions, ongoing territorial disputes
   - Confidence: Medium (50-60%)

3. H3: Climate change impacts will become the dominant economic factor by Q4 2025, overshadowing other considerations.
   - Supporting evidence: IPCC report on agricultural productivity impacts
   - Confidence: Medium-low (30-40%)

These hypotheses represent competing explanations that account for the available evidence. The first hypothesis currently has the strongest evidential support, but significant uncertainties remain that could shift the balance toward the alternative explanations.""",
                "query_id": "mock-query-id-24680",
                "model": "mock-sonar"
            }
        
        else:
            return {
                "content": "This is a mock response from the Perplexity API. In a real implementation, this would contain comprehensive research results based on your query.",
                "query_id": "mock-query-id-13579",
                "model": "mock-sonar"
            }
    
    def _research(self, input_data: Dict) -> Dict:
        """
        Conduct comprehensive research on a question.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Research results
        """
        logger.info("Conducting research")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Prepare research query
        research_query = f"Conduct comprehensive research on the following analytical question. Provide detailed findings with sources and evidence: {question}"
        
        # Call Perplexity API
        research_response = self._call_perplexity_api(research_query, self.model)
        
        # Check for errors
        if "error" in research_response:
            return research_response
        
        # Extract research data
        research_data = {
            "question": question,
            "research_content": research_response["content"],
            "model": research_response["model"],
            "query_id": research_response.get("query_id"),
            "timestamp": time.time()
        }
        
        # Extract key insights
        insights_query = f"Based on the following research, extract the 5-7 most important insights that are relevant to answering this analytical question: {question}\n\nResearch:\n{research_response['content']}"
        
        insights_response = self._call_perplexity_api(
            insights_query, 
            self.model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract key insights
        key_insights = []
        if "error" not in insights_response:
            # Try to parse insights from the response
            insights_content = insights_response["content"]
            
            # Simple parsing for numbered lists
            import re
            insights_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', insights_content, re.DOTALL)
            if insights_matches:
                key_insights = [insight.strip() for insight in insights_matches]
            else:
                # If no numbered list, try to split by newlines
                lines = insights_content.split('\n')
                key_insights = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Formulate initial hypotheses
        hypotheses_query = f"Based on the following research, formulate 3-5 competing hypotheses that could answer this analytical question. For each hypothesis, provide supporting evidence and an initial confidence assessment: {question}\n\nResearch:\n{research_response['content']}"
        
        hypotheses_response = self._call_perplexity_api(
            hypotheses_query, 
            self.model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract initial hypotheses
        initial_hypotheses = []
        if "error" not in hypotheses_response:
            # Try to parse hypotheses from the response
            hypotheses_content = hypotheses_response["content"]
            
            # Simple parsing for hypotheses
            import re
            hypotheses_matches = re.findall(r'(?:H\d+|Hypothesis \d+):\s+(.*?)(?=(?:H\d+|Hypothesis \d+):|$)', hypotheses_content, re.DOTALL)
            if hypotheses_matches:
                initial_hypotheses = [hypothesis.strip() for hypothesis in hypotheses_matches]
            else:
                # If no clear hypothesis format, try to split by numbered list
                hypotheses_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', hypotheses_content, re.DOTALL)
                if hypotheses_matches:
                    initial_hypotheses = [hypothesis.strip() for hypothesis in hypotheses_matches]
        
        # Recommend workflow
        workflow_query = f"Based on the following research and the nature of this analytical question, recommend an optimal workflow of analytical techniques to answer it. Consider the question type, complexity, and available evidence: {question}\n\nResearch:\n{research_response['content']}"
        
        workflow_response = self._call_perplexity_api(
            workflow_query, 
            self.model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract workflow recommendation
        recommended_workflow = {}
        if "error" not in workflow_response:
            recommended_workflow = {
                "recommendation": workflow_response["content"],
                "timestamp": time.time()
            }
        
        # Compile results
        results = {
            "research_data": research_data,
            "key_insights": key_insights,
            "initial_hypotheses": initial_hypotheses,
            "recommended_workflow": recommended_workflow
        }
        
        return results
    
    def _deep_research(self, input_data: Dict) -> Dict:
        """
        Conduct deep research on a question using the deep research model.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Deep research results
        """
        logger.info("Conducting deep research")
        
        # Get question
        question = input_data.get("question", "")
        if not question:
            error_msg = "No question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get focus areas if provided
        focus_areas = input_data.get("focus_areas", [])
        
        # Prepare research query
        research_query = f"Conduct comprehensive, in-depth research on the following analytical question. Provide detailed findings with sources, evidence, and expert perspectives: {question}"
        
        if focus_areas:
            research_query += "\n\nPlease focus on the following specific areas:\n"
            for area in focus_areas:
                research_query += f"- {area}\n"
        
        # Call Perplexity API with deep research model
        research_response = self._call_perplexity_api(research_query, self.deep_research_model)
        
        # Check for errors
        if "error" in research_response:
            return research_response
        
        # Extract research data
        research_data = {
            "question": question,
            "focus_areas": focus_areas,
            "research_content": research_response["content"],
            "model": research_response["model"],
            "query_id": research_response.get("query_id"),
            "timestamp": time.time()
        }
        
        # Extract key insights
        insights_query = f"Based on the following in-depth research, extract the 7-10 most important insights that are relevant to answering this analytical question. For each insight, provide the supporting evidence and source: {question}\n\nResearch:\n{research_response['content']}"
        
        insights_response = self._call_perplexity_api(
            insights_query, 
            self.deep_research_model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract key insights
        key_insights = []
        if "error" not in insights_response:
            # Try to parse insights from the response
            insights_content = insights_response["content"]
            
            # Simple parsing for numbered lists
            import re
            insights_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', insights_content, re.DOTALL)
            if insights_matches:
                key_insights = [insight.strip() for insight in insights_matches]
            else:
                # If no numbered list, try to split by newlines
                lines = insights_content.split('\n')
                key_insights = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Formulate initial hypotheses
        hypotheses_query = f"Based on the following in-depth research, formulate 4-6 competing hypotheses that could answer this analytical question. For each hypothesis, provide supporting evidence, potential counterevidence, and an initial confidence assessment: {question}\n\nResearch:\n{research_response['content']}"
        
        hypotheses_response = self._call_perplexity_api(
            hypotheses_query, 
            self.deep_research_model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract initial hypotheses
        initial_hypotheses = []
        if "error" not in hypotheses_response:
            # Try to parse hypotheses from the response
            hypotheses_content = hypotheses_response["content"]
            
            # Simple parsing for hypotheses
            import re
            hypotheses_matches = re.findall(r'(?:H\d+|Hypothesis \d+):\s+(.*?)(?=(?:H\d+|Hypothesis \d+):|$)', hypotheses_content, re.DOTALL)
            if hypotheses_matches:
                initial_hypotheses = [hypothesis.strip() for hypothesis in hypotheses_matches]
            else:
                # If no clear hypothesis format, try to split by numbered list
                hypotheses_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', hypotheses_content, re.DOTALL)
                if hypotheses_matches:
                    initial_hypotheses = [hypothesis.strip() for hypothesis in hypotheses_matches]
        
        # Identify key uncertainties
        uncertainties_query = f"Based on the following in-depth research, identify the 5-7 key uncertainties that affect our ability to answer this analytical question with high confidence. For each uncertainty, explain its impact and whether additional research could reduce it: {question}\n\nResearch:\n{research_response['content']}"
        
        uncertainties_response = self._call_perplexity_api(
            uncertainties_query, 
            self.deep_research_model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract key uncertainties
        key_uncertainties = []
        if "error" not in uncertainties_response:
            # Try to parse uncertainties from the response
            uncertainties_content = uncertainties_response["content"]
            
            # Simple parsing for numbered lists
            import re
            uncertainties_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', uncertainties_content, re.DOTALL)
            if uncertainties_matches:
                key_uncertainties = [uncertainty.strip() for uncertainty in uncertainties_matches]
            else:
                # If no numbered list, try to split by newlines
                lines = uncertainties_content.split('\n')
                key_uncertainties = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Recommend workflow
        workflow_query = f"Based on the following in-depth research and the nature of this analytical question, recommend an optimal workflow of analytical techniques to answer it. Consider the question type, complexity, available evidence, and key uncertainties. Explain why each technique is appropriate and how they should be sequenced: {question}\n\nResearch:\n{research_response['content']}"
        
        workflow_response = self._call_perplexity_api(
            workflow_query, 
            self.deep_research_model, 
            follow_up=True, 
            previous_query_id=research_response.get("query_id")
        )
        
        # Extract workflow recommendation
        recommended_workflow = {}
        if "error" not in workflow_response:
            recommended_workflow = {
                "recommendation": workflow_response["content"],
                "timestamp": time.time()
            }
        
        # Compile results
        results = {
            "research_data": research_data,
            "key_insights": key_insights,
            "initial_hypotheses": initial_hypotheses,
            "key_uncertainties": key_uncertainties,
            "recommended_workflow": recommended_workflow
        }
        
        return results
    
    def _extract_insights(self, input_data: Dict) -> Dict:
        """
        Extract key insights from research content.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Extracted insights
        """
        logger.info("Extracting insights")
        
        # Get research content
        research_content = input_data.get("research_content", "")
        if not research_content:
            error_msg = "No research content provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get question
        question = input_data.get("question", "")
        
        # Prepare query
        if question:
            query = f"Based on the following research, extract the 5-7 most important insights that are relevant to answering this analytical question: {question}\n\nResearch:\n{research_content}"
        else:
            query = f"Extract the 5-7 most important insights from the following research content:\n\n{research_content}"
        
        # Call Perplexity API
        response = self._call_perplexity_api(query, self.model)
        
        # Check for errors
        if "error" in response:
            return response
        
        # Extract insights
        insights_content = response["content"]
        
        # Try to parse insights from the response
        import re
        insights = []
        
        # Try to find numbered list
        insights_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', insights_content, re.DOTALL)
        if insights_matches:
            insights = [insight.strip() for insight in insights_matches]
        else:
            # If no numbered list, try to split by newlines
            lines = insights_content.split('\n')
            insights = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Compile results
        results = {
            "insights": insights,
            "raw_response": insights_content,
            "model": response["model"],
            "timestamp": time.time()
        }
        
        return results
    
    def _formulate_hypotheses(self, input_data: Dict) -> Dict:
        """
        Formulate hypotheses based on research content.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Formulated hypotheses
        """
        logger.info("Formulating hypotheses")
        
        # Get research content
        research_content = input_data.get("research_content", "")
        if not research_content:
            error_msg = "No research content provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get question
        question = input_data.get("question", "")
        
        # Prepare query
        if question:
            query = f"Based on the following research, formulate 3-5 competing hypotheses that could answer this analytical question. For each hypothesis, provide supporting evidence and an initial confidence assessment: {question}\n\nResearch:\n{research_content}"
        else:
            query = f"Formulate 3-5 competing hypotheses based on the following research content. For each hypothesis, provide supporting evidence and an initial confidence assessment:\n\n{research_content}"
        
        # Call Perplexity API
        response = self._call_perplexity_api(query, self.model)
        
        # Check for errors
        if "error" in response:
            return response
        
        # Extract hypotheses
        hypotheses_content = response["content"]
        
        # Try to parse hypotheses from the response
        import re
        hypotheses = []
        
        # Try to find hypothesis format (H1, Hypothesis 1, etc.)
        hypotheses_matches = re.findall(r'(?:H\d+|Hypothesis \d+):\s+(.*?)(?=(?:H\d+|Hypothesis \d+):|$)', hypotheses_content, re.DOTALL)
        if hypotheses_matches:
            hypotheses = [hypothesis.strip() for hypothesis in hypotheses_matches]
        else:
            # If no clear hypothesis format, try to split by numbered list
            hypotheses_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', hypotheses_content, re.DOTALL)
            if hypotheses_matches:
                hypotheses = [hypothesis.strip() for hypothesis in hypotheses_matches]
        
        # Compile results
        results = {
            "hypotheses": hypotheses,
            "raw_response": hypotheses_content,
            "model": response["model"],
            "timestamp": time.time()
        }
        
        return results
    
    def _recommend_workflow(self, input_data: Dict) -> Dict:
        """
        Recommend an analytical workflow based on research content.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Workflow recommendation
        """
        logger.info("Recommending workflow")
        
        # Get research content
        research_content = input_data.get("research_content", "")
        if not research_content:
            error_msg = "No research content provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get question
        question = input_data.get("question", "")
        
        # Get available techniques if provided
        available_techniques = input_data.get("available_techniques", [])
        
        # Prepare query
        if question:
            query = f"Based on the following research and the nature of this analytical question, recommend an optimal workflow of analytical techniques to answer it. Consider the question type, complexity, and available evidence: {question}\n\nResearch:\n{research_content}"
        else:
            query = f"Recommend an optimal workflow of analytical techniques based on the following research content. Consider the content type, complexity, and available evidence:\n\n{research_content}"
        
        if available_techniques:
            query += "\n\nAvailable analytical techniques:\n"
            for technique in available_techniques:
                query += f"- {technique}\n"
        
        # Call Perplexity API
        response = self._call_perplexity_api(query, self.model)
        
        # Check for errors
        if "error" in response:
            return response
        
        # Extract workflow recommendation
        workflow_content = response["content"]
        
        # Try to parse workflow steps if available techniques were provided
        workflow_steps = []
        if available_techniques:
            # Try to find techniques mentioned in the response
            for technique in available_techniques:
                if technique.lower() in workflow_content.lower():
                    workflow_steps.append(technique)
        
        # Compile results
        results = {
            "recommendation": workflow_content,
            "workflow_steps": workflow_steps,
            "model": response["model"],
            "timestamp": time.time()
        }
        
        return results
    
    def _gather_evidence(self, input_data: Dict) -> Dict:
        """
        Gather evidence related to a specific hypothesis or question.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Gathered evidence
        """
        logger.info("Gathering evidence")
        
        # Get hypothesis or question
        hypothesis = input_data.get("hypothesis", "")
        question = input_data.get("question", "")
        
        if not hypothesis and not question:
            error_msg = "No hypothesis or question provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Prepare query
        if hypothesis:
            query = f"Gather specific evidence related to the following hypothesis. Include supporting evidence, contradicting evidence, and sources: {hypothesis}"
        else:
            query = f"Gather specific evidence related to the following analytical question. Include relevant facts, data points, expert opinions, and sources: {question}"
        
        # Call Perplexity API with deep research model for better evidence gathering
        response = self._call_perplexity_api(query, self.deep_research_model)
        
        # Check for errors
        if "error" in response:
            return response
        
        # Extract evidence
        evidence_content = response["content"]
        
        # Try to parse evidence items
        import re
        evidence_items = []
        
        # Try to find numbered list
        evidence_matches = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', evidence_content, re.DOTALL)
        if evidence_matches:
            evidence_items = [evidence.strip() for evidence in evidence_matches]
        else:
            # If no numbered list, try to split by newlines or bullet points
            bullet_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', evidence_content, re.DOTALL)
            if bullet_matches:
                evidence_items = [evidence.strip() for evidence in bullet_matches]
            else:
                # If no clear format, try to split by newlines
                lines = evidence_content.split('\n')
                evidence_items = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        # Compile results
        results = {
            "evidence_items": evidence_items,
            "raw_response": evidence_content,
            "model": response["model"],
            "query_id": response.get("query_id"),
            "timestamp": time.time()
        }
        
        return results
    
    def _evaluate_evidence(self, input_data: Dict) -> Dict:
        """
        Evaluate the credibility and relevance of evidence.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Evidence evaluation
        """
        logger.info("Evaluating evidence")
        
        # Get evidence items
        evidence_items = input_data.get("evidence_items", [])
        if not evidence_items:
            error_msg = "No evidence items provided"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get hypothesis or question if provided
        hypothesis = input_data.get("hypothesis", "")
        question = input_data.get("question", "")
        
        # Prepare query
        query = "Evaluate the credibility and relevance of the following evidence items. For each item, assess source reliability, potential biases, and strength of evidence."
        
        if hypothesis:
            query += f" Consider how each item supports or contradicts this hypothesis: {hypothesis}"
        elif question:
            query += f" Consider how each item helps answer this question: {question}"
        
        query += "\n\nEvidence items:\n"
        for i, item in enumerate(evidence_items):
            query += f"{i+1}. {item}\n"
        
        # Call Perplexity API
        response = self._call_perplexity_api(query, self.model)
        
        # Check for errors
        if "error" in response:
            return response
        
        # Extract evaluation
        evaluation_content = response["content"]
        
        # Compile results
        results = {
            "evaluation": evaluation_content,
            "model": response["model"],
            "timestamp": time.time()
        }
        
        return results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this MCP.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "models": [self.model, self.deep_research_model],
            "operations": list(self.operation_handlers.keys()),
            "api_available": bool(self.api_key)
        }
