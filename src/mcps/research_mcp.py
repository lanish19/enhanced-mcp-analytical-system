"""
Research MCP implementation.
This module provides the ResearchMCP class for conducting research.
"""

import logging
import json
import time
import os
import hashlib
import requests
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import urllib.parse
import re

from src.base_mcp import BaseMCP
from src.utils.llm_integration import call_llm, extract_content, parse_json_response, MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchMCP(BaseMCP):
    """
    Research MCP for conducting research.

    This MCP provides capabilities for:
    1. Conducting web searches
    2. Extracting content from web pages
    3. Analyzing research findings
    4. Generating hypotheses based on research
    """
    
    def __init__(self, config=None):
        """
        Initialize the ResearchMCP.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("research_mcp", "Conducts research and generates hypotheses")
        
        # Initialize configuration
        self.config = config or {}
        
        # API configuration
        self.brave_api_key = self.config.get("brave_api_key", os.environ.get("BRAVE_API_KEY", ""))
        self.brave_api_base = "https://api.search.brave.com/res/v1/web/search"
        self.semantic_scholar_api_base = "https://api.semanticscholar.org/graph/v1"
        
        # Cache configuration
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_dir = self.config.get("cache_dir", "cache/research")
        self.search_cache = {}
        self.content_cache = {}
        
        # Retry configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1)
        
        # Create cache directory if it doesn't exist
        if self.cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized ResearchMCP with cache {'enabled' if self.cache_enabled else 'disabled'}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get a list of capabilities provided by this MCP.
        
        Returns:
            List of capability names
        """
        return ["web_search", "academic_search", "news_search", "content_extraction", "hypothesis_generation"]
    
    def get_research(self, query: str) -> Dict[str, Any]:
        """
        Conduct a research based on the provided query
        
        Args:
            query (str): The search query string.
        
        Returns:
            Dict[str, Any]: A dictionary containing a list of research findings.
                Example:
                {
                    "research_results": [
                        "Research finding 1",
                        "Research finding 2",
                        "Research finding 3"
                    ]
                }
        """
        logger.info(f"Conducting research for query: {query}")

        prompt = f"""
            Provide a list of concise and relevant research findings related to the following topic: {query}. 
            Each finding should be a single sentence.
            
            Return a JSON object with a single key named `research_results`, whose value is a list of strings, where each string is a research finding.
            Example:
            {{
                "research_results": [
                    "Research finding 1",
                    "Research finding 2",
                    "Research finding 3"
                ]
            }}
        """

        model_config = MODEL_CONFIG["llama4"]
        model_config["temperature"] = 0.7

        try:
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            parsed_response = parse_json_response(content)

            if parsed_response.get("fallback_generated"):
                logger.error(f"Error in LLM research call: {parsed_response.get('error')}")
                return {"research_results": []}
            
            research_results = parsed_response.get("research_results", [])
        except Exception as e:
            logger.error(f"Error during LLM research call: {e}")
            return {"research_results": []}

        return {"research_results": research_results}

    def process(self, context: Dict) -> Dict:
        """
        Process a research request.
        
        Args:
            context: Dictionary containing research parameters
            
        Returns:
            Dictionary containing research results
        """
        operation = context.get("operation", "search")
        
        if operation == "search":
            return self._process_search(context)
        elif operation == "research_to_hypothesis":
            return self._process_research_to_hypothesis(context)
        elif operation == "extract_content":
            return self._process_extract_content(context)
        elif operation == "analyze_results":
            return self._process_analyze_results(context)
        else:
            error_msg = f"Unknown operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "failed"}
    
    def _process_search(self, context: Dict) -> Dict:
        """
        Process a search request.
        
        Args:
            context: Dictionary containing search parameters
            
        Returns:
            Dictionary containing search results
        """
        query = context.get("query")
        search_type = context.get("search_type", "web")
        num_results = context.get("num_results", 10)
        
        if not query:
            return {"error": "No query provided", "status": "failed"}
        
        # Conduct search
        search_results = self.conduct_search(query, search_type, num_results)
        
        # Extract content from top results if requested
        if context.get("extract_content", False):
            for i, result in enumerate(search_results.get("results", [])[:3]):  # Extract content from top 3 results
                url = result.get("url")
                if url:
                    content = self.extract_content(url)
                    search_results["results"][i]["extracted_content"] = content
        
        # Analyze results if requested
        if context.get("analyze_results", False):
            analysis = self.analyze_search_results(search_results)
            search_results["analysis"] = analysis
        
        search_results["status"] = "success"
        return search_results
    
    def _process_research_to_hypothesis(self, context: Dict) -> Dict:
        """
        Process a research to hypothesis request.
        
        Args:
            context: Dictionary containing research parameters
            
        Returns:
            Dictionary containing research results and generated hypotheses
        """
        question = context.get("question", "")
        analysis_context = context.get("context", {})
        
        if not question:
            return {"error": "No question provided", "status": "failed"}
        
        try:
            # Step 1: Conduct research on the question
            logger.info(f"Conducting research for question: {question}")
            search_results = self.conduct_search(question, "web", 5)
            
            # Step 2: Extract key findings from search results
            key_findings = []
            for result in search_results.get("results", [])[:3]:
                url = result.get("url")
                if url:
                    try:
                        content_result = self.extract_content(url)
                        if content_result.get("status") == "success":
                            content = content_result.get("content", "")
                            # Truncate content to avoid token limits
                            if len(content) > 2000:
                                content = content[:2000] + "..."
                            key_findings.append({
                                "source": url,
                                "title": result.get("title", ""),
                                "content": content
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting content from {url}: {e}")
            
            # Step 3: Generate hypotheses based on research findings
            hypotheses = self._generate_hypotheses(question, key_findings)
            
            # Step 4: Prepare the result
            result = {
                "question": question,
                "search_results": search_results.get("results", []),
                "key_findings": key_findings,
                "hypotheses": hypotheses,
                "confidence": self._assess_confidence(key_findings, hypotheses),
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in research_to_hypothesis: {e}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed",
                "question": question,
                "hypotheses": [],
                "confidence": "low"
            }
    
    def _process_extract_content(self, context: Dict) -> Dict:
        """
        Process a content extraction request.
        
        Args:
            context: Dictionary containing extraction parameters
            
        Returns:
            Dictionary containing extracted content
        """
        url = context.get("url")
        
        if not url:
            return {"error": "No URL provided", "status": "failed"}
        
        content_result = self.extract_content(url)
        content_result["status"] = content_result.get("status", "success")
        return content_result
    
    def _process_analyze_results(self, context: Dict) -> Dict:
        """
        Process a results analysis request.
        
        Args:
            context: Dictionary containing analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        search_results = context.get("search_results", {})
        
        if not search_results or not search_results.get("results"):
            return {"error": "No search results provided", "status": "failed"}
        
        analysis = self.analyze_search_results(search_results)
        analysis["status"] = "success"
        return analysis
    
    def conduct_search(self, query: str, search_type: str = "web", num_results: int = 10) -> Dict[str, Any]:
        """
        Conduct a search using the specified search type.
        
        Args:
            query: Search query
            search_type: Type of search ("web", "academic", "news")
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Conducting {search_type} search for: {query}")
        
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = hashlib.md5(f"{query}:{search_type}:{num_results}".encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Using cached search results for: {query}")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}")
        
        try:
            # Select appropriate search method based on type
            if search_type == "academic":
                results = self._academic_search(query, num_results)
            elif search_type == "news":
                results = self._brave_search(query, num_results, "news")
            else:  # Default to web search
                results = self._brave_search(query, num_results)
            
            # Cache results if enabled
            if self.cache_enabled:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(results, f)
                except Exception as e:
                    logger.warning(f"Error writing cache file: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error conducting search: {e}")
            return self._handle_search_error(e, query)
    
    def _brave_search(self, query: str, num_results: int, search_type: str = "web") -> Dict[str, Any]:
        """
        Conduct a search using Brave Search API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            search_type: Type of search ("web" or "news")
            
        Returns:
            Dictionary containing search results
        """
        if not self.brave_api_key:
            logger.warning("Brave API key not provided, falling back to simulated results")
            return self._simulate_search_results(query, num_results)
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": min(num_results, 20),  # API limit is 20 per request
            "search_lang": "en"
        }
        
        if search_type == "news":
            params["search_type"] = "news"
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.brave_api_base, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Process and structure the results
                results = []
                for item in data.get("web", {}).get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("description", ""),
                        "source": "brave_search"
                    })
                
                return {
                    "query": query,
                    "search_type": search_type,
                    "timestamp": time.time(),
                    "results": results,
                    "total_results": len(results)
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Brave Search API request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
        
        # This should not be reached due to the raise in the loop
        return self._simulate_search_results(query, num_results)
    
    def _academic_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Conduct an academic search using Semantic Scholar API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # URL encode the query
        encoded_query = urllib.parse.quote(query)
        
        # Construct the API URL
        url = f"{self.semantic_scholar_api_base}/paper/search?query={encoded_query}&limit={num_results}&fields=title,url,abstract,year,authors,venue"
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Process and structure the results
                results = []
                for item in data.get("data", []):
                    authors = [author.get("name", "") for author in item.get("authors", [])]
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("abstract", ""),
                        "year": item.get("year"),
                        "authors": authors,
                        "venue": item.get("venue", ""),
                        "source": "semantic_scholar"
                    })
                
                return {
                    "query": query,
                    "search_type": "academic",
                    "timestamp": time.time(),
                    "results": results,
                    "total_results": len(results)
                }
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Semantic Scholar API request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Try Google Scholar as fallback
                    return self._google_scholar_search(query, num_results)
        
        # This should not be reached due to the fallback
        return self._simulate_search_results(query, num_results)
    
    def _google_scholar_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Conduct an academic search using Google Scholar (as fallback).
        Note: This is a simple scraping approach and may be blocked by Google.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Falling back to Google Scholar search for: {query}")
        
        # URL encode the query
        encoded_query = urllib.parse.quote(query)
        
        # Construct the URL
        url = f"https://scholar.google.com/scholar?q={encoded_query}&hl=en&as_sdt=0,5&num={num_results}"
        
        # Set user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            for item in soup.select('.gs_ri')[:num_results]:
                title_elem = item.select_one('.gs_rt')
                title = title_elem.text if title_elem else ""
                
                url_elem = title_elem.select_one('a') if title_elem else None
                url = url_elem['href'] if url_elem and 'href' in url_elem.attrs else ""
                
                desc_elem = item.select_one('.gs_rs')
                description = desc_elem.text if desc_elem else ""
                
                authors_year_elem = item.select_one('.gs_a')
                authors_year_text = authors_year_elem.text if authors_year_elem else ""
                
                # Extract year using regex
                year_match = re.search(r'\b(19|20)\d{2}\b', authors_year_text)
                year = year_match.group(0) if year_match else None
                
                results.append({
                    "title": title,
                    "url": url,
                    "description": description,
                    "authors_year": authors_year_text,
                    "year": year,
                    "source": "google_scholar"
                })
            
            return {
                "query": query,
                "search_type": "academic",
                "timestamp": time.time(),
                "results": results,
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            return self._simulate_search_results(query, num_results)
    
    def _handle_search_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Handle search errors and provide fallback results.
        
        Args:
            error: The exception that occurred
            query: The search query
            
        Returns:
            Dictionary containing fallback search results
        """
        logger.error(f"Search error for query '{query}': {error}")
        
        # Return simulated results as fallback
        return self._simulate_search_results(query, 5)
    
    def _simulate_search_results(self, query: str, num_results: int) -> Dict[str, Any]:
        """
        Generate simulated search results when real search fails.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Dictionary containing simulated search results
        """
        logger.warning(f"Generating simulated search results for: {query}")
        
        # Create simulated results
        results = []
        for i in range(min(num_results, 5)):
            results.append({
                "title": f"Simulated Result {i+1} for {query}",
                "url": f"https://example.com/result{i+1}",
                "description": f"This is a simulated search result for the query: {query}. Real search functionality is currently unavailable.",
                "source": "simulated"
            })
        
        return {
            "query": query,
            "search_type": "web",
            "timestamp": time.time(),
            "results": results,
            "total_results": len(results),
            "simulated": True,
            "error_message": "Real search functionality is currently unavailable. Using simulated results."
        }
    
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a web page.
        
        Args:
            url: URL of the web page
            
        Returns:
            Dictionary containing extracted content
        """
        logger.info(f"Extracting content from: {url}")
        
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"content_{cache_key}.json")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Using cached content for: {url}")
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading content cache file: {e}")
        
        try:
            # Set user agent to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise
            
            # Extract content using trafilatura
            extracted_text = trafilatura.extract(response.text, include_links=True, include_images=False, include_tables=False)
            
            if not extracted_text:
                # Fallback to BeautifulSoup if trafilatura fails
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                extracted_text = soup.get_text(separator='\n')
                
                # Clean up text
                lines = (line.strip() for line in extracted_text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                extracted_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(extracted_text) > 10000:
                extracted_text = extracted_text[:10000] + "...[content truncated]"
            
            # Prepare result
            result = {
                "url": url,
                "content": extracted_text,
                "timestamp": time.time(),
                "status": "success"
            }
            
            # Cache result if enabled
            if self.cache_enabled:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                except Exception as e:
                    logger.warning(f"Error writing content cache file: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "content": "",
                "timestamp": time.time(),
                "status": "failed"
            }
    
    def analyze_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results to extract key insights.
        
        Args:
            search_results: Dictionary containing search results
            
        Returns:
            Dictionary containing analysis of search results
        """
        logger.info("Analyzing search results")
        
        results = search_results.get("results", [])
        if not results:
            return {
                "key_topics": [],
                "sentiment": "neutral",
                "credibility": "unknown",
                "timestamp": time.time()
            }
        
        # Extract titles and descriptions
        texts = []
        for result in results:
            if result.get("title"):
                texts.append(result["title"])
            if result.get("description"):
                texts.append(result["description"])
        
        combined_text = " ".join(texts)
        
        # Use LLM to analyze the results
        prompt = f"""
        Analyze the following search results and provide insights:

        Search Query: {search_results.get('query', 'Unknown')}
        
        Search Results:
        {combined_text}
        
        Please provide the following analysis in JSON format:
        1. key_topics: List of 3-5 key topics or themes present in these results
        2. sentiment: Overall sentiment of the results (positive, negative, neutral, or mixed)
        3. credibility: Assessment of the overall credibility of the sources (high, medium, low, or unknown)
        4. information_gaps: List of 2-3 potential information gaps or areas needing more research
        
        Return your analysis in the following JSON format:
        {{
            "key_topics": ["topic1", "topic2", "topic3"],
            "sentiment": "sentiment_assessment",
            "credibility": "credibility_assessment",
            "information_gaps": ["gap1", "gap2", "gap3"]
        }}
        """
        
        try:
            model_config = MODEL_CONFIG["llama4"]
            model_config["temperature"] = 0.3
            
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            analysis = parse_json_response(content)
            
            if analysis.get("fallback_generated"):
                logger.warning(f"Error in LLM analysis: {analysis.get('error')}")
                # Provide a basic fallback analysis
                analysis = {
                    "key_topics": ["Topic 1", "Topic 2", "Topic 3"],
                    "sentiment": "neutral",
                    "credibility": "unknown",
                    "information_gaps": ["More specific information needed", "Recent data may be missing"]
                }
            
            # Add timestamp
            analysis["timestamp"] = time.time()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing search results: {e}")
            return {
                "key_topics": ["Error in analysis"],
                "sentiment": "unknown",
                "credibility": "unknown",
                "information_gaps": ["Analysis failed due to error"],
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _generate_hypotheses(self, question: str, key_findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on research findings.
        
        Args:
            question: The research question
            key_findings: List of key findings from research
            
        Returns:
            List of generated hypotheses
        """
        logger.info(f"Generating hypotheses for question: {question}")
        
        # Prepare findings text
        findings_text = ""
        for i, finding in enumerate(key_findings):
            findings_text += f"Source {i+1}: {finding.get('title', 'Untitled')}\n"
            findings_text += f"URL: {finding.get('source', 'No URL')}\n"
            content = finding.get('content', '')
            # Truncate content if too long
            if len(content) > 500:
                content = content[:500] + "..."
            findings_text += f"Content: {content}\n\n"
        
        # Limit findings text length
        if len(findings_text) > 4000:
            findings_text = findings_text[:4000] + "...[content truncated]"
        
        prompt = f"""
        Based on the following research question and findings, generate 3-5 plausible hypotheses.
        
        Question: {question}
        
        Research Findings:
        {findings_text}
        
        For each hypothesis, provide:
        1. A clear statement of the hypothesis
        2. Key supporting evidence from the research findings
        3. Potential counterarguments or limitations
        4. A confidence level (high, medium, low) with brief justification
        
        Return your hypotheses in the following JSON format:
        {{
            "hypotheses": [
                {{
                    "statement": "Hypothesis statement",
                    "supporting_evidence": "Key evidence supporting this hypothesis",
                    "counterarguments": "Potential counterarguments or limitations",
                    "confidence": "confidence_level",
                    "confidence_justification": "Brief justification for confidence level"
                }},
                ...
            ]
        }}
        """
        
        try:
            model_config = MODEL_CONFIG["llama4"]
            model_config["temperature"] = 0.7
            
            response = call_llm(prompt, model_config)
            content = extract_content(response)
            result = parse_json_response(content)
            
            if result.get("fallback_generated"):
                logger.warning(f"Error in LLM hypothesis generation: {result.get('error')}")
                # Provide a basic fallback hypothesis
                return [{
                    "statement": "Insufficient data to generate reliable hypotheses",
                    "supporting_evidence": "Limited research findings available",
                    "counterarguments": "More research needed to form proper hypotheses",
                    "confidence": "low",
                    "confidence_justification": "Generated as fallback due to processing error"
                }]
            
            hypotheses = result.get("hypotheses", [])
            
            # Validate hypotheses format
            for i, hypothesis in enumerate(hypotheses):
                if not isinstance(hypothesis, dict):
                    logger.warning(f"Invalid hypothesis format at index {i}")
                    hypotheses[i] = {
                        "statement": str(hypothesis),
                        "supporting_evidence": "Format error - no structured data available",
                        "counterarguments": "Format error - no structured data available",
                        "confidence": "low",
                        "confidence_justification": "Generated from improperly formatted data"
                    }
                elif not all(k in hypothesis for k in ["statement", "supporting_evidence", "counterarguments", "confidence"]):
                    missing_keys = [k for k in ["statement", "supporting_evidence", "counterarguments", "confidence"] if k not in hypothesis]
                    logger.warning(f"Missing keys in hypothesis at index {i}: {missing_keys}")
                    for key in missing_keys:
                        hypothesis[key] = f"Missing {key} data"
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return [{
                "statement": "Error occurred during hypothesis generation",
                "supporting_evidence": "Process failed due to technical error",
                "counterarguments": "Not applicable due to error",
                "confidence": "none",
                "confidence_justification": f"Error: {str(e)}"
            }]
    
    def _assess_confidence(self, key_findings: List[Dict[str, Any]], hypotheses: List[Dict[str, Any]]) -> str:
        """
        Assess overall confidence in research results.
        
        Args:
            key_findings: List of key findings from research
            hypotheses: List of generated hypotheses
            
        Returns:
            Confidence level (high, medium, low)
        """
        # Check if we have sufficient findings
        if len(key_findings) < 2:
            return "low"
        
        # Check if hypotheses were generated successfully
        if not hypotheses or len(hypotheses) < 2:
            return "low"
        
        # Check confidence levels in hypotheses
        confidence_levels = [h.get("confidence", "low").lower() for h in hypotheses]
        high_count = confidence_levels.count("high")
        medium_count = confidence_levels.count("medium")
        low_count = confidence_levels.count("low")
        
        # Determine overall confidence
        if high_count > len(confidence_levels) / 2:
            return "high"
        elif high_count + medium_count > len(confidence_levels) / 2:
            return "medium"
        else:
            return "low"
