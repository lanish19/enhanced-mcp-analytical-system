"""
Research MCP for conducting web searches and content extraction.
This module provides the ResearchMCP class with real search capabilities.
"""

import logging
import json
import time
import os
import hashlib
import requests
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import trafilatura
import urllib.parse
import re

from src.base_mcp import BaseMCP

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchMCP(BaseMCP):
    """
    Research MCP for conducting web searches and content extraction.
    
    This MCP provides capabilities for:
    1. Conducting web searches using Brave Search API
    2. Conducting academic searches using Semantic Scholar or Google Scholar
    3. Extracting content from web pages
    4. Analyzing search results
    """
    
    def __init__(self, config=None):
        """
        Initialize the Research MCP.
        
        Args:
            config: Optional configuration dictionary with API keys
        """
        super().__init__(name="research_mcp", config=config)
        
        # Configuration
        self.brave_api_key = config.get("brave_api_key") if config else os.environ.get("BRAVE_API_KEY")
        self.brave_api_base = config.get("brave_api_base", "https://api.search.brave.com/res/v1/web/search")
        self.semantic_scholar_api_base = config.get("semantic_scholar_api_base", "https://api.semanticscholar.org/graph/v1")
        self.max_retries = config.get("max_retries", 3) if config else 3
        self.retry_delay = config.get("retry_delay", 2) if config else 2
        self.cache_enabled = config.get("cache_enabled", True) if config else True
        
        # Cache for search results and content extraction
        self.search_cache = {}
        self.content_cache = {}
        
        logger.info(f"Initialized ResearchMCP with cache enabled: {self.cache_enabled}")
    
    def process(self, context: Dict) -> Dict:
        """
        Process a research request.
        
        Args:
            context: Dictionary containing research parameters
            
        Returns:
            Dictionary containing research results
        """
        query = context.get("query")
        search_type = context.get("search_type", "web")
        num_results = context.get("num_results", 10)
        
        if not query:
            return {"error": "No query provided"}
        
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
        
        return search_results
    
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
            if cache_key in self.search_cache:
                logger.info(f"Using cached search results for: {query}")
                return self.search_cache[cache_key]
        
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
                self.search_cache[cache_key] = results
            
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
            if cache_key in self.content_cache:
                logger.info(f"Using cached content for: {url}")
                return self.content_cache[cache_key]
        
        try:
            # Try extraction with Trafilatura first
            content = self._extract_with_trafilatura(url)
            
            # If Trafilatura fails, try BeautifulSoup
            if not content.get("success", False):
                content = self._extract_with_beautifulsoup(url)
            
            # Cache results if enabled
            if self.cache_enabled:
                self.content_cache[cache_key] = content
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return self._handle_extraction_error(e, url)
    
    def _extract_with_trafilatura(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a web page using Trafilatura.
        
        Args:
            url: URL of the web page
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            # Set user agent to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Download the web page
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Extract content using Trafilatura
            downloaded = response.text
            result = trafilatura.extract(downloaded, include_comments=False, include_tables=True, output_format="json")
            
            if result:
                # Parse JSON result
                content_dict = json.loads(result)
                
                return {
                    "url": url,
                    "timestamp": time.time(),
                    "title": content_dict.get("title", ""),
                    "text": content_dict.get("text", ""),
                    "author": content_dict.get("author", ""),
                    "date": content_dict.get("date", ""),
                    "categories": content_dict.get("categories", []),
                    "tags": content_dict.get("tags", []),
                    "success": True,
                    "extractor": "trafilatura"
                }
            else:
                return {"url": url, "success": False, "error": "Trafilatura extraction failed"}
                
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed for {url}: {e}")
            return {"url": url, "success": False, "error": str(e)}
    
    def _extract_with_beautifulsoup(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a web page using BeautifulSoup.
        
        Args:
            url: URL of the web page
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            # Set user agent to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Download the web page
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.text.strip() if soup.title else ""
            
            # Extract main content
            # This is a simple heuristic and may not work for all websites
            content = ""
            
            # Try to find main content container
            main_content = None
            for container in ["main", "article", "#content", ".content", "#main", ".main", ".post", ".article"]:
                main_content = soup.select_one(container)
                if main_content:
                    break
            
            # If main content container found, extract text from it
            if main_content:
                # Remove script and style elements
                for script in main_content(["script", "style"]):
                    script.extract()
                
                # Get text
                content = main_content.get_text(separator="\n", strip=True)
            else:
                # Fallback: extract text from body
                body = soup.body
                if body:
                    # Remove script and style elements
                    for script in body(["script", "style"]):
                        script.extract()
                    
                    # Get text
                    content = body.get_text(separator="\n", strip=True)
            
            # Extract metadata
            meta_tags = {}
            for meta in soup.find_all("meta"):
                if meta.get("name") and meta.get("content"):
                    meta_tags[meta.get("name")] = meta.get("content")
                elif meta.get("property") and meta.get("content"):
                    meta_tags[meta.get("property")] = meta.get("content")
            
            # Extract author
            author = meta_tags.get("author", "")
            if not author:
                author = meta_tags.get("og:author", "")
            
            # Extract date
            date = meta_tags.get("article:published_time", "")
            if not date:
                date = meta_tags.get("date", "")
            
            return {
                "url": url,
                "timestamp": time.time(),
                "title": title,
                "text": content,
                "author": author,
                "date": date,
                "meta_tags": meta_tags,
                "success": True,
                "extractor": "beautifulsoup"
            }
                
        except Exception as e:
            logger.warning(f"BeautifulSoup extraction failed for {url}: {e}")
            return {"url": url, "success": False, "error": str(e)}
    
    def _handle_extraction_error(self, error: Exception, url: str) -> Dict[str, Any]:
        """
        Handle extraction errors and provide fallback results.
        
        Args:
            error: The exception that occurred
            url: The URL
            
        Returns:
            Dictionary containing fallback extraction results
        """
        logger.error(f"Extraction error for URL '{url}': {error}")
        
        return {
            "url": url,
            "timestamp": time.time(),
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "fallback_message": "Content extraction failed. Please try accessing the URL directly."
        }
    
    def analyze_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results to extract key information.
        
        Args:
            results: Dictionary containing search results
            
        Returns:
            Dictionary containing analysis of search results
        """
        logger.info("Analyzing search results")
        
        # Check if results are valid
        if not results or "results" not in results or not results["results"]:
            return {
                "success": False,
                "error": "No search results to analyze"
            }
        
        try:
            # Extract search items
            items = results["results"]
            
            # Count sources
            sources = {}
            for item in items:
                source = item.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            
            # Extract domains
            domains = {}
            for item in items:
                url = item.get("url", "")
                if url:
                    try:
                        domain = urllib.parse.urlparse(url).netloc
                        domains[domain] = domains.get(domain, 0) + 1
                    except:
                        pass
            
            # Extract years for academic results
            years = {}
            for item in items:
                year = item.get("year")
                if year:
                    years[str(year)] = years.get(str(year), 0) + 1
            
            # Extract common terms from titles and descriptions
            terms = {}
            for item in items:
                title = item.get("title", "")
                description = item.get("description", "")
                text = f"{title} {description}".lower()
                
                # Simple tokenization and counting
                words = re.findall(r'\b[a-z]{3,}\b', text)
                for word in words:
                    if word not in ["the", "and", "for", "with", "that", "this", "from", "have", "are", "not"]:
                        terms[word] = terms.get(word, 0) + 1
            
            # Sort terms by frequency
            sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)
            top_terms = dict(sorted_terms[:20])
            
            return {
                "success": True,
                "total_results": len(items),
                "sources": sources,
                "domains": domains,
                "years": years if years else None,
                "top_terms": top_terms,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing search results: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
