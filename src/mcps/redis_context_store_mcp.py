"""
Redis Context Store MCP for storing and retrieving analysis context.
This module provides the RedisContextStoreMCP class with enhanced context management capabilities.
"""

import logging
import time
import os
import json
import uuid
from typing import Dict, List, Any, Optional

import redis

from src.base_mcp import BaseMCP
from src.analysis_context import AnalysisContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedisContextStoreMCP(BaseMCP):
    """
    Redis Context Store MCP for storing and retrieving analysis context.
    
    This MCP provides capabilities for:
    1. Creating and managing analysis sessions
    2. Storing and retrieving context data
    3. Maintaining analysis history
    4. Supporting collaborative analysis
    """
    
    def __init__(self, config=None):
        """
        Initialize the Redis Context Store MCP.
        
        Args:
            config: Optional configuration dictionary with Redis connection details
        """
        super().__init__(name="redis_context_store", config=config)
        
        # Configuration
        self.redis_host = config.get("redis_host", "localhost") if config else os.environ.get("REDIS_HOST", "localhost")
        self.redis_port = config.get("redis_port", 6379) if config else int(os.environ.get("REDIS_PORT", 6379))
        self.redis_password = config.get("redis_password") if config else os.environ.get("REDIS_PASSWORD")
        self.redis_db = config.get("redis_db", 0) if config else int(os.environ.get("REDIS_DB", 0))
        self.ttl = config.get("ttl", 86400) if config else int(os.environ.get("REDIS_TTL", 86400))  # Default 24 hours
        
        # Initialize Redis client
        self._initialize_redis_client()
        
        logger.info(f"Initialized RedisContextStoreMCP with host: {self.redis_host}, port: {self.redis_port}")
    
    def _initialize_redis_client(self):
        """Initialize the Redis client with connection pooling and error handling."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                db=self.redis_db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            logger.warning("Using in-memory fallback for context storage")
            # Initialize in-memory storage as fallback
            self.memory_storage = {}
    
    def process(self, context: Dict) -> Dict:
        """
        Process a context store request.
        
        Args:
            context: Dictionary containing request parameters
            
        Returns:
            Dictionary containing processing results
        """
        operation = context.get("operation")
        namespace = context.get("namespace")
        key = context.get("key")
        data = context.get("data")
        
        if not operation:
            return {"error": "No operation specified"}
        
        if operation == "create_session":
            question = context.get("question")
            if not question:
                return {"error": "No question provided for session creation"}
            return {"session_id": self.create_analysis_session(question)}
        
        elif operation == "get":
            if not namespace or not key:
                return {"error": "Namespace and key required for get operation"}
            return {"data": self.get_context(namespace, key)}
        
        elif operation == "update":
            if not namespace or not key or data is None:
                return {"error": "Namespace, key, and data required for update operation"}
            self.update_context(namespace, key, data)
            return {"success": True}
        
        elif operation == "delete":
            if not namespace or not key:
                return {"error": "Namespace and key required for delete operation"}
            self.delete_context(namespace, key)
            return {"success": True}
        
        elif operation == "get_all_sessions":
            return {"sessions": self.get_all_sessions()}
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    def create_analysis_session(self, question: str) -> str:
        """
        Create a new analysis session.
        
        Args:
            question: The analytical question for this session
            
        Returns:
            Session ID
        """
        logger.info(f"Creating analysis session for question: {question[:100]}...")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session data
        session_data = {
            "session_id": session_id,
            "question": question,
            "creation_time": time.time(),
            "last_updated": time.time(),
            "status": "created"
        }
        
        # Store session data
        self._store_data("analysis_sessions", session_id, session_data)
        
        return session_id
    
    def get_context(self, namespace: str, key: str) -> Dict:
        """
        Get context data.
        
        Args:
            namespace: Context namespace
            key: Context key
            
        Returns:
            Context data
        """
        logger.info(f"Getting context for {namespace}:{key}")
        
        return self._get_data(namespace, key)
    
    def update_context(self, namespace: str, key: str, data: Dict) -> None:
        """
        Update context data.
        
        Args:
            namespace: Context namespace
            key: Context key
            data: Context data
        """
        logger.info(f"Updating context for {namespace}:{key}")
        
        # Get existing data
        existing_data = self._get_data(namespace, key)
        
        # If updating an analysis session, update last_updated timestamp
        if namespace == "analysis_sessions":
            data["last_updated"] = time.time()
        
        # If existing data is a dict and new data is a dict, merge them
        if isinstance(existing_data, dict) and isinstance(data, dict):
            # Deep merge
            merged_data = self._deep_merge(existing_data, data)
            self._store_data(namespace, key, merged_data)
        else:
            # Otherwise, just replace
            self._store_data(namespace, key, data)
    
    def delete_context(self, namespace: str, key: str) -> None:
        """
        Delete context data.
        
        Args:
            namespace: Context namespace
            key: Context key
        """
        logger.info(f"Deleting context for {namespace}:{key}")
        
        if self.redis_client:
            try:
                redis_key = f"{namespace}:{key}"
                self.redis_client.delete(redis_key)
            except redis.RedisError as e:
                logger.error(f"Redis error in delete_context: {e}")
                # Fall back to memory storage
                if namespace in self.memory_storage and key in self.memory_storage[namespace]:
                    del self.memory_storage[namespace][key]
        else:
            # Use memory storage
            if namespace in self.memory_storage and key in self.memory_storage[namespace]:
                del self.memory_storage[namespace][key]
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get all analysis sessions.
        
        Returns:
            List of session data dictionaries
        """
        logger.info("Getting all analysis sessions")
        
        if self.redis_client:
            try:
                # Get all keys in the analysis_sessions namespace
                keys = self.redis_client.keys("analysis_sessions:*")
                
                # Get data for each key
                sessions = []
                for key in keys:
                    # Extract session ID from key
                    session_id = key.split(":", 1)[1]
                    session_data = self._get_data("analysis_sessions", session_id)
                    if session_data:
                        sessions.append(session_data)
                
                return sessions
            except redis.RedisError as e:
                logger.error(f"Redis error in get_all_sessions: {e}")
                # Fall back to memory storage
                if "analysis_sessions" in self.memory_storage:
                    return list(self.memory_storage["analysis_sessions"].values())
                return []
        else:
            # Use memory storage
            if "analysis_sessions" in self.memory_storage:
                return list(self.memory_storage["analysis_sessions"].values())
            return []
    
    def _store_data(self, namespace: str, key: str, data: Any) -> None:
        """
        Store data in Redis or memory storage.
        
        Args:
            namespace: Data namespace
            key: Data key
            data: Data to store
        """
        if self.redis_client:
            try:
                redis_key = f"{namespace}:{key}"
                serialized_data = json.dumps(data)
                self.redis_client.set(redis_key, serialized_data, ex=self.ttl)
            except redis.RedisError as e:
                logger.error(f"Redis error in _store_data: {e}")
                # Fall back to memory storage
                if namespace not in self.memory_storage:
                    self.memory_storage[namespace] = {}
                self.memory_storage[namespace][key] = data
        else:
            # Use memory storage
            if namespace not in self.memory_storage:
                self.memory_storage[namespace] = {}
            self.memory_storage[namespace][key] = data
    
    def _get_data(self, namespace: str, key: str) -> Any:
        """
        Get data from Redis or memory storage.
        
        Args:
            namespace: Data namespace
            key: Data key
            
        Returns:
            Stored data, or empty dict if not found
        """
        if self.redis_client:
            try:
                redis_key = f"{namespace}:{key}"
                data = self.redis_client.get(redis_key)
                if data:
                    return json.loads(data)
                return {}
            except redis.RedisError as e:
                logger.error(f"Redis error in _get_data: {e}")
                # Fall back to memory storage
                if namespace in self.memory_storage and key in self.memory_storage[namespace]:
                    return self.memory_storage[namespace][key]
                return {}
        else:
            # Use memory storage
            if namespace in self.memory_storage and key in self.memory_storage[namespace]:
                return self.memory_storage[namespace][key]
            return {}
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
