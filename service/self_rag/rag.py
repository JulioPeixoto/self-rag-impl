"""
Service layer for the Self-RAG agent.
"""
from typing import Dict, Any, Optional
import os
from .core import SelfRAGAgent
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class SelfRAGService:
    """
    Service class for the Self-RAG agent.
    
    This class provides a service interface for the Self-RAG agent,
    handling the initialization and query processing.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'SelfRAGService':
        """
        Get the singleton instance of the service.
        
        Returns:
            The service instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the service."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.agent = SelfRAGAgent(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
        
        self.has_ingested = False
    
    def process_query(self, query: str, webpage_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query using the Self-RAG agent.
        
        Args:
            query: The query to process.
            webpage_url: Optional URL to ingest before processing the query.
            
        Returns:
            A dictionary containing the query result.
        """
        if webpage_url:
            self.agent.ingest_webpage(webpage_url)
            self.has_ingested = True
        
        if not self.has_ingested:
            return {
                "error": "No webpage has been ingested. Please provide a webpage_url."
            }
        
        result = self.agent.query(query)
        
        return result 