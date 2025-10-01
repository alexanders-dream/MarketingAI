"""
Interface definitions to break circular dependencies in Marketing AI v3
"""

from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod


class MarketAnalyzerInterface(Protocol):
    """Interface for market analysis functionality"""
    
    @abstractmethod
    def generate_guided_market_analysis_with_context(self, llm, 
                                                   company_name: str, 
                                                   industry: str,
                                                   target_audience: str,
                                                   products_services: str,
                                                   brand_description: str,
                                                   marketing_goals: str,
                                                   use_guided_research: bool = True) -> Dict[str, str]:
        """Generate guided market analysis using business context"""
        pass
    
    @abstractmethod
    def generate_market_analysis(self, llm, vector_store, industry: str = None,
                               company_name: str = None, use_web_scraping: bool = True) -> Dict[str, str]:
        """Generate comprehensive market analysis"""
        pass
    
    @abstractmethod
    def generate_market_strategy(self, llm, analysis_data: Dict[str, str]) -> str:
        """Generate market strategy based on analysis"""
        pass


class LLMClientInterface(Protocol):
    """Interface for LLM clients"""
    
    @abstractmethod
    def invoke(self, prompt: str) -> Any:
        """Invoke the LLM with a prompt"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name"""
        pass


class BusinessContextInterface(Protocol):
    """Interface for business context data"""
    
    @property
    @abstractmethod
    def company_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def industry(self) -> str:
        pass
    
    @property
    @abstractmethod
    def target_audience(self) -> str:
        pass
    
    @property
    @abstractmethod
    def products_services(self) -> str:
        pass
    
    @property
    @abstractmethod
    def brand_description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def marketing_goals(self) -> str:
        pass
