"""
Guided Research Agents for Market Analysis - Inspired by open_deep_research
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS

from web_scraper import WebScraper
from config import AppConfig

logger = logging.getLogger(__name__)


class ResearchTaskType(Enum):
    """Types of research tasks"""
    MARKET_ANALYSIS = "market_analysis"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    INDUSTRY_TRENDS = "industry_trends"
    CUSTOMER_INSIGHTS = "customer_insights"
    OPPORTUNITY_ANALYSIS = "opportunity_analysis"


@dataclass
class BusinessContext:
    """Business context extracted from uploaded documents"""
    company_name: str = ""
    industry: str = ""
    products_services: str = ""
    target_audience: str = ""
    brand_description: str = ""
    competitive_advantages: str = ""
    market_goals: str = ""
    existing_content: str = ""
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class ResearchQuestion(BaseModel):
    """Structured research question"""
    question: str = Field(description="The research question to investigate")
    task_type: str = Field(description="Type of research task")
    priority: int = Field(description="Priority level (1-5, 5 being highest)")
    search_queries: List[str] = Field(description="Specific search queries to use")
    expected_insights: str = Field(description="What insights we expect to gain")


class ResearchPlan(BaseModel):
    """Complete research plan with multiple questions"""
    research_questions: List[ResearchQuestion] = Field(description="List of research questions")
    research_focus: str = Field(description="Main focus of the research")
    success_criteria: str = Field(description="How to measure research success")


class ResearchFinding(BaseModel):
    """Individual research finding"""
    question: str = Field(description="The research question this addresses")
    finding: str = Field(description="The key finding or insight")
    sources: List[str] = Field(description="Sources of information")
    confidence: float = Field(description="Confidence level (0-1)")
    implications: str = Field(description="Business implications of this finding")


class ResearchSupervisor:
    """Supervisor agent that coordinates research activities"""
    
    def __init__(self):
        self.web_scraper = WebScraper()
    
    async def create_research_plan(self, llm, business_context: BusinessContext, 
                                 vector_store: Optional[FAISS] = None) -> ResearchPlan:
        """
        Create a comprehensive research plan based on business context
        
        Args:
            llm: Language model instance
            business_context: Business information extracted from documents
            vector_store: Optional vector store for document context
            
        Returns:
            Structured research plan
        """
        # Get document context if available
        document_context = ""
        if vector_store:
            document_context = await self._get_document_context(vector_store, business_context)
        
        planning_prompt = """
        You are a senior market research strategist tasked with creating a comprehensive research plan.
        
        **Business Context:**
        Company: {company_name}
        Industry: {industry}
        Products/Services: {products_services}
        Target Audience: {target_audience}
        Brand Description: {brand_description}
        Competitive Advantages: {competitive_advantages}
        Market Goals: {market_goals}
        Keywords: {keywords}
        
        **Document Context:**
        {document_context}
        
        **Task:**
        Create a structured research plan with 8-12 specific research questions that will provide comprehensive market intelligence. Focus on:
        
        1. Market size, trends, and growth opportunities
        2. Competitive landscape and positioning
        3. Customer needs, pain points, and behavior
        4. Industry challenges and disruptions
        5. Pricing strategies and market dynamics
        6. Distribution channels and partnerships
        7. Regulatory environment and barriers
        8. Technology trends affecting the market
        
        For each research question, provide:
        - Clear, specific question
        - Task type (market_analysis, competitor_analysis, etc.)
        - Priority level (1-5)
        - 2-3 specific search queries
        - Expected insights to be gained
        
        Ensure questions are actionable and will lead to concrete business insights.
        
        {format_instructions}
        """
        
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        prompt = ChatPromptTemplate.from_template(planning_prompt)
        
        chain = prompt | llm | parser
        
        try:
            research_plan = await chain.ainvoke({
                "company_name": business_context.company_name,
                "industry": business_context.industry,
                "products_services": business_context.products_services,
                "target_audience": business_context.target_audience,
                "brand_description": business_context.brand_description,
                "competitive_advantages": business_context.competitive_advantages,
                "market_goals": business_context.market_goals,
                "keywords": ", ".join(business_context.keywords),
                "document_context": document_context,
                "format_instructions": parser.get_format_instructions()
            })
            
            logger.info(f"Created research plan with {len(research_plan.research_questions)} questions")
            return research_plan
            
        except Exception as e:
            logger.error(f"Failed to create research plan: {str(e)}")
            # Return a basic plan as fallback
            return self._create_fallback_plan(business_context)
    
    async def _get_document_context(self, vector_store: FAISS, business_context: BusinessContext) -> str:
        """Extract relevant context from uploaded documents"""
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            # Search for relevant business information
            search_queries = [
                f"{business_context.company_name} business model",
                f"{business_context.industry} market analysis",
                "competitive advantages and positioning",
                "target customers and market segments"
            ]
            
            context_parts = []
            for query in search_queries:
                docs = await retriever.ainvoke(query)
                for doc in docs[:2]:  # Limit to top 2 docs per query
                    context_parts.append(doc.page_content[:500])
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to extract document context: {str(e)}")
            return ""
    
    def _create_fallback_plan(self, business_context: BusinessContext) -> ResearchPlan:
        """Create a basic research plan as fallback"""
        questions = [
            ResearchQuestion(
                question=f"What is the current market size and growth rate for {business_context.industry}?",
                task_type="market_analysis",
                priority=5,
                search_queries=[
                    f"{business_context.industry} market size 2024",
                    f"{business_context.industry} growth rate trends",
                    f"{business_context.industry} market forecast"
                ],
                expected_insights="Market size, growth trends, and future projections"
            ),
            ResearchQuestion(
                question=f"Who are the main competitors in {business_context.industry}?",
                task_type="competitor_analysis",
                priority=5,
                search_queries=[
                    f"{business_context.industry} top companies",
                    f"{business_context.company_name} competitors",
                    f"{business_context.industry} market leaders"
                ],
                expected_insights="Key competitors, market share, and competitive positioning"
            ),
            ResearchQuestion(
                question=f"What are the key trends shaping {business_context.industry}?",
                task_type="industry_trends",
                priority=4,
                search_queries=[
                    f"{business_context.industry} trends 2024",
                    f"{business_context.industry} innovation",
                    f"{business_context.industry} disruption"
                ],
                expected_insights="Industry trends, innovations, and disruptions"
            )
        ]
        
        return ResearchPlan(
            research_questions=questions,
            research_focus=f"Market analysis for {business_context.company_name} in {business_context.industry}",
            success_criteria="Comprehensive understanding of market dynamics, competition, and opportunities"
        )


class MarketResearcher:
    """Agent specialized in market research and analysis"""
    
    def __init__(self):
        self.web_scraper = WebScraper()
    
    async def research_question(self, llm, question: ResearchQuestion, 
                              business_context: BusinessContext) -> ResearchFinding:
        """
        Research a specific question and return findings
        
        Args:
            llm: Language model instance
            question: Research question to investigate
            business_context: Business context for focused research
            
        Returns:
            Research finding with insights and sources
        """
        try:
            # Perform web searches
            search_results = await self._perform_searches(question.search_queries, business_context)
            
            # Analyze and synthesize findings
            finding = await self._analyze_search_results(llm, question, search_results, business_context)
            
            return finding
            
        except Exception as e:
            logger.error(f"Research failed for question '{question.question}': {str(e)}")
            return ResearchFinding(
                question=question.question,
                finding=f"Research failed: {str(e)}",
                sources=[],
                confidence=0.0,
                implications="Unable to determine due to research failure"
            )
    
    async def _perform_searches(self, search_queries: List[str], 
                              business_context: BusinessContext) -> Dict[str, Any]:
        """Perform web searches for the given queries"""
        all_results = {}
        
        for query in search_queries:
            try:
                # Enhance query with business context
                enhanced_query = f"{query} {business_context.industry}"
                if business_context.company_name:
                    enhanced_query += f" {business_context.company_name}"
                
                results = await self.web_scraper.scrape_market_data(
                    industry=business_context.industry,
                    company_name=business_context.company_name,
                    max_pages=3
                )
                
                all_results[query] = results
                
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {str(e)}")
                all_results[query] = {}
        
        return all_results
    
    async def _analyze_search_results(self, llm, question: ResearchQuestion, 
                                    search_results: Dict[str, Any], 
                                    business_context: BusinessContext) -> ResearchFinding:
        """Analyze search results and generate insights"""
        
        # Format search results for analysis
        formatted_results = self._format_search_results(search_results)
        
        analysis_prompt = """
        You are a market research analyst analyzing web search results to answer a specific research question.
        
        **Research Question:** {question}
        **Task Type:** {task_type}
        **Business Context:** {company_name} in {industry}
        
        **Search Results:**
        {search_results}
        
        **Analysis Requirements:**
        1. Provide a clear, concise answer to the research question
        2. Identify key insights and trends
        3. Assess the reliability and confidence level of findings (0-1 scale)
        4. Explain business implications for {company_name}
        5. List the most relevant sources used
        
        Focus on actionable insights that can inform marketing strategy and business decisions.
        Be specific with numbers, dates, and concrete examples where available.
        
        {format_instructions}
        """
        
        parser = PydanticOutputParser(pydantic_object=ResearchFinding)
        prompt = ChatPromptTemplate.from_template(analysis_prompt)
        
        chain = prompt | llm | parser
        
        try:
            finding = await chain.ainvoke({
                "question": question.question,
                "task_type": question.task_type,
                "company_name": business_context.company_name,
                "industry": business_context.industry,
                "search_results": formatted_results,
                "format_instructions": parser.get_format_instructions()
            })
            
            return finding
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return ResearchFinding(
                question=question.question,
                finding="Analysis could not be completed due to processing error",
                sources=[],
                confidence=0.0,
                implications="Unable to determine implications"
            )
    
    def _format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results for LLM analysis"""
        formatted = []
        
        for query, results in search_results.items():
            formatted.append(f"**Query: {query}**")
            
            if isinstance(results, dict) and 'content' in results:
                content = results['content'][:2000]  # Limit content length
                formatted.append(f"Content: {content}")
            elif isinstance(results, list):
                for i, result in enumerate(results[:3]):  # Limit to top 3 results
                    if isinstance(result, dict):
                        title = result.get('title', 'No title')
                        content = result.get('content', '')[:500]
                        formatted.append(f"Result {i+1}: {title}\n{content}")
            
            formatted.append("---")
        
        return "\n".join(formatted)


class ResearchSynthesizer:
    """Agent that synthesizes all research findings into comprehensive analysis"""
    
    async def synthesize_research(self, llm, research_findings: List[ResearchFinding], 
                                business_context: BusinessContext, 
                                research_plan: ResearchPlan) -> str:
        """
        Synthesize all research findings into a comprehensive market analysis report
        
        Args:
            llm: Language model instance
            research_findings: List of individual research findings
            business_context: Business context
            research_plan: Original research plan
            
        Returns:
            Comprehensive market analysis report
        """
        
        # Organize findings by task type
        findings_by_type = self._organize_findings_by_type(research_findings)
        
        synthesis_prompt = """
        You are a senior market research consultant creating a comprehensive market analysis report.
        
        **Business Context:**
        Company: {company_name}
        Industry: {industry}
        Products/Services: {products_services}
        Target Audience: {target_audience}
        
        **Research Focus:** {research_focus}
        
        **Research Findings by Category:**
        {organized_findings}
        
        **Task:**
        Create a comprehensive, executive-level market analysis report with the following structure:
        
        # Executive Summary
        Key findings and strategic recommendations (2-3 paragraphs)
        
        # Market Landscape Analysis
        ## Market Size and Growth
        ## Key Market Trends
        ## Market Opportunities and Threats
        
        # Competitive Intelligence
        ## Competitive Landscape Overview
        ## Key Competitors Analysis
        ## Competitive Positioning Recommendations
        
        # Customer and Industry Insights
        ## Target Market Analysis
        ## Customer Needs and Pain Points
        ## Industry Dynamics and Forces
        
        # Strategic Recommendations
        ## Market Entry/Expansion Strategy
        ## Competitive Differentiation
        ## Marketing and Positioning Strategy
        ## Risk Mitigation
        
        # Implementation Priorities
        ## Immediate Actions (0-3 months)
        ## Medium-term Initiatives (3-12 months)
        ## Long-term Strategic Goals (1-3 years)
        
        **Requirements:**
        - Use specific data points and insights from the research
        - Provide actionable recommendations
        - Highlight confidence levels for key findings
        - Include relevant market numbers and trends
        - Focus on strategic implications for {company_name}
        """
        
        prompt = ChatPromptTemplate.from_template(synthesis_prompt)
        chain = prompt | llm | StrOutputParser()
        
        try:
            report = await chain.ainvoke({
                "company_name": business_context.company_name,
                "industry": business_context.industry,
                "products_services": business_context.products_services,
                "target_audience": business_context.target_audience,
                "research_focus": research_plan.research_focus,
                "organized_findings": self._format_organized_findings(findings_by_type)
            })
            
            return report
            
        except Exception as e:
            logger.error(f"Research synthesis failed: {str(e)}")
            return f"Failed to synthesize research findings: {str(e)}"
    
    def _organize_findings_by_type(self, findings: List[ResearchFinding]) -> Dict[str, List[ResearchFinding]]:
        """Organize findings by research task type"""
        organized = {}
        
        for finding in findings:
            # Extract task type from question or use default
            task_type = "general"
            if "market" in finding.question.lower():
                task_type = "market_analysis"
            elif "competitor" in finding.question.lower():
                task_type = "competitor_analysis"
            elif "trend" in finding.question.lower():
                task_type = "industry_trends"
            elif "customer" in finding.question.lower():
                task_type = "customer_insights"
            
            if task_type not in organized:
                organized[task_type] = []
            organized[task_type].append(finding)
        
        return organized
    
    def _format_organized_findings(self, findings_by_type: Dict[str, List[ResearchFinding]]) -> str:
        """Format organized findings for the synthesis prompt"""
        formatted = []
        
        for task_type, findings in findings_by_type.items():
            formatted.append(f"## {task_type.replace('_', ' ').title()}")
            
            for finding in findings:
                formatted.append(f"**Question:** {finding.question}")
                formatted.append(f"**Finding:** {finding.finding}")
                formatted.append(f"**Confidence:** {finding.confidence:.2f}")
                formatted.append(f"**Implications:** {finding.implications}")
                if finding.sources:
                    formatted.append(f"**Sources:** {', '.join(finding.sources[:3])}")
                formatted.append("---")
        
        return "\n".join(formatted)


class GuidedMarketResearch:
    """Main orchestrator for guided market research"""
    
    def __init__(self):
        self.supervisor = ResearchSupervisor()
        self.market_researcher = MarketResearcher()
        self.synthesizer = ResearchSynthesizer()
    
    async def conduct_comprehensive_research(self, llm, business_context: BusinessContext, 
                                           vector_store: Optional[FAISS] = None,
                                           max_concurrent_research: int = 3) -> str:
        """
        Conduct comprehensive guided market research
        
        Args:
            llm: Language model instance
            business_context: Business context from uploaded documents
            vector_store: Optional vector store for document context
            max_concurrent_research: Maximum concurrent research tasks
            
        Returns:
            Comprehensive market analysis report
        """
        try:
            logger.info("Starting guided market research process")
            
            # Step 1: Create research plan
            research_plan = await self.supervisor.create_research_plan(
                llm, business_context, vector_store
            )
            
            # Step 2: Execute research questions concurrently
            research_findings = await self._execute_research_plan(
                llm, research_plan, business_context, max_concurrent_research
            )
            
            # Step 3: Synthesize findings into comprehensive report
            final_report = await self.synthesizer.synthesize_research(
                llm, research_findings, business_context, research_plan
            )
            
            logger.info("Completed guided market research process")
            return final_report
            
        except Exception as e:
            logger.error(f"Guided research failed: {str(e)}")
            return f"Research process failed: {str(e)}"
    
    async def _execute_research_plan(self, llm, research_plan: ResearchPlan, 
                                   business_context: BusinessContext,
                                   max_concurrent: int) -> List[ResearchFinding]:
        """Execute research plan with controlled concurrency"""
        
        # Sort questions by priority
        sorted_questions = sorted(
            research_plan.research_questions, 
            key=lambda q: q.priority, 
            reverse=True
        )
        
        findings = []
        
        # Process questions in batches
        for i in range(0, len(sorted_questions), max_concurrent):
            batch = sorted_questions[i:i + max_concurrent]
            
            # Execute batch concurrently
            batch_tasks = [
                self.market_researcher.research_question(llm, question, business_context)
                for question in batch
            ]
            
            batch_findings = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for finding in batch_findings:
                if isinstance(finding, Exception):
                    logger.error(f"Research task failed: {str(finding)}")
                else:
                    findings.append(finding)
        
        return findings