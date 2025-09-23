"""
Market Analysis Agent for Marketing AI v3
"""
import logging
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from config import AppConfig
from web_scraper import scrape_market_data_sync, scrape_competitor_data_sync

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Analyzes business documents to generate market insights"""

    def __init__(self):
        self.analysis_fields = {
            "brand_description": "Extract and summarize the company's brand description, mission, values, and unique selling points.",
            "target_audience": "Identify and describe the target audience or customer segments, including demographics, psychographics, and key characteristics.",
            "products_services": "List and briefly describe the main products and/or services offered by the business.",
            "marketing_goals": "Identify the key marketing goals or objectives. If not explicitly stated, suggest reasonable goals based on the business type.",
            "existing_content": "Summarize any existing marketing content, campaigns, or channels mentioned in the document.",
            "keywords": "Generate 10-15 relevant keywords for marketing purposes, formatted as a comma-separated list.",
            "suggested_topics": "Suggest 5-7 content topics that would be relevant for this business's marketing strategy, formatted as a numbered list.",
            "market_opportunities": "Identify potential market opportunities, gaps, or areas for growth based on the business information.",
            "competitive_advantages": "Analyze and describe the business's competitive advantages and differentiators.",
            "customer_pain_points": "Identify customer pain points or problems that this business solves."
        }

    def generate_insight(self, llm, vector_store: FAISS, field_name: str) -> str:
        """
        Generate a specific market insight using RAG

        Args:
            llm: Language model instance
            vector_store: FAISS vector store
            field_name: Name of the insight field to generate

        Returns:
            Generated insight text
        """
        if field_name not in self.analysis_fields:
            raise ValueError(f"Unknown analysis field: {field_name}")

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        prompt_template = """
        You are a senior marketing strategist analyzing business documents to extract key insights.

        Task: {input}

        Context from business documents:
        {context}

        Provide a clear, concise, and actionable response based only on the information provided in the context.
        If the information is not available in the context, make reasonable inferences based on the business type and industry.
        """

        document_chain = create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_template(prompt_template),
            document_variable_name="context"
        )

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        try:
            result = retrieval_chain.invoke({
                "input": self.analysis_fields[field_name]
            })
            return self._parse_insight(field_name, result["answer"])

        except Exception as e:
            logger.error(f"Insight generation failed for {field_name}: {str(e)}")
            return ""

    def generate_market_analysis(self, llm, vector_store: FAISS, industry: str = None,
                                company_name: str = None, use_web_scraping: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive market analysis with optional web scraping

        Args:
            llm: Language model instance
            vector_store: FAISS vector store
            industry: Industry sector for web research
            company_name: Company name for web research
            use_web_scraping: Whether to include web-scraped data

        Returns:
            Dictionary of analysis insights
        """
        analysis_results = {}

        # Generate base insights from documents
        for field_name in self.analysis_fields.keys():
            insight = self.generate_insight(llm, vector_store, field_name)
            analysis_results[field_name] = insight

        # Enhance with web data if requested
        if use_web_scraping and (industry or company_name):
            try:
                web_data = self._scrape_market_web_data(industry, company_name)
                enhanced_results = self._enhance_analysis_with_web_data(llm, analysis_results, web_data)
                analysis_results.update(enhanced_results)
                logger.info("Successfully enhanced market analysis with web data")
            except Exception as e:
                logger.error(f"Failed to enhance analysis with web data: {str(e)}")

        return analysis_results

    def _parse_insight(self, field_name: str, text: str) -> str:
        """
        Parse and clean generated insight text

        Args:
            field_name: Name of the insight field
            text: Raw LLM response

        Returns:
            Cleaned and formatted text
        """
        text = text.strip()

        # Special handling for list-based fields
        if field_name in ["keywords", "suggested_topics"]:
            lines = [line.strip("-• ") for line in text.splitlines() if line.strip()]
            return ", ".join(lines)

        return text

    def generate_market_strategy(self, llm, analysis_data: Dict[str, str]) -> str:
        """
        Generate comprehensive market strategy based on analysis

        Args:
            llm: Language model instance
            analysis_data: Market analysis insights

        Returns:
            Complete market strategy
        """
        # Ensure all required fields are present with defaults
        required_fields = [
            "brand_description", "target_audience", "products_services",
            "marketing_goals", "existing_content", "keywords",
            "market_opportunities", "competitive_advantages", "customer_pain_points"
        ]

        complete_data = {}
        for field in required_fields:
            complete_data[field] = analysis_data.get(field, "Not available")

        strategy_prompt = """
        You are a senior marketing strategist creating a comprehensive marketing plan.

        Based on the following business analysis, develop a detailed, actionable marketing strategy:

        **Business Analysis:**
        Brand Description: {brand_description}
        Target Audience: {target_audience}
        Products/Services: {products_services}
        Marketing Goals: {marketing_goals}
        Existing Content: {existing_content}
        Keywords: {keywords}
        Market Opportunities: {market_opportunities}
        Competitive Advantages: {competitive_advantages}
        Customer Pain Points: {customer_pain_points}

        **Required Strategy Structure:**

        ## Executive Summary
        Brief overview of the entire strategy (2-3 paragraphs)

        ## Market Analysis
        Industry trends, competitive landscape, and market positioning

        ## Target Audience Segmentation
        Detailed profiles of key customer segments

        ## Value Proposition & Positioning
        Unique selling points and brand positioning strategy

        ## Marketing Channels & Tactics
        Prioritized channels by ROI potential with specific tactics

        ## Content Strategy
        Content pillars, formats, distribution plan, and editorial calendar

        ## Budget Allocation
        Recommended spending by channel with justification

        ## Implementation Timeline
        30-60-90 day action plan with milestones

        ## KPIs & Success Metrics
        Specific, measurable success indicators

        ## Risk Assessment & Contingency Plans
        Potential challenges and mitigation strategies

        Focus on creating a strategy that is specific, measurable, achievable, relevant, and time-bound (SMART).
        Ensure all recommendations are practical and aligned with the business's resources and goals.
        """

        prompt = ChatPromptTemplate.from_template(strategy_prompt)
        chain = prompt | llm | StrOutputParser()

        try:
            return chain.invoke(complete_data)
        except Exception as e:
            logger.error(f"Market strategy generation failed: {str(e)}")
            return f"Error generating market strategy: {str(e)}"

    def generate_competitor_analysis(self, llm, analysis_data: Dict[str, str],
                                   competitor_data: Optional[str] = None,
                                   company_name: str = None, industry: str = None,
                                   use_web_scraping: bool = True) -> str:
        """
        Generate competitor analysis with optional web scraping

        Args:
            llm: Language model instance
            analysis_data: Market analysis insights
            competitor_data: Optional competitor information
            company_name: Company name for web research
            industry: Industry sector for web research
            use_web_scraping: Whether to include web-scraped competitor data

        Returns:
            Competitor analysis report
        """
        # Get web-scraped competitor data if requested
        web_competitor_data = ""
        if use_web_scraping and company_name and industry:
            try:
                web_data = self._scrape_competitor_web_data(company_name, industry)
                web_competitor_data = self._format_competitor_web_data(web_data)
                logger.info("Successfully scraped competitor data from web")
            except Exception as e:
                logger.error(f"Failed to scrape competitor data: {str(e)}")

        competitor_prompt = """
        You are a competitive intelligence analyst.

        **Business Context:**
        Brand Description: {brand_description}
        Target Audience: {target_audience}
        Products/Services: {products_services}
        Competitive Advantages: {competitive_advantages}
        Market Opportunities: {market_opportunities}

        **Additional Competitor Data:**
        {competitor_data}

        **Web Research Data:**
        {web_competitor_data}

        **Analysis Requirements:**

        ## Competitive Landscape
        Overview of key competitors and market structure

        ## Competitor Profiling
        Detailed profiles of 3-5 main competitors including:
        - Target markets and positioning
        - Product/service offerings
        - Pricing strategies
        - Marketing channels and tactics
        - Strengths and weaknesses

        ## Competitive Advantages Analysis
        SWOT analysis comparing your business to competitors

        ## Market Gaps & Opportunities
        Unserved customer needs and market gaps

        ## Recommended Positioning Strategy
        How to differentiate from competitors

        ## Competitive Monitoring Plan
        How to track competitor activities ongoing

        Provide actionable insights that inform marketing strategy development.
        """

        prompt = ChatPromptTemplate.from_template(competitor_prompt)
        chain = prompt | llm | StrOutputParser()

        analysis_data_copy = analysis_data.copy()
        analysis_data_copy["competitor_data"] = competitor_data or "No specific competitor data provided"
        analysis_data_copy["web_competitor_data"] = web_competitor_data

        try:
            return chain.invoke(analysis_data_copy)
        except Exception as e:
            logger.error(f"Competitor analysis generation failed: {str(e)}")
            return f"Error generating competitor analysis: {str(e)}"

    def _scrape_market_web_data(self, industry: str = None, company_name: str = None) -> Dict[str, Any]:
        """
        Scrape market data from the web

        Args:
            industry: Industry sector
            company_name: Company name

        Returns:
            Scraped market data
        """
        if not industry and not company_name:
            return {}

        try:
            return scrape_market_data_sync(industry or "general business", company_name, max_pages=3)
        except Exception as e:
            logger.error(f"Web scraping failed: {str(e)}")
            return {}

    def _scrape_competitor_web_data(self, company_name: str, industry: str) -> Dict[str, Any]:
        """
        Scrape competitor data from the web

        Args:
            company_name: Company name
            industry: Industry sector

        Returns:
            Scraped competitor data
        """
        try:
            return scrape_competitor_data_sync(company_name, industry, max_pages=3)
        except Exception as e:
            logger.error(f"Competitor web scraping failed: {str(e)}")
            return {}

    def _enhance_analysis_with_web_data(self, llm, analysis_data: Dict[str, str],
                                       web_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Enhance analysis data with web-scraped information

        Args:
            llm: Language model instance
            analysis_data: Original analysis data
            web_data: Web-scraped data

        Returns:
            Enhanced analysis data
        """
        enhanced_data = {}

        # Format web data for LLM consumption
        web_context = self._format_web_data_for_llm(web_data)

        enhancement_prompt = """
        You are a marketing analyst enhancing business insights with web research data.

        **Original Analysis:**
        {original_analysis}

        **Web Research Data:**
        {web_context}

        **Task:**
        Enhance the original analysis by incorporating relevant information from the web research.
        Maintain the same format and structure, but add depth and accuracy where the web data provides additional insights.
        If the web data conflicts with the original analysis, prioritize the more comprehensive or recent information.
        Keep the response format consistent with the original analysis style.

        Enhanced Analysis:
        """

        for field, original_value in analysis_data.items():
            if field in ["keywords", "suggested_topics"]:
                # Skip list-based fields that don't need enhancement
                enhanced_data[field] = original_value
                continue

            try:
                prompt = ChatPromptTemplate.from_template(enhancement_prompt)
                chain = prompt | llm | StrOutputParser()

                result = chain.invoke({
                    "original_analysis": f"{field}: {original_value}",
                    "web_context": web_context
                })

                # Extract just the enhanced value
                if ":" in result:
                    enhanced_value = result.split(":", 1)[1].strip()
                else:
                    enhanced_value = result.strip()

                enhanced_data[field] = enhanced_value

            except Exception as e:
                logger.error(f"Failed to enhance {field}: {str(e)}")
                enhanced_data[field] = original_value

        return enhanced_data

    def _format_web_data_for_llm(self, web_data: Dict[str, Any]) -> str:
        """
        Format web data for LLM consumption

        Args:
            web_data: Raw web data

        Returns:
            Formatted string for LLM
        """
        if not web_data:
            return "No web data available"

        formatted = []

        for key, value in web_data.items():
            if isinstance(value, list) and value:
                formatted.append(f"**{key.replace('_', ' ').title()}:**")
                for item in value[:3]:  # Limit to 3 items per category
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        summary = item.get("summary", "")
                        if title and summary:
                            formatted.append(f"- {title}: {summary}")
                    else:
                        formatted.append(f"- {item}")
                formatted.append("")
            elif isinstance(value, str) and value:
                formatted.append(f"**{key.replace('_', ' ').title()}:** {value}")
                formatted.append("")

        return "\n".join(formatted)

    def _format_competitor_web_data(self, web_data: Dict[str, Any]) -> str:
        """
        Format competitor web data for LLM consumption

        Args:
            web_data: Raw competitor web data

        Returns:
            Formatted string for LLM
        """
        if not web_data:
            return "No competitor web data available"

        formatted = []

        # Format different competitor categories
        categories = ["direct_competitors", "indirect_competitors", "market_leaders", "alternatives"]

        for category in categories:
            if category in web_data and web_data[category]:
                formatted.append(f"**{category.replace('_', ' ').title()}:**")
                for comp in web_data[category][:3]:  # Limit to 3 per category
                    names = ", ".join(comp.get("names", []))
                    title = comp.get("title", "")
                    summary = comp.get("summary", "")
                    if names:
                        formatted.append(f"- {names}")
                        if title:
                            formatted.append(f"  Title: {title}")
                        if summary:
                            formatted.append(f"  Summary: {summary}")
                formatted.append("")

        if web_data.get("sources"):
            formatted.append("**Sources:**")
            for source in web_data["sources"][:5]:  # Limit sources
                formatted.append(f"- {source}")

        return "\n".join(formatted)
