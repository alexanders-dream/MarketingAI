"""
Market Analysis Agent for Marketing AI v3 - Enhanced with Guided Research
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

from config import AppConfig
from web_scraper import scrape_market_data_sync, scrape_competitor_data_sync
from research_agents import GuidedMarketResearch, BusinessContext

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Analyzes business documents to generate market insights"""

    def __init__(self):
        self.guided_research = GuidedMarketResearch()
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
        Scrape comprehensive market web data using advanced research
        """
        try:
            # Use guided research for comprehensive market analysis
            market_data = self.guided_research.conduct_guided_research(industry or "general business", company_name)
            
            # Supplement with traditional web scraping
            traditional_market = scrape_market_data_sync(industry or "general business", company_name, max_pages=3)
            traditional_competitors = scrape_competitor_data_sync(company_name, industry or "general business", max_pages=3) if company_name else {}
            
            # Combine both research approaches
            return {
                "guided_research": market_data,
                "traditional_scraping": {
                    "market_research": traditional_market,
                    "competitor_analysis": traditional_competitors
                },
                "research_sources": market_data.get("research_sources", []) + traditional_market.get("sources", []),
                "analysis_date": market_data.get("analysis_date", "")
            }
            
        except Exception as e:
            logger.error(f"Advanced market research failed: {str(e)}")
            # Fallback to traditional scraping only
            market_data = scrape_market_data_sync(industry or "general business", company_name, max_pages=3)
            competitor_data = scrape_competitor_data_sync(company_name, industry or "general business", max_pages=3) if company_name else {}
            
            return {
                "market_research": market_data,
                "competitor_analysis": competitor_data,
                "research_sources": market_data.get("sources", []),
                "analysis_date": "",
                "method": "fallback"
            }

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
        Enhance analysis data with comprehensive web research information

        Args:
            llm: Language model instance
            analysis_data: Original analysis data
            web_data: Advanced web-scraped data including guided research

        Returns:
            Enhanced analysis data with research-backed insights
        """
        try:
            # Format both guided research and traditional scraping data
            web_context = self._format_advanced_web_data_for_llm(web_data)
            
            # Create enhanced analysis prompt
            enhancement_prompt = f"""
            You are a senior marketing strategist with access to comprehensive market research data.
            
            **Original Business Analysis:**
            {json.dumps(analysis_data, indent=2)}
            
            **Advanced Web Research Data:**
            {web_context}
            
            **Task:** Significantly enhance the original business analysis with insights from the comprehensive web research data.
            
            **Enhancement Requirements:**
            1. **Market Trends**: Update with latest industry trends and market dynamics
            2. **Competitive Landscape**: Add detailed competitor analysis and positioning
            3. **Target Audience**: Enhance with demographic and behavioral insights from market research
            4. **Market Opportunities**: Identify specific, data-backed opportunities
            5. **Keywords**: Add high-value, research-backed keywords
            6. **Content Strategy**: Suggest trending topics and content gaps
            7. **Marketing Channels**: Recommend channels based on industry best practices
            8. **Budget Considerations**: Include industry-standard budget allocations
            
            **Data Sources to Consider:**
            - Market research reports and industry publications
            - Competitor analysis and market positioning data
            - Trend analysis and consumer behavior insights
            - Industry benchmarks and performance metrics
            - Emerging market opportunities and growth areas
            
            **Output Format:** Return a JSON object with enhanced versions of all original fields, significantly improved with web research insights.
            
            **Quality Standards:**
            - Use specific data points and statistics where available
            - Reference industry trends and market dynamics
            - Include actionable recommendations based on research
            - Highlight competitive advantages and market positioning
            - Identify specific opportunities for growth and expansion
            """
            
            from langchain_core.output_parsers import JsonOutputParser
            
            enhancement_chain = ChatPromptTemplate.from_template(enhancement_prompt) | llm | JsonOutputParser()
            enhanced_analysis = enhancement_chain.invoke({})
            
            # Add research metadata
            enhanced_analysis["research_sources"] = web_data.get("research_sources", [])
            enhanced_analysis["analysis_date"] = web_data.get("analysis_date", "")
            enhanced_analysis["enhancement_method"] = "advanced_web_research"
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Failed to enhance analysis with web data: {str(e)}")
            # Return original data with error flag
            analysis_data["enhancement_error"] = str(e)
            return analysis_data

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

    def _format_advanced_web_data_for_llm(self, web_data: Dict[str, Any]) -> str:
        """
        Format advanced web research data for LLM consumption

        Args:
            web_data: Advanced web research data including guided research

        Returns:
            Formatted string for LLM
        """
        formatted_data = []
        
        # Research metadata
        if "research_sources" in web_data:
            formatted_data.append("## Research Sources")
            for source in web_data["research_sources"][:5]:
                formatted_data.append(f"- {source}")
        
        # Guided research results
        if "guided_research" in web_data:
            research = web_data["guided_research"]
            formatted_data.append("\n## Comprehensive Market Research")
            
            if "market_overview" in research:
                formatted_data.append("### Market Overview")
                formatted_data.append(f"{research['market_overview']}")
            
            if "industry_trends" in research:
                formatted_data.append("\n### Industry Trends")
                formatted_data.append(f"{research['industry_trends']}")
            
            if "competitive_analysis" in research:
                formatted_data.append("\n### Competitive Analysis")
                formatted_data.append(f"{research['competitive_analysis']}")
            
            if "target_audience_insights" in research:
                formatted_data.append("\n### Target Audience Insights")
                formatted_data.append(f"{research['target_audience_insights']}")
            
            if "market_opportunities" in research:
                formatted_data.append("\n### Market Opportunities")
                formatted_data.append(f"{research['market_opportunities']}")
            
            if "content_strategy_recommendations" in research:
                formatted_data.append("\n### Content Strategy Recommendations")
                formatted_data.append(f"{research['content_strategy_recommendations']}")
            
            if "budget_recommendations" in research:
                formatted_data.append("\n### Budget Recommendations")
                formatted_data.append(f"{research['budget_recommendations']}")
        
        # Traditional web scraping data (fallback)
        if "market_trends" in web_data and web_data["market_trends"]:
            formatted_data.append("\n## Additional Market Data")
            formatted_data.append("### Market Trends")
            for trend in web_data["market_trends"][:5]:
                formatted_data.append(f"- {trend}")
        
        if "competitors" in web_data and web_data["competitors"]:
            formatted_data.append("\n### Key Competitors")
            for competitor in web_data["competitors"][:8]:
                formatted_data.append(f"- {competitor}")
        
        if "market_size" in web_data and web_data["market_size"]:
            formatted_data.append("\n### Market Size Information")
            for size in web_data["market_size"][:3]:
                formatted_data.append(f"- {size}")
        
        return "\n".join(formatted_data)

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

    def generate_guided_market_analysis(self, business_context: Dict[str, str], use_guided_research: bool = True, use_web_scraping: bool = True) -> Dict[str, Any]:
        """Generate comprehensive market analysis with advanced research capabilities"""
        try:
            # Initialize components
            llm = self.llm_manager.get_llm()
            
            # Extract business context
            business_info = self._extract_business_context(business_context)
            
            # Generate keywords first
            keywords = self._generate_market_keywords(llm, business_info)
            
            # Initialize web data
            web_data = {}
            
            # Use guided research if available and enabled
            if use_guided_research and hasattr(self, 'guided_research') and self.guided_research:
                try:
                    research_results = self.guided_research.research_market(
                        business_info.company_name,
                        business_info.industry,
                        business_info.target_audience,
                        keywords
                    )
                    
                    if research_results and research_results.get("status") == "success":
                        web_data = research_results
                        logger.info("Successfully used guided research for market analysis")
                except Exception as e:
                    logger.warning(f"Guided research failed, falling back to web scraping: {str(e)}")
            
            # Fallback to traditional web scraping if guided research failed or disabled
            if not web_data and use_web_scraping:
                try:
                    web_data = self._scrape_market_web_data(keywords, business_info.industry)
                    logger.info("Successfully used web scraping for market analysis")
                except Exception as e:
                    logger.warning(f"Web scraping failed: {str(e)}")
            
            # Generate comprehensive analysis with enhanced research data
            analysis_prompt = f"""
            You are a senior market research analyst with access to comprehensive market intelligence.
            
            **Business Context:**
            Company: {business_info.company_name}
            Industry: {business_info.industry}
            Target Audience: {business_info.target_audience}
            Brand Description: {business_info.brand_description}
            Products/Services: {business_info.products_services}
            Marketing Goals: {business_info.marketing_goals}
            
            **Keywords:** {', '.join(keywords[:10])}
            
            **Advanced Market Research Data:**
            {self._format_advanced_web_data_for_llm(web_data) if web_data else 'No additional market research data available.'}
            
            **Task:** Generate a comprehensive market analysis report with the following sections:
            
            1. **Market Overview**: Industry landscape, market size, growth potential, and key market dynamics
            2. **Target Audience Analysis**: Detailed demographic, psychographic, and behavioral profiles with data-backed insights
            3. **Competitive Landscape**: Comprehensive competitor analysis, market positioning, and competitive advantages
            4. **Market Trends**: Current and emerging trends with specific data points and industry statistics
            5. **Opportunities & Threats**: Specific opportunities and potential challenges with actionable insights
            6. **Marketing Strategy Recommendations**: Data-driven marketing recommendations based on research
            7. **Content Strategy**: Suggested content themes, trending topics, and content gaps to fill
            8. **Budget Considerations**: Industry-standard budget allocations and ROI expectations
            
            **Research-Based Analysis Requirements:**
            - Use specific data points, statistics, and industry benchmarks where available
            - Reference current market trends, consumer behavior insights, and industry dynamics
            - Include actionable recommendations based on comprehensive research data
            - Focus on specific opportunities for growth, expansion, and market penetration
            - Highlight competitive advantages and market positioning opportunities
            - Provide budget recommendations based on industry standards and ROI data
            - Suggest content themes based on trending topics and audience interests
            
            **Output Format:** Return a JSON object with the above sections as keys, providing detailed, research-backed analysis for each.
            """
            
            from langchain_core.output_parsers import JsonOutputParser
            
            analysis_chain = ChatPromptTemplate.from_template(analysis_prompt) | llm | JsonOutputParser()
            analysis_result = analysis_chain.invoke({})
            
            # Add comprehensive metadata
            analysis_result["keywords"] = keywords
            analysis_result["research_method"] = "guided_research" if web_data.get("status") == "success" else "web_scraping"
            analysis_result["analysis_date"] = datetime.now().isoformat()
            analysis_result["research_sources"] = web_data.get("research_sources", [])
            analysis_result["data_quality_score"] = web_data.get("data_quality_score", 0)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to generate guided market analysis: {str(e)}")
            
            # Fallback to original analysis method
            return self._generate_original_analysis(business_context)

    def _extract_business_context(self, llm, vector_store: FAISS, 
                                company_name: str = None, industry: str = None) -> BusinessContext:
        """
        Extract business context from uploaded documents
        
        Args:
            llm: Language model instance
            vector_store: FAISS vector store
            company_name: Optional company name
            industry: Optional industry
            
        Returns:
            BusinessContext object with extracted information
        """
        try:
            # Generate basic analysis to extract context
            analysis_data = {}
            for field_name in self.analysis_fields.keys():
                insight = self.generate_insight(llm, vector_store, field_name)
                analysis_data[field_name] = insight
            
            # Extract keywords from the keywords field
            keywords = []
            if analysis_data.get("keywords"):
                keywords = [k.strip() for k in analysis_data["keywords"].split(",")]
            
            # Create business context
            business_context = BusinessContext(
                company_name=company_name or self._extract_company_name(analysis_data),
                industry=industry or self._extract_industry(analysis_data),
                products_services=analysis_data.get("products_services", ""),
                target_audience=analysis_data.get("target_audience", ""),
                brand_description=analysis_data.get("brand_description", ""),
                competitive_advantages=analysis_data.get("competitive_advantages", ""),
                market_goals=analysis_data.get("marketing_goals", ""),
                existing_content=analysis_data.get("existing_content", ""),
                keywords=keywords
            )
            
            return business_context
            
        except Exception as e:
            logger.error(f"Failed to extract business context: {str(e)}")
            # Return minimal context
            return BusinessContext(
                company_name=company_name or "Unknown Company",
                industry=industry or "General Business"
            )

    def _extract_company_name(self, analysis_data: Dict[str, str]) -> str:
        """Extract company name from analysis data"""
        brand_desc = analysis_data.get("brand_description", "")
        # Simple extraction - look for company name patterns
        if brand_desc:
            # This is a simple heuristic - could be improved with NER
            words = brand_desc.split()
            for i, word in enumerate(words):
                if word.lower() in ["company", "corp", "inc", "ltd", "llc"]:
                    if i > 0:
                        return words[i-1]
        return "Unknown Company"

    def _extract_industry(self, analysis_data: Dict[str, str]) -> str:
        """Extract industry from analysis data"""
        products = analysis_data.get("products_services", "")
        brand_desc = analysis_data.get("brand_description", "")
        
        # Simple industry detection based on keywords
        text = f"{products} {brand_desc}".lower()
        
        industry_keywords = {
            "technology": ["software", "tech", "digital", "app", "platform", "saas"],
            "healthcare": ["health", "medical", "hospital", "clinic", "pharmaceutical"],
            "finance": ["financial", "bank", "investment", "insurance", "fintech"],
            "retail": ["retail", "store", "shop", "ecommerce", "marketplace"],
            "education": ["education", "school", "university", "learning", "training"],
            "manufacturing": ["manufacturing", "factory", "production", "industrial"],
            "consulting": ["consulting", "advisory", "professional services"],
            "food": ["food", "restaurant", "catering", "beverage", "culinary"]
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                return industry.title()
        
        return "General Business"

    def generate_guided_market_analysis_with_context(self, llm, 
                                                   company_name: str, 
                                                   industry: str,
                                                   target_audience: str,
                                                   products_services: str,
                                                   brand_description: str,
                                                   marketing_goals: str,
                                                   use_guided_research: bool = True) -> Dict[str, str]:
        """
        Generate guided market analysis using business context from step-by-step workflow
        
        Args:
            llm: Language model instance
            company_name: Company name
            industry: Industry sector
            target_audience: Target audience description
            products_services: Products/services description
            brand_description: Brand description
            marketing_goals: Marketing goals
            use_guided_research: Whether to use guided research
            
        Returns:
            Dictionary with market analysis results
        """
        try:
            # Create business context from provided information
            business_context = BusinessContext(
                company_name=company_name,
                industry=industry,
                products_services=products_services,
                target_audience=target_audience,
                brand_description=brand_description,
                market_goals=marketing_goals,
                keywords=[]  # Will be generated
            )
            
            # Generate keywords based on business context
            keywords_prompt = """
            Based on this business information, generate 10-15 relevant marketing keywords:
            
            Company: {company_name}
            Industry: {industry}
            Products/Services: {products_services}
            Target Audience: {target_audience}
            Brand: {brand_description}
            
            Return only the keywords as a comma-separated list.
            """
            
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            prompt = ChatPromptTemplate.from_template(keywords_prompt)
            chain = prompt | llm | StrOutputParser()
            
            keywords_str = chain.invoke({
                "company_name": company_name,
                "industry": industry,
                "products_services": products_services,
                "target_audience": target_audience,
                "brand_description": brand_description
            })
            
            business_context.keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            
            # Run guided research if enabled
            if use_guided_research:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    comprehensive_report = loop.run_until_complete(
                        self.guided_research.conduct_comprehensive_research(
                            llm, business_context, None  # No vector store, using provided context
                        )
                    )
                    
                    # Parse the comprehensive report into structured data
                    return self._parse_comprehensive_report(comprehensive_report, business_context)
                    
                finally:
                    loop.close()
            else:
                # Basic analysis without guided research
                return self._generate_basic_analysis(llm, business_context)
                
        except Exception as e:
            logger.error(f"Guided market analysis with context failed: {str(e)}")
            return {
                "error": f"Market analysis failed: {str(e)}",
                "competitors": "Analysis failed",
                "market_trends": "Analysis failed",
                "market_opportunities": "Analysis failed"
            }
    
    def _parse_comprehensive_report(self, report: str, business_context: BusinessContext) -> Dict[str, str]:
        """Parse comprehensive report into structured data"""
        # This is a simplified parser - you could enhance it with more sophisticated parsing
        sections = {
            "competitors": "Competitive Landscape",
            "market_trends": "Industry Trends",
            "market_opportunities": "Market Opportunities",
            "competitive_advantages": "Competitive Advantages",
            "market_size": "Market Size",
            "target_segments": "Target Segments"
        }
        
        parsed_data = {}
        
        for key, section_title in sections.items():
            # Simple parsing - look for section headers
            if section_title in report:
                # Extract content between this section and the next
                start_idx = report.find(section_title)
                next_section_idx = len(report)
                
                # Find the next section
                for other_title in sections.values():
                    if other_title != section_title:
                        other_idx = report.find(other_title, start_idx + len(section_title))
                        if other_idx > start_idx and other_idx < next_section_idx:
                            next_section_idx = other_idx
                
                section_content = report[start_idx:next_section_idx].strip()
                # Remove the section title
                section_content = section_content.replace(section_title, "").strip()
                parsed_data[key] = section_content
            else:
                # Fallback for missing sections
                parsed_data[key] = f"Analysis for {section_title.lower()} not available"
        
        # Add business context
        parsed_data.update({
            "company_name": business_context.company_name,
            "industry": business_context.industry,
            "target_audience": business_context.target_audience,
            "products_services": business_context.products_services,
            "brand_description": business_context.brand_description
        })
        
        return parsed_data
    
    def _generate_basic_analysis(self, llm, business_context: BusinessContext) -> Dict[str, str]:
        """Generate basic market analysis without guided research"""
        basic_prompt = """
        Based on this business information, provide a market analysis covering:
        
        1. Key competitors and competitive landscape
        2. Current market trends and opportunities
        3. Market size and growth potential
        4. Target customer segments
        5. Competitive positioning recommendations
        
        Business Information:
        Company: {company_name}
        Industry: {industry}
        Products/Services: {products_services}
        Target Audience: {target_audience}
        Brand: {brand_description}
        Marketing Goals: {market_goals}
        
        Provide detailed, actionable insights for each area.
        """
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_template(basic_prompt)
        chain = prompt | llm | StrOutputParser()
        
        analysis = chain.invoke({
            "company_name": business_context.company_name,
            "industry": business_context.industry,
            "products_services": business_context.products_services,
            "target_audience": business_context.target_audience,
            "brand_description": business_context.brand_description,
            "market_goals": business_context.market_goals
        })
        
        return {
            "comprehensive_analysis": analysis,
            "company_name": business_context.company_name,
            "industry": business_context.industry,
            "target_audience": business_context.target_audience,
            "products_services": business_context.products_services,
            "brand_description": business_context.brand_description
        }