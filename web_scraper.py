"""
Web Scraper Module for Marketing AI v3
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import json

logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper using crawl4ai for market research and competitor analysis"""

    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
        )

    async def scrape_market_data(self, industry: str, company_name: str = None,
                                max_pages: int = 5) -> Dict[str, Any]:
        """
        Scrape market data for industry analysis

        Args:
            industry: Industry sector to research
            company_name: Optional specific company to focus on
            max_pages: Maximum number of pages to scrape

        Returns:
            Dictionary containing market research data
        """
        queries = [
            f"{industry} market trends 2024",
            f"{industry} industry analysis",
            f"{industry} market size and growth",
            f"{industry} competitive landscape"
        ]

        if company_name:
            queries.extend([
                f"{company_name} competitors",
                f"{company_name} market position",
                f"{company_name} industry analysis"
            ])

        all_results = []
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            for query in queries[:max_pages]:
                try:
                    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=10"
                    result = await crawler.arun(
                        url=search_url,
                        config=CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS,
                            extraction_strategy=JsonCssExtractionStrategy({
                                "title": "h3",
                                "link": "a[href]",
                                "snippet": ".VwiC3b, .s3v9rd, .kvH3mc"
                            })
                        )
                    )

                    if result.success:
                        data = json.loads(result.extracted_content)
                        all_results.extend(data)
                        logger.info(f"Successfully scraped data for query: {query}")

                except Exception as e:
                    logger.error(f"Error scraping for query '{query}': {str(e)}")
                    continue

        return self._process_market_data(all_results, industry, company_name)

    async def scrape_competitor_data(self, company_name: str, industry: str,
                                    max_pages: int = 5) -> Dict[str, Any]:
        """
        Scrape competitor data for analysis

        Args:
            company_name: Company to find competitors for
            industry: Industry sector
            max_pages: Maximum number of pages to scrape

        Returns:
            Dictionary containing competitor research data
        """
        queries = [
            f"{company_name} competitors",
            f"alternatives to {company_name}",
            f"{company_name} vs competitors",
            f"{industry} market leaders",
            f"top {industry} companies"
        ]

        all_results = []
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            for query in queries[:max_pages]:
                try:
                    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=10"
                    result = await crawler.arun(
                        url=search_url,
                        config=CrawlerRunConfig(
                            cache_mode=CacheMode.BYPASS,
                            extraction_strategy=JsonCssExtractionStrategy({
                                "title": "h3",
                                "link": "a[href]",
                                "snippet": ".VwiC3b, .s3v9rd, .kvH3mc"
                            })
                        )
                    )

                    if result.success:
                        data = json.loads(result.extracted_content)
                        all_results.extend(data)
                        logger.info(f"Successfully scraped competitor data for query: {query}")

                except Exception as e:
                    logger.error(f"Error scraping competitors for query '{query}': {str(e)}")
                    continue

        return self._process_competitor_data(all_results, company_name, industry)

    def _process_market_data(self, raw_data: List[Dict], industry: str,
                           company_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process raw scraped market data into structured format

        Args:
            raw_data: Raw scraped data
            industry: Industry sector
            company_name: Optional company name

        Returns:
            Processed market data
        """
        processed_data = {
            "industry": industry,
            "company": company_name,
            "market_trends": [],
            "industry_analysis": [],
            "market_size": [],
            "competitive_landscape": [],
            "sources": []
        }

        for item in raw_data:
            title = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()
            link = item.get("link", "")

            # Categorize content based on keywords
            if any(keyword in title + snippet for keyword in ["trend", "2024", "future", "emerging"]):
                processed_data["market_trends"].append({
                    "title": item.get("title", ""),
                    "summary": item.get("snippet", ""),
                    "source": link
                })

            elif any(keyword in title + snippet for keyword in ["analysis", "overview", "sector", "industry"]):
                processed_data["industry_analysis"].append({
                    "title": item.get("title", ""),
                    "summary": item.get("snippet", ""),
                    "source": link
                })

            elif any(keyword in title + snippet for keyword in ["market size", "growth", "revenue", "valuation"]):
                processed_data["market_size"].append({
                    "title": item.get("title", ""),
                    "summary": item.get("snippet", ""),
                    "source": link
                })

            elif any(keyword in title + snippet for keyword in ["competitor", "competition", "market share", "leader"]):
                processed_data["competitive_landscape"].append({
                    "title": item.get("title", ""),
                    "summary": item.get("snippet", ""),
                    "source": link
                })

            processed_data["sources"].append(link)

        # Remove duplicates and limit results
        for key in ["market_trends", "industry_analysis", "market_size", "competitive_landscape"]:
            processed_data[key] = self._deduplicate_items(processed_data[key])[:5]

        processed_data["sources"] = list(set(processed_data["sources"]))[:10]

        return processed_data

    def _process_competitor_data(self, raw_data: List[Dict], company_name: str,
                               industry: str) -> Dict[str, Any]:
        """
        Process raw scraped competitor data into structured format

        Args:
            raw_data: Raw scraped data
            company_name: Company name
            industry: Industry sector

        Returns:
            Processed competitor data
        """
        processed_data = {
            "target_company": company_name,
            "industry": industry,
            "direct_competitors": [],
            "indirect_competitors": [],
            "market_leaders": [],
            "alternatives": [],
            "sources": []
        }

        for item in raw_data:
            title = item.get("title", "").lower()
            snippet = item.get("snippet", "").lower()
            link = item.get("link", "")

            # Extract competitor names from titles and snippets
            competitor_names = self._extract_competitor_names(title + " " + snippet, company_name)

            if competitor_names:
                competitor_info = {
                    "names": competitor_names,
                    "title": item.get("title", ""),
                    "summary": item.get("snippet", ""),
                    "source": link
                }

                # Categorize competitors
                if "vs" in title or "versus" in title or "comparison" in title:
                    processed_data["direct_competitors"].append(competitor_info)
                elif "alternative" in title or "option" in title:
                    processed_data["alternatives"].append(competitor_info)
                elif "leader" in title or "top" in title:
                    processed_data["market_leaders"].append(competitor_info)
                else:
                    processed_data["indirect_competitors"].append(competitor_info)

            processed_data["sources"].append(link)

        # Remove duplicates and limit results
        for key in ["direct_competitors", "indirect_competitors", "market_leaders", "alternatives"]:
            processed_data[key] = self._deduplicate_competitors(processed_data[key])[:5]

        processed_data["sources"] = list(set(processed_data["sources"]))[:10]

        return processed_data

    def _extract_competitor_names(self, text: str, exclude_company: str) -> List[str]:
        """
        Extract potential competitor names from text

        Args:
            text: Text to analyze
            exclude_company: Company name to exclude

        Returns:
            List of potential competitor names
        """
        # Simple extraction - in a real implementation, this could use NLP
        words = text.split()
        competitors = []

        # Look for capitalized words that might be company names
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2 and word.lower() != exclude_company.lower():
                # Check if it's part of a company name (multiple capitalized words)
                company_name = word
                j = i + 1
                while j < len(words) and words[j][0].isupper() and len(words[j]) > 1:
                    company_name += " " + words[j]
                    j += 1

                if company_name.lower() != exclude_company.lower():
                    competitors.append(company_name)

        return list(set(competitors))[:3]  # Limit to 3 per item

    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate items based on title"""
        seen_titles = set()
        deduplicated = []

        for item in items:
            title = item.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                deduplicated.append(item)

        return deduplicated

    def _deduplicate_competitors(self, competitors: List[Dict]) -> List[Dict]:
        """Remove duplicate competitors based on names"""
        seen_names = set()
        deduplicated = []

        for comp in competitors:
            names_tuple = tuple(sorted(comp.get("names", [])))
            if names_tuple and names_tuple not in seen_names:
                seen_names.add(names_tuple)
                deduplicated.append(comp)

        return deduplicated


# Synchronous wrapper functions for easy use
def scrape_market_data_sync(industry: str, company_name: str = None, max_pages: int = 5) -> Dict[str, Any]:
    """Synchronous wrapper for market data scraping"""
    scraper = WebScraper()
    return asyncio.run(scraper.scrape_market_data(industry, company_name, max_pages))


def scrape_competitor_data_sync(company_name: str, industry: str, max_pages: int = 5) -> Dict[str, Any]:
    """Synchronous wrapper for competitor data scraping"""
    scraper = WebScraper()
    return asyncio.run(scraper.scrape_competitor_data(company_name, industry, max_pages))
