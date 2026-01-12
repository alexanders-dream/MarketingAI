"""
Web Scraper Module for Marketing AI v3
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import json
import requests
from bs4 import BeautifulSoup
import time
from firecrawl import FirecrawlApp
from config import AppConfig

logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper using crawl4ai for market research and competitor analysis"""

    def __init__(self):
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
        )
        # Cache playwright availability to avoid repeated checks
        self._playwright_available = None
        
        # Initialize Firecrawl if API key is available
        self.firecrawl_api_key = AppConfig.get_api_key("FIRECRAWL")
        self.firecrawl_app = FirecrawlApp(api_key=self.firecrawl_api_key) if self.firecrawl_api_key else None


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
        
        # Check if Playwright/crawl4ai is likely to work
        playwright_available = self._check_playwright_availability()
        
        if playwright_available:
            try:
                # Try using crawl4ai first
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
                            
            except Exception as e:
                logger.warning(f"Crawl4ai failed, using fallback method: {str(e)}")
                # Use fallback method when crawl4ai fails
                all_results = await self._fallback_scrape(queries[:max_pages])
        else:
            # Skip crawl4ai and use fallback directly
            logger.info("Playwright not available, using fallback scraping method")
            all_results = await self._fallback_scrape(queries[:max_pages])

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
        
        # Check if Playwright/crawl4ai is likely to work
        playwright_available = self._check_playwright_availability()
        
        if playwright_available:
            try:
                # Try using crawl4ai first
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
                            
            except Exception as e:
                logger.warning(f"Crawl4ai failed for competitor scraping, using fallback method: {str(e)}")
                # Use fallback method when crawl4ai fails
                all_results = await self._fallback_scrape(queries[:max_pages])
        else:
            # Skip crawl4ai and use fallback directly
            logger.info("Playwright not available for competitor scraping, using fallback method")
            all_results = await self._fallback_scrape(queries[:max_pages])

        return self._process_competitor_data(all_results, company_name, industry)

    async def scrape_website_content(self, url: str) -> Dict[str, Any]:
        """
        Scrape comprehensive website content for business analysis
        
        Args:
            url: Website URL to scrape
            
        Returns:
            Dictionary containing website content and metadata
        """
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                # Configure crawler for comprehensive content extraction
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        extraction_strategy=JsonCssExtractionStrategy({
                            "title": "title",
                            "meta_description": "meta[name='description']",
                            "meta_keywords": "meta[name='keywords']",
                            "headings": "h1, h2, h3, h4, h5, h6",
                            "content": "p, div[class*='content'], div[class*='text'], article, section",
                            "links": "a[href]",
                            "images": "img[src]",
                            "social_links": "a[href*='facebook.com'], a[href*='twitter.com'], a[href*='linkedin.com'], a[href*='instagram.com']"
                        })
                    )
                )
                
                if result.success:
                    extracted_data = json.loads(result.extracted_content)
                    
                    # Process and structure the data
                    processed_data = {
                        "url": url,
                        "title": self._extract_text(extracted_data, "title"),
                        "meta_description": self._extract_text(extracted_data, "meta_description"),
                        "meta_keywords": self._extract_text(extracted_data, "meta_keywords"),
                        "headings": self._extract_headings(extracted_data),
                        "content": self._extract_main_content(extracted_data),
                        "links": self._extract_links(extracted_data),
                        "social_links": self._extract_social_links(extracted_data),
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    logger.info(f"Successfully scraped website: {url}")
                    return processed_data
                else:
                    logger.error(f"Failed to scrape website: {url}")
                    return self._fallback_website_scrape(url)
                    
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            return self._fallback_website_scrape(url)
    
    def _extract_text(self, data: List[Dict], field_name: str) -> str:
        """Extract text from scraped data"""
        for item in data:
            if field_name in item:
                return item[field_name].strip()
        return ""
    
    def _extract_headings(self, data: List[Dict]) -> List[str]:
        """Extract headings from scraped data"""
        headings = []
        for item in data:
            if "headings" in item:
                heading_text = item["headings"].strip()
                if heading_text and heading_text not in headings:
                    headings.append(heading_text)
        return headings[:10]  # Limit to top 10 headings
    
    def _extract_main_content(self, data: List[Dict]) -> str:
        """Extract main content from scraped data"""
        content_parts = []
        for item in data:
            if "content" in item:
                content_text = item["content"].strip()
                if content_text and len(content_text) > 50:  # Filter out short content
                    content_parts.append(content_text)
        
        # Combine and limit content length
        combined_content = " ".join(content_parts)
        return combined_content[:5000]  # Limit to 5000 characters
    
    def _extract_links(self, data: List[Dict]) -> List[str]:
        """Extract links from scraped data"""
        links = []
        for item in data:
            if "links" in item and item["links"] not in links:
                links.append(item["links"])
        return links[:20]  # Limit to 20 links
    
    def _extract_social_links(self, data: List[Dict]) -> List[str]:
        """Extract social media links from scraped data"""
        social_links = []
        for item in data:
            if "social_links" in item and item["social_links"] not in social_links:
                social_links.append(item["social_links"])
        return social_links
    
    def _fallback_website_scrape(self, url: str) -> Dict[str, Any]:
        """
        Fallback website scraping using Firecrawl or requests
        Used when crawl4ai fails
        """
        # Try Firecrawl first if available
        if self.firecrawl_app:
            try:
                logger.info(f"Attempting to scrape {url} using Firecrawl")
                # Use 'scrape' direct arguments
                scrape_result = self.firecrawl_app.scrape(url, formats=['markdown', 'html'])
                
                if scrape_result and 'markdown' in scrape_result:
                    # Firecrawl returns markdown, which is great for LLMs
                    # We might need to do some light parsing if we strictly need the structure above
                    # For now, let's map it as best as we can
                    metadata = scrape_result.get('metadata', {})
                    
                    return {
                        "url": url,
                        "title": metadata.get('title', ''),
                        "meta_description": metadata.get('description', ''),
                        "meta_keywords": "", # Firecrawl might not return this explicitly
                        "headings": [], # Extracted from markdown if we wanted to parse it
                        "content": scrape_result.get('markdown', '')[:20000], # Firecrawl gives good markdown
                        "links": [], 
                        "social_links": [],
                        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "method": "firecrawl"
                    }
            except Exception as e:
                logger.warning(f"Firecrawl scraping failed for {url}: {e}, falling back to requests")
                
        # Fallback to requests/BeautifulSoup
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            meta_description = soup.find('meta', attrs={'name': 'description'})
            
            # Extract main content
            content_elements = soup.find_all(['p', 'div', 'article', 'section'])
            content = ' '.join([elem.get_text(strip=True) for elem in content_elements if len(elem.get_text(strip=True)) > 50])
            
            # Extract headings
            headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])]
            
            # Extract social links
            social_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(social in href for social in ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']):
                    social_links.append(href)
            
            return {
                "url": url,
                "title": title.get_text(strip=True) if title else "",
                "meta_description": meta_description.get('content', '') if meta_description else "",
                "meta_keywords": "",
                "headings": headings[:10],
                "content": content[:5000],
                "links": [],
                "social_links": social_links[:10],
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": "fallback_soup"
            }
            
        except Exception as e:
            logger.error(f"Fallback website scraping failed for {url}: {str(e)}")
            return {
                "url": url,
                "title": "",
                "meta_description": "",
                "meta_keywords": "",
                "headings": [],
                "content": f"Failed to scrape website: {str(e)}",
                "links": [],
                "social_links": [],
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": "failed",
                "error": str(e)
            }

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
            "sources": [],
            "warnings": []
        }

        # Check if we have any data to process
        if not raw_data:
            processed_data["warnings"].append("No market data could be scraped. Please check your internet connection or try again later.")
            return processed_data

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

        # Add warning if no meaningful data was found
        meaningful_data_found = any(
            len(processed_data[key]) > 0 
            for key in ["market_trends", "industry_analysis", "market_size", "competitive_landscape"]
        )
        
        if not meaningful_data_found:
            processed_data["warnings"].append("Market data scraping completed but no relevant information was found. The results may be limited.")

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
            "sources": [],
            "warnings": []
        }

        # Check if we have any data to process
        if not raw_data:
            processed_data["warnings"].append("No competitor data could be scraped. Please check your internet connection or try again later.")
            return processed_data

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

        # Add warning if no meaningful data was found
        meaningful_data_found = any(
            len(processed_data[key]) > 0 
            for key in ["direct_competitors", "indirect_competitors", "market_leaders", "alternatives"]
        )
        
        if not meaningful_data_found:
            processed_data["warnings"].append("Competitor data scraping completed but no relevant information was found. The results may be limited.")

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

    # Sync wrapper methods for Streamlit compatibility
    def scrape_website_content_sync(self, url: str) -> Dict[str, Any]:
        """Synchronous wrapper for scrape_website_content"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.scrape_website_content(url))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Sync website scraping failed for {url}: {str(e)}")
            return self._fallback_website_scrape(url)

    def scrape_market_data_sync(self, industry: str, company_name: str = None, max_pages: int = 5) -> Dict[str, Any]:
        """Synchronous wrapper for scrape_market_data"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.scrape_market_data(industry, company_name, max_pages))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Sync market data scraping failed: {str(e)}")
            return {"industry": industry, "company": company_name, "error": str(e)}

    def scrape_competitor_data_sync(self, company_name: str, industry: str, max_pages: int = 5) -> Dict[str, Any]:
        """Synchronous wrapper for scrape_competitor_data"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.scrape_competitor_data(company_name, industry, max_pages))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Sync competitor data scraping failed: {str(e)}")
            return {"target_company": company_name, "industry": industry, "error": str(e)}

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

    def _check_playwright_availability(self) -> bool:
        """
        Check if Playwright is available and working on this system.
        Returns False if Playwright is likely to fail due to Windows asyncio issues.
        Uses cached value to avoid repeated checks.
        """
        # Return cached value if already checked
        if self._playwright_available is not None:
            return self._playwright_available
            
        try:
            # Try to import and create a simple Playwright instance to test availability
            import asyncio
            from playwright.async_api import async_playwright
            
            # Test if asyncio subprocess works on this Windows system
            # This is a known issue with Python 3.10+ on Windows
            try:
                # Simple test to see if subprocess creation works
                proc = asyncio.create_subprocess_exec('echo', 'test')
                proc.close()  # Clean up if it worked
                self._playwright_available = True
                logger.info("Playwright is available on this system")
            except NotImplementedError:
                logger.warning("Playwright not available due to asyncio subprocess issues on Windows")
                self._playwright_available = False
                
        except ImportError:
            logger.warning("Playwright not installed")
            self._playwright_available = False
        except Exception as e:
            logger.warning(f"Playwright availability check failed: {e}")
            self._playwright_available = False
            
        return self._playwright_available

    async def _fallback_scrape(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Fallback scraping method using Firecrawl if available, otherwise requests/BeautifulSoup
        Used when Playwright/crawl4ai fails
        """
        results = []
        
        # Try Firecrawl first
        if self.firecrawl_app:
            try:
                logger.info("Using Firecrawl for fallback scraping")
                for query in queries:
                    try:
                        # Use Firecrawl's native search method with direct arguments
                        logger.info(f"Searching with Firecrawl: {query}")
                        search_results = self.firecrawl_app.search(query, limit=5, scrape_options={'formats': ['markdown']})
                        
                        if search_results and isinstance(search_results, dict) and 'data' in search_results:
                            # Handle dictionary response with 'data' key
                            for item in search_results['data']:
                                results.append({
                                    'title': item.get('title', ''),
                                    'link': item.get('url', ''),
                                    'snippet': item.get('description', '') or item.get('markdown', '')[:200]
                                })
                        elif search_results and isinstance(search_results, list):
                            # Handle list response
                            for item in search_results:
                                results.append({
                                    'title': item.get('title', ''),
                                    'link': item.get('url', ''),
                                    'snippet': item.get('description', '') or item.get('markdown', '')[:200]
                                })
                                    
                        time.sleep(1) # Rate limiting
                        
                    except Exception as e:
                         logger.error(f"Firecrawl search failed for query '{query}': {e}")
                         continue
                
                if results:
                    return results
                    
            except Exception as e:
                logger.warning(f"Firecrawl fallback failed: {e}, falling back to requests")
        
        # Fallback to requests (DuckDuckGo HTML)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for query in queries:
            try:
                # Use DuckDuckGo instead of Google to avoid blocking
                search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
                
                response = requests.get(search_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract search results
                    search_results = soup.find_all('div', class_='result')
                    
                    for result in search_results[:5]:  # Limit to top 5 results
                        title_elem = result.find('a', class_='result__a')
                        snippet_elem = result.find('a', class_='result__snippet')
                        
                        if title_elem:
                            results.append({
                                'title': title_elem.get_text(strip=True),
                                'link': title_elem.get('href', ''),
                                'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                            })
                
                # Add delay to be respectful
                time.sleep(1)
                logger.info(f"Fallback scraping completed for query: {query}")
                
            except Exception as e:
                logger.error(f"Fallback scraping failed for query '{query}': {str(e)}")
                # Don't add mock data - let the calling function handle the empty results
                # This ensures data reliability and prevents misleading information
                continue
        
        return results


# Synchronous wrapper functions for easy use
def scrape_market_data_sync(industry: str, company_name: str = None, max_pages: int = 5) -> Dict[str, Any]:
    """Synchronous wrapper for market data scraping"""
    scraper = WebScraper()
    return asyncio.run(scraper.scrape_market_data(industry, company_name, max_pages))


def scrape_competitor_data_sync(company_name: str, industry: str, max_pages: int = 5) -> Dict[str, Any]:
    """Synchronous wrapper for competitor data scraping"""
    scraper = WebScraper()
    return asyncio.run(scraper.scrape_competitor_data(company_name, industry, max_pages))
