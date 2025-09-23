"""
Content Generation Engine for Marketing AI v3
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class ContentGenerator:
    """Generates various types of marketing content"""

    def __init__(self):
        self.content_templates = self._load_content_templates()

    def _load_content_templates(self) -> Dict[str, str]:
        """Load content generation templates for different marketing tasks"""
        return {
            "Marketing Strategy": """
                You are a senior marketing strategist tasked with creating a comprehensive marketing plan.

                ## Business Context
                Brand Description: {brand_description}
                Target Audience: {target_audience}
                Products/Services: {products_services}
                Marketing Goals: {marketing_goals}
                Existing Content: {existing_content}
                Keywords: {keywords}

                ## Instructions
                Develop a detailed, actionable marketing strategy that aligns with the business goals.
                Focus on creating a strategy that is specific, measurable, achievable, relevant, and time-bound.

                ## Required Output Structure
                1. Executive Summary (brief overview of the entire strategy)
                2. Market Analysis (industry trends, competitive landscape)
                3. Target Audience Segmentation (detailed profiles of key segments)
                4. Value Proposition & Positioning (unique selling points, brand positioning)
                5. Marketing Channels & Tactics (prioritized by ROI potential)
                6. Content Strategy (topics, formats, distribution, calendar)
                7. Budget Allocation (recommended spending by channel)
                8. Implementation Timeline (30-60-90 day plan)
                9. KPIs & Success Metrics (specific measurements for each goal)
                10. Risk Assessment & Contingency Plans
            """,

            "Campaign Strategy": """
                You are a creative campaign director tasked with developing innovative marketing campaigns.

                ## Business Context
                Brand Description: {brand_description}
                Target Audience: {target_audience}
                Products/Services: {products_services}
                Marketing Goals: {marketing_goals}
                Keywords: {keywords}
                Suggested Topics: {suggested_topics}
                Tone: {tone}

                ## Instructions
                Generate 5 distinct, creative campaign concepts that align with the brand identity and will resonate with the target audience.
                Each campaign should be achievable with realistic resources and have clear business impact.

                ## Required Output Structure
                For each of the 5 campaigns, provide:

                ### Campaign [Number]: [Creative Name]
                * Concept: Brief explanation of the campaign idea and creative angle
                * Target Segment: Specific audience segment this will appeal to most
                * Core Message: The primary takeaway for the audience
                * Campaign Elements: List of deliverables (videos, posts, emails, etc.)
                * Channels: Primary platforms for distribution
                * Timeline: Suggested duration and key milestones
                * Success Metrics: How to measure campaign effectiveness
                * Estimated Impact: Expected outcomes tied to marketing goals
            """,

            "Social Media Content Strategy": """
                You are an expert social media manager creating platform-specific content.

                ## Business Context
                Brand Description: {brand_description}
                Target Audience: {target_audience}
                Products/Services: {products_services}
                Marketing Goals: {marketing_goals}
                Keywords: {keywords}
                Suggested Topics: {suggested_topics}
                Tone: {tone}
                Post Type: {post_type}

                ## Instructions
                Create a comprehensive social media content plan optimized for {post_type}.
                Focus on engaging the target audience with content that drives specific marketing goals.
                Ensure all content maintains the brand's {tone} tone of voice.

                ## Required Output Structure
                1. Platform Strategy
                   * Why {post_type} is effective for this audience
                   * Best practices specific to this platform
                   * Posting frequency recommendations

                2. Content Pillars (3-4 key themes aligned with business goals)

                3. Content Calendar (2-week sample)
                   * Week 1:
                     * Day 1: [Content type] - [Example post with exact copy]
                     * Day 2: [Content type] - [Example post with exact copy]
                     [Continue for all week]
                   * Week 2: [Same format]

                4. Engagement Strategy
                   * Response templates for common interactions
                   * Community-building tactics
                   * User-generated content opportunities

                5. Growth Tactics
                   * Hashtag strategy (10-15 targeted hashtags grouped by purpose)
                   * Collaboration opportunities
                   * Cross-promotion ideas

                6. Analytics Focus
                   * Key metrics to track for this specific platform
                   * Benchmarks for success
            """,

            "SEO Optimization Strategy": """
                You are an SEO specialist developing a comprehensive search optimization strategy.

                ## Business Context
                Brand Description: {brand_description}
                Target Audience: {target_audience}
                Products/Services: {products_services}
                Marketing Goals: {marketing_goals}
                Keywords: {keywords}
                Existing Content: {existing_content}

                ## Instructions
                Create a detailed SEO strategy that will improve organic visibility and drive qualified traffic.
                Focus on both quick wins and long-term sustainable growth.
                Provide specific, actionable recommendations rather than general advice.

                ## Required Output Structure
                1. Keyword Strategy
                   * Primary Keywords (5-7 high-priority terms with search volume estimates)
                   * Secondary Keywords (10-15 supporting terms)
                   * Long-tail Opportunities (7-10 specific phrases)
                   * Semantic/Topic Clusters (group related terms by topic)

                2. On-Page Optimization
                   * Title Tag Templates
                   * Meta Description Frameworks
                   * Heading Structure Recommendations
                   * Content Length and Formatting Guidelines
                   * Internal Linking Strategy

                3. Technical SEO Checklist
                   * Site Speed Optimization
                   * Mobile Usability
                   * Schema Markup Recommendations
                   * Indexation Controls
                   * URL Structure Guidelines

                4. Content Strategy
                   * Content Gaps Analysis
                   * Content Update Priorities
                   * New Content Recommendations (5-7 specific pieces)
                   * Content Calendar Framework

                5. Off-Page Strategy
                   * Link Building Tactics (specific to industry)
                   * Digital PR Opportunities
                   * Local Citation Opportunities (if applicable)

                6. Measurement Plan
                   * Key Performance Indicators
                   * Tracking Setup Recommendations
                   * Reporting Schedule and Format
            """,

            "Post Composer": """
                You are a professional copywriter creating compelling {post_type} content.

                ## Business Context
                Brand Description: {brand_description}
                Target Audience: {target_audience}
                Products/Services: {products_services}
                Marketing Goals: {marketing_goals}
                Keywords: {keywords}
                Existing Content: {existing_content}
                Suggested Topics: {suggested_topics}

                ## Instructions
                Create high-converting {post_type} copy that speaks directly to the target audience.
                Maintain a {tone} tone throughout while incorporating strategic keywords {keywords} naturally.
                The copy should directly support the stated marketing goals and the selected topics {suggested_topics}.

                ## Platform-Specific Guidelines
                ### Instagram:
                - a visual element description
                - Create a captivating caption (max 150 words)
                - Include a strong hook in the first line
                - Use 5-10 relevant hashtags
                - Add a clear call-to-action

                ### LinkedIn:
                - Professional but engaging tone
                - 3-5 short paragraphs with white space
                - Include industry insights or data points
                - End with a thoughtful question or clear CTA

                ### Twitter:
                - Concise messaging under 280 characters
                - Include relevant hashtags (2-3)
                - Consider a visual element description
                - Create a compelling reason to click/engage

                ### Blog:
                - Compelling headline with primary keyword
                - 800-1200 words with clear structure
                - H2 and H3 subheadings containing keywords
                - Introduction with hook and thesis
                - Body with valuable insights/examples
                - Conclusion with next steps or CTA

                ### Podcast:
                - Episode title and description
                - Show notes with timestamps
                - Key talking points and questions
                - Call-to-action for listeners

                ### Media Brief:
                - Headline and subheadline
                - Key message points (3-5 bulletpoints)
                - Supporting facts/statistics
                - Quote from company representative
                - Call-to-action and contact information

                ## Required Output Structure
                - Headline/Title: Attention-grabbing, keyword-rich
                - Main Content: Formatted appropriately for {post_type}
                - Call-to-Action: Clear next step for the audience
                - [For social posts] Hashtags: Strategically selected for reach
            """,

            "Market Analysis": """
                You are a market research analyst providing comprehensive business intelligence.

                ## Business Context
                Brand Description: {brand_description}
                Target Audience: {target_audience}
                Products/Services: {products_services}
                Marketing Goals: {marketing_goals}
                Keywords: {keywords}
                Market Opportunities: {market_opportunities}
                Competitive Advantages: {competitive_advantages}
                Customer Pain Points: {customer_pain_points}

                ## Instructions
                Provide a comprehensive market analysis that gives actionable insights for marketing strategy development.
                Focus on identifying opportunities, understanding the competitive landscape, and providing strategic recommendations.

                ## Required Output Structure

                ## Executive Summary
                Key findings and strategic implications (2-3 paragraphs)

                ## Industry Overview
                Current market size, growth trends, and key drivers

                ## Target Market Analysis
                Detailed segmentation and customer profiling

                ## Competitive Landscape
                Key competitors, market share analysis, and positioning

                ## SWOT Analysis
                Strengths, Weaknesses, Opportunities, Threats

                ## Market Opportunities
                Growth areas, underserved segments, and strategic gaps

                ## Customer Insights
                Pain points, buying behavior, and decision drivers

                ## Strategic Recommendations
                Actionable steps based on the analysis

                ## Risk Assessment
                Potential challenges and mitigation strategies
            """
        }

    def generate_content(self, llm, task: str, form_data: Dict[str, str]) -> str:
        """
        Generate content for a specific marketing task

        Args:
            llm: Language model instance
            task: Marketing task type
            form_data: Form data with business context

        Returns:
            Generated content
        """
        if task not in self.content_templates:
            raise ValueError(f"Unknown task type: {task}")

        try:
            # Create the prompt
            prompt = ChatPromptTemplate.from_template(self.content_templates[task])

            # Create the chain
            chain = prompt | llm | StrOutputParser()

            # Execute the chain
            response = chain.invoke(form_data)

            return response

        except Exception as e:
            logger.error(f"Content generation failed for task {task}: {str(e)}")
            return f"Error generating content: {str(e)}"

    def enhance_content_with_market_data(self, llm, base_content: str,
                                       market_analysis: Dict[str, str],
                                       enhancement_type: str = "comprehensive") -> str:
        """
        Enhance existing content with market analysis insights

        Args:
            llm: Language model instance
            base_content: Original content to enhance
            market_analysis: Market analysis data
            enhancement_type: Type of enhancement (comprehensive, targeted, etc.)

        Returns:
            Enhanced content
        """
        enhancement_prompt = f"""
        You are a content strategist tasked with enhancing marketing content using market intelligence.

        ## Original Content
        {base_content}

        ## Market Analysis Context
        Brand Description: {market_analysis.get('brand_description', '')}
        Target Audience: {market_analysis.get('target_audience', '')}
        Competitive Advantages: {market_analysis.get('competitive_advantages', '')}
        Market Opportunities: {market_analysis.get('market_opportunities', '')}
        Customer Pain Points: {market_analysis.get('customer_pain_points', '')}
        Keywords: {market_analysis.get('keywords', '')}

        ## Enhancement Instructions
        Enhance the original content by:
        1. Incorporating relevant market insights naturally
        2. Strengthening value propositions with competitive advantages
        3. Addressing customer pain points more effectively
        4. Optimizing keyword integration for better SEO
        5. Making the content more targeted to the specific audience
        6. Adding data-driven credibility where appropriate

        Maintain the original content's structure and tone while making it more compelling and conversion-focused.
        Do not change the fundamental message or call-to-action - enhance and strengthen it.

        ## Enhanced Content
        """

        try:
            prompt = ChatPromptTemplate.from_template(enhancement_prompt)
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({})

        except Exception as e:
            logger.error(f"Content enhancement failed: {str(e)}")
            return base_content  # Return original content if enhancement fails


class ContentPerformanceScorer:
    """Scores content performance potential"""

    def __init__(self):
        self.scoring_criteria = {
            "engagement_potential": [
                "hook_strength", "value_proposition", "call_to_action",
                "emotional_appeal", "relevance_to_audience"
            ],
            "seo_optimization": [
                "keyword_integration", "title_optimization", "content_length",
                "internal_linking", "mobile_friendliness"
            ],
            "shareability": [
                "uniqueness", "timeliness", "controversial_elements",
                "visual_appeal", "practical_value"
            ],
            "conversion_potential": [
                "urgency_creation", "scarcity_elements", "social_proof",
                "risk_reduction", "clear_next_steps"
            ]
        }

    def score_content(self, llm, content: str, content_type: str = "general") -> Dict[str, Any]:
        """
        Score content performance across multiple dimensions

        Args:
            llm: Language model instance
            content: Content to score
            content_type: Type of content (blog, social, email, etc.)

        Returns:
            Performance scores and recommendations
        """
        scoring_prompt = f"""
        You are a content marketing expert evaluating content performance potential.

        ## Content to Evaluate
        {content}

        ## Content Type: {content_type}

        ## Evaluation Criteria
        Rate each of the following aspects on a scale of 1-10 (10 being excellent):

        **Engagement Potential:**
        - Hook Strength: How compelling is the opening?
        - Value Proposition: How clearly does it communicate benefits?
        - Call to Action: How clear and compelling is the CTA?
        - Emotional Appeal: How well does it connect emotionally?
        - Audience Relevance: How well does it target the intended audience?

        **SEO Optimization:**
        - Keyword Integration: How naturally are keywords incorporated?
        - Title Optimization: How SEO-friendly is the title/headline?
        - Content Length: Is the length appropriate for the content type?
        - Readability: How easy is the content to read and scan?

        **Shareability:**
        - Uniqueness: How unique/original is the content?
        - Timeliness: How timely/relevant is the topic?
        - Practical Value: How useful is the information provided?

        **Conversion Potential:**
        - Urgency Creation: Does it create a sense of urgency?
        - Social Proof: Does it include credibility elements?
        - Risk Reduction: Does it address potential objections?
        - Clear Next Steps: Are action steps clearly defined?

        ## Output Format
        Provide scores for each criterion and a brief explanation.
        Then give 3 specific recommendations for improvement.
        """

        try:
            prompt = ChatPromptTemplate.from_template(scoring_prompt)
            chain = prompt | llm | StrOutputParser()
            analysis = chain.invoke({})

            # Parse the analysis into structured scores
            return self._parse_performance_scores(analysis)

        except Exception as e:
            logger.error(f"Content scoring failed: {str(e)}")
            return {"error": str(e)}

    def _parse_performance_scores(self, analysis: str) -> Dict[str, Any]:
        """
        Parse the LLM analysis into structured scores

        Args:
            analysis: Raw LLM analysis text

        Returns:
            Structured scoring data
        """
        # This is a simplified parser - in production, you'd want more robust parsing
        scores = {
            "overall_score": 0,
            "engagement_score": 0,
            "seo_score": 0,
            "shareability_score": 0,
            "conversion_score": 0,
            "recommendations": [],
            "raw_analysis": analysis
        }

        # Extract numeric scores from the analysis (simplified)
        import re
        score_pattern = r'(\d+)/10'
        found_scores = re.findall(score_pattern, analysis)

        if found_scores:
            numeric_scores = [int(score) for score in found_scores[:4]]  # Take first 4 scores
            scores.update({
                "engagement_score": numeric_scores[0] if len(numeric_scores) > 0 else 0,
                "seo_score": numeric_scores[1] if len(numeric_scores) > 1 else 0,
                "shareability_score": numeric_scores[2] if len(numeric_scores) > 2 else 0,
                "conversion_score": numeric_scores[3] if len(numeric_scores) > 3 else 0,
            })
            scores["overall_score"] = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0

        return scores
