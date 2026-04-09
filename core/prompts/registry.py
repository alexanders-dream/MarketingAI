AGENT_PROMPT_TEMPLATES = {
    "content_agent": {
        "blog_post": """
You are a content marketing specialist for {company_name}, a {industry} company.

BRAND VOICE: {brand_voice}
TARGET AUDIENCE: {target_audience}
STRATEGY GOAL: {strategy_goal}

Write a comprehensive blog post about: {topic}

Requirements:
- Length: {word_count} words
- Include SEO keywords: {keywords}
- Tone: {tone}
- Include a compelling title, meta description, and CTA
- Format in Markdown with proper headings (H2, H3)
- Include 2-3 internal linking suggestions
""",
        "social_post": """
You are a social media specialist for {company_name}.

PLATFORM: {platform}
BRAND VOICE: {brand_voice}
TARGET AUDIENCE: {target_audience}
CONTENT TYPE: {content_type}

Create a {content_type} post about: {topic}

Platform-specific requirements:
{platform_requirements}

Include: caption, hashtags (5-10), CTA, and posting time suggestion.
""",
    },
    
    "seo_agent": {
        "seo_audit": """
Analyze the SEO performance for {company_name} ({website_url}).

Focus areas:
- Keyword opportunities in {industry}
- Content gaps vs competitors: {competitors}
- Technical SEO recommendations
- Backlink strategy suggestions

Provide actionable recommendations ranked by impact.
""",
    },
    
    "research_agent": {
        "competitor": """
Research the latest activities of these competitors in {industry}:
{competitors}

Analyze:
- Recent content published (last 30 days)
- Social media activity and engagement
- New product/service announcements
- Marketing messaging changes

Provide insights {company_name} can act on.
""",
    },
}

PLATFORM_REQUIREMENTS = {
    "instagram": "Max 2200 chars caption. Visual-first. Use line breaks. Carousel = 10 slides max.",
    "linkedin": "Professional tone. 1300 char sweet spot. Use bullet points. Tag relevant people/companies.",
    "tiktok": "Hook in first 3 seconds. Trending audio reference. 150 char caption max. Hashtags critical.",
    "x": "280 char limit. Thread if needed. Conversational. Quote-tweet worthy.",
    "facebook": "Longer posts OK. Community-focused. Ask questions for engagement.",
    "threads": "Conversational. 500 char limit. Chain for longer content.",
}

SOCIAL_POST_WITH_PRODUCT_PROMPT = """
You are creating a {content_type} for {platform} for {company_name}.

PRODUCT TO FEATURE:
- Name: {product_name}
- Price: {product_price}
- Description: {product_description}
- Categories: {product_categories}

AVAILABLE IMAGES (select the best one for this post):
{image_list}

Write a compelling {content_type} that showcases this product.
Include: caption, hashtags, CTA.
Specify which image index to use: image_index: <number>

The selected image URL will be passed directly to Upload-Post for publishing.
"""
