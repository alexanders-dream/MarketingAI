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

def _sanitize_for_template(value) -> str:
    """Escape curly braces in user-supplied content to prevent prompt injection."""
    if not isinstance(value, str):
        value = str(value)
    return value.replace("{", "{{").replace("}", "}}")

def generate_fallback_prompt(task: dict, context: dict) -> str:
    """Generate a fallback prompt if no specific template exists."""
    s = _sanitize_for_template
    return f"""
You are an AI assistant for {s(context.get('company_name', ''))}.
Your task is to: {s(task.get('description', task.get('title', '')))}.
Please complete this task according to the best marketing practices.
"""

def generate_dynamic_prompt(task: dict, context: dict, strategy_goals: list) -> str:
    """Generate a unique, context-rich prompt for a specific task."""
    agent = task.get("assigned_agent", "content_agent")
    task_type = task.get("task_type", "blog_post")

    template = AGENT_PROMPT_TEMPLATES.get(agent, {}).get(task_type, "")
    if not template:
        return generate_fallback_prompt(task, context)

    s = _sanitize_for_template
    
    # Extract comma separated lists
    keywords = context.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    competitors = context.get("competitors", [])
    if not isinstance(competitors, list):
        competitors = []

    return template.format(
        company_name=s(context.get("company_name", "")),
        industry=s(context.get("industry", "")),
        brand_voice=s(context.get("brand_voice", "professional")),
        target_audience=s(context.get("target_audience", "")),
        strategy_goal=s(", ".join(strategy_goals) if strategy_goals else ""),
        topic=s(task.get("title", "")),
        word_count=task.get("word_count", 1500),
        keywords=s(", ".join(keywords)),
        tone=s(task.get("priority", "medium")), # fallback to priority if tone isn't supplied
        platform=s(task.get("target_platform", "")),
        content_type=s(task.get("target_content_type", "")),
        platform_requirements=s(PLATFORM_REQUIREMENTS.get(task.get("target_platform", ""), "")),
        website_url=s(context.get("website_url", "")),
        competitors=s(", ".join(competitors)),
    )
