import json
from .registry import AGENT_PROMPT_TEMPLATES, PLATFORM_REQUIREMENTS, SOCIAL_POST_WITH_PRODUCT_PROMPT
from .sanitizer import _sanitize_for_template

def generate_fallback_prompt(task: dict, context: dict) -> str:
    """Generate a generic prompt if no specific template exists."""
    s = _sanitize_for_template
    return f"""
You are an AI assistant executing a marketing task.
TASK TITLE: {s(task.get("title", "Marketing Task"))}
TASK TYPE: {s(task.get("task_type", "General"))}
COMPANY: {s(context.get("company_name", ""))}
INDUSTRY: {s(context.get("industry", ""))}

Please execute the task requirements.
"""

def generate_dynamic_prompt(task: dict, context: dict, strategy: dict) -> str:
    """Generate a unique, context-rich prompt for a specific task.

    Security: all values derived from the user's WordPress site are sanitized
    via _sanitize_for_template() before str.format() interpolation to prevent
    both KeyError crashes and adversarial prompt-injection attacks.
    """
    agent = task.get("assigned_agent")
    task_type = task.get("task_type")

    template = AGENT_PROMPT_TEMPLATES.get(agent, {}).get(task_type, "")
    
    # Check if this is a product-specific social post
    if task_type == "social_post" and task.get("product_context"):
        template = SOCIAL_POST_WITH_PRODUCT_PROMPT

    if not template:
        # Fallback: generate prompt for unknown task types
        return generate_fallback_prompt(task, context)

    s = _sanitize_for_template

    kwargs = {
        "company_name": s(context.get("company_name", "")),
        "industry": s(context.get("industry", "")),
        "brand_voice": s(context.get("brand_voice", "professional")),
        "target_audience": s(context.get("target_audience", "")),
        "strategy_goal": s(strategy.get("primary_kpi", "")),
        "topic": s(task.get("title", "")),
        "word_count": task.get("word_count", 1500),
        "keywords": s(", ".join(context.get("keywords", []))),
        "tone": s(task.get("tone", "professional")),
        "platform": s(task.get("target_platform", "")),
        "content_type": s(task.get("target_content_type", "")),
        "platform_requirements": s(PLATFORM_REQUIREMENTS.get(task.get("target_platform", ""), "")),
        "website_url": s(context.get("website_url", "")),
        "competitors": s(", ".join(context.get("competitors", []))),
    }

    # Add product context if applicable
    if task.get("product_context"):
        product = task["product_context"]
        kwargs.update({
            "product_name": s(product.get("name", "")),
            "product_price": s(product.get("price", "")),
            "product_description": s(product.get("description", "")),
            "product_categories": s(", ".join(product.get("categories", []))),
            "image_list": s("\n".join([f"[{i}]: {url}" for i, url in enumerate(product.get("image_urls", []))]))
        })

    return template.format(**kwargs)
