import json
import logging
from typing import Dict, Any, List
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    pass
    
logger = logging.getLogger(__name__)

async def decompose_strategy(strategy: Dict[str, Any], context: Dict[str, Any], llm) -> List[Dict[str, Any]]:
    """Use LLM to break strategy into atomic, schedulable tasks"""
    from backend.services.prompt_templates import generate_dynamic_prompt
    import uuid
    import datetime

    # Simple fallback parsing without needing langchain core parser if desired, but we try to use LLM.
    decompose_prompt = """
You are a marketing operations manager. Decompose this marketing strategy
into specific, executable tasks that AI agents can perform autonomously.

BUSINESS CONTEXT:
- Company: {company_name}
- Industry: {industry}
- Target Audience: {target_audience}
- Brand Voice: {brand_voice}

STRATEGY:
{strategy_content}

TIMEFRAME: {timeframe}

For each task, provide:
1. title: Short task name
2. description: What exactly needs to be done
3. task_type: One of [blog_post, social_post, seo_audit, research, email, competitor, analytics]
4. target_platform: Target platform (wordpress, instagram, linkedin, tiktok, facebook, x, email)
5. target_content_type: Specific format (blog_post, carousel, reel, story, thread, newsletter)
6. priority: high/medium/low
7. days_from_start: integer (e.g. 1 means tomorrow)
8. dependencies: List of task TITLES that must complete first (optional)

Output strictly as a valid JSON array of objects. Do NOT wrap in markdown code blocks.
"""

    prompt = decompose_prompt.format(
        company_name=context.get("company_name", ""),
        industry=context.get("industry", ""),
        target_audience=context.get("target_audience", ""),
        brand_voice=context.get("brand_voice", "professional"),
        strategy_content=json.dumps(strategy.get("strategy_data", {})),
        timeframe=strategy.get("timeframe", "90_days")
    )

    try:
        if callable(getattr(llm, 'ainvoke', None)):
            result = await llm.ainvoke(prompt)
            content = result.content
        else:
            result = llm.invoke(prompt)
            content = result.content
            
        # Cleanup
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        parsed_tasks = json.loads(content.strip())
    except Exception as e:
        logger.error(f"Failed to decompose strategy: {e}")
        parsed_tasks = [
            {
                "title": "Initial Marketing Analysis",
                "description": "Fallback task generated due to parsing error",
                "task_type": "research",
                "days_from_start": 0
            }
        ]
        
    AGENT_MAPPING = {
        "blog_post":     "content",
        "social_post":   "social",
        "seo_audit":     "seo",
        "research":      "research",
        "email":         "content",
        "competitor":    "research",
        "analytics":     "research",
        "design":        "content",
    }
    
    # Second pass: Process dependencies and generate IDs, dates, dynamic prompts
    task_map = {}
    final_tasks = []
    
    now = datetime.datetime.utcnow().date()
    
    for pt in parsed_tasks:
        uuid_val = str(uuid.uuid4())
        pt["id"] = uuid_val
        task_map[pt.get("title")] = uuid_val
        
        days = pt.get("days_from_start", 0)
        pt["scheduled_date"] = now + datetime.timedelta(days=days)
        pt["assigned_agent"] = AGENT_MAPPING.get(pt.get("task_type"), "content")

    for pt in parsed_tasks:
        # Resolve dependencies to IDs
        deps = pt.get("dependencies", [])
        dep_ids = []
        for d in deps:
            if d in task_map:
                dep_ids.append(task_map[d])
        pt["depends_on"] = dep_ids
        
        # Generate dynamic prompt
        pt["generated_prompt"] = generate_dynamic_prompt(pt, context, strategy.get("kpis", []))
        pt["prompt_context"] = {
            "company_name": context.get("company_name"),
            "industry": context.get("industry"),
            "brand_voice": context.get("brand_voice"),
            "target_audience": context.get("target_audience"),
            "strategy_goals": strategy.get("kpis", []),
        }

        final_tasks.append(pt)
        
    return final_tasks
