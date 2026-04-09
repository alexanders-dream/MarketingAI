from celery import shared_task
import logging
import asyncio
import asyncpg
from workers.agents.agents import ContentAgent, SocialAgent, SEOAgent, ResearchAgent
from backend.db.models import TaskRepository, ContextRepository, StrategyRepository
from backend.services.webhook_notifier import notify_wordpress
from core.config import get_settings
from langchain_community.chat_models import ChatOpenAI # Example

logger = logging.getLogger(__name__)

AGENT_MAPPING = {
    "content": ContentAgent,
    "social": SocialAgent,
    "seo": SEOAgent,
    "research": ResearchAgent
}

async def _do_execute_agent(task_id: str):
    settings = get_settings()
    # Ad-hoc connection for safety in Celery
    conn = await asyncpg.connect(settings.database_url)
    try:
        task_repo = TaskRepository(conn)
        task = await task_repo.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
            
        await conn.execute("SET LOCAL app.current_user_id = $1;", str(task['user_id']))
        await task_repo.update_task(task_id, status='in_progress', started_at=asyncio.get_event_loop().time())
        
        agent_type = task['assigned_agent']
        agent_class = AGENT_MAPPING.get(agent_type.lower(), ContentAgent)
        
        # Instantiate agent
        llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini", api_key=settings.openai_api_key)
        agent = agent_class(task_id=task_id, llm=llm, conn=conn)
        
        result = await agent.run(dict(task))
        
        # Mark completed
        import datetime
        await task_repo.update_task(task_id, status='completed', execution_log=str(result), completed_at=datetime.datetime.utcnow())
        
        # Notify WP
        ctx_repo = ContextRepository(conn)
        strat_repo = StrategyRepository(conn)
        strat = await strat_repo.get_strategy(str(task['strategy_id']))
        ctx = await ctx_repo.get_context(str(strat['context_id']))
        
        if ctx.get('webhook_url') and ctx.get('webhook_secret'):
            await notify_wordpress(
                webhook_url=ctx['webhook_url'],
                webhook_secret=ctx['webhook_secret'],
                event_name="task.completed",
                task_id=task_id,
                payload={"status": "completed"}
            )
            
        return result
    finally:
        await conn.close()

@shared_task(bind=True, max_retries=3)
def execute_agent_task(self, task_id: str):
    logger.info(f"Starting agent task {task_id}")
    try:
        result = asyncio.run(_do_execute_agent(task_id))
        return {"task_id": task_id, "status": "success", "result": result}
    except Exception as exc:
        logger.error(f"Agent task {task_id} failed: {exc}")
        self.retry(exc=exc, countdown=60)

async def _do_batch_due_tasks():
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        import datetime
        cutoff = datetime.datetime.utcnow().date()
        repo = TaskRepository(conn)
        # Bypassing RLS for system-level polling worker, so we use superuser or we don't set it 
        # (Assuming the DB connection string has read access without RLS or we bypass)
        # A simpler way in models to not strictly require user_id for fetch is done.
        due = await conn.fetch("SELECT id FROM strategy_tasks WHERE status = 'pending' AND scheduled_date <= $1", cutoff)
        for t in due:
            execute_agent_task.delay(str(t['id']))
    finally:
        await conn.close()

@shared_task
def execute_due_tasks_batch():
    logger.info("Executing periodic due tasks batch")
    asyncio.run(_do_batch_due_tasks())

