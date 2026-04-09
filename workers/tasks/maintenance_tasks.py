from celery import shared_task
import logging
import asyncio
import asyncpg
from core.config import get_settings

logger = logging.getLogger(__name__)

async def _do_promote_stalled():
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        # Check tasks that have all dependencies met
        logger.info("Checking for stalled DAG tasks with resolved dependencies")
        # For simplicity, if we have a task with a status that needs checking:
        # In a real system, you query where NOT array_length(depends_on, 1) > 0 OR 
        # NOT EXISTS (select 1 from strategy_tasks st2 where st2.id = any(strategy_tasks.depends_on) and st2.status != 'completed')
        query = """
        UPDATE strategy_tasks
        SET status = 'pending'
        WHERE status = 'waiting_for_deps' AND
        NOT EXISTS (
            SELECT 1 FROM strategy_tasks st2 
            WHERE st2.id = ANY(strategy_tasks.depends_on) AND st2.status != 'completed'
        )
        RETURNING id;
        """
        promoted = await conn.fetch(query)
        for p in promoted:
            from workers.tasks.agent_tasks import execute_agent_task
            execute_agent_task.delay(str(p['id']))
    finally:
        await conn.close()

@shared_task
def promote_stalled_tasks():
    logger.info("Promoting stalled tasks")
    asyncio.run(_do_promote_stalled())
    return {"status": "checked"}

@shared_task
def keepalive_ping():
    logger.info("Oracle Cloud keepalive ping emitted")
    return {"status": "alive"}

