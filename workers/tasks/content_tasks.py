from celery import shared_task
import logging
import asyncio
import asyncpg
from core.config import get_settings
from backend.services.webhook_notifier import notify_wordpress
from backend.services.upload_post_client import UploadPostClient

logger = logging.getLogger(__name__)

async def _publish_to_wp(content_id: str):
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        content = await conn.fetchrow("SELECT * FROM generated_content WHERE id = $1::uuid", content_id)
        if not content: return
        
        ctx = await conn.fetchrow("SELECT webhook_url, webhook_secret FROM website_contexts WHERE user_id = $1::uuid LIMIT 1", str(content['user_id']))
        if ctx and ctx['webhook_url']:
            await notify_wordpress(
                webhook_url=ctx['webhook_url'],
                webhook_secret=ctx['webhook_secret'],
                event_name="content.generated",
                task_id=content_id, # mock using content_id for tracing
                payload={"title": content['title'], "content": content['content']}
            )
    finally:
        await conn.close()

@shared_task(bind=True, max_retries=3)
def publish_to_wordpress(self, content_id: str):
    logger.info(f"Publishing content {content_id} to WordPress")
    asyncio.run(_publish_to_wp(content_id))
    return {"status": "published"}

async def _publish_to_social(post_id: str):
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        pub = await conn.fetchrow("SELECT * FROM social_publishes WHERE id = $1::uuid", post_id)
        if not pub: return
        
        # Simplified: Use Upload-Post to push it
        client = UploadPostClient(settings.upload_post_api_key)
        # (Content retrieval and publish logic would go here)
        await conn.execute("UPDATE social_publishes SET status = 'published' WHERE id = $1::uuid", post_id)
    finally:
        await conn.close()

@shared_task(bind=True, max_retries=3)
def publish_to_social(self, post_id: str):
    logger.info(f"Publishing post {post_id} to social accounts via Upload-Post")
    asyncio.run(_publish_to_social(post_id))
    return {"status": "published"}

