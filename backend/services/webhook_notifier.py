import httpx
import json
import logging
import hashlib
import hmac

logger = logging.getLogger(__name__)

async def notify_wordpress(webhook_url: str, webhook_secret: str, event_name: str, task_id: str, payload: dict = None) -> None:
    """POST a task-completion webhook to the originating WordPress site."""
    if not webhook_url or not webhook_secret:
        return

    body = {
        "event": event_name,
        "task_id": task_id,
        "payload": payload or {},
    }
    
    # Sign the payload so WordPress can verify authenticity
    sig = hmac.new(webhook_secret.encode(), json.dumps(body).encode(), hashlib.sha256).hexdigest()

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            await client.post(webhook_url, json=body, headers={"X-MARKETINGAI-SIGNATURE": sig})
            logger.info(f"Webhook delivered for task {task_id} to WP")
        except Exception as e:
            logger.warning(f"Webhook delivery failed for task {task_id}: {e}")
