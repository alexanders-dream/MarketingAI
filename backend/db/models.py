import json
from typing import Dict, Any, List, Optional
import asyncpg
import logging

logger = logging.getLogger(__name__)

class BaseRepository:
    """Base repository with shared asyncpg connection"""
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

class UserRepository(BaseRepository):
    async def get_by_email(self, email: str) -> Optional[asyncpg.Record]:
        return await self.conn.fetchrow(
            "SELECT * FROM users WHERE email = $1", email
        )

    async def get_by_api_key_hash(self, api_key_hash: str) -> Optional[asyncpg.Record]:
        return await self.conn.fetchrow(
            "SELECT * FROM users WHERE api_key_hash = $1", api_key_hash
        )

    async def create_user(self, email: str, password_hash: str, api_key_hash: str) -> asyncpg.Record:
        return await self.conn.fetchrow(
            """
            INSERT INTO users (email, password_hash, api_key_hash)
            VALUES ($1, $2, $3)
            RETURNING *
            """,
            email, password_hash, api_key_hash
        )

class ContextRepository(BaseRepository):
    async def get_context(self, context_id: str) -> Optional[asyncpg.Record]:
        return await self.conn.fetchrow(
            "SELECT * FROM website_contexts WHERE id = $1::uuid", context_id
        )

    async def create_context(self, user_id: str, website_url: str, data: Dict[str, Any]) -> asyncpg.Record:
        # Note: RLS makes passing user_id explicitly sometimes redundant if it's set in the session, 
        # but good for insertion explicitly.
        return await self.conn.fetchrow(
            """
            INSERT INTO website_contexts (user_id, website_url, company_name, industry, context_data)
            VALUES ($1::uuid, $2, $3, $4, $5)
            RETURNING *
            """,
            user_id, website_url, data.get("company_name"), data.get("industry"), json.dumps(data)
        )
        
    async def get_webhook_url(self, user_id: str) -> Optional[str]:
        # Usually from contexts for a user
        row = await self.conn.fetchrow(
            "SELECT webhook_url FROM website_contexts WHERE user_id = $1::uuid LIMIT 1", user_id
        )
        return row['webhook_url'] if row else None

    async def get_webhook_secret(self, user_id: str) -> Optional[str]:
        row = await self.conn.fetchrow(
            "SELECT webhook_secret FROM website_contexts WHERE user_id = $1::uuid LIMIT 1", user_id
        )
        return row['webhook_secret'] if row else None

class StrategyRepository(BaseRepository):
    async def create_strategy(self, user_id: str, context_id: str, title: str, goal: str, strategy_data: Dict[str, Any]) -> asyncpg.Record:
        return await self.conn.fetchrow(
            """
            INSERT INTO strategies (user_id, context_id, title, goal, strategy_data)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5)
            RETURNING *
            """,
            user_id, context_id, title, goal, json.dumps(strategy_data)
        )
        
    async def get_strategy(self, strategy_id: str) -> Optional[asyncpg.Record]:
        return await self.conn.fetchrow("SELECT * FROM strategies WHERE id = $1::uuid", strategy_id)

class TaskRepository(BaseRepository):
    async def create_task(self, strategy_id: str, user_id: str, task_data: Dict[str, Any]) -> asyncpg.Record:
        depends_on = task_data.get("depends_on", [])
        return await self.conn.fetchrow(
            """
            INSERT INTO strategy_tasks 
            (strategy_id, user_id, title, description, task_type,
             assigned_agent, scheduled_date, target_platform,
             target_content_type, generated_prompt, prompt_context, priority, depends_on)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
            """,
            strategy_id, user_id, task_data["title"], task_data.get("description"),
            task_data["task_type"], task_data["assigned_agent"],
            task_data["scheduled_date"], task_data.get("target_platform"),
            task_data.get("target_content_type"), task_data.get("generated_prompt"),
            json.dumps(task_data.get("prompt_context", {})), task_data.get("priority", "medium"),
            depends_on
        )
        
    async def get_task(self, task_id: str) -> Optional[asyncpg.Record]:
        return await self.conn.fetchrow("SELECT * FROM strategy_tasks WHERE id = $1::uuid", task_id)
        
    async def get_tasks_for_strategy(self, strategy_id: str) -> List[asyncpg.Record]:
        return await self.conn.fetch("SELECT * FROM strategy_tasks WHERE strategy_id = $1::uuid", strategy_id)

    async def get_all_tasks(self) -> List[asyncpg.Record]:
        return await self.conn.fetch("SELECT * FROM strategy_tasks ORDER BY scheduled_date ASC")

    async def update_task(self, task_id: str, **kwargs) -> Optional[asyncpg.Record]:
        if not kwargs:
            return None
            
        set_clauses = []
        values = []
        idx = 1
        
        # Start at 2 since task_id is 1
        for k, v in kwargs.items():
            set_clauses.append(f"{k} = ${idx + 1}")
            values.append(v)
            idx += 1
            
        query = f"UPDATE strategy_tasks SET {', '.join(set_clauses)} WHERE id = $1::uuid RETURNING *"
        return await self.conn.fetchrow(query, task_id, *values)

    async def get_due_tasks(self, cutoff: str) -> List[asyncpg.Record]:
        # cutoff format 'YYYY-MM-DD'
        return await self.conn.fetch(
            "SELECT * FROM strategy_tasks WHERE status = 'pending' AND scheduled_date <= $1",
            cutoff
        )

class UsageRepository(BaseRepository):
    async def record_usage(self, user_id: str, event_type: str, count: int = 1):
        import datetime
        period = datetime.datetime.utcnow().strftime("%Y-%m")
        await self.conn.execute(
            """
            INSERT INTO usage_events (user_id, event_type, event_count, period_month)
            VALUES ($1::uuid, $2, $3, $4)
            """,
            user_id, event_type, count, period
        )
        
    async def get_usage_stats(self, user_id: str, period: str) -> Dict[str, int]:
        rows = await self.conn.fetch(
            """
            SELECT event_type, SUM(event_count) as total
            FROM usage_events
            WHERE user_id = $1::uuid AND period_month = $2
            GROUP BY event_type
            """,
            user_id, period
        )
        return {row['event_type']: row['total'] for row in rows}

class SubscriptionRepository(BaseRepository):
    async def create_or_update(self, user_id: str, data: Dict[str, Any]) -> asyncpg.Record:
        return await self.conn.fetchrow(
            """
            INSERT INTO subscriptions 
            (user_id, payment_provider, provider_subscription_id, pesapal_order_tracking_id, status, current_period_start, current_period_end)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            user_id,
            data.get('payment_provider', 'stripe'),
            data.get('provider_subscription_id'),
            data.get('pesapal_order_tracking_id'),
            data.get('status', 'active'),
            data.get('current_period_start'),
            data.get('current_period_end')
        )
