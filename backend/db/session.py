import logging
import asyncpg
from typing import Optional, AsyncGenerator
from fastapi import Request, HTTPException, status
from core.config import get_settings

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

async def init_db_pool():
    """Initialize the asyncpg connection pool."""
    global _pool
    settings = get_settings()
    try:
        # Strip sqlalchemy prefix if present
        url = settings.database_url
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://")
        
        _pool = await asyncpg.create_pool(url, min_size=5, max_size=20)
        logger.info("Database connection pool initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        # In a real setup, we might raise or allow retry
        
async def close_db_pool():
    """Close the asyncpg connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Database connection pool closed.")

async def get_db(request: Request) -> AsyncGenerator[asyncpg.Connection, None]:
    """Dependency to get a database connection from the pool."""
    if _pool is None:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
        
    async with _pool.acquire() as conn:
        # If the request auth middleware set a user_id, 
        # we configure RLS for this connection connection
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            await conn.execute("SET LOCAL app.current_user_id = $1;", str(user_id))
            
        yield conn
