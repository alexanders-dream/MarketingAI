from celery.signals import worker_process_init, worker_process_shutdown
import logging
from core.config import get_settings
# import psycopg2 # Using sync DB driver for celery workers since celery is synchronous

logger = logging.getLogger(__name__)

# Global sync connection pool for the worker
db_pool = None

@worker_process_init.connect
def init_worker_db_connection(**kwargs):
    """
    Initialize a synchronous PostgreSQL connection pool for the Celery worker process.
    This runs once per worker child process.
    """
    global db_pool
    # settings = get_settings()
    # url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    # db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, url)
    logger.info("Worker process initialized DB connection pool")

@worker_process_shutdown.connect
def shutdown_worker_db_connection(**kwargs):
    """Cleanup DB pool on worker shutdown."""
    global db_pool
    if db_pool:
        # db_pool.closeall()
        logger.info("Worker process closed DB connection pool")
