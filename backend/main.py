from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from backend.db.session import init_db_pool, close_db_pool
from backend.routers import auth, context, strategy, tasks, content, social, billing

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI - MarketingAI...")
    await init_db_pool()
    yield
    logger.info("Shutting down FastAPI - MarketingAI...")
    await close_db_pool()

app = FastAPI(
    title="MarketingAI API",
    version="1.0.0",
    description="Backend API for MarketingAI WordPress plugin integration",
    lifespan=lifespan
)

# Configure CORS for WordPress clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "https://marketingai.com", 
        "https://upload-post.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(context.router, prefix="/api/v1/context", tags=["context"])
app.include_router(strategy.router, prefix="/api/v1/strategy", tags=["strategy"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(content.router, prefix="/api/v1/content", tags=["content"])
app.include_router(social.router, prefix="/api/v1/social", tags=["social"])
app.include_router(billing.router, prefix="/api/v1/billing", tags=["billing"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
