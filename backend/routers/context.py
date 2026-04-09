from fastapi import APIRouter, Depends
from backend.db.session import get_db
from backend.db.models import ContextRepository
from backend.schemas.context import ContextCreateRequest
from backend.routers.auth import get_current_user
import uuid

router = APIRouter()

@router.post("/extract")
async def extract_context(req: ContextCreateRequest, user = Depends(get_current_user), conn = Depends(get_db)):
    repo = ContextRepository(conn)
    
    # In a full flow, we might run Firecrawl here or run an agent async.
    # For now, we save WP data 
    wp_data = req.wp_data or {}
    
    context = await repo.create_context(str(user['id']), req.website_url, wp_data)
    
    return {"status": "ok", "context_id": str(context['id'])}

@router.get("/{context_id}")
async def get_context(context_id: str, user = Depends(get_current_user), conn = Depends(get_db)):
    repo = ContextRepository(conn)
    ctx = await repo.get_context(context_id)
    if not ctx:
        return {"status": "error", "message": "not found"}
    # ctx belongs to user if RLS allows reading it
    # We must convert it to dict since it's an asyncpg.Record
    ctx_dict = dict(ctx)
    return {"status": "ok", "data": ctx_dict}
