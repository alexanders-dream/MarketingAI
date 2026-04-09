from fastapi import APIRouter, Depends
from backend.db.session import get_db
from backend.routers.auth import get_current_user
from backend.schemas.content import ContentGenerateRequest, ContentScheduleRequest

router = APIRouter()

@router.post("/generate")
async def generate_content(req: ContentGenerateRequest, user = Depends(get_current_user), conn = Depends(get_db)):
    prompt = "Generate content"
    if req.update_type == "refresh":
        prompt = f"Refresh following content:\n{req.content}"
        
    # Call to GenAI omitted for simplicty, return placeholder update
    return {"status": "ok", "content": f"{req.content}\n\n[Refreshed by MarketingAI]"}

@router.post("/schedule")
async def schedule_content(req: ContentScheduleRequest, user = Depends(get_current_user), conn = Depends(get_db)):
    from workers.tasks.content_tasks import publish_to_wordpress
    # Delay logic
    publish_to_wordpress.delay(req.content_id)
    return {"status": "ok", "message": "Scheduled"}
