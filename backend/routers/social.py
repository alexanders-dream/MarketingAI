from fastapi import APIRouter, Depends, HTTPException
from backend.db.session import get_db
from backend.routers.auth import get_current_user
from backend.schemas.social import ConnectSocialResponse, PublishSocialRequest
from backend.services.upload_post_client import UploadPostClient
from core.config import get_settings

router = APIRouter()

def get_upload_post_client():
    settings = get_settings()
    return UploadPostClient(settings.upload_post_api_key)

@router.post("/accounts/connect", response_model=ConnectSocialResponse)
async def connect_social_accounts(user = Depends(get_current_user)):
    client = get_upload_post_client()
    try:
        up_user = await client.create_user(str(user['id']), user['email'])
        jwt = await client.get_jwt(str(user['id']))
        return ConnectSocialResponse(oauth_url=f"https://upload-post.com/connect?token={jwt}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/accounts")
async def list_social_accounts(user = Depends(get_current_user)):
    # Fetch from Upload-Post via some endpoint, or keep a local copy
    return {"accounts": []}

@router.post("/publish")
async def publish_social(req: PublishSocialRequest, user = Depends(get_current_user)):
    client = get_upload_post_client()
    # Usually this happens async in a celery worker. For endpoint parity:
    from workers.tasks.content_tasks import publish_to_social
    publish_to_social.delay(req.content_id)
    return {"status": "ok", "message": "Queued for social publishing"}

@router.get("/analytics/{profile_id}")
async def get_social_analytics(profile_id: str, user = Depends(get_current_user)):
    client = get_upload_post_client()
    try:
        analytics = await client.get_analytics(profile_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
