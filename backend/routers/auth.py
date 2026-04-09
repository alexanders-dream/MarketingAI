from fastapi import APIRouter, Depends, HTTPException, Header, Request
from typing import Optional
from backend.db.session import get_db
from backend.db.models import UserRepository, UsageRepository, SubscriptionRepository
from backend.schemas.auth import UserResponse, UserCreate
from core.security import generate_api_key, hash_api_key

router = APIRouter()

async def get_current_user(x_api_key: str = Header(...), conn = Depends(get_db)):
    repo = UserRepository(conn)
    hashed = hash_api_key(x_api_key)
    user = await repo.get_by_api_key_hash(hashed)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # We must enforce RLS by using the connection we got. 
    # Luckily, `get_db` already sets it if `request.state.user_id` is set, but we need the request to set it or set it here.
    await conn.execute("SET LOCAL app.current_user_id = $1;", str(user['id']))
    return user

@router.get("/verify")
async def verify_auth(user = Depends(get_current_user)):
    return {"status": "ok", "message": "API key valid", "plan": user['plan'], "user_id": str(user['id'])}

@router.post("/register")
async def register(req: UserCreate, conn = Depends(get_db)):
    repo = UserRepository(conn)
    existing = await repo.get_by_email(req.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # In reality, hash req.password using bcrypt. 
    # For now, simplistic
    pwd_hash = hash_api_key(req.password) 
    
    new_api_key = generate_api_key()
    key_hash = hash_api_key(new_api_key)
    
    user = await repo.create_user(req.email, pwd_hash, key_hash)
    
    return {
        "status": "success",
        "api_key": new_api_key,
        "user_id": str(user['id'])
    }

@router.get("/usage/stats")
async def get_usage(user = Depends(get_current_user), conn = Depends(get_db)):
    import datetime
    period = datetime.datetime.utcnow().strftime("%Y-%m")
    repo = UsageRepository(conn)
    stats = await repo.get_usage_stats(str(user['id']), period)
    return stats
