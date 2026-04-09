from pydantic import BaseModel
from typing import Optional, Dict, Any

class ContentGenerateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    update_type: Optional[str] = None
    post_id: Optional[int] = None
    platform: Optional[str] = None
    
class ContentScheduleRequest(BaseModel):
    content_id: str
    scheduled_date: str
    platform: str
