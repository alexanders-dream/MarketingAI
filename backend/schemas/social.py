from pydantic import BaseModel
from typing import List, Optional

class ConnectSocialRequest(BaseModel):
    pass # Currently only requires current user context

class ConnectSocialResponse(BaseModel):
    oauth_url: str
    
class PublishSocialRequest(BaseModel):
    content_id: str
    platforms: List[str]
    image_url: Optional[str] = None
