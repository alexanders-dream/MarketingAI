from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class ContextCreateRequest(BaseModel):
    website_url: str
    wp_data: Optional[Dict[str, Any]] = None

class ContextResponse(BaseModel):
    id: str
    status: str
    data: Dict[str, Any]
