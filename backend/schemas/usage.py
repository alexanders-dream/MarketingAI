from pydantic import BaseModel
from typing import Dict

class UsageStatsResponse(BaseModel):
    api_calls: int = 0
    social_posts: int = 0
    content_generation: int = 0
