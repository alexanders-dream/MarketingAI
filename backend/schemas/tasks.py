from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class TaskUpdateStatusRequest(BaseModel):
    status: str

class TaskResponse(BaseModel):
    id: str
    strategy_id: str
    title: str
    description: Optional[str]
    task_type: str
    assigned_agent: str
    scheduled_date: Any # date
    status: str
    depends_on: List[str]
    celery_task_id: Optional[str]
