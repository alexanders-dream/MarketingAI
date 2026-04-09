from pydantic import BaseModel
from typing import List, Dict, Any

class StrategyGenerateRequest(BaseModel):
    context_id: str
    goal: str
    
class StrategyResponse(BaseModel):
    id: str
    status: str
    title: str
    goal: str
    strategy_data: Dict[str, Any]

class ExecuteStrategyResponse(BaseModel):
    strategy_id: str
    total_tasks: int
    task_ids: List[str]
    calendar_url: str
