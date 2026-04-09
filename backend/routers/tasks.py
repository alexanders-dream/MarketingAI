from fastapi import APIRouter, Depends, HTTPException
from backend.db.session import get_db
from backend.db.models import TaskRepository
from backend.routers.auth import get_current_user
from backend.schemas.tasks import TaskUpdateStatusRequest
from typing import List

router = APIRouter()

@router.get("/")
async def get_tasks(strategy_id: str = None, user = Depends(get_current_user), conn = Depends(get_db)):
    repo = TaskRepository(conn)
    if strategy_id:
        tasks = await repo.get_tasks_for_strategy(strategy_id)
    else:
        tasks = await repo.get_all_tasks()
        
    return [dict(t) for t in tasks]

@router.get("/calendar")
async def get_calendar(month: str = None, year: str = None, user = Depends(get_current_user), conn = Depends(get_db)):
    repo = TaskRepository(conn)
    tasks = await repo.get_all_tasks()
    
    # Very basic grouping
    from collections import defaultdict
    calendar_view = defaultdict(list)
    for t in tasks:
        dt_str = t['scheduled_date'].strftime("%Y-%m-%d") if t['scheduled_date'] else "unscheduled"
        calendar_view[dt_str].append(dict(t))
        
    return calendar_view

@router.patch("/{task_id}")
async def update_task(task_id: str, req: TaskUpdateStatusRequest, user = Depends(get_current_user), conn = Depends(get_db)):
    repo = TaskRepository(conn)
    updated = await repo.update_task(task_id, status=req.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found")
    return dict(updated)

@router.post("/execute-due")
async def execute_due(user = Depends(get_current_user)):
    # Triggered by WP cron
    from workers.tasks.agent_tasks import execute_due_tasks_batch
    execute_due_tasks_batch.delay()
    return {"status": "triggered"}
