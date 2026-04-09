from fastapi import APIRouter, Depends, HTTPException
from backend.db.session import get_db
from backend.db.models import StrategyRepository, ContextRepository, TaskRepository
from backend.schemas.strategy import StrategyGenerateRequest, StrategyResponse, ExecuteStrategyResponse
from backend.routers.auth import get_current_user

router = APIRouter()

@router.post("/generate", response_model=StrategyResponse)
async def generate_strategy(req: StrategyGenerateRequest, user = Depends(get_current_user), conn = Depends(get_db)):
    ctx_repo = ContextRepository(conn)
    ctx = await ctx_repo.get_context(req.context_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Context not found")
        
    # Ideally invoke MarketAnalyzer here
    # from core.analysis.market_analyzer import MarketAnalyzer
    # result = await MarketAnalyzer().generate_strategy(...)
    strategy_data = {"goal": req.goal, "kpis": ["Increase organic traffic", "Grow social engagement"], "channels": ["wordpress", "instagram", "linkedin"]}
    
    repo = StrategyRepository(conn)
    strategy = await repo.create_strategy(
        str(user['id']), req.context_id, f"Strategy for {req.goal}", req.goal, strategy_data
    )
    
    return StrategyResponse(
        id=str(strategy['id']),
        status="generated",
        title=strategy['title'],
        goal=strategy['goal'],
        strategy_data=strategy_data
    )

@router.post("/{strategy_id}/execute", response_model=ExecuteStrategyResponse)
async def execute_strategy(strategy_id: str, user = Depends(get_current_user), conn = Depends(get_db)):
    strat_repo = StrategyRepository(conn)
    ctx_repo = ContextRepository(conn)
    task_repo = TaskRepository(conn)
    
    strategy = await strat_repo.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    ctx = await ctx_repo.get_context(str(strategy['context_id']))
    
    from backend.services.strategy_decomposer import decompose_strategy
    from langchain_core.prompts import ChatPromptTemplate
    from core.config import get_settings
    
    # Mock LLM for decomposition, you can inject real langchain instance
    class MockLLM:
        async def ainvoke(self, prompt):
            import json
            class Res:
                content = json.dumps([
                    {
                        "title": "Initial Blog Post",
                        "description": "Write a blog post about products",
                        "task_type": "blog_post",
                        "target_platform": "wordpress",
                        "target_content_type": "blog_post",
                        "priority": "high",
                        "days_from_start": 1,
                        "dependencies": []
                    }
                ])
            return Res()
    
    tasks_schema = await decompose_strategy(dict(strategy), dict(ctx), MockLLM())
    
    saved_task_ids = []
    # Save tasks
    for task_data in tasks_schema:
        created = await task_repo.create_task(strategy_id, str(user['id']), task_data)
        saved_task_ids.append(str(created['id']))
        
    # Trigger Executor logic via Celery
    from workers.tasks.agent_tasks import execute_due_tasks_batch
    execute_due_tasks_batch.delay() # Trigger immediate check
    
    return ExecuteStrategyResponse(
        strategy_id=strategy_id,
        total_tasks=len(saved_task_ids),
        task_ids=saved_task_ids,
        calendar_url=f"/api/v1/tasks/calendar?strategy_id={strategy_id}"
    )
