import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StrategyExecutor:
    """
    Executes a generated marketing strategy by orchestrating tasks as a DAG.
    Uses Kahn's algorithm for topological sorting to ensure dependencies are respected.
    """

    def __init__(self, db_session = None):
        """
        Args:
            db_session: Database session for task retrieval/updates. 
                       In Phase 1, this can be None or a mock while we transition to FastAPI.
        """
        self.db = db_session

    def resolve_execution_order(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Validates task dependencies and returns a safe execution order (DAG sort).
        
        Tasks should be a list of dictionaries, each containing:
        - "id": unique string ID
        - "depends_on": list of string IDs this task depends on
        
        Raises:
            ValueError: If a cyclic dependency is detected or if a dependency is missing.
        """
        task_ids = {t["id"] for t in tasks}
        adj_list = {t_id: [] for t_id in task_ids}
        in_degree = {t_id: 0 for t_id in task_ids}
        
        for t in tasks:
            t_id = t["id"]
            depends_on = t.get("depends_on", [])
            for dep in depends_on:
                if dep not in task_ids:
                    raise ValueError(f"Task {t_id} depends on unknown task {dep}")
                
                adj_list[dep].append(t_id)
                in_degree[t_id] += 1
                
        # Queue initialized with tasks having 0 in-degree
        queue = [t_id for t_id in task_ids if in_degree[t_id] == 0]
        execution_order = []
        
        while queue:
            curr = queue.pop(0)
            execution_order.append(curr)
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(execution_order) != len(tasks):
            raise ValueError("Cyclic dependency detected in strategy tasks.")
            
        return execution_order

    def execute_strategy(self, strategy_id: str, tasks: Optional[List[Dict[str, Any]]] = None):
        """
        Resolves task dependencies and queues the initial, independent tasks.
        Subsequent dependent tasks will be queued recursively by the Celery workers
        as their dependencies complete.
        """
        # In a real db context, we fetch: tasks = self.db.get_tasks(strategy_id)
        if tasks is None and self.db:
            # placeholders for db abstraction
            tasks = []
            
        if not tasks:
            logger.warning(f"No tasks found for strategy {strategy_id}")
            return
            
        try:
            execution_order = self.resolve_execution_order(tasks)
            logger.info(f"Strategy {strategy_id} DAG resolved. Safe execution order: {execution_order}")
        except ValueError as e:
            logger.error(f"Failed to execute strategy {strategy_id}: {str(e)}")
            raise
            
        # Only queue the tasks that have NO dependencies (in_degree == 0).
        # Depending tasks are handled by process observers (e.g. celery chord/callback)
        # or the promote_stalled_tasks beat schedule.
        queued_count = 0
        for task in tasks:
            if not task.get("depends_on"):
                self._queue_task(task["id"])
                queued_count += 1
                
        logger.info(f"Strategy {strategy_id} initiated: {queued_count} root tasks queued.")
        return execution_order

    def _queue_task(self, task_id: str):
        """
        Queues a single task into the Celery worker pool.
        """
        try:
            # When Celery is available:
            # from workers.tasks.agent_tasks import execute_agent_task
            # execute_agent_task.delay(task_id)
            logger.info(f"[Mock Queue] Task {task_id} queued for execution.")
        except Exception as e:
            logger.error(f"Failed to queue task {task_id}: {str(e)}")
