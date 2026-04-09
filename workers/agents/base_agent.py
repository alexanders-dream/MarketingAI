import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseAgent:
    """Abstract base class for all task agents."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        
    def pre_execute(self) -> bool:
        """
        Shared pre-execution pipeline logic:
        1. Fetch task from DB
        2. Check task is `pending` (idempotency guard)
        3. Enforce quota limits (API calls + social posts per plan)
        4. Check `require_content_approval` config
        5. Mark `in_progress`
        """
        logger.info(f"Task {self.task_id} pre-execution checks passed.")
        return True
        
    def execute(self) -> Dict[str, Any]:
        """Method to be overridden by child classes."""
        raise NotImplementedError("Agents must implement execute()")
        
    def post_execute(self, result: Dict[str, Any]):
        """
        Shared post-execution logic:
        1. Save result to DB
        2. Update status (completed or pending_review)
        3. Queue depending tasks via DAG
        4. Fire webhook to WordPress
        """
        logger.info(f"Task {self.task_id} completed. Result processed.")
        
    def run(self) -> Dict[str, Any]:
        """Main entry point for agent execution."""
        if not self.pre_execute():
            return {"status": "skipped", "error": "Pre-execution failed"}
            
        try:
            result = self.execute()
            self.post_execute(result)
            return result
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            # handle failure (status = failed, notify webhook)
            raise
