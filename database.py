"""
Database operations for Marketing AI v3
"""
import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from config import DatabaseConfig, AppConfig


class DatabaseManager:
    """SQLite database manager for persistent storage"""

    def __init__(self, db_path: str = AppConfig.DATABASE_PATH):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(DatabaseConfig.PROJECTS_SCHEMA)
            cursor.execute(DatabaseConfig.CONTENT_SCHEMA)
            cursor.execute(DatabaseConfig.USER_PREFERENCES_SCHEMA)
            cursor.execute(DatabaseConfig.CONTEXT_VERSIONS_SCHEMA)

            # Add indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_project_id ON generated_content (project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_project_id ON context_versions (project_id)")
            
            conn.commit()

    def create_project(self, name: str, description: str = "") -> int:
        """Create a new project and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                (name, description)
            )
            project_id = cursor.lastrowid
            conn.commit()
            return project_id

    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM projects
                ORDER BY updated_at DESC
            """)
            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3],
                    "updated_at": row[4]
                }
                for row in rows
            ]

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific project by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM projects
                WHERE id = ?
            """, (project_id,))
            row = cursor.fetchone()

            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "created_at": row[3],
                    "updated_at": row[4]
                }
            return None

    def update_project(self, project_id: int, name: str = None, description: str = None) -> bool:
        """Update project details"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                params.append(project_id)
                query = f"UPDATE projects SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount > 0
            return False

    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all its content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete associated content first
            cursor.execute("DELETE FROM generated_content WHERE project_id = ?", (project_id,))
            # Delete project
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            return cursor.rowcount > 0

    def save_content(self, project_id: int, task_type: str, content: str, metadata: Dict[str, Any] = None) -> int:
        """Save generated content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute(
                "INSERT INTO generated_content (project_id, task_type, content, metadata) VALUES (?, ?, ?, ?)",
                (project_id, task_type, content, metadata_json)
            )
            content_id = cursor.lastrowid
            conn.commit()
            return content_id

    def get_project_content(self, project_id: int, task_type: str = None) -> List[Dict[str, Any]]:
        """Get content for a project, optionally filtered by task type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if task_type:
                cursor.execute("""
                    SELECT id, task_type, content, metadata, created_at
                    FROM generated_content
                    WHERE project_id = ? AND task_type = ?
                    ORDER BY created_at DESC
                """, (project_id, task_type))
            else:
                cursor.execute("""
                    SELECT id, task_type, content, metadata, created_at
                    FROM generated_content
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                """, (project_id,))

            rows = cursor.fetchall()
            results = []
            for row in rows:
                metadata = None
                if row[3]:  # metadata field
                    try:
                        metadata = json.loads(row[3])
                    except (json.JSONDecodeError, TypeError):
                        # Handle invalid JSON gracefully
                        metadata = {"error": "Invalid metadata format", "raw_data": row[3]}
                
                results.append({
                    "id": row[0],
                    "task_type": row[1],
                    "content": row[2],
                    "metadata": metadata,
                    "created_at": row[4]
                })
            
            return results

    def get_user_preference(self, key: str, user_id: str = "default") -> Optional[str]:
        """Get a user preference"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM user_preferences WHERE user_id = ? AND key = ?",
                (user_id, key)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_user_preference(self, key: str, value: str, user_id: str = "default") -> None:
        """Set a user preference"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_preferences (user_id, key, value)
                VALUES (?, ?, ?)
            """, (user_id, key, value))
            conn.commit()

    def get_content_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent content history across all projects"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.task_type, c.content, c.metadata, c.created_at,
                       p.name as project_name
                FROM generated_content c
                LEFT JOIN projects p ON c.project_id = p.id
                ORDER BY c.created_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            results = []
            for row in rows:
                metadata = None
                if row[3]:  # metadata field
                    try:
                        metadata = json.loads(row[3])
                    except (json.JSONDecodeError, TypeError):
                        # Handle invalid JSON gracefully
                        metadata = {"error": "Invalid metadata format", "raw_data": row[3]}
                
                results.append({
                    "id": row[0],
                    "task_type": row[1],
                    "content": row[2],
                    "metadata": metadata,
                    "created_at": row[4],
                    "project_name": row[5]
                })
            
            return results

    # Context Versioning Methods
    def save_context_version(self, project_id: int, context: Dict[str, Any], source_type: str = "manual") -> int:
        """Save a version of business context"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            context_json = json.dumps(context, default=str)
            cursor.execute(
                "INSERT INTO context_versions (project_id, context_json, source_type) VALUES (?, ?, ?)",
                (project_id, context_json, source_type)
            )
            version_id = cursor.lastrowid
            conn.commit()
            return version_id

    def get_all_context_versions(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all context versions for a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, context_json, source_type, created_at
                FROM context_versions
                WHERE project_id = ?
                ORDER BY created_at DESC
            """, (project_id,))
            rows = cursor.fetchall()
            
            return [
                {
                    "version_id": row[0],
                    "context": json.loads(row[1]) if row[1] else {},
                    "source_type": row[2],
                    "created_at": row[3]
                }
                for row in rows
            ]

    def get_latest_context(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest context version for a project"""
        versions = self.get_all_context_versions(project_id)
        return versions[0]["context"] if versions else None

    def delete_context_versions(self, project_id: int) -> bool:
        """Delete all context versions for a project"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM context_versions WHERE project_id = ?", (project_id,))
            conn.commit()
            return cursor.rowcount > 0
