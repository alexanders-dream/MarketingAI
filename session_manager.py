"""
Session state management for Marketing AI v3
Replaces database operations with Streamlit's session_state for in-memory storage.
"""
import streamlit as st
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

class SessionManager:
    """Manages application state using Streamlit's session_state."""

    def __init__(self):
        self._initialize_session_storage()

    def _initialize_session_storage(self) -> None:
        """Initialize the session state structure for data storage."""
        if "projects" not in st.session_state:
            st.session_state.projects = {}
        if "content" not in st.session_state:
            st.session_state.content = {}
        if "context_versions" not in st.session_state:
            st.session_state.context_versions = {}
        if "user_preferences" not in st.session_state:
            st.session_state.user_preferences = {}

    def create_project(self, name: str, description: str = "") -> str:
        """Create a new project and return its ID."""
        project_id = str(uuid.uuid4())
        st.session_state.projects[project_id] = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        return project_id

    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects."""
        projects = list(st.session_state.projects.values())
        return sorted(projects, key=lambda p: p["updated_at"], reverse=True)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific project by ID."""
        return st.session_state.projects.get(project_id)

    def update_project(self, project_id: str, name: str = None, description: str = None) -> bool:
        """Update project details."""
        if project_id in st.session_state.projects:
            if name is not None:
                st.session_state.projects[project_id]["name"] = name
            if description is not None:
                st.session_state.projects[project_id]["description"] = description
            st.session_state.projects[project_id]["updated_at"] = datetime.now().isoformat()
            return True
        return False

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its content."""
        if project_id in st.session_state.projects:
            del st.session_state.projects[project_id]
            # Delete associated content
            st.session_state.content = {
                cid: c for cid, c in st.session_state.content.items() if c["project_id"] != project_id
            }
            # Delete associated context versions
            st.session_state.context_versions = {
                vid: v for vid, v in st.session_state.context_versions.items() if v["project_id"] != project_id
            }
            return True
        return False

    def save_content(self, project_id: str, task_type: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Save generated content."""
        content_id = str(uuid.uuid4())
        st.session_state.content[content_id] = {
            "id": content_id,
            "project_id": project_id,
            "task_type": task_type,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        return content_id

    def get_project_content(self, project_id: str, task_type: str = None) -> List[Dict[str, Any]]:
        """Get content for a project, optionally filtered by task type."""
        project_content = [
            c for c in st.session_state.content.values() if c["project_id"] == project_id
        ]
        if task_type:
            project_content = [c for c in project_content if c["task_type"] == task_type]
        
        return sorted(project_content, key=lambda c: c["created_at"], reverse=True)

    def get_user_preference(self, key: str, user_id: str = "default") -> Optional[str]:
        """Get a user preference."""
        return st.session_state.user_preferences.get(user_id, {}).get(key)

    def set_user_preference(self, key: str, value: str, user_id: str = "default") -> None:
        """Set a user preference."""
        if user_id not in st.session_state.user_preferences:
            st.session_state.user_preferences[user_id] = {}
        st.session_state.user_preferences[user_id][key] = value

    def get_content_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent content history across all projects."""
        all_content = list(st.session_state.content.values())
        sorted_content = sorted(all_content, key=lambda c: c["created_at"], reverse=True)
        
        history = []
        for content_item in sorted_content[:limit]:
            project = self.get_project(content_item["project_id"])
            history.append({
                **content_item,
                "project_name": project["name"] if project else "Unknown Project"
            })
        return history

    def save_context_version(self, project_id: str, context: Dict[str, Any], source_type: str = "manual") -> str:
        """Save a version of business context."""
        version_id = str(uuid.uuid4())
        st.session_state.context_versions[version_id] = {
            "version_id": version_id,
            "project_id": project_id,
            "context": context,
            "source_type": source_type,
            "created_at": datetime.now().isoformat()
        }
        return version_id

    def get_all_context_versions(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all context versions for a project."""
        project_versions = [
            v for v in st.session_state.context_versions.values() if v["project_id"] == project_id
        ]
        return sorted(project_versions, key=lambda v: v["created_at"], reverse=True)

    def get_latest_context(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest context version for a project."""
        versions = self.get_all_context_versions(project_id)
        return versions[0]["context"] if versions else None

    def delete_context_versions(self, project_id: str) -> bool:
        """Delete all context versions for a project."""
        initial_count = len(st.session_state.context_versions)
        st.session_state.context_versions = {
            vid: v for vid, v in st.session_state.context_versions.items() if v["project_id"] != project_id
        }
        return len(st.session_state.context_versions) < initial_count

    def get_all_data(self) -> Dict[str, Any]:
        """Get all data from the session for download."""
        return {
            "projects": st.session_state.projects,
            "content": st.session_state.content,
            "context_versions": st.session_state.context_versions,
            "user_preferences": st.session_state.user_preferences
        }
