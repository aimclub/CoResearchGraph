import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .base import BaseStateManager


class SQLiteStateManager(BaseStateManager):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS article_steps (
                article_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (article_id, step_name)
            )
            """)
            
    def get_status(self, article_id: str, step: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT status FROM article_steps WHERE article_id=? AND step_name=?",
            (article_id, step),
        )
        row = cur.fetchone()
        return row['status'] if row else None
    
    def set_status(self, article_id: str, step: str, status: str, error: str | None = None):
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("""
            INSERT INTO article_steps(article_id, step_name, status, updated_at, error)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(article_id, step_name)
            DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, error=excluded.error
            """, (article_id, step, status, now, error))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def list_states(
            self,
            article_id: Optional[str] = None,
            status: Optional[str] = None,
            step: Optional[str] = None
    ) -> List[Dict[str | Any, Any]]:
        query = "SELECT * FROM article_steps"
        params = []
        conditions = []
        
        if article_id:
            conditions.append("article_id = ?")
            params.append(article_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if step:
            conditions.append("step_name = ?")
            params.append(step)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cur = self.conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]
    
    def clear_data(self, article_id: Optional[str] = None):
        with self.conn:
            if article_id:
                self.conn.execute("DELETE FROM article_steps WHERE article_id = ?", (article_id,))
            else:
                self.conn.execute("DELETE FROM article_steps")
    
    def reset_running_states(self, message: str = "Interrupted by system restart"):
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("""
                UPDATE article_steps
                SET status = 'failed', error = ?, updated_at = ?
                WHERE status = 'running'
            """, (message, now))
        