import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from .base import BaseStateManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgreSQLStateManager(BaseStateManager):
    def __init__(self, dsn: str):
        self.conn: psycopg.Connection

        try:
            self.conn = psycopg.connect(dsn)
            self._init_schema()
        except Exception:
            logger.exception("Failed to initialize database connection")
            self._rollback()
            self.close()
            raise

    def _rollback(self):
        try:
            if self.conn is not None and not self.conn.closed:
                self.conn.rollback()
        except Exception:
            logger.exception("Failed to rollback transaction")

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS article_steps (
                    article_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    error TEXT,
                    PRIMARY KEY (article_id, step_name)
                )
                """
            )
        self.conn.commit()

    def get_status(self, article_id: str, step: str) -> Optional[str]:
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM article_steps WHERE article_id=%s AND step_name=%s",
                    (article_id, step),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            self._rollback()
            logger.exception("Failed to get status")
            raise

    def set_status(
        self, article_id: str, step: str, status: str, error: Optional[str] = None,
    ):
        try:
            now = datetime.now(timezone.utc)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO article_steps(article_id, step_name, status, updated_at, error)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(article_id, step_name)
                    DO UPDATE SET
                        status=EXCLUDED.status,
                        updated_at=EXCLUDED.updated_at,
                        error=EXCLUDED.error
                    """,
                    (article_id, step, status, now, error),
                )
            self.conn.commit()
        except Exception:
            self._rollback()
            logger.exception("Failed to set status")
            raise

    def list_states(
        self, article_id: Optional[str] = None, status: Optional[str] = None, step: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            query = "SELECT * FROM article_steps"
            params: List[Any] = []
            conditions: List[str] = []

            if article_id:
                conditions.append("article_id = %s")
                params.append(article_id)

            if status:
                conditions.append("status = %s")
                params.append(status)

            if step:
                conditions.append("step_name = %s")
                params.append(step)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            with self.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)  # noqa
                return cur.fetchall()
        except Exception:
            self._rollback()
            logger.exception("Failed to list states")
            raise

    def clear_data(self, article_id: Optional[str] = None):
        try:
            with self.conn.cursor() as cur:
                if article_id:
                    cur.execute(
                        "DELETE FROM article_steps WHERE article_id = %s",
                        (article_id,),
                    )
                else:
                    cur.execute("DELETE FROM article_steps")
            self.conn.commit()
        except Exception:
            self._rollback()
            logger.exception("Failed to clear data")
            raise

    def reset_running_states(self, message: str = "Interrupted by system restart"):
        try:
            now = datetime.now(timezone.utc)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE article_steps
                    SET status = 'failed', error = %s, updated_at = %s
                    WHERE status = 'running'
                    """,
                    (message, now),
                )
            self.conn.commit()
        except Exception:
            self._rollback()
            logger.exception("Failed to reset running states")
            raise

    def close(self):
        try:
            if self.conn is not None and not self.conn.closed:
                self.conn.close()
        except Exception:
            logger.exception("Failed to close database connection")
        finally:
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error("Exiting context due to exception: %s", exc_val)

        self.close()
        return False
