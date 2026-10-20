"""
Workflow DAG engine for Trove agent orchestration.

Defines multi-step task pipelines with dependencies, executes them
in topological order, and tracks progress. Claude reads next available
steps and acts on them — the engine only tracks state, not execution.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional


class WorkflowEngine:
    """SQLite-backed workflow DAG tracker.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: Path, connection: sqlite3.Connection = None,
                 lock: threading.Lock = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = lock or threading.Lock()
        self._owns_conn = connection is None
        if connection is not None:
            self._conn = connection
        else:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_steps (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    description TEXT,
                    depends_on TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result TEXT,
                    started_at REAL,
                    completed_at REAL,
                    sort_order INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_steps_workflow
                    ON workflow_steps(workflow_id);
            """)
            self._conn.commit()

    def create(self, name: str, steps: list[dict]) -> str:
        """Create a workflow with dependency-ordered steps.

        Args:
            name: Workflow name.
            steps: List of dicts with keys: name, description (optional),
                   depends_on (optional list of step names).

        Returns:
            The workflow ID.

        Raises:
            ValueError: If the dependency graph has cycles or references
                       unknown step names.
        """
        if not steps:
            raise ValueError("Workflow must have at least one step")

        # Assign UUIDs and build name -> id mapping
        step_map: dict[str, dict] = {}
        for step in steps:
            step_id = str(uuid.uuid4())
            step_name = step["name"]
            if step_name in step_map:
                raise ValueError(f"Duplicate step name: {step_name}")
            step_map[step_name] = {
                "id": step_id,
                "name": step_name,
                "description": step.get("description", ""),
                "depends_on_names": step.get("depends_on", []) or [],
            }

        # Validate depends_on references
        for step_name, info in step_map.items():
            for dep_name in info["depends_on_names"]:
                if dep_name not in step_map:
                    raise ValueError(
                        f"Step '{step_name}' depends on unknown step '{dep_name}'"
                    )

        # Topological sort via Kahn's algorithm
        in_degree: dict[str, int] = {name: 0 for name in step_map}
        adjacency: dict[str, list[str]] = {name: [] for name in step_map}
        for step_name, info in step_map.items():
            for dep_name in info["depends_on_names"]:
                adjacency[dep_name].append(step_name)
                in_degree[step_name] += 1

        queue: deque[str] = deque()
        for name_key, degree in in_degree.items():
            if degree == 0:
                queue.append(name_key)

        sorted_names: list[str] = []
        while queue:
            current = queue.popleft()
            sorted_names.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(step_map):
            raise ValueError("Workflow has circular dependencies")

        # Resolve depends_on names to IDs
        for info in step_map.values():
            info["depends_on_ids"] = [
                step_map[dep_name]["id"]
                for dep_name in info["depends_on_names"]
            ]

        # Persist
        now = time.time()
        workflow_id = str(uuid.uuid4())

        with self._lock:
            self._conn.execute(
                "INSERT INTO workflows (id, name, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (workflow_id, name, "pending", now, now),
            )
            for sort_order, step_name in enumerate(sorted_names):
                info = step_map[step_name]
                self._conn.execute(
                    """INSERT INTO workflow_steps
                       (id, workflow_id, name, description, depends_on,
                        status, sort_order)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        info["id"],
                        workflow_id,
                        info["name"],
                        info["description"],
                        json.dumps(info["depends_on_ids"]),
                        "pending",
                        sort_order,
                    ),
                )
            self._conn.commit()

        return workflow_id

    def get(self, workflow_id: str) -> Optional[dict]:
        """Get a workflow with its steps.

        Returns:
            Dict with workflow info and nested steps list, or None.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, status, created_at, updated_at FROM workflows WHERE id = ?",
                (workflow_id,),
            ).fetchone()

            if not row:
                return None

            step_rows = self._conn.execute(
                """SELECT id, name, description, depends_on, status,
                          result, started_at, completed_at, sort_order
                   FROM workflow_steps WHERE workflow_id = ?
                   ORDER BY sort_order""",
                (workflow_id,),
            ).fetchall()

        steps = []
        for sr in step_rows:
            steps.append({
                "id": sr[0],
                "name": sr[1],
                "description": sr[2],
                "depends_on": json.loads(sr[3]) if sr[3] else [],
                "status": sr[4],
                "result": sr[5],
                "started_at": sr[6],
                "completed_at": sr[7],
                "sort_order": sr[8],
            })

        return {
            "id": row[0],
            "name": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "steps": steps,
        }

    def list(self, status: str = None) -> list[dict]:
        """List all workflows with step counts and progress.

        Args:
            status: Optional status filter.

        Returns:
            List of workflow summary dicts.
        """
        where = ""
        params: list = []
        if status:
            where = "WHERE status = ?"
            params.append(status)

        with self._lock:
            rows = self._conn.execute(
                f"""SELECT id, name, status, created_at, updated_at
                    FROM workflows {where}
                    ORDER BY created_at DESC""",
                params,
            ).fetchall()

            result = []
            for row in rows:
                step_counts = self._conn.execute(
                    """SELECT status, COUNT(*) FROM workflow_steps
                       WHERE workflow_id = ? GROUP BY status""",
                    (row[0],),
                ).fetchall()

                total = sum(c[1] for c in step_counts)
                completed = sum(c[1] for c in step_counts if c[0] == "completed")

                result.append({
                    "id": row[0],
                    "name": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "total_steps": total,
                    "completed_steps": completed,
                })

        return result

    def next_steps(self, workflow_id: str) -> list[dict]:
        """Get steps that are ready to run (pending with all deps completed).

        Returns:
            List of step dicts that can be started.
        """
        with self._lock:
            step_rows = self._conn.execute(
                """SELECT id, name, description, depends_on, status,
                          sort_order
                   FROM workflow_steps WHERE workflow_id = ?
                   ORDER BY sort_order""",
                (workflow_id,),
            ).fetchall()

        if not step_rows:
            return []

        # Build status lookup
        step_status: dict[str, str] = {}
        for sr in step_rows:
            step_status[sr[0]] = sr[4]

        ready = []
        for sr in step_rows:
            if sr[4] != "pending":
                continue
            deps = json.loads(sr[3]) if sr[3] else []
            all_deps_done = all(
                step_status.get(dep_id) in ("completed", "skipped") for dep_id in deps
            )
            if all_deps_done:
                ready.append({
                    "id": sr[0],
                    "name": sr[1],
                    "description": sr[2],
                    "depends_on": deps,
                    "status": sr[4],
                    "sort_order": sr[5],
                })

        return ready

    VALID_STEP_STATUSES = {"pending", "running", "completed", "failed", "skipped"}

    def update_step(self, workflow_id: str, step_id: str,
                    status: str, result: str = None) -> bool:
        """Update a step's status and auto-update workflow status.

        Args:
            workflow_id: The workflow ID.
            step_id: The step ID.
            status: New status (pending, running, completed, failed, skipped).
            result: Optional result text.

        Returns:
            True if updated, False if step not found.

        Raises:
            ValueError: If status is not a valid step status.
        """
        if status not in self.VALID_STEP_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_STEP_STATUSES))}"
            )
        now = time.time()

        with self._lock:
            # Verify step exists
            row = self._conn.execute(
                "SELECT id FROM workflow_steps WHERE id = ? AND workflow_id = ?",
                (step_id, workflow_id),
            ).fetchone()
            if not row:
                return False

            # Update step
            _ALLOWED_COLUMNS = {"status", "result", "started_at", "completed_at"}
            updates = ["status = ?", "result = ?"]
            params: list = [status, result]

            if status == "running":
                updates.append("started_at = ?")
                params.append(now)
            elif status in ("completed", "failed", "skipped"):
                updates.append("completed_at = ?")
                params.append(now)

            # Verify all columns are from the whitelist
            for clause in updates:
                col = clause.split(" = ")[0].strip()
                assert col in _ALLOWED_COLUMNS, f"Unexpected column in update: {col}"

            params.extend([step_id, workflow_id])
            self._conn.execute(
                f"UPDATE workflow_steps SET {', '.join(updates)} WHERE id = ? AND workflow_id = ?",
                params,
            )

            # Auto-update workflow status
            step_statuses = self._conn.execute(
                "SELECT status FROM workflow_steps WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchall()

            statuses = [s[0] for s in step_statuses]
            terminal = {"completed", "skipped"}
            has_failed = any(s == "failed" for s in statuses)
            has_running = any(s == "running" for s in statuses)
            has_pending = any(s == "pending" for s in statuses)

            if all(s in terminal for s in statuses):
                wf_status = "completed"
            elif has_running:
                wf_status = "running"
            elif has_failed and not has_pending:
                # Failed with no pending work remaining
                wf_status = "failed"
            elif any(s in terminal for s in statuses) or has_failed:
                # Some progress made (or a failure with pending work) — still in progress
                wf_status = "running"
            else:
                wf_status = "pending"

            self._conn.execute(
                "UPDATE workflows SET status = ?, updated_at = ? WHERE id = ?",
                (wf_status, now, workflow_id),
            )
            self._conn.commit()

        return True

    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow and cascade-delete its steps.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM workflows WHERE id = ?",
                (workflow_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self):
        """Close the SQLite connection (only if owned by this instance)."""
        if self._owns_conn:
            with self._lock:
                self._conn.close()
