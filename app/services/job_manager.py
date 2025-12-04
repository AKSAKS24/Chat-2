"""
In-memory job manager to track async jobs (Celery tasks).
Jobs have: id, status, logs, result, metadata.
"""

import asyncio
import uuid
from typing import Any, Dict, Optional


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        async with self._lock:
            job_id = str(uuid.uuid4())
            self._jobs[job_id] = {
                "id": job_id,
                "status": JobStatus.QUEUED,
                "logs": [],
                "result": None,
                "metadata": metadata or {},
            }
            return job_id

    async def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        log: Optional[str] = None,
        result: Any = None,
    ) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if status:
                job["status"] = status
            if log:
                job["logs"].append(log)
            if result is not None:
                job["result"] = result

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._jobs.get(job_id)


job_manager = JobManager()
