"""
memory_store.py
---------------
Simple in-memory persistence layer for Chats, Messages, and Jobs.

⚠️ This is NOT a database – it is only suitable for development and
single-process deployments. Replace with Redis/Postgres in production.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


MessageRole = Literal["user", "assistant", "system"]
JobStatusType = Literal["queued", "running", "completed", "failed"]


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime


@dataclass
class Chat:
    id: str
    provider: str
    model: str
    agent_id: str
    title: Optional[str] = None
    messages: List[Message] = field(default_factory=list)


@dataclass
class Job:
    id: str
    chat_id: str
    prompt: str

    status: JobStatusType = "queued"
    logs: List[str] = field(default_factory=list)
    result_message: Optional[str] = None

    # Optional: for downloadable generated DOCX/ZIP files
    output_docx_path: Optional[str] = None

    # Optional: preserve entire structured result (text, json, paths etc.)
    output_payload: Optional[Dict[str, Any]] = None

    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    """
    Generic in-memory storage with a simple asyncio lock for jobs.
    Chats are mutated in-place but job lifecycle updates go through
    `update_job` for consistent logging and status management.
    """

    def __init__(self) -> None:
        self.chats: Dict[str, Chat] = {}
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    # ---------- Chat helpers ----------

    def save_chat(self, chat: Chat) -> Chat:
        self.chats[chat.id] = chat
        return chat

    def get_chat(self, chat_id: str) -> Chat:
        chat = self.chats.get(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} not found")
        return chat

    # ---------- Job helpers ----------

    def create_job(
        self,
        chat_id: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(
            id=job_id,
            chat_id=chat_id,
            prompt=prompt,
            status="queued",
            metadata=metadata or {},
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found")
        return job

    async def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatusType] = None,
        log: Optional[str] = None,
        result_message: Optional[str] = None,
        output_docx_path: Optional[str] = None,
        output_payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Job:
        """
        Atomically update a job. All fields are optional and only
        non-None values are applied.
        """
        async with self._lock:
            job = self.get_job(job_id)

            if status is not None:
                job.status = status
            if log:
                job.logs.append(log)
            if result_message is not None:
                job.result_message = result_message
            if output_docx_path is not None:
                job.output_docx_path = output_docx_path
            if output_payload is not None:
                job.output_payload = output_payload
            if error is not None:
                job.error = error

            return job


# Global singleton used throughout the app
MEMORY_STORE = MemoryStore()
