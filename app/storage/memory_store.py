"""
memory_store.py
---------------
In-memory persistence layer for Chats, Messages, and Jobs.
Suitable for development & single-process deployments.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


MessageRole = Literal["user", "assistant", "system"]
JobStatusType = Literal["queued", "running", "completed", "failed"]


# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------

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
    agent_id: Optional[str] = None
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

    output_docx_path: Optional[str] = None
    output_payload: Optional[Dict[str, Any]] = None

    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# MemoryStore Implementation
# ------------------------------------------------------------

class MemoryStore:
    """
    In-memory storage for Chats, Messages, Jobs.
    """

    def __init__(self) -> None:
        self.chats: Dict[str, Chat] = {}
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    # --------------------------------------------------------
    # CHAT HELPERS
    # --------------------------------------------------------

    def create_chat(self, provider: str, model: str, agent_id: Optional[str], title: Optional[str]) -> Chat:
        chat_id = f"chat_{uuid.uuid4().hex[:12]}"
        chat = Chat(
            id=chat_id,
            provider=provider,
            model=model,
            agent_id=agent_id,
            title=title,
        )
        self.chats[chat_id] = chat
        return chat

    def save_chat(self, chat: Chat) -> Chat:
        self.chats[chat.id] = chat
        return chat

    def get_chat(self, chat_id: str) -> Chat:
        chat = self.chats.get(chat_id)
        if not chat:
            raise KeyError(f"Chat {chat_id} not found")
        return chat

    def get_all_chats(self) -> List[Chat]:
        return list(self.chats.values())

    def add_message(self, chat_id: str, role: MessageRole, content: str) -> Message:
        chat = self.get_chat(chat_id)
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
        )
        chat.messages.append(message)
        return message

    # --------------------------------------------------------
    # JOB HELPERS
    # --------------------------------------------------------

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


# Global singleton
MEMORY_STORE = MemoryStore()
