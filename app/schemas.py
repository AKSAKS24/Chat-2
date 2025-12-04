from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel


# ---------------------------------------------------------
# MESSAGE SCHEMA (used inside chats)
# ---------------------------------------------------------

class MessageSchema(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime


# ---------------------------------------------------------
# CHAT CREATION REQUEST
# ---------------------------------------------------------

class ChatCreateRequest(BaseModel):
    provider: str
    model: str
    agent_id: Optional[str] = None   # None → Normal Chat Mode
    title: Optional[str] = None


# ---------------------------------------------------------
# CHAT RESPONSE (used for GET /chats & GET /chats/{id})
# ---------------------------------------------------------

class ChatResponse(BaseModel):
    id: str
    provider: str
    model: str
    agent_id: Optional[str] = None
    title: Optional[str] = None
    messages: List[MessageSchema]

    @classmethod
    def from_chat(cls, chat):
        """
        Convert Chat dataclass → ChatResponse pydantic model.
        This is required because our Chat is a dataclass stored
        inside the MemoryStore, not a Pydantic model.
        """
        return cls(
            id=chat.id,
            provider=chat.provider,
            model=chat.model,
            agent_id=chat.agent_id,
            title=chat.title,
            messages=[
                MessageSchema(
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                )
                for m in chat.messages
            ]
        )


# ---------------------------------------------------------
# NORMAL CHAT MODE REQUEST/RESPONSE
# ---------------------------------------------------------

class MessageRequest(BaseModel):
    prompt: str


class MessageResponse(BaseModel):
    chat_id: str
    message: str


# ---------------------------------------------------------
# CHAT HISTORY (optional, for separate endpoint)
# ---------------------------------------------------------

class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: List[MessageSchema]


# ---------------------------------------------------------
# JOB REQUEST (Agent Mode)
# ---------------------------------------------------------

class JobCreateRequest(BaseModel):
    prompt: str


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    chat_id: str
    result_message: Optional[str] = None
    output_docx_url: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------
# PROVIDER / MODEL LISTING
# ---------------------------------------------------------

class ProviderModelsResponse(BaseModel):
    data: dict


# ---------------------------------------------------------
# AGENT LISTING
# ---------------------------------------------------------

class AgentInfo(BaseModel):
    id: str
    name: str
    description: str


class AgentsListResponse(BaseModel):
    data: List[AgentInfo]
