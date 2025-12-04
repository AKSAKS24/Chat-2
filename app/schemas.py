from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel


class MessageSchema(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime


class ChatCreateRequest(BaseModel):
    provider: str
    model: str
    agent_id: str
    title: Optional[str] = None


class ChatResponse(BaseModel):
    chat_id: str
    provider: str
    model: str
    agent_id: str
    title: Optional[str]
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: List[MessageSchema]


class JobCreateRequest(BaseModel):
    prompt: str  # ABAP code input


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    chat_id: str
    result_message: Optional[str] = None
    output_docx_url: Optional[str] = None
    error: Optional[str] = None


class ProviderModelsResponse(BaseModel):
    data: dict


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str


class AgentsListResponse(BaseModel):
    data: list[AgentInfo]
