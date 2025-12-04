from fastapi import APIRouter, HTTPException
from ..schemas import ChatCreateRequest, ChatResponse, ChatHistoryResponse, MessageSchema
from ..services import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def create_chat(req: ChatCreateRequest):
    chat = chat_service.create_chat(
        provider=req.provider,
        model=req.model,
        agent_id=req.agent_id,
        title=req.title,
    )
    return ChatResponse(
        chat_id=chat.id,
        provider=chat.provider,
        model=chat.model,
        agent_id=chat.agent_id,
        title=chat.title,
        created_at=chat.created_at,
    )


@router.get("/{chat_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(chat_id: str):
    try:
        chat = chat_service.get_chat(chat_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = [
        MessageSchema(role=m.role, content=m.content, timestamp=m.timestamp)
        for m in chat.messages
    ]
    return ChatHistoryResponse(chat_id=chat.id, messages=messages)
