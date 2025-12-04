from fastapi import APIRouter, HTTPException
from ..services.chat_service import (
    create_chat,
    get_chat,
    get_all_chats,
    add_message,
    run_normal_chat
)
from ..schemas import ChatCreateRequest, ChatResponse, MessageRequest, MessageResponse

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.post("", response_model=ChatResponse)
async def create_chat_endpoint(req: ChatCreateRequest):
    chat = create_chat(
        provider=req.provider,
        model=req.model,
        agent_id=req.agent_id,
        title=req.title,
    )
    return ChatResponse.from_chat(chat)


@router.get("", response_model=list[ChatResponse])
async def list_chats_endpoint():
    chats = get_all_chats()
    return [ChatResponse.from_chat(c) for c in chats]


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat_endpoint(chat_id: str):
    chat = get_chat(chat_id)
    return ChatResponse.from_chat(chat)


# ⭐ NEW — NORMAL CHAT MODE (NO AGENT, NO JOB)
@router.post("/{chat_id}/message", response_model=MessageResponse)
async def normal_chat_endpoint(chat_id: str, req: MessageRequest):
    chat = get_chat(chat_id)

    # If chat has agent_id → user must use /jobs endpoint instead
    if chat.agent_id:
        raise HTTPException(
            status_code=400,
            detail="This chat is configured for agent mode. Use /jobs/{chat_id} instead."
        )

    # Run direct LLM chat
    assistant_message = await run_normal_chat(chat_id, req.prompt)

    return MessageResponse(
        chat_id=chat_id,
        message=assistant_message
    )
