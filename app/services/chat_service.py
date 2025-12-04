from datetime import datetime
from typing import List

from ..storage.memory_store import MEMORY_STORE, Chat, Message
from ..utils.ids import new_id


def create_chat(provider: str, model: str, agent_id: str, title: str | None = None) -> Chat:
    """
    Create and persist a new chat session in the in-memory store.
    """
    chat_id = new_id("chat")
    chat = Chat(
        id=chat_id,
        provider=provider,
        model=model,
        agent_id=agent_id,
        title=title,
    )
    MEMORY_STORE.save_chat(chat)
    return chat


def get_chat(chat_id: str) -> Chat:
    """
    Fetch an existing chat or raise KeyError if not found.
    """
    return MEMORY_STORE.get_chat(chat_id)


def add_message(chat: Chat, role: str, content: str) -> Message:
    """
    Append a message to the given chat and return it.
    """
    message = Message(
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=datetime.utcnow(),
    )
    chat.messages.append(message)
    return message


def get_history(chat: Chat) -> List[Message]:
    return chat.messages
