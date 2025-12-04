from datetime import datetime
from .job_manager import job_manager
from ..storage.memory_store import MEMORY_STORE
from ..utils.ids import new_id
from ..llm.provider_registry import create_llm_client


def create_chat(provider: str, model: str, agent_id: str | None, title: str | None):
    chat = MEMORY_STORE.create_chat(provider, model, agent_id, title)
    return chat


def get_chat(chat_id: str):
    return MEMORY_STORE.get_chat(chat_id)


def get_all_chats():
    return MEMORY_STORE.get_all_chats()


def add_message(chat, role: str, content: str):
    return MEMORY_STORE.add_message(chat.id, role, content)


# ⭐ NORMAL CHAT MODE (NO AGENT, NO JOB ENGINE)
async def run_normal_chat(chat_id: str, prompt: str) -> str:
    chat = get_chat(chat_id)

    # Build full history for LLM
    history = [
        {"role": m.role, "content": m.content}
        for m in chat.messages
    ]
    history.append({"role": "user", "content": prompt})

    llm = create_llm_client(chat.provider, chat.model)
    response = await llm.chat(history)

    assistant_reply = response.text

    # Save in chat history
    add_message(chat, "user", prompt)
    add_message(chat, "assistant", assistant_reply)

    return assistant_reply
