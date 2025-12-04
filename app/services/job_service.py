from typing import Optional

from ..storage.memory_store import MEMORY_STORE, Job
from ..services.chat_service import get_chat
from ..services.job_manager import job_manager
from ..utils.logger import logger


async def create_job(chat_id: str, prompt: str) -> Job:
    """
    Create a Job object for a given chat and register it with the
    in-memory JobManager used for streaming status/logs.
    """
    chat = get_chat(chat_id)

    job = MEMORY_STORE.create_job(
        chat_id=chat_id,
        prompt=prompt,
        metadata={"chat_id": chat_id, "agent_id": chat.agent_id},
    )

    # Mirror job into the lightweight job_manager structure that powers SSE
    await job_manager.create_job(
        metadata={"chat_id": chat_id, "agent_id": chat.agent_id}
    )

    logger.info("[JobService] Created job %s for chat %s", job.id, chat_id)
    return job


def get_job(job_id: str) -> Job:
    return MEMORY_STORE.get_job(job_id)


def start_job(job_id: str) -> None:
    """
    Fire-and-forget kick-off of the async agent execution.

    This replaces the previous Celery-based enqueue mechanism and runs
    the work inside the FastAPI process using asyncio.
    """
    from ..tasks import launch_job  # local import to avoid cycles

    logger.info("[JobService] Starting job %s", job_id)
    launch_job(job_id)
