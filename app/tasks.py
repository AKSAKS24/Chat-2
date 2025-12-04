"""
Async job runner for ChatPwC backend.

This module executes BOTH:
1) Agent Mode (when agent_id is provided)
2) Normal Chat Mode (when agent_id is None)

No Celery is required — everything runs inside asyncio.
"""

import asyncio

from .services.job_manager import job_manager, JobStatus
from .services.chat_service import get_chat, add_message
from .services.agent_registry import get_agent
from .llm.provider_registry import create_llm_client
from .storage.memory_store import MEMORY_STORE
from .utils.logger import logger


async def _run_job(job_id: str) -> None:
    """
    Core async worker:
    - Loads job & chat
    - Detects agent mode vs normal chat mode
    - Calls LLM or agent
    - Updates MemoryStore + JobManager
    """
    job = MEMORY_STORE.get_job(job_id)
    chat = get_chat(job.chat_id)

    llm_client = create_llm_client(chat.provider, chat.model)

    await job_manager.update_job(
        job_id,
        status=JobStatus.RUNNING,
        log="Job started",
    )

    try:
        prompt = job.prompt

        # --------------------------
        # ⭐ MODE 1: NORMAL CHAT MODE
        # --------------------------
        if not chat.agent_id:  # None or empty string
            logger.info("[Worker] Normal chat mode for job %s", job_id)

            # Build conversation history for LLM
            messages = [
                {"role": m.role, "content": m.content}
                for m in chat.messages
            ]
            messages.append({"role": "user", "content": prompt})

            response = await llm_client.chat(messages)

            result_text = response.text

            # Save assistant message into chat history
            add_message(chat, "assistant", result_text)

            await MEMORY_STORE.update_job(
                job_id,
                status="completed",
                log="Normal chat completed",
                result_message=result_text,
                output_payload={"text": result_text},
            )

            await job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                log="Normal chat completed",
                result={"text": result_text},
            )
            return

        # --------------------------
        # ⭐ MODE 2: AGENT MODE
        # --------------------------
        agent = get_agent(chat.agent_id)
        logger.info("[Worker] Running agent %s for job %s", agent.name, job_id)

        result = await agent.run(
            chat=chat,
            llm_client=llm_client,
            job_id=job_id,
            prompt=prompt,
        )

        # Update chat with final answer
        add_message(chat, "assistant", result.text)

        await MEMORY_STORE.update_job(
            job_id,
            status="completed",
            log="Agent job completed",
            result_message=result.text,
            output_docx_path=result.output_docx_path,
            output_payload={
                "text": result.text,
                "output_docx_path": result.output_docx_path,
            },
        )

        await job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            log="Agent job completed",
            result={
                "text": result.text,
                "output_docx_path": result.output_docx_path,
            },
        )

    except Exception as exc:
        logger.exception("Job %s failed", job_id)

        await MEMORY_STORE.update_job(
            job_id,
            status="failed",
            log=f"Job failed: {exc}",
            error=str(exc),
        )
        await job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            log=f"Job failed: {exc}",
            result=None,
        )


def launch_job(job_id: str) -> None:
    """
    Fire-and-forget launcher.
    If event loop exists → create_task()
    Else → asyncio.run()
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(_run_job(job_id))
    else:
        asyncio.run(_run_job(job_id))
