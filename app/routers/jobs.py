from pathlib import Path
from typing import AsyncGenerator

import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from ..schemas import JobCreateRequest, JobResponse
from ..services import job_service, job_manager
from ..utils.logger import logger

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/{chat_id}", response_model=JobResponse)
async def create_job_for_chat(chat_id: str, req: JobCreateRequest) -> JobResponse:
    """
    Create a job for a given chat and start background execution.
    """
    job = await job_service.create_job(chat_id=chat_id, prompt=req.prompt)

    # fire-and-forget execution inside this process
    job_service.start_job(job.id)

    return JobResponse(
        job_id=job.id,
        chat_id=job.chat_id,
        status=job.status,
        result_message=job.result_message,
        output_docx_url=f"/jobs/{job.id}/docx" if job.output_docx_path else None,
        error=job.error,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str) -> JobResponse:
    """
    Retrieve the latest job state from the in-memory store.
    """
    try:
        job = job_service.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.id,
        chat_id=job.chat_id,
        status=job.status,
        result_message=job.result_message,
        output_docx_url=f"/jobs/{job.id}/docx" if job.output_docx_path else None,
        error=job.error,
    )


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str):
    """
    Server-Sent Events (SSE) endpoint to stream job logs and final result.
    The frontend can connect to this endpoint to receive incremental updates.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        last_len = 0
        while True:
            state = await job_manager.job_manager.get_job(job_id)  # type: ignore[attr-defined]
            if not state:
                raise HTTPException(status_code=404, detail="Job not found")

            logs = state.get("logs", [])
            if len(logs) > last_len:
                for line in logs[last_len:]:
                    yield f"data: {line}\n\n"
                last_len = len(logs)

            if state["status"] in ("completed", "failed"):
                import json

                payload = json.dumps(
                    {"status": state["status"], "result": state.get("result")}
                )
                yield f"data: {payload}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{job_id}/docx")
async def download_job_docx(job_id: str):
    """
    Download the generated DOCX for a completed job.
    """
    try:
        job = job_service.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.output_docx_path:
        raise HTTPException(status_code=404, detail="No DOCX generated for this job")

    path = Path(job.output_docx_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="DOCX file not found")

    return FileResponse(
        path,
        filename=path.name,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )
