from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routers import chat, jobs, meta
from .utils.logger import logger

app = FastAPI(title="AI Agent Wrapper Backend")

# CORS – adjust origins for your frontend (React/Vue/etc.)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount static for generated DOCX
generated_dir = Path(__file__).resolve().parents[1] / "generated"
generated_dir.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(generated_dir)), name="generated")

# include routers
app.include_router(meta.router)
app.include_router(chat.router)
app.include_router(jobs.router)


@app.get("/")
async def root():
    return {"message": "AI Agent Wrapper Backend is running"}
