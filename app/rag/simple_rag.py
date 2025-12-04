from pathlib import Path
from functools import lru_cache


KB_PATH = Path(__file__).resolve().parents[2] / "data" / "ts_rag_kb.txt"


@lru_cache
def _load_kb() -> str:
    if KB_PATH.exists():
        return KB_PATH.read_text(encoding="utf-8")
    return ""


def get_context_for_abap(abap_code: str) -> str:
    """
    Simple RAG: return the entire KB; you can later do semantic search.
    """
    kb = _load_kb()
    return kb
