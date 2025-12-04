from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from ..llm.base import ChatMessage
from ..rag.simple_rag import get_context_for_abap
from ..storage.memory_store import Chat
from ..llm.base import LLMClient
from ..utils.docx_generator import create_ts_docx


class TSFSAgent(BaseAgent):
    id = "ts_fs_agent"
    name = "TS/FS Generator"
    description = "Takes ABAP code and generates Technical Specification using RAG KB (outputs DOCX)."

    async def run(
        self,
        job_id: str,
        prompt: str,
        llm_client: LLMClient,
        chat: Chat,
    ) -> AgentResult:
        # prompt = ABAP code
        abap_code = prompt
        rag_ctx = get_context_for_abap(abap_code)

        system_msg = ChatMessage(
            role="system",
            content=(
                "You are an expert SAP ABAP Technical Architect. "
                "Generate a comprehensive Technical Specification (TS) for the given ABAP code.\n"
                "Use the RAG knowledge base context for structure and best practices.\n"
                "Output as well structured markdown with clear sections like:\n"
                "## Introduction\n## Business Requirement Overview\n## Solution Overview\n"
                "## SAP Objects\n## Data Model\n## Processing Logic\n## Error Handling\n## Performance & Security\n"
                "## Transport & Dependencies\n"
            ),
        )

        user_msg = ChatMessage(
            role="user",
            content=(
                "ABAP code:\n\n"
                f"{abap_code}\n\n"
                "RAG context:\n\n"
                f"{rag_ctx}\n"
            ),
        )

        ts_markdown = await llm_client.chat([system_msg, user_msg])

        # Save DOCX
        output_dir = Path(__file__).resolve().parents[2] / "generated"
        output_path = output_dir / f"{job_id}_ts.docx"
        create_ts_docx(ts_markdown, output_path)

        return AgentResult(
            text=ts_markdown,
            output_docx_path=str(output_path),
        )
