from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A student question about university policy")


class SourceReference(BaseModel):
    source: str
    page: Optional[Any] = None


class ChatResponse(BaseModel):
    answer: str
    source_references: List[SourceReference]
    status: str


class IngestResponse(BaseModel):
    success: bool
    chunks_processed: int
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline()
    yield


app = FastAPI(
    title="MirAI Student Policy Advisor API",
    version="1.0.0",
    description="Backend for the MirAI Student Policy Advisor RAG application.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    pipeline = getattr(app.state, "pipeline", None)
    initialized = bool(pipeline and pipeline.is_ready())
    return {
        "status": "ok",
        "vector_db_initialized": initialized,
        "backend": "MirAI Student Policy Advisor API",
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A PDF file is required for ingestion.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted for ingestion.")

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = file.filename.replace(" ", "_")
    temp_path = data_dir / safe_name

    try:
        content = await file.read()
        temp_path.write_bytes(content)
        summary = await asyncio.to_thread(app.state.pipeline.ingest_pdf, temp_path)
        return IngestResponse(
            success=bool(summary.get("success", False)),
            chunks_processed=int(summary.get("chunks_processed", 0)),
            message=str(summary.get("message", "The PDF was processed successfully.")),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise HTTPException(status_code=500, detail=f"Failed to ingest the handbook: {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(
            status_code=503,
            detail="The vector database is not initialized yet. Upload the official handbook PDF first.",
        )

    try:
        response = await asyncio.to_thread(pipeline.answer_question, request.question)
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise HTTPException(status_code=500, detail=f"The chatbot could not answer that question: {exc}") from exc

    source_references = [
        SourceReference(**item) for item in response.get("source_references", [])
    ]
    return ChatResponse(
        answer=response.get("answer", "I could not find enough information in the official handbook to answer that question."),
        source_references=source_references,
        status=response.get("status", "ok"),
    )
