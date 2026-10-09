from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent


def get_project_settings() -> Dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "google_model": os.getenv("GOOGLE_MODEL", "gemini-3.8-flash"),
        "google_embedding_model": os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004"),
        "chroma_db_dir": os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db")),
        "handbook_path": os.getenv("HANDBOOK_PATH", str(PROJECT_ROOT / "data" / "Mirai_SoT_Policy_Handbook_2026.pdf")),
        "backend_url": os.getenv("BACKEND_URL", "http://127.0.0.1:8000"),
    }


class RAGPipeline:
    def __init__(self) -> None:
        self.settings = get_project_settings()
        self.project_root = PROJECT_ROOT
        self.chroma_dir = Path(self.settings["chroma_db_dir"]).resolve()
        self.handbook_path = Path(self.settings["handbook_path"]).resolve()
        self.data_dir = self.handbook_path.parent
        self.embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self.vector_store: Optional[Chroma] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.retriever: Optional[MultiQueryRetriever] = None
        self.prompt = self._build_prompt()
        self.rag_chain = None
        self.initialized = False
        self.initialize()

    def initialize(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        api_key = self.settings["google_api_key"]
        if not api_key:
            self.initialized = False
            return

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.settings["google_embedding_model"],
            google_api_key=api_key,
        )
        self.vector_store = Chroma(
            persist_directory=str(self.chroma_dir),
            embedding_function=self.embeddings,
            collection_name="mirai_policy_handbook",
        )
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings["google_model"],
            temperature=0.0,
            google_api_key=api_key,
        )
        self.retriever = MultiQueryRetriever.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        )
        self.rag_chain = self._build_rag_chain()
        self.initialized = True

    def is_ready(self) -> bool:
        return self.initialized and self.vector_store is not None and self.retriever is not None and self.llm is not None

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful academic assistant for the official MirAI Student Policy Handbook. "
                    "Answer only using the retrieved handbook context. If the retrieved context does not support the answer, "
                    "respond with: 'I could not find enough information in the official MirAI Student Policy Handbook to answer this confidently.' "
                    "Do not use outside knowledge. Mention available source references when they are present.",
                ),
                (
                    "user",
                    "Question: {question}\n\nRetrieved handbook context:\n{context}",
                ),
            ]
        )

    def _build_rag_chain(self):
        if self.llm is None:
            return None

        def get_context(question: str) -> str:
            docs = self.retrieve_documents(question)
            if not docs:
                return ""
            return self._format_context(docs)

        return (
            RunnableParallel(
                question=RunnablePassthrough(),
                context=RunnableLambda(get_context),
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_context(self, documents: List[Document]) -> str:
        return "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content.strip()}"
            for doc in documents
        )

    def _source_references(self, documents: List[Document]) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for doc in documents:
            metadata = doc.metadata or {}
            refs.append(
                {
                    "source": metadata.get("source", "unknown_source"),
                    "page": metadata.get("page"),
                }
            )
        return refs

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        seen: Set[str] = set()
        unique_docs: List[Document] = []
        for doc in documents:
            key = (doc.page_content.strip(), doc.metadata.get("page"), doc.metadata.get("source"))
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    def retrieve_documents(self, question: str) -> List[Document]:
        if not self.is_ready() or self.retriever is None:
            return []
        try:
            docs = self.retriever.get_relevant_documents(question)
            return self._deduplicate_documents(docs)
        except Exception as exc:  # pragma: no cover - depends on external API and DB state
            raise ValueError(f"Failed to retrieve handbook context: {exc}") from exc

    def ingest_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        pdf_file = Path(file_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"The handbook PDF was not found at: {pdf_file}")
        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported for ingestion.")
        if pdf_file.stat().st_size == 0:
            raise ValueError(f"The PDF at {pdf_file} is empty.")

        self.initialize()
        if not self.is_ready():
            raise ValueError(
                "Google API credentials are not configured. Please add GOOGLE_API_KEY to the .env file before ingesting."
            )

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
        except Exception as exc:  # pragma: no cover - external library issue
            raise ValueError(f"Unable to read the PDF file: {exc}") from exc

        if not documents:
            raise ValueError(f"The PDF at {pdf_file} appears to be empty or unreadable.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)

        source_name = pdf_file.name
        existing_ids = []
        try:
            existing = self.vector_store.get(where={"source": source_name})
            existing_ids = existing.get("ids", []) if existing else []
            if existing_ids:
                self.vector_store.delete(ids=existing_ids)
        except Exception:
            pass

        for chunk in chunks:
            metadata = chunk.metadata or {}
            metadata["source"] = source_name
            metadata["source_path"] = str(pdf_file.resolve())
            chunk.metadata = metadata

        try:
            self.vector_store.add_documents(chunks)
            self.vector_store.persist()
        except Exception as exc:  # pragma: no cover - external API or Chroma issue
            raise ValueError(f"Failed during vector embedding and storage: {exc}") from exc

        summary = {
            "success": True,
            "chunks_processed": len(chunks),
            "message": f"Successfully ingested {len(chunks)} chunks from {source_name} into the local ChromaDB.",
        }
        return summary

    def answer_question(self, question: str) -> Dict[str, Any]:
        if not question or not question.strip():
            return {
                "answer": "Please provide a question to ask the MirAI Student Policy Advisor.",
                "source_references": [],
                "retrieved_context": [],
                "status": "invalid_request",
            }

        if not self.is_ready() or self.rag_chain is None:
            return {
                "answer": "The policy handbook is not yet indexed. Please upload the official handbook PDF using the ingestion endpoint first.",
                "source_references": [],
                "retrieved_context": [],
                "status": "not_ready",
            }

        try:
            docs = self.retrieve_documents(question)
        except ValueError as exc:
            return {
                "answer": f"I could not answer that question because the retrieval step failed: {exc}",
                "source_references": [],
                "retrieved_context": [],
                "status": "retrieval_error",
            }

        if not docs:
            return {
                "answer": "I could not find enough information in the official MirAI Student Policy Handbook to answer that question confidently.",
                "source_references": [],
                "retrieved_context": [],
                "status": "no_context",
            }

        try:
            answer_text = self.rag_chain.invoke(question)
        except Exception as exc:  # pragma: no cover - external API failure
            return {
                "answer": f"The document model could not generate an answer right now: {exc}",
                "source_references": self._source_references(docs),
                "retrieved_context": [doc.page_content for doc in docs],
                "status": "generation_error",
            }

        answer = (answer_text or "").strip()
        if not answer:
            answer = "I could not find enough information in the official MirAI Student Policy Handbook to answer this confidently."

        return {
            "answer": answer,
            "source_references": self._source_references(docs),
            "retrieved_context": [doc.page_content for doc in docs],
            "status": "ok",
        }
