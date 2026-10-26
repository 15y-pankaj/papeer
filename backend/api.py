import json
import logging
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from pydantic import AnyHttpUrl, BaseModel, Field

from backend.btw_handler import handle_btw

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DB_PATH = Path(
    os.getenv("PAPEER_SESSIONS_DB", str(PROJECT_ROOT / "sessions.sqlite3"))
)
LEGACY_SESSIONS_PATH = PROJECT_ROOT / "sessions.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_REQUEST = 5
ALLOWED_FILE_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


@contextmanager
def _connection(path: Path):
    connection = sqlite3.connect(path, timeout=10)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


class SessionStore:
    """Small SQLite repository for durable session metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _connection(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_named INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        self._import_legacy_sessions()

    def _import_legacy_sessions(self) -> None:
        if not LEGACY_SESSIONS_PATH.is_file():
            return
        try:
            legacy = json.loads(LEGACY_SESSIONS_PATH.read_text(encoding="utf-8"))
            rows = [
                (
                    str(item["id"]),
                    str(item.get("name") or "New Session"),
                    str(item["created_at"]),
                    int(bool(item.get("is_named", False))),
                )
                for item in legacy.values()
            ]
        except (OSError, AttributeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.exception("Unable to import legacy sessions.json")
            return

        with _connection(self.path) as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO sessions (id, name, created_at, is_named)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def list(self) -> list[dict[str, Any]]:
        with _connection(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, name, created_at, is_named FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [
            {**dict(row), "is_named": bool(row["is_named"])}
            for row in rows
        ]

    def get(self, session_id: str) -> dict[str, Any] | None:
        with _connection(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT id, name, created_at, is_named FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {**dict(row), "is_named": bool(row["is_named"])}

    def create(self) -> dict[str, Any]:
        session = {
            "id": str(uuid.uuid4()),
            "name": "New Session",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_named": False,
        }
        with _connection(self.path) as connection:
            connection.execute(
                "INSERT INTO sessions (id, name, created_at, is_named) VALUES (?, ?, ?, ?)",
                (
                    session["id"],
                    session["name"],
                    session["created_at"],
                    int(session["is_named"]),
                ),
            )
        return session

    def set_name(self, session_id: str, name: str) -> None:
        with _connection(self.path) as connection:
            connection.execute(
                "UPDATE sessions SET name = ?, is_named = 1 WHERE id = ?",
                (name, session_id),
            )


session_store = SessionStore(SESSIONS_DB_PATH)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "PAPEER_CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Papeer API",
    version="1.0.0",
    description=(
        "API for the Papeer research assistant. This unauthenticated MVP is "
        "intended for trusted/local use only."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)


class UrlLoadRequest(BaseModel):
    urls: list[AnyHttpUrl] = Field(min_length=1, max_length=5)


class ArxivLoadRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


def _graph():
    from backend.rag_graph import build_graph

    return build_graph()


@lru_cache(maxsize=1)
def get_graph():
    return _graph()


def _require_session(session_id: str) -> dict[str, Any]:
    try:
        normalized_id = str(uuid.UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc
    session = session_store.get(normalized_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _serialize_state(values: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in values.items():
        if key == "messages":
            serialized[key] = [
                {
                    "type": type(message).__name__,
                    "content": _content_text(message.content)[:300]
                    or repr(message.content)[:300],
                }
                for message in value or []
            ]
        elif key == "retrieved_docs":
            serialized[key] = [
                {"content": doc.page_content[:300], "metadata": doc.metadata}
                for doc in value or []
            ]
        else:
            serialized[key] = value
    return serialized


def _history_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    turn = 0
    awaiting_answer = False
    for message in messages:
        if isinstance(message, HumanMessage):
            if not awaiting_answer:
                turn += 1
                history.append(
                    {"role": "user", "content": _content_text(message.content)}
                )
                awaiting_answer = True
        elif isinstance(message, (AIMessage, AIMessageChunk)) and not getattr(
            message, "tool_calls", []
        ):
            content = _content_text(message.content)
            if awaiting_answer and content:
                history.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "turn": turn,
                        "graph_state": {},
                    }
                )
                awaiting_answer = False
    return history


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _name_session(session_id: str, query: str) -> None:
    session = session_store.get(session_id)
    if session is None or session["is_named"]:
        return
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        title = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite").invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title."
                    ),
                },
                {"role": "user", "content": query[:500]},
            ]
        ).content
        title = _content_text(title).strip().strip('"')[:100]
        session_store.set_name(session_id, title or query[:60] or "New Session")
    except Exception:
        logger.exception("Could not generate a title for session %s", session_id)
        session_store.set_name(session_id, query[:60] or "New Session")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness check; it does not make calls to external providers."""
    return {"status": "ok"}


@app.get("/api/ready")
def readiness() -> dict[str, str]:
    """Check required configuration and Qdrant connectivity."""
    missing = [
        key
        for key in ("GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY")
        if not os.getenv(key)
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Service configuration is incomplete: {', '.join(missing)}",
        )
    try:
        from backend.vector_store import qdrant_client

        qdrant_client.get_collections()
    except Exception as exc:
        logger.exception("Qdrant readiness check failed")
        raise HTTPException(status_code=503, detail="Qdrant is unavailable.") from exc
    return {"status": "ready"}


@app.get("/api/sessions")
def list_sessions() -> list[dict[str, Any]]:
    return session_store.list()


@app.post("/api/sessions", status_code=201)
def create_session() -> dict[str, Any]:
    return session_store.create()


@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str) -> dict[str, Any]:
    _require_session(session_id)
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = get_graph().get_state(config)
    except Exception as exc:
        logger.exception("Could not load chat history for session %s", session_id)
        raise HTTPException(status_code=503, detail="Could not load chat history.") from exc
    messages = state.values.get("messages", []) if state and state.values else []
    return {"messages": _history_from_messages(messages)}


def _chat_events(session_id: str, query: str):
    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "query": query,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
    }
    response_text = ""
    try:
        graph = get_graph()
        for chunk, metadata in graph.stream(
            input_state, config, stream_mode="messages"
        ):
            if metadata.get("langgraph_node") != "generate_answer":
                continue
            text = _content_text(getattr(chunk, "content", None))
            if text:
                response_text += text
                yield _sse("token", {"text": text})

        final_values = graph.get_state(config).values
        answer = _content_text(final_values.get("answer")) or response_text
        if not response_text and answer:
            yield _sse("token", {"text": answer})
        yield _sse(
            "done",
            {
                "answer": answer or "No response generated.",
                "graph_state": _serialize_state(final_values),
            },
        )
    except Exception:
        logger.exception("Chat pipeline failed for session %s", session_id)
        yield _sse("error", {"detail": "The chat request failed. Check API logs."})


@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: str, request: ChatRequest) -> StreamingResponse:
    _require_session(session_id)
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")
    _name_session(session_id, query)
    return StreamingResponse(
        _chat_events(session_id, query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _btw_events(query: str):
    try:
        for text in handle_btw(query):
            yield _sse("token", {"text": text})
        yield _sse("done", {})
    except Exception:
        logger.exception("Side-channel request failed")
        yield _sse("error", {"detail": "The /btw request failed. Check API logs."})


@app.post("/api/btw")
def btw(request: ChatRequest) -> StreamingResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")
    return StreamingResponse(
        _btw_events(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sessions/{session_id}/documents")
def upload_documents(
    session_id: str,
    files: Annotated[list[UploadFile], File(min_length=1, max_length=MAX_FILES_PER_REQUEST)],
) -> dict[str, Any]:
    _require_session(session_id)
    from backend.paper_loader import load_document
    from backend.vector_store import add_paper

    results = []
    for upload in files:
        filename = Path(upload.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_FILE_EXTENSIONS:
            results.append(
                {"filename": filename, "ok": False, "error": "Unsupported file type."}
            )
            continue
        temp_path: str | None = None
        try:
            content = upload.file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} exceeds the 20 MiB per-file limit.",
                )
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(content)
                temp_path = temporary.name
            documents = load_document(temp_path)
            for document in documents:
                document.metadata["title"] = Path(filename).stem
            add_paper(documents, session_id)
            results.append(
                {
                    "filename": filename,
                    "ok": True,
                    "title": Path(filename).stem,
                    "chunks": len(documents),
                }
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Could not ingest uploaded file %s", filename)
            results.append(
                {"filename": filename, "ok": False, "error": "Document ingestion failed."}
            )
        finally:
            upload.file.close()
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
    return {"results": results}


@app.post("/api/sessions/{session_id}/documents/urls")
def load_urls(session_id: str, request: UrlLoadRequest) -> dict[str, Any]:
    _require_session(session_id)
    from backend.paper_loader import load_webpage
    from backend.vector_store import add_paper

    results = []
    for item in request.urls:
        url = str(item)
        try:
            documents = load_webpage(url)
            add_paper(documents, session_id)
            results.append({"url": url, "ok": True, "chunks": len(documents)})
        except Exception:
            logger.exception("Could not ingest URL %s", url)
            results.append({"url": url, "ok": False, "error": "URL ingestion failed."})
    return {"results": results}


@app.post("/api/sessions/{session_id}/documents/arxiv")
def load_arxiv(session_id: str, request: ArxivLoadRequest) -> dict[str, Any]:
    _require_session(session_id)
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="ArXiv query cannot be blank.")
    from backend.paper_loader import load_arxiv as load_arxiv_documents
    from backend.vector_store import add_paper

    try:
        documents = load_arxiv_documents(request.query.strip())
        add_paper(documents, session_id)
    except Exception as exc:
        logger.exception("Could not ingest ArXiv query")
        raise HTTPException(status_code=502, detail="ArXiv document ingestion failed.") from exc
    title = (
        documents[0].metadata.get("title", request.query.strip())
        if documents
        else request.query.strip()
    )
    return {"title": title, "chunks": len(documents)}


@app.get("/api/sessions/{session_id}/documents")
def list_documents(session_id: str) -> dict[str, list[str]]:
    _require_session(session_id)
    from backend.vector_store import list_papers

    try:
        return {"documents": list_papers(session_id)}
    except Exception as exc:
        logger.exception("Could not list documents for session %s", session_id)
        raise HTTPException(status_code=503, detail="Could not load document list.") from exc
