# Papeer

Papeer is a Streamlit research assistant backed by a FastAPI service. The Streamlit
application communicates with the API over HTTP; the API owns session persistence,
document ingestion, vector search, and the LangGraph RAG pipeline.

## Run locally

Configure `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`
in the project-root `.env` file. Keep that file private.

Install the dependencies from `requirements.txt`, then start the API and UI in
separate terminals from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The Streamlit client defaults to `http://127.0.0.1:8000`. Set `PAPEER_API_URL`
to use a different API address. Configure browser origins with the comma-separated
`PAPEER_CORS_ORIGINS` environment variable. Session metadata is stored in
`sessions.sqlite3`; the API imports existing entries from `sessions.json` on startup.
Graph checkpoints remain in `checkpoints.db`.

The API exposes OpenAPI documentation at `http://127.0.0.1:8000/docs`, a
liveness check at `/api/health`, and a dependency readiness check at `/api/ready`.

## Security and deployment

This API currently has **no authentication or user-level authorization**. It is
for trusted local use only and must not be exposed to the public internet. Before
deploying it for multiple users, add authentication and per-user session/document
authorization, rate limits, request-size limits at the reverse proxy, and SSRF
protections for URL ingestion. SQLite-backed sessions and checkpoints are suitable
for a single-node deployment, not horizontal scaling.