<!-- apps/api/README.md -->
## `api` (FastAPI) — RAG + LangGraph orchestration

This service exposes a small HTTP API for a RAG shopping assistant. Internally it uses:

- **LangGraph** for agent/tool orchestration
- **Qdrant** for retrieval (hybrid search)
- **Postgres** for LangGraph checkpointing/state
- **OpenAI** for LLM + embeddings (plus optional Cohere reranking depending on the pipeline)

### Endpoints

- **`POST /rag/`**: streams progress + final answer using **SSE** (`text/event-stream`)
  - Request body matches `RAGRequest` (`apps/api/src/api/api/models.py`):
    - `query`: string
    - `thread_id`: string (conversation/session identifier)
- **`POST /submit_feedback/`**: submits feedback tied to a `trace_id`
- **`GET /docs`**: Swagger UI

> Note: `POST /rag` (no trailing slash) will 307-redirect to `/rag/`.
> For streaming clients, call `/rag/` directly to avoid redirect edge cases.

### Streaming format (SSE)

The `/rag/` endpoint emits a sequence of SSE frames:

- **Progress frames**: plain text (not JSON), e.g. `Analysing the question...`
- **Final frame**: JSON string with shape:
  - `{"type":"final_answer","data":{"answer": "...", "used_context":[...], "trace_id":"..."}}`

Each frame is sent as:

```text
data: <payload>\n\n
```

### Run (Docker Compose)

From repo root:

```bash
make run-docker-compose
```

This starts the API on `http://localhost:8000` and its dependencies (`qdrant`, `postgres`).

### Run (local, with `uv`)

From repo root:

```bash
uv sync
OPENAI_API_KEY=... uv run --package api uvicorn api.app:app --reload --port 8000
```

If you run locally (not in Docker), ensure the service can reach Qdrant/Postgres (by default the code uses Docker hostnames like `qdrant` and `postgres`).

### Quick test (streaming)

```bash
curl -N -X POST "http://localhost:8000/rag/" \
  -H "Content-Type: application/json" \
  -d '{"query":"Suggest earphones for the gym","thread_id":"demo-thread-1"}'
```

