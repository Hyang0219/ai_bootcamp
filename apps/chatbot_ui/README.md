<!-- apps/chatbot_ui/README.md -->
## `chatbot_ui` (Streamlit) — Ecommerce Assistant UI

Streamlit UI for chatting with the RAG backend (`apps/api`). It consumes the backend’s **SSE streaming** responses and renders:

- intermediate “status” updates (planning / tool usage)
- the final answer + retrieved items (used context)
- optional feedback submission

### Configuration

The UI reads `API_URL` from environment (see `apps/chatbot_ui/src/chatbot_ui/core/config.py`):

- **Docker Compose default**: `http://api:8000`
- **Local development**: set `API_URL=http://localhost:8000`

### Run (Docker Compose)

From repo root:

```bash
make run-docker-compose
```

Then open `http://localhost:8501`.

### Run (local, with `uv`)

From repo root:

```bash
uv sync
API_URL="http://localhost:8000" uv run --package chatbot_ui streamlit run apps/chatbot_ui/src/chatbot_ui/app.py
```

### API expectations

- The backend streaming endpoint is `POST /rag/` (note trailing slash).
- The stream includes non-JSON progress frames and a final JSON frame with `type="final_answer"`.

