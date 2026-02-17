<!-- README.md -->
# End-to-End AI Engineering Bootcamp

This repository contains the code and course materials for the [End-to-End AI Engineering Bootcamp](https://maven.com/swirl-ai/end-to-end-ai-engineering). It is designed to be a comprehensive guide to building production-ready AI applications, featuring a full-stack RAG (Retrieval-Augmented Generation) application and a series of educational notebooks.

## 🏗️ Project Architecture

The project is structured as a monorepo containing a modern AI application stack:

*   **Frontend**: A [Streamlit](https://streamlit.io/) chatbot interface (`apps/chatbot_ui`) for interacting with the RAG pipeline.
*   **Backend**: A [FastAPI](https://fastapi.tiangolo.com/) service (`apps/api`) that handles retrieval, generation, and orchestration.
*   **Vector Database**: [Qdrant](https://qdrant.tech/) for storing and searching vector embeddings.
*   **State / Checkpointing**: Postgres (LangGraph checkpoint store).
*   **MCP Servers (optional)**: Two [FastMCP](https://github.com/jlowin/fastmcp) servers for tool calls in Week 4 notebooks:
    *   `apps/items_mcp_server` (items retrieval)
    *   `apps/reviews_mcp_server` (reviews retrieval)
*   **Observability**: Integrated with [LangSmith](https://www.langchain.com/langsmith) for tracing, monitoring, and evaluation.
*   **Package Management**: Uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

## 📂 Repository Structure

```
.
├── apps/
│   ├── api/            # FastAPI backend service
│   ├── chatbot_ui/     # Streamlit frontend application
│   ├── items_mcp_server/   # MCP server (items tools)
│   └── reviews_mcp_server/ # MCP server (reviews tools)
├── notebooks/          # Educational notebooks organized by curriculum
│   ├── prerequisites/  # Intro to LLM APIs
│   ├── week_1/         # RAG foundations: Ingestion, Pipeline, Evals
│   ├── week_2/         # Advanced RAG: Hybrid Search, Reranking, Structured Outputs
│   └── week_4/         # Agents + LangGraph + MCP + streaming
├── docker-compose.yml  # Container orchestration
├── Makefile            # Convenience commands
├── pyproject.toml      # Project dependencies and configuration
└── env.example         # Template for environment variables
```

## 🚀 Getting Started

### Prerequisites

*   [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.
*   [uv](https://github.com/astral-sh/uv) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
*   API Keys for:
    *   **OpenAI** (for embeddings and LLMs)
    *   **Cohere** (for reranking)
    *   **LangSmith** (optional but recommended for observability)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd ai_bootcamp
    ```

2.  **Configure Environment Variables:**
    Copy the example environment file and fill in your API keys.
    ```bash
    cp env.example .env
    ```
    *   Edit `.env` and add your keys (`OPENAI_API_KEY`, `CO_API_KEY`, `LANGSMITH_API_KEY`, etc.).

3.  **Install Dependencies:**
    Use `uv` to sync the project dependencies.
    ```bash
    uv sync
    ```

### Running the Application (Docker)

The easiest way to spin up the entire stack (API, Frontend, Database) is using Docker Compose.

```bash
make run-docker-compose
```

This will build the images and start the services:
*   **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
*   **FastAPI Backend**: [http://localhost:8000](http://localhost:8000) (Docs at [/docs](http://localhost:8000/docs))
*   **RAG streaming endpoint**: `POST http://localhost:8000/rag/` (SSE: `text/event-stream`)
*   **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
*   **Items MCP server**: `http://localhost:8001/mcp`
*   **Reviews MCP server**: `http://localhost:8002/mcp`

*Note: Qdrant data is persisted in the `./qdrant_storage` directory.*

### Troubleshooting

- **`POST /rag` returns 307 redirect**: the FastAPI route is mounted at `/rag/` (trailing slash). Prefer calling `POST /rag/` directly for streaming clients.
- **Notebooks after code changes**: if you edit helper modules (e.g. `utils.py`), restart the Jupyter kernel or reload the module to avoid import caching.

### Running Notebooks

To explore the course materials interactively:

1.  Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```
2.  Start Jupyter Lab (or Notebook):
    ```bash
    uv run jupyter lab
    ```

## 📚 Curriculum Overview

### Prerequisites
- **LLM APIs** (`notebooks/prerequisites/01-llm-apis.ipynb`): calling LLMs, basic prompting, and setup.

### Week 1: Foundations of RAG
- **Explore Amazon dataset** (`notebooks/week_1/01-explore-amazon-dataset.ipynb`)
- **Preprocessing** (`notebooks/week_1/02-RAG-preprocessing-Amazon.ipynb`)
- **RAG pipeline** (`notebooks/week_1/03-RAG-pipeline.ipynb`)
- **Evaluation dataset** (`notebooks/week_1/04-evaluation-dataset.ipynb`)
- **RAG evals** (`notebooks/week_1/05-RAG-evals.ipynb`)

### Week 2: Advanced RAG Techniques
- **Structured Outputs intro** (`notebooks/week_2/01-Structured-Outputs-Intro.ipynb`)
- **Structured Outputs + RAG pipeline** (`notebooks/week_2/02-Structured_Outputs-RAG-pipeline.ipynb`)
- **Hybrid search** (`notebooks/week_2/03-Hybrid-Search.ipynb`)
- **Reranking** (`notebooks/week_2/04-Reranking.ipynb`)
- **Prompt versioning** (`notebooks/week_2/05-Prompt-Versioning.ipynb`)

### Week 3: Agents & LangGraph Foundations
- **LangGraph intro** (`notebooks/week_3/01-LangGraph-Intro.ipynb`)
- **Query rewriting** (`notebooks/week_3/02-Query-Rewriting.ipynb`)
- **Routing** (`notebooks/week_3/03-Router.ipynb`)
- **Single-turn agent** (`notebooks/week_3/04-Agent-Single-Turn.ipynb`)

### Week 4: Multi-tool Agents, MCP, and Streaming
- **Multi-turn agent** (`notebooks/week_4/01-Multi-turn-Agent.ipynb`)
- **Multiple tools** (`notebooks/week_4/02-Multiple-Tools.ipynb`)
- **Human feedback** (`notebooks/week_4/03-Human-Feedback.ipynb`)
- **MCP** (`notebooks/week_4/04-MCP.ipynb`)
- **Streaming state** (`notebooks/week_4/05-Streaming-State.ipynb`)

## 🛠️ Development

- **Clean Notebooks**: Strip output from notebooks before committing.
  ```bash
  make clean-notebook-outputs
  ```
- **Run Evaluations**: Execute the retriever evaluation script.
  ```bash
  make run-evals-retriever
  ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
