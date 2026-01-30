# End-to-End AI Engineering Bootcamp

This repository contains the code and course materials for the [End-to-End AI Engineering Bootcamp](https://maven.com/swirl-ai/end-to-end-ai-engineering). It is designed to be a comprehensive guide to building production-ready AI applications, featuring a full-stack RAG (Retrieval-Augmented Generation) application and a series of educational notebooks.

## 🏗️ Project Architecture

The project is structured as a monorepo containing a modern AI application stack:

*   **Frontend**: A [Streamlit](https://streamlit.io/) chatbot interface (`apps/chatbot_ui`) for interacting with the RAG pipeline.
*   **Backend**: A [FastAPI](https://fastapi.tiangolo.com/) service (`apps/api`) that handles retrieval, generation, and orchestration.
*   **Vector Database**: [Qdrant](https://qdrant.tech/) for storing and searching vector embeddings.
*   **Observability**: Integrated with [LangSmith](https://www.langchain.com/langsmith) for tracing, monitoring, and evaluation.
*   **Package Management**: Uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

## 📂 Repository Structure

```
.
├── apps/
│   ├── api/            # FastAPI backend service
│   └── chatbot_ui/     # Streamlit frontend application
├── notebooks/          # Educational notebooks organized by curriculum
│   ├── prerequisites/  # Intro to LLM APIs
│   ├── week_1/         # RAG foundations: Ingestion, Pipeline, Evals
│   └── week_2/         # Advanced RAG: Hybrid Search, Reranking, Structured Outputs
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
*   **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

*Note: Qdrant data is persisted in the `./qdrant_storage` directory.*

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

### Week 1: Foundations of RAG
*   **Exploratory Data Analysis**: Understanding the Amazon dataset.
*   **Preprocessing**: Cleaning and preparing data for vectorization.
*   **Pipeline Construction**: Building a basic RAG retrieval and generation flow.
*   **Evaluation**: Creating synthetic datasets and evaluating RAG performance using RAGAS/LangSmith.

### Week 2: Advanced RAG Techniques
*   **Structured Outputs**: Forcing LLMs to return JSON/Pydantic models.
*   **Hybrid Search**: Combining dense vectors with sparse (BM25) keyword search.
*   **Reranking**: Improving retrieval quality with Cohere rerankers.
*   **Prompt Engineering**: Versioning and managing prompts as code.

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
