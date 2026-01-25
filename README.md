## ai-engineering-bootcamp-prerequisites

This repository bootstraps the [End-to-End AI Engineering Bootcamp](https://maven.com/swirl-ai/end-to-end-ai-engineering). It wires together:
- a **FastAPI backend** that runs a RAG pipeline (Qdrant + OpenAI) and exposes `/rag` for querying;
- a **Streamlit UI** that calls that API, shows chat history, and displays LangSmith evaluation results;
- supporting scripts/notebooks for Qdrant ingestion, evaluation dataset creation, and LangSmith experiments.

### Key setup notes
1. Copy the template and add API keys:
   ```bash
   cp env.example .env
   ```
   Then fill `.env` with your OpenAI / Groq / Google keys, plus LangSmith credentials (project/endpoint/API key).

2. Start everything with Docker:
   ```bash
   make run-docker-compose
   ```
   - FastAPI listens on `http://localhost:8000`.
   - Streamlit is at `http://localhost:8501`.
   - Qdrant persists vectors under `./qdrant_storage`, so you can restart the stack without losing your collection.

3. Dataset / evaluation workflow:
   - Use `notebooks/week_1/04-evaluation-dataset.ipynb` (corrected to write `question`) to create `rag-evaluation-dataset-v3` in LangSmith.
   - Run `apps/api/evals/eval_retriever.py` against that dataset to log LangSmith experiments. The evaluator now gracefully skips runs with missing chunks but scores every valid retriever output.

4. LangSmith tracing:
   - Each function in `apps/api/src/api/agents/retrieval_generation.py` is decorated with `@traceable`. LangSmith captures embeddings, prompt construction, and the overall `rag_pipeline`.
   - Set `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`, and `LANGSMITH_API_KEY` in `.env` so traces land in the right project.

### Tidying things up
- `documentation/development-environment` and `qdrant_storage` are excluded in `.gitignore`.
- Hidden files such as `.DS_Store` are ignored as well.

Once everything is running and the dataset is rebuilt, you can inspect LangSmith’s “Precision/Recall,” “Faithfulness,” and “Response Relevancy” metrics under the `rag-evaluation-dataset-v3` project.
