<!-- apps/reviews_mcp_server/README.md -->
## `reviews_mcp_server` (FastMCP) — Reviews retrieval tools

This is a small **MCP server** (powered by `fastmcp`) that exposes tools for retrieving and formatting *product reviews* context from Qdrant, filtered to a set of candidate items.

### Exposed tools

- **`get_formatted_reviews_context(query: str, item_list: list, top_k: int = 15) -> str`**
  - Retrieves top-\(k\) matching reviews from Qdrant for the provided `item_list` and returns a newline-delimited formatted string:
    - `- ID: <parent_asin>, Review: <review_text>`

Implementation lives in:
- `src/reviews_mcp_server/main.py`
- `src/reviews_mcp_server/utils.py`

### Run (Docker Compose)

From repo root:

```bash
make run-docker-compose
```

The server will be available at:

- **MCP endpoint**: `http://localhost:8002/mcp` (container listens on `:8000`, mapped to `8002`)

### Run (local, with `uv`)

From repo root:

```bash
uv sync
OPENAI_API_KEY=... uv run --package reviews_mcp_server python -m reviews_mcp_server.main
```

> This service expects Qdrant reachable at `http://qdrant:6333` (Docker hostname). If running outside Docker, ensure networking matches or adjust the code accordingly.

### Quick client example

```python
from fastmcp import Client

client = Client("http://localhost:8002/mcp")

async with client:
    result = await client.call_tool(
        "get_formatted_reviews_context",
        {"query": "durable", "item_list": ["B000123", "B000456"], "top_k": 20},
    )
    print(result.content[0].text)
```

