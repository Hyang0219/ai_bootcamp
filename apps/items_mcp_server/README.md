<!-- apps/items_mcp_server/README.md -->
## `items_mcp_server` (FastMCP) — Items retrieval tools

This is a small **MCP server** (powered by `fastmcp`) that exposes tools for retrieving and formatting *inventory item* context from Qdrant.

### Exposed tools

- **`get_formatted_items_context(query: str, top_k: int = 5) -> str`**
  - Retrieves top-\(k\) matching items from Qdrant (hybrid retrieval) and returns a newline-delimited formatted string:
    - `- ID: <parent_asin>, Rating: <average_rating>, Description: <description>`

Implementation lives in:
- `src/items_mcp_server/main.py`
- `src/items_mcp_server/utils.py`

### Run (Docker Compose)

From repo root:

```bash
make run-docker-compose
```

The server will be available at:

- **MCP endpoint**: `http://localhost:8001/mcp` (container listens on `:8000`, mapped to `8001`)

### Run (local, with `uv`)

From repo root:

```bash
uv sync
OPENAI_API_KEY=... uv run --package items_mcp_server python -m items_mcp_server.main
```

> This service expects Qdrant reachable at `http://qdrant:6333` (Docker hostname). If running outside Docker, ensure networking matches or adjust the code accordingly.

### Quick client example

```python
from fastmcp import Client

client = Client("http://localhost:8001/mcp")

async with client:
    tools = await client.list_tools()
    result = await client.call_tool(
        "get_formatted_items_context",
        {"query": "gym earphones", "top_k": 5},
    )
    print(result.content[0].text)
```

