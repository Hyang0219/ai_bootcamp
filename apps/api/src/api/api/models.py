from pydantic import BaseModel, Field
from typing import Optional

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline.")

class RAGUsedContext(BaseModel):
    id: str = Field(..., description="The ID of the referenced item")
    description: str = Field(..., description="The description of the item")
    image_url: str = Field(..., description="The image URL of the item")
    price: Optional[float] = Field(..., description="The price of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID.")
    answer: str = Field(..., description="The answer to the query.")
    used_context: list[RAGUsedContext] = Field(..., description="The information about the items used to answer the query.")