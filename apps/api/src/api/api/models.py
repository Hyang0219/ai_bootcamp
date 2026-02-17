from pydantic import BaseModel, Field
from typing import Optional, Union

class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline.")
    thread_id: str = Field(..., description="The thread ID")

class RAGUsedContext(BaseModel):
    id: str = Field(..., description="The ID of the referenced item")
    description: str = Field(..., description="The description of the item")
    image_url: str = Field(..., description="The image URL of the item")
    price: Optional[float] = Field(..., description="The price of the item")

class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID.")
    answer: str = Field(..., description="The answer to the query.")
    used_context: list[RAGUsedContext] = Field(..., description="The information about the items used to answer the query.")
    trace_id: str = Field(..., description="The trace ID of the run.")

class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="The trace ID of the run.")
    feedback_score: Union[int, None] = Field(..., description="The feedback score.")
    feedback_text: str = Field(..., description="The feedback text.")
    feedback_source_type: str = Field(..., description="The feedback source type. Human or API.")
    thread_id: str = Field(..., description="The thread ID.")

class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID.")
    status: str = Field(..., description="The status of the feedback submission.")