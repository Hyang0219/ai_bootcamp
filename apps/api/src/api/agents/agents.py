from openai import OpenAI

from langsmith import traceable
from langchain_core.messages import convert_to_openai_messages
import instructor

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message

from pydantic import BaseModel, Field
from typing import List


### Intent Router Response Model
class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str

### QnA Agent Response Model
class ToolCall(BaseModel):
    name: str
    arguments: dict

class RAGUsedContext(BaseModel):
    id: str = Field(description="The id of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class AgentResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    references: List[RAGUsedContext] = Field(description="The list of the items used to answer the question")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []

### QnA Agent Node
@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"}
)
def agent_node(state) -> dict:

    template = prompt_template_config("api/agents/prompts/qa_agent.yaml", "qa_agent")

    prompt = template.render(
        available_tools=state.available_tools
    )

    messsages = state.messages

    conversations =[]

    for message in messsages:
        conversations.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4o-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversations],
        temperature=0.5
    )

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references
    }


### Intent Router Agent Node

@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def intent_router_node(state):

    template = prompt_template_config("api/agents/prompts/intent_router_agent.yaml", "intent_router_agent")

    prompt = template.render()

    messsages = state.messages

    conversations =[]

    for message in messsages:
        conversations.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
            *conversations
        ],
        response_model=IntentRouterResponse,
    )

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }