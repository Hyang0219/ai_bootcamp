import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field
import instructor
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Document, FusionQuery
from api.agents.utils.prompt_management import prompt_template_config


class RAGUsedContext(BaseModel):
    id: str = Field(description="The id of the item used to answer the question")
    description: str = Field(description="Short description of the item used to answer the question")

class RAGGenerationRepsonse(BaseModel):
    answer: str = Field(description="The suggested items to the question following the format: 'Name of the item (ID of the item)'")
    reference: list[RAGUsedContext] = Field(description="The list of items used to answer the question")  

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }

    return response.data[0].embedding

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, k=5):
    query_embedding = get_embeddings(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search-v2",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(text=query, model="Qdrant/bm25"),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context= []
    similiarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context.append(result.payload['description'])
        similiarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload['average_rating'])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similiarity_scores": similiarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context['retrieved_context_ids'], context['retrieved_context'], context['retrieved_context_ratings']):
        formatted_context += f"- ID: {id}, Rating: {rating}, Description: {chunk}\n"

    return formatted_context

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question):

    template = prompt_template_config("api/agents/prompts/retrieval_generation.yaml", "retrieval_generation")
    prompt = template.render(preprocessed_context=preprocessed_context, question=question)

    return prompt

@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def generate_answer(prompt):

    client = instructor.from_openai(openai.OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        response_model=RAGGenerationRepsonse
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response


@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, k=5):

    retrieved_context = retrieve_data(question, qdrant_client, k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    final_result = {
        "original_output": answer,
        "answer": answer.answer,
        "reference": answer.reference,
        'question': question,
        "retrieved_context_ids": retrieved_context['retrieved_context_ids'],
        "retrieved_context": retrieved_context['retrieved_context'],
        "similiarity_scores": retrieved_context['similiarity_scores']
    }

    return final_result

def rag_pipeline_wrapper(question, k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(question, qdrant_client, k)

    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get('reference', []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search-v2",
            query=dummy_vector,
            limit=1,
            using="text-embedding-3-small",
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        ).points[0].payload
        image_url = payload.get('image', '')
        price = payload.get('price', '')
        if image_url:
            used_context.append({
                "description": item.description,
                "image_url": image_url,
                "price": price,
                "id": item.id
            })

    return {
        "answer": result['answer'],
        "used_context": used_context
    }