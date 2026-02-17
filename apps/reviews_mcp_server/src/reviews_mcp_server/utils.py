import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document, FusionQuery, Filter, FieldCondition, MatchAny


def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding


def retrieve_reviews_data(query, item_list, k=5):

    query_embedding = get_embeddings(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01-reviews",
        prefetch=Prefetch(
            query=query_embedding,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchAny(any=item_list)
                    )
                ]
            ),
            limit=20
        ),
        query=FusionQuery(fusion='rrf'),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context= []
    similiarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context.append(result.payload['text'])
        similiarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similiarity_scores": similiarity_scores,
    }


def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- ID: {id}, Review: {chunk}\n"

    return formatted_context