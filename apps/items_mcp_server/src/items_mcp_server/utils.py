import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document, FusionQuery


def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding


### Item Description Retrieval Tool
def retrieve_items_data(query, k=5):

    query_embedding = get_embeddings(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

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


def process_items_data(context):

    formatted_context = ""

    for id, chunk, rating in zip(context['retrieved_context_ids'], context['retrieved_context'], context['retrieved_context_ratings']):
        formatted_context += f"- ID: {id}, Rating: {rating}, Description: {chunk}\n"

    return formatted_context