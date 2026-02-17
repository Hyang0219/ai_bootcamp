import openai
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery, Document, VectorParams, Distance, PayloadSchemaType, PointStruct, MatchAny


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


### Item Description Retrieval Tool

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
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

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_items_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context['retrieved_context_ids'], context['retrieved_context'], context['retrieved_context_ratings']):
        formatted_context += f"- ID: {id}, Rating: {rating}, Description: {chunk}\n"

    return formatted_context


def get_formatted_items_context(query: str, top_k: int =5) -> str:

    """
    Get the top k context, each representing an inventory item for a given query.

    Args:
        query (str): The query to get the context for.
        top_k (int): The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending to each chunk, each represending an inventory item a given query.
    """

    context = retrieve_items_data(query, top_k)

    formatted_context = process_items_context(context)

    return formatted_context


### Item Reviews Retrieval Tool


@traceable(
    name="retrieve_reviews_data",
    run_type="retriever"
)
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

@traceable(
    name="format_retrieved_reviews_context",
    run_type="prompt"
)
def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(context['retrieved_context_ids'], context['retrieved_context']):
        formatted_context += f"- ID: {id}, Review: {chunk}\n"

    return formatted_context


def get_formatted_reviews_context(query: str, item_list: list, top_k: int =15) -> str:

    """
    Get the top k reviews mathcing a query for a list of prefiltered items.

    Args:
        query (str): The query to get the reiviews for.
        item_list (list): The list of items to prefilter the reviews for.
        top_k (int): The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered.

    Returns:
        A string of the top k reviews with IDs prepending to each review, each represending a review for a given query and item.
    """

    context = retrieve_reviews_data(query, item_list, top_k)

    formatted_context = process_reviews_context(context)

    return formatted_context