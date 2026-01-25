import openai

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

import asyncio

from api.agents.retrieval_generation import rag_pipeline

ls_client = Client()
qdrant_client = QdrantClient(
    url="http://localhost:6333"
)

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

def _skip_metric(key: str, reason: str):
    return {
        "key": key,
        "value": "skipped",
        "metadata": {"reason": reason},
    }


def ragas_faithfulness(run, example):
    question = run.outputs.get("question")
    answer = run.outputs.get("answer")
    contexts = run.outputs.get("retrieved_context")
    if not question or not answer or not contexts:
        return _skip_metric(
            "ragas_faithfulness", "missing question/answer/retrieved_context"
        )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    async def inner():
        return await Faithfulness(llm=ragas_llm).single_turn_ascore(sample)

    return asyncio.run(inner())


def ragas_response_relevancy(run, example):
    question = run.outputs.get("question")
    answer = run.outputs.get("answer")
    contexts = run.outputs.get("retrieved_context")
    if not question or not answer or not contexts:
        return _skip_metric(
            "ragas_response_relevancy", "missing question/answer/retrieved_context"
        )

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    async def inner():
        scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        return await scorer.single_turn_ascore(sample)

    return asyncio.run(inner())


def ragas_context_precision_id_based(run, example):
    retrieved_ids = run.outputs.get("retrieved_context_ids")
    reference_ids = example.outputs.get("reference_context_ids")
    if not retrieved_ids or not reference_ids:
        return None

    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=reference_ids
    )

    async def inner():
        scorer = IDBasedContextPrecision()
        return await scorer.single_turn_ascore(sample)

    return asyncio.run(inner())


def ragas_context_recall_id_based(run, example):
    retrieved_ids = run.outputs.get("retrieved_context_ids")
    reference_ids = example.outputs.get("reference_context_ids")
    if not retrieved_ids or not reference_ids:
        return None

    sample = SingleTurnSample(
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=reference_ids
    )

    async def inner():
        scorer = IDBasedContextRecall()
        return await scorer.single_turn_ascore(sample)

    return asyncio.run(inner())


results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset-v3",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based
    ],
    experiment_prefix="retriever",
    max_concurrency=10
)