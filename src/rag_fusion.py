"""
So called "fusion" technique involves generating multiple queries from a single
query by asking a LLM to generate multiple queries.
"""

import json
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document


def dump_doc(doc: Document) -> str:
    return json.dumps(doc.model_dump())


def load_doc(doc: str) -> Document:
    return Document(**json.loads(doc))


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dump_doc(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (load_doc(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that generates multiple search queries based on a single input query.",
        ),
        ("human", "Generate multiple search queries related to: {original_query}"),
        ("human", "OUTPUT (4 queries):"),
    ]
)


def fusion(llm, retriever) -> Runnable:
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    return generate_queries | retriever.map() | reciprocal_rank_fusion
