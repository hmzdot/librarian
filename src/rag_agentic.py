"""
Agentic RAG is a technique that involves using a LLM to generate a dummy answer to a
question.

Let's say we have a query:
> "What are next steps for my latest project?"

We can generate a dummy answer to this query:
> "The next steps for your latest project are to finish the implementation of the
> feature and to test it thoroughly."

This dummy answer can be used to generate embeddings that match better with the
embeddings of the documents.
"""

from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that generates dummy answers to questions.\n\n",
        ),
        (
            "human",
            "Generate a dummy answer to the following question: {original_query}",
        ),
        ("human", "OUTPUT (100 words):"),
    ]
)


def agentic(llm, retriever) -> Runnable:
    generate_answer = prompt | llm | StrOutputParser()
    return generate_answer | retriever
