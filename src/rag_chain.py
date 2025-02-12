from typing import Iterator
from loader import load_vault
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

embedding_model = OpenAIEmbeddings()
llm = init_chat_model(model="gpt-4o", model_provider="openai")

template = """
You are an assistant that helps answer questions using the following notes:

{context}

Answer the user's query.
Reference the notes by their path whenever applicable. Reference them like
[1], [2] and have a "References" section at the end that matches the reference
paths.

## Examples
If there's a reference to /Vault/RustNotes.md in a sentence about Rust,
it should look like:

```md
Rust is a systems programming language that is fast, safe, and easy to use.[1]
References:
- [1]: /Vault/RustNotes.md

Question: {input}
```
"""
prompt = ChatPromptTemplate.from_template(template)


def generate_response(query: str, vault_path: str) -> Iterator[str]:
    docs = load_vault(vault_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    # Vector store is a place to store the embeddings of the documents.
    # FAISS::from_documents first embeds the documents, then stores them in
    # FAISS (Facebook AI Similarity Search) store
    store = FAISS.from_documents(docs, embedding_model)

    # Retriever is an interface with `get_relevant_documents` method.
    retriever = store.as_retriever()

    # LLMChainExtractor simply has a prompt that asks LLM to extract relevant
    # parts of the context. Then it chains (prompt |> llm |> NoOutputParser)
    # to get the output.
    #
    # NoOutputParser is lambda txt: "" if txt == "NO_OUTPUT" else txt
    compressor = LLMChainExtractor.from_llm(llm)

    # ContextualCompressionRetriever first calls vecstore.similarity_search
    # to get the documents, then calls compressor to compress the documents.
    #
    # A whole class just for this. lol
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

    # In Langchain, a chain is a sequence of runnables.
    # A runnable is an abstraction over a function. hahaha
    # Upside is that you can chain them or run them in parallel.
    #
    # RunnablePassthrough is identity function. You can pass extra data as
    # runnable by using RunnablePassthrough().
    #
    # StrOutputParser is lambda chunk: chunk.content, hahahaha
    rag_chain = (
        {"context": compression_retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.stream(query)
