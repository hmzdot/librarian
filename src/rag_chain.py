from typing import Iterator
from loader import load_vault
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessageChunk

embedding_model = OpenAIEmbeddings()
llm = init_chat_model(model="gpt-4o", model_provider="openai")


def generate_messages(query: str, docs: list[Document], verbose=False):
    # Group docs by path
    docs_by_path = {}
    for doc in docs:
        path = doc.metadata["source"]
        if path not in docs_by_path:
            docs_by_path[path] = f"# {path}\n\n"
        docs_by_path[path] += doc.page_content + "\n\n"

    context = "\n\n".join(docs_by_path.values())

    if verbose:
        print("=== Context ===")
        print(context)

    system_message = SystemMessage(
        content=f"""
        You are an assistant that helps answer questions using the following notes:

        {context}

        Answer the user's query.
        Reference the notes by their path whenever applicable. Reference them like
        [1], [2] and have a "References" section at the end that matches the reference
        paths.
        
        For example, if there's a reference to /Vault/RustNotes.md in a sentence about
        Rust, it should look like:
        
        ```md
        Rust is a systems programming language that is fast, safe, and easy to use.[1]

        References:
        - [1]: /Vault/RustNotes.md
        ```
        
        """
    )
    user_message = HumanMessage(content=query)

    return [system_message, user_message]


def generate_response(query: str, vault_path: str) -> Iterator[BaseMessageChunk]:
    docs = load_vault(vault_path)
    store = FAISS.from_documents(docs, embedding_model)
    similar_docs = store.similarity_search(query)
    messages = generate_messages(query, similar_docs)
    return llm.stream(messages)
