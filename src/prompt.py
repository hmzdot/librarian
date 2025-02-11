from vecdb import VecDB
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

llm = init_chat_model(model="gpt-4o", model_provider="openai")


def generate_response(query, vecdb: VecDB, verbose=False):
    retrieved_docs = vecdb.search(query)

    # Group docs by path
    docs_by_path = {}
    for doc in retrieved_docs:
        path = doc["path"]
        if path not in docs_by_path:
            docs_by_path[path] = f"# {path}\n\n"
        docs_by_path[path] += doc["text"] + f"({doc['score']})" + "\n\n"

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
    messages = [system_message, user_message]

    for token in llm.stream(messages):
        print(token.content, end="", flush=True)
