import os
import hashlib
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import TypedDict
from tqdm import tqdm

os.makedirs("cache", exist_ok=True)

embedding_model = OpenAIEmbeddings()

Embedded = TypedDict(
    "Embedded",
    {
        "path": str,
        "chunk": str,
        "embedding": list[float],
    },
)


def md5_checksum(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def strip_hash(path: str) -> str:
    return ":".join(path.split(":")[:-1])


def load_markdown(vault_path: str) -> list[dict[str, str]]:
    docs = []
    for root, _, files in os.walk(vault_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".md"):
                file_hash = md5_checksum(file_path)
                with open(file_path, "r") as f:
                    docs.append(
                        {
                            "path": f"{file_path}:{file_hash}",
                            "content": f"# {file_path}\n\n" + f.read(),
                        }
                    )
    return docs


def chunk_documents(docs, chunk_size=512, chunk_overlap=50, progress=True):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = []
    for doc in tqdm(docs, desc="Chunking documents", disable=not progress):
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_docs.append({"path": doc["path"], "chunk": chunk})
    return chunked_docs


def embed_documents(docs, progress=True) -> list[Embedded]:
    return [
        {
            "path": strip_hash(doc["path"]),
            "chunk": doc["chunk"],
            "embedding": embedding_model.embed_query(doc["chunk"]),
        }
        for doc in tqdm(docs, desc="Embedding documents", disable=not progress)
    ]


def load_embeddings(cache_path: str) -> list[Embedded]:
    with open(cache_path, "r") as f:
        return [
            {
                "path": strip_hash(doc["path"]),
                "chunk": doc["chunk"],
                "embedding": doc["embedding"],
            }
            for doc in json.load(f)
        ]


def load_vault(
    vault_path: str,
    progress=True,
    cache_prefix: str = "embeddings",
) -> list[Embedded]:
    docs = load_markdown(vault_path)
    chunked_docs = chunk_documents(
        docs, chunk_size=1024, chunk_overlap=0, progress=progress
    )
    cache_path = f"cache/{cache_prefix}_embeddings.json"

    # Load embeddings from disk
    # TODO: Validate cache
    if os.path.exists(cache_path):
        embedded_docs = load_embeddings(cache_path)
    else:
        embedded_docs = embed_documents(chunked_docs, progress=progress)
        with open(cache_path, "w") as f:
            json.dump(embedded_docs, f)

    return embedded_docs
