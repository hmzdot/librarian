import os
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("vault_path", type=str)
    return parser.parse_args()


def load_markdown(vault_path: str) -> str:
    docs = []
    for root, _, files in os.walk(vault_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".md"):
                with open(file_path, "r") as f:
                    docs.append({"path": file_path, "content": f.read()})
    return docs


def chunk_documents(docs, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_docs.append({"path": doc["path"], "chunk": chunk})
    return chunked_docs


def embed_documents(docs):
    embeddings = OpenAIEmbeddings()
    for doc in docs:
        embeddings.embed_query(doc["chunk"])


if __name__ == "__main__":
    args = parse_args()
    md_files = load_markdown(args.vault_path)
    chunked_docs = chunk_documents(md_files)
    print(chunked_docs[0])
