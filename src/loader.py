import os
from langchain.schema import Document


def load_vault(vault_path: str) -> list[Document]:
    docs = []
    for root, _, files in os.walk(vault_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".md"):
                with open(file_path, "r") as f:
                    docs.append(
                        Document(
                            page_content=f"# {file_path}\n\n" + f.read(),
                            metadata={"source": file_path},
                        )
                    )
    return docs
