# Librarian

Turn Markdown folders to RAG-enabled assistants

![Demo](https://github.com/user-attachments/assets/9b37832e-9cf9-4934-9159-daa9f320e5e3)

## Approaches

- [X] Simple RAG
- [X] Fusion RAG (Generate similar queries before asking LLM)
- [ ] Agentic RAG (Generate dummy answer before searching for better embeddings)

## How to Run

with `uv` installed

```bash
git clone git@github.com:hmzdot/librarian.git
cd librarian

uv sync

uv run src/main.py {path/to/folder} {query}
# Example: uv run src/main.py ~/Documents/Vaults/General "What should be the next step for my OCR project?"
```
