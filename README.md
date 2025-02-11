# Librarian

Turn Markdown folders to RAG-enabled assistants

## Approaches

- [X] Simple RAG
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
