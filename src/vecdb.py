import faiss
import numpy as np
from loader import Embedded, embedding_model


class VecDB:
    def __init__(self, index: faiss.IndexIDMap, mapping: dict):
        self.index = index
        self.mapping = mapping

    def search(self, query: str, top_k=5) -> list[dict]:
        query_vector = (
            np.array(embedding_model.embed_query(query))
            .astype("float32")
            .reshape(1, -1)
        )
        distances, indices = self.index.search(query_vector, top_k)
        mappings = [
            {**self.mapping[map_id], "score": 1 - distances[0][i]}
            for i, map_id in enumerate(indices[0])
        ]

        return mappings

    @staticmethod
    def from_embeddings(embedding_list: list[Embedded]) -> "VecDB":
        embeddings = [embedding["embedding"] for embedding in embedding_list]
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)

        ids = np.array(range(len(embeddings)))
        index.add_with_ids(embeddings, ids)

        # Build mapping
        mapping = {
            i: {
                "path": embedding["path"],
                "text": embedding["chunk"],
            }
            for i, embedding in enumerate(embedding_list)
        }
        return VecDB(index, mapping)
