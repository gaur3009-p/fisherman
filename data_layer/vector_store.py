import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.texts = []

    def add(self, text: str):
        emb = self.model.encode([text], normalize_embeddings=True)
        emb = np.array(emb, dtype="float32")  # ✅ REQUIRED
        self.index.add(emb)
        self.texts.append(text)

    def search(self, query: str, k=3):
        if len(self.texts) == 0:
            return []

        q = self.model.encode([query], normalize_embeddings=True)
        q = np.array(q, dtype="float32")  # ✅ REQUIRED

        _, idx = self.index.search(q, k)
        return [self.texts[i] for i in idx[0] if i < len(self.texts)]
