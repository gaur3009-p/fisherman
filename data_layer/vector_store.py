import faiss, numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add(self, text):
        emb = self.model.encode([text])
        self.index.add(np.array(emb))
        self.texts.append(text)

    def search(self, query, k=3):
        if not self.texts:
            return []
        q = self.model.encode([query])
        _, idx = self.index.search(np.array(q), k)
        return [self.texts[i] for i in idx[0] if i < len(self.texts)]
