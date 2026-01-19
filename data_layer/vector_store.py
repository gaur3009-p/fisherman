import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384

        # 📌 Persistence paths
        self.index_path = "data/faiss.index"
        self.text_path = "data/texts.pkl"

        # 📌 Load existing memory if present
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.text_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.texts = []

    def _persist(self):
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.text_path, "wb") as f:
            pickle.dump(self.texts, f)

    def add(self, text: str):
        emb = self.model.encode(
            [text],
            normalize_embeddings=True
        )
        emb = np.array(emb, dtype="float32")  # ✅ FAISS SAFE

        self.index.add(emb)
        self.texts.append(text)
        self._persist()  # ✅ SAVE TO DISK

    def search(self, query: str, k=3):
        if not self.texts:
            return []

        q = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        q = np.array(q, dtype="float32")

        _, idx = self.index.search(q, k)
        return [self.texts[i] for i in idx[0] if i < len(self.texts)]
