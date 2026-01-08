class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query):
        results = self.vector_store.search(query)
        return "\n".join(results) if results else "No similar past cases found."
