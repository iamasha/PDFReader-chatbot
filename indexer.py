import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List
import os

# Use local model path
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "local_model")

class FaissIndex:
    def __init__(self, emb_model_path=LOCAL_MODEL_PATH):
        # Load the model from the local folder instead of downloading
        print(f"📂 Loading local model from: {emb_model_path}")
        self.model = SentenceTransformer(emb_model_path)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts = []

    def add(self, chunks: List[str]):
        embs = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.texts.extend(chunks)

    def save(self, index_path="faiss.index", meta_path="meta.pkl"):
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, index_path="faiss.index", meta_path="meta.pkl"):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.texts = pickle.load(f)

    def query(self, q: str, top_k: int = 5):
        q_emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = [self.texts[idx] for idx in I[0]]
        return results
