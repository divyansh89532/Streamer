import pymongo
import faiss
import numpy as np
import os 


# MongoDB Setup
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv('MONGO_DB')]
face_collection = db["RegisteredUsers"]

# FAISS Setup
embedding_dim = 512  # InsightFace outputs 512-d embeddings

# Load existing index or create a new one
if os.path.exists(os.getenv("FAISS_INDEX_PATH")):
    index = faiss.read_index(os.getenv("FAISS_INDEX_PATH"))  # ✅ Load existing index
else:
    index = faiss.IndexFlatL2(512)  # ✅ 512-dimensional embeddings
    index = faiss.IndexIDMap(index)  # ✅ Allows add_with_ids

def add_to_faiss(embedding: np.ndarray, user_id: int):
    """Adds an embedding with a unique ID to the FAISS index and saves it."""
    index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([user_id], dtype=np.int64))
    faiss.write_index(index, os.getenv("FAISS_INDEX_PATH"))  # ✅ Save index after update