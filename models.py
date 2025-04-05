from sentence_transformers import SentenceTransformer
import faiss
import json
import torch

def load_model(device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

def load_movies(file_path="data/5000_movies.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_faiss_index(file_path="data/5000_movies.index"):
    return faiss.read_index(file_path)