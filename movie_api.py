from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from json import dumps, loads

from kafka import KafkaProducer
import time

app = FastAPI(title="Movie Recommendation API", version="1.0")

# Cargar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2",
                            device='cuda:0')

# Cargar datos de películas
with open("data/5000_movies.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

# Cargar índice FAISS
index = faiss.read_index("data/5000_movies.index")

# Configurar el productor Kafka
producer = KafkaProducer(
    #bootstrap_servers='192.168.0.109:9092',  # IP de VM
    bootstrap_servers=["192.168.0.109:9092"],
    value_serializer=lambda v: dumps(v).encode('utf-8')
)

# @app.get("/")
# def home():
#     return {"message": "Bienvenido"}

@app.get("/recommend/")
def recommend(query: str = Query(..., description="Descripción de la película"), top_k: int = 20):
    """ Devuelve las películas más similares a la descripción dada """
    
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [{"title": movies[i]["title"], "overview": movies[i]["overview"], "score": float(dist)}
               for i, dist in zip(indices[0], distances[0])]

    latency = (time.time() - start_time) * 1000

    message = f"{time.time()}, 200, {latency}"

    # message = {
    #         "timestamp": time.time(),
    #         "endpoint": "recommendation request",
    #         "status": "200",  # si fue exitoso
    #         "query": query,
    #         "latency_ms": latency
    #     }
    
    producer.send('movielogN', value=message)

    return {"query": query, "recommendations": results}

