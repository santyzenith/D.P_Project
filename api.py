from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from models import load_model, load_movies, load_faiss_index
from recommendation import get_recommendations
from telemetry import init_producer, log_request
import time
# from kafka import KafkaProducer
# from json import dumps, loads

app = FastAPI(title="Movie Recommendation API", version="1.0")

model = load_model()
movies = load_movies()
index = load_faiss_index()
producer = init_producer()

class QueryRequest(BaseModel):
    query: str = Field(..., 
                       min_length=1, 
                       description="Descripción de la película")
    top_k: int = Field(20, 
                       ge=1, 
                       le=50, 
                       description="Número de recomendaciones")

# producer = KafkaProducer(
#     bootstrap_servers=["192.168.0.104:9092"],
#     value_serializer=lambda v: dumps(v).encode('utf-8')
# )

@app.post("/recommend/")
def recommend(request: QueryRequest):
    start_time = time.time()
    
    # Validar tamaño de la consulta (contexto máximo)
    tokens = model.tokenizer.tokenize(request.query)
    token_count = len(tokens)
    max_context = model.max_seq_length
    exceeds_limit = token_count > 5

    results = get_recommendations(request.query, model, index, movies, request.top_k)
    latency = (time.time() - start_time) * 1000
    
    # Registrar telemetría
    message = {
        "timestamp": time.time(),
        "endpoint": "recommendation request",
        "status": "200" if not exceeds_limit else "200-warning",
        "query": request.query,
        "latency_ms": latency,
        "token_count": token_count,
        "exceeds_max_context": exceeds_limit
    }

    try:
        producer.send("movielogN", value=message)
        print(f"Mensaje enviado a Kafka: {message}")
    except Exception as e:
        print(f"Error enviando mensaje a Kafka: {e}")

    if exceeds_limit:
        return {
            "query": request.query,
            "recommendations": results,
            "warning": f"Query exceeds max context length ({max_context} tokens), truncated to {token_count} tokens"
        }
    return {"query": request.query, "recommendations": results}