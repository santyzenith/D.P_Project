import numpy as np

def get_recommendations(query, model, index, movies, top_k=20):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [{"title": movies[i]["title"], "overview": movies[i]["overview"], "score": float(dist)}
            for i, dist in zip(indices[0], distances[0])]