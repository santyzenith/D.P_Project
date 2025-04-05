import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from recommendation import get_recommendations
from models import load_model, load_movies, load_faiss_index
from unittest.mock import Mock

@pytest.fixture
def setup():
    # felixble para usar CPU
    model = load_model(device="cpu")
    movies = load_movies()
    index = load_faiss_index()
    return model, movies, index

def test_recommendations(setup):
    model, movies, index = setup
    query = "A movie with dogs"
    results = get_recommendations(query, model, index, movies, top_k=5)
    assert len(results) == 5
    assert all("title" in r and "overview" in r and "score" in r for r in results)

# Alternativa mock si no se quiere cargar el modelo
# def test_recommendations_mock():
#     mock_model = Mock()
#     mock_model.encode.return_value = [[0.1, 0.2, 0.3]]  # Embedding simulado
#     mock_movies = [{"title": "Test Movie", "overview": "Test Overview"}]
#     mock_index = Mock()
#     mock_index.search.return_value = ([0.5], [0])  # Distancia e índice simulados
    
#     results = get_recommendations("test query", mock_model, mock_index, mock_movies, top_k=1)
#     assert len(results) == 1
#     assert results[0]["title"] == "Test Movie"
