from sentence_transformers import SentenceTransformer
from models import load_movies
import numpy as np

def analyze_context_length(model, movies):
    tokenizer = model.tokenizer
    max_context = model.max_seq_length  # 256 para all-MiniLM-L6-v2
    lengths = []

    for movie in movies:
        tokens = tokenizer.tokenize(movie["overview"])
        length = len(tokens)
        lengths.append(length)

    max_length = max(lengths)
    avg_length = np.mean(lengths)
    exceed_count = sum(1 for l in lengths if l > max_context)

    print(f"Máximo tamaño en tokens: {max_length}")
    print(f"Promedio tamaño en tokens: {avg_length:.2f}")
    print(f"Descripciones que exceden {max_context} tokens: {exceed_count} ({(exceed_count/len(movies))*100:.2f}%)")
    return max_length, avg_length, exceed_count

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    movies = load_movies()
    analyze_context_length(model, movies)

if __name__ == "__main__":
    main()