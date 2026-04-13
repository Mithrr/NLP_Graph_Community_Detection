import pandas as pd
from src.preprocessing import clean_text
from src.embedding import get_bert, get_tfidf
from src.similarity import compute_similarity
from src.graph_builder import build_graph
from src.community import detect_communities
from src.config import *

def run_pipeline():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    
    texts = df.iloc[:200]["text"].apply(clean_text).tolist()

    if USE_BERT:
        embeddings = get_bert(texts)
    else:
        embeddings = get_tfidf(texts)

    sim_matrix = compute_similarity(embeddings)

    G = build_graph(sim_matrix, texts, SIMILARITY_THRESHOLD)

    partition = detect_communities(G)

    return G, partition, texts