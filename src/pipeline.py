import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from src.preprocessing import clean_text
from src.embedding import get_bert, get_tfidf
from src.similarity import compute_similarity
from src.graph_builder import build_graph
from src.community import detect_communities
from src.config import *

def load_data():
    if DATASET_TYPE == "twitter":
        df = pd.read_csv(DATA_PATH, encoding='latin-1', header=None)
        texts = df.iloc[:MAX_ROWS, 5].astype(str).tolist()

    elif DATASET_TYPE == "news":
        data = fetch_20newsgroups(subset='train')
        texts = data.data[:MAX_ROWS]

    return texts


def run_pipeline():
    texts = load_data()

    texts = [clean_text(t) for t in texts]

    if USE_BERT:
        embeddings = get_bert(texts)
    else:
        embeddings = get_tfidf(texts)

    sim_matrix = compute_similarity(embeddings)

    G = build_graph(sim_matrix, texts, SIMILARITY_THRESHOLD)

    partition = detect_communities(G)

    return G, partition, texts