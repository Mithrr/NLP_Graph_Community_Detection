from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

def get_tfidf(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(texts)

def get_bert(texts):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return model.encode(texts, device=device)