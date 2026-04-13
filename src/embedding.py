from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def get_tfidf(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(texts)

def get_bert(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)