DATASET_TYPE = "news"   # options: "twitter", "news"

DATA_PATH = "data/raw/tweets.csv"
if DATASET_TYPE == "twitter":
    SIMILARITY_THRESHOLD = 0.1
else:
    SIMILARITY_THRESHOLD = 0.25
USE_BERT = True
MAX_ROWS = 200