import json
import os
DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = PROJECT_ROOT + "/cache"
BM25_K1 = 1.5
BM25_B = 0.75
def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORD_PATH, "r") as file:
        stop_words = file.read().splitlines()
    return stop_words
