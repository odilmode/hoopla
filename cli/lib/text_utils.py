# python
import string
from nltk.stem import SnowballStemmer

def to_token(text: str) -> list[str]:
    return [t for t in text.split() if t.strip()]

def to_stem(words: list[str]) -> list[str]:
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w) for w in words]

def preprocess_text(text: str) -> str:
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))
