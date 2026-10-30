import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def tokenize(text):
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return [word for word in words if word not in ENGLISH_STOP_WORDS]
