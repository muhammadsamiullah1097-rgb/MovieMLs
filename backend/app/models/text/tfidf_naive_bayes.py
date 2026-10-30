from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .text_preprocessing import tokenize


def get_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, tokenizer=tokenize, token_pattern=None, lowercase=False)),
        ("model", MultinomialNB(alpha=0.5, fit_prior=False)),
    ])
