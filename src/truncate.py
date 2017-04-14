import exploration
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ratings, jokes = exploration.get_ratings_and_jokes(filepath='data/ratings.dat')

vector = TfidfVectorizer(stop_words='english')
X = vector.fit_transform(jokes.joke_text)
vocab = vector.get_feature_names()

trunc = TruncatedSVD(n_components=2)
trunc.fit(X)
T = trunc.transform(X)

def trunc_func(desired_features):
    trunc = TruncatedSVD(n_components=desired_features)
    trunc.fit(X)
    T = trunc.transform(X)
    print_top_words(trunc, vocab, 20)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
