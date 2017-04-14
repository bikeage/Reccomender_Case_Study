import exploration
import graphlab
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def get_jokes_with_lf_scores(jokes_df, n_components=5, model=NMF):
    vect = TfidfVectorizer(stop_words='english')
    tf_idf_matrix = vect.fit_transform(jokes_df.joke_text)

    trained_model = model(n_components=n_components).fit(tf_idf_matrix)

    latent_feature_scores = trained_model.transform(tf_idf_matrix)
    lfs_df = pd.DataFrame(latent_feature_scores)
    jokes_with_lfs = pd.concat((jokes_df, lfs_df), axis=1)

    return graphlab.SFrame(jokes_with_lfs)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
