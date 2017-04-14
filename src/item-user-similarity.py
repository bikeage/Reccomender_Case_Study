# Various recommend_jokes_for_users
# https://turi.com/products/create/docs/graphlab.toolkits.recommender.html

import graphlab
import pandas as pd
import numpy as np
import exploration as ex

#ratings_contents = pd.read_table("../data/ratings.dat", names=["user", "joke", "rating"])
#df = pd.read_table('../data/ratings.dat')

# data
ratings_df, jokes_df = ex.get_ratings_and_jokes()
sample_df = pd.read_csv('data/sample_submission.csv')

u1 = ratings_df[ratings_df['user_id'] == 1].sort('joke_id')
s1 = sample_df[sample_df['user_id'] == 1].sort('joke_id')
test_df = pd.read_csv('data/test_ratings.csv')
t1 = test_df[test_df['user_id'] == 1].sort('joke_id')


def get_sframe(df):
    return graphlab.SFrame(df)


def get_user_rec_model(sf):
    '''
    input:
        SFrame of users, items, and ratings

    Graphlab auto-determines model and fits data
    returns ItemSimilarityRecommender
    https://turi.com/products/create/docs/generated/graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender.html
    '''
    m = graphlab.recommender.create(sf, user_id='user_id', item_id='joke_id')

    return m


def recommend_jokes_for_users(u_list=[1,2], k=5): # Run predict on test set
    '''
    input:  list of users
    return  sfeame of list of recommndations users

    https://turi.com/products/create/docs/generated/graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender.recommend.html#graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender.recommend
    '''
    r = m.recommend(users=u_list,k=k)

    return r


def get_user_rec_model(sf):
    m = graphlab.recommender.item_content_recommender.create(sf , "joke_id")

def get_tfidf(jokes_df):
    sa = graphlab.SArray(jokes_df.joke_text)
    docs_tfidf = graphlab.text_analytics.tf_idf(sa)  # returns an SArray


m = get_user_rec_model(get_sframe(ratings_df))

# get scores

# print results

# plot some data
