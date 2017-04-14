 #pair.py

import graphlab
import pandas as pd
import numpy as np
import exploration as ex

#ratings_contents = pd.read_table("../data/ratings.dat", names=["user", "joke", "rating"])
#df = pd.read_table('../data/ratings.dat')

# test set
#u1 = ratings_df[ratings_df['user_id'] == 1].sort('joke_id')



def get_data_as_sframe():
    '''
    Get the data
    '''
    ratings_df, jokes_df = ex.get_ratings_and_jokes()
    test_df = pd.read_csv('data/sample_submission.csv')

    #create the SFrame
    rsdf = graphlab.SFrame(ratings_df)
    tsdf = graphlab.SFrame(test_df)

    return ratings_df, jokes_df, test_df


def get_model():
    '''
    Graphlab auto-determines model and fits data
    returns ItemSimilarityRecommender
    https://turi.com/products/create/docs/generated/graphlab.recommender.item_similarity_recommender.ItemSimilarityRecommender.html
    '''
    m = graphlab.recommender.create(sdf, user_id='user_id', item_id='joke_id')

    return m


def recommend_jokes_for_users(u_list=[1,2], k=5): # Run predict on test set
    '''
    input:  list of users
    return  sfeame of list of recommndations users
    '''
    r = m.recommend(users=u_list,k=k)

    return r


# get scores

# print results

# plot some data

ratings_df, jokes_df, test_df = get_data_as_sframe()
