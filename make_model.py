import graphlab
import pandas as pd

def get_data(filepath):
    ratings = pd.read_table(filepath)
    return ratings

def main():
    filepath = "../data/ratings.dat"
    ratings = get_data()
    ratings = graphlab.SFrame(ratings_contents)
    rank_model = graphlab.ranking_factorization_recommender.create(ratings, user_id='user_id', item_id='joke_id', target='rating', ranking_regularization=1)
    print rank_model
