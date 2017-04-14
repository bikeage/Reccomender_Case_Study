#pair.py

import graphlab
import pandas as pd
import numpy as np

#ratings_contents = pd.read_table("../data/ratings.dat", names=["user", "joke", "rating"])
df = pd.read_table('../data/ratings.dat')

#1 create the SFrame
sdf = graphlab.SFrame(ratings_contents)


# Graphlab auto-determines model and fits data
# returns ItemSimilarityRecommender
m = graphlab.recommender.create(sdf, user_id='user_id', item_id='joke_id')

# Run predict on test set

# get scores

# print results

# plot some data
