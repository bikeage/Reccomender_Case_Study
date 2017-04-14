import graphlab
import pandas as pd
import os
import numpy as np

def get_data(train_filepath, test_filepath, test_mode=False):
    ratings = pd.read_table(train_filepath)
    ratings = graphlab.SFrame(ratings)
    if test_mode:
        test_ratings = pd.read_csv(test_filepath)
        return ratings, graphlab.SFrame(test_ratings)
    else:
        train, validate = graphlab.recommender.util.random_split_by_user(ratings, user_id='user_id', item_id='joke_id')
        return train, validate


def train_model(train, **kwargs):
    # import ipdb; ipdb.set_trace()
    model = graphlab.factorization_recommender.create(train, user_id='user_id', item_id='joke_id', target='rating', **kwargs)
    return model

def export_recs(recs_for_csv):
    recs_for_csv = recs_for_csv['user_id', 'joke_id', 'score']
    recs_for_csv.rename(names={"score": "rating"})
    recs_for_csv.export_csv('../data/predicted_ratings.csv')

def grid_search(train_ratings, val_ratings):
    """params = {'user_id':'user_id', 'item_id':'joke_id', 'target':'rating',
                'solver':['sgd'],
                'regularization':[0],
                'num_factors': [2]}
    job = graphlab.model_parameter_search.create((train_ratings, val_ratings), \
            graphlab.recommender.factorization_recommender.create, params)
    results = job.get_results()
    rmses = np.array(results['validation_rmse'])
    best_result_index = np.argsort(rmses)[0]
    best_solver = results['solver'][best_result_index]
    best_reg = results['regularization'][best_result_index]
    best_n_factors = results['num_factors'][best_result_index]"""
    results = "hi"
    best_solver = 'sgd'
    best_reg = 0
    best_n_factors = 2

    return results, best_solver, best_reg, best_n_factors

def main():
    train_filepath = "../data/ratings.dat"
    test_filepath = "../data/test_ratings.csv"
    train_ratings, test_ratings = get_data(train_filepath, test_filepath, test_mode=True)

    if os.path.exists('model_file'):
        model = graphlab.load_model('model_file')
    else:
        results, best_solver, best_reg, best_n_factors = grid_search(train_ratings, test_ratings)
        model = train_model(train_ratings, solver=best_solver, regularization=best_reg, num_factors=best_n_factors)
        model.save('model_file')

    test_users = test_ratings['user_id']
    test_items = test_ratings['joke_id']
    recs = model.recommend(users=test_users, items=test_items, k=1)

    export_recs(recs)



if __name__ == '__main__':
    main()
