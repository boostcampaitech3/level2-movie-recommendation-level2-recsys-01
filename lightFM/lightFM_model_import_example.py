import numpy as np
import pandas as pd

import joblib

import argparse
from sklearn import preprocessing

from tqdm import tqdm

from scipy import sparse

from lightfm.evaluation import recall_at_k

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightFM")
    parser.add_argument("--submission", type=str, default="lightFM_model_import_example")
    args = parser.parse_args()

    model_wo_item_feature = joblib.load("model/" + args.model + '.pkl')

    print("imported ", args.model, "model")

    ratings_df = pd.read_csv("train_ratings.csv")
    ratings_df["rating"] = 1
    ratings_df = ratings_df[["user", "item", "rating"]]
    ratings_df.columns = ['user_id', 'movie_id', 'rating']

    df = ratings_df.copy()
    users = df["user_id"]
    movies = df["movie_id"]

    le_users = preprocessing.LabelEncoder()
    le_users.fit(users)
    user_index = le_users.transform(users)

    le_movies = preprocessing.LabelEncoder()
    le_movies.fit(movies)
    movie_index = le_movies.transform(movies)

    coo_shape = (len(le_users.classes_), len(le_movies.classes_))

    ratings_coo = sparse.coo_matrix((df["rating"], (user_index, movie_index)), shape=coo_shape, dtype = np.int32)

    ratings_recall = recall_at_k(model_wo_item_feature,
                        ratings_coo,
                        k = 10,
                        num_threads=2).mean()
    print(args.model, " ratings set recall@10: ", ratings_recall)

    item_output = []
    ratings_array = ratings_coo.toarray()

    for i in tqdm(range(len(le_users.classes_))):
        predict_item_ids = np.where(ratings_array[i] == 0)[0]
        predict_user_ids = np.full((len(predict_item_ids)), i)
        prediction = model_wo_item_feature.predict(predict_user_ids, predict_item_ids)
        
        ind = prediction.argpartition(-10)[-10:]
        item_output.extend(predict_item_ids[ind])

    submission = pd.read_csv("sample_submission.csv")
    submission["item"] = le_movies.inverse_transform(item_output)
    submission.to_csv("output/" + args.submission + ".csv", index=False)


if __name__ == "__main__":
    main()
 