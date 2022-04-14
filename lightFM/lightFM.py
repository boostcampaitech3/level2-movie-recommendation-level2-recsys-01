import numpy as np
import pandas as pd

import joblib

import argparse
from sklearn import preprocessing

from tqdm import tqdm

from scipy import sparse

from lightfm import LightFM
from lightfm.evaluation import recall_at_k


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="warp")
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--num_components", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--item_alpha", type=float, default=1e-6)
    parser.add_argument("--submission", type=str, default="lightFM")

    args = parser.parse_args()

    print(args.submission)

    # 평점 데이터
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
    ratings_coo.shape

    # warp model without item feature

    # Set the number of threads; you can increase this
    # if you have more physical cores available.
    NUM_THREADS = args.num_threads
    NUM_COMPONENTS = args.num_components
    NUM_EPOCHS = args.num_epochs
    ITEM_ALPHA = args.item_alpha

    # Define a new model instance
    model = LightFM(loss=args.model,
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    model_wo_item_feature = model.fit(ratings_coo,
                    epochs=NUM_EPOCHS,
                    num_threads=NUM_THREADS,
                    verbose = True)

    ratings_recall = recall_at_k(model_wo_item_feature,
                        ratings_coo,
                        k = 10,
                        num_threads=NUM_THREADS).mean()
    print(args.submission, " ratings set recall@10: ", ratings_recall)

    joblib.dump(model_wo_item_feature, "model/" + args.submission + '.pkl')

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