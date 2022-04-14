import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import joblib

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightFM")
    parser.add_argument("--submission", type=str, default="DeepCTR_model_import_example")
    args = parser.parse_args()

    model = joblib.load("model/" + args.model + '.pkl')
    embedding_dim = int(args.model.split('_')[1])
    print("imported ", args.model, "model")

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data = pd.read_csv("train_ratings.csv")
    data["rating"] = 1

    genre_data = pd.read_csv("genres.tsv", sep="\t")

    genre = {}
    for i in range(genre_data.shape[0]):
        it, ge = genre_data["item"][i], genre_data["genre"][i]
        if it in genre:
            genre[it] = genre[it] + '|' + ge
        else:
            genre[it] = ge
        
    genre = pd.DataFrame(genre, index = [0]).T
    genre.reset_index(inplace = True)
    genre.columns = ["item", "genre"]

    # Negative instance 생성
    print("Create Nagetive instances")
    num_negative = 50
    user_group_dfs = list(data.groupby('user')['item'])
    first_row = True
    user_neg_dfs = pd.DataFrame()
    items = set(data["item"])

    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)
        
        i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

    data = pd.concat([data, user_neg_dfs], axis = 0, sort=False)
    data = data.merge(genre, on = ["item"], how = "left")

    sparse_features = ["user", "item"]

    # 1.Label Encoding for sparse features,and process sequence features
    lbe1 = LabelEncoder()
    data["user"] = lbe1.fit_transform(data["user"])
    lbe2 = LabelEncoder()
    data["item"] = lbe2.fit_transform(data["item"])

    # preprocess the sequence feature

    key2index = {}
    genre_list = list(map(split, data['genre'].values))
    genre_length = np.array(list(map(len, genre_list)))
    max_len = max(genre_length)
    # Notice : padding=`post`
    genre_list = pad_sequences(genre_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                                for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genre', vocabulary_size=len(
        key2index) + 1, embedding_dim=embedding_dim), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    data = shuffle(data)
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["genre"] = genre_list

    data_dummy = pd.read_csv("train_ratings.csv")
    data_dummy["user"] = lbe1.fit_transform(data_dummy["user"])
    data_dummy["item"] = lbe2.fit_transform(data_dummy["item"])
    data_dummy = data_dummy[["user", "item"]]
    data_dummy["value"] = 1
    data_dummy = data_dummy.pivot(index='user', columns='item', values='value')
    data_dummy = data_dummy.fillna(0.0).astype(int)

    predict_genre = data[["item", "genre"]].drop_duplicates().sort_values(["item"])
    predict_genre.reset_index(drop=True, inplace=True)
    genre_list = list(map(split, predict_genre['genre'].values))
    genre_length = np.array(list(map(len, genre_list)))
    max_len = max(genre_length)
    genre_list = pad_sequences(genre_list, maxlen=max_len, padding='post', )
    genre_list = pd.DataFrame(genre_list)
    predict_genre = pd.concat([predict_genre, genre_list], axis=1)
    predict_genre.drop(["genre"], axis=1, inplace=True)

    item_output = []

    for i in tqdm(range(len(lbe1.classes_))):
        pred_df = pd.DataFrame()
        predict_item_ids = np.where(data_dummy.iloc[i] == 0)[0]
        predict_user_ids = np.full((len(predict_item_ids)), i)
        pred_df["user"] = predict_user_ids
        pred_df["item"] = predict_item_ids
        pred_df = pred_df.merge(predict_genre, on = ["item"], how = "left")
        # display(pred_df)
        
        pred_model_input = {name: pred_df[name] for name in sparse_features}  #
        pred_model_input["genre"] = np.array(pred_df.iloc[:, 2:])
        # print(pred_model_input)

        pred_ans = model.predict(pred_model_input, 256)
        pred_ans = pred_ans.reshape(-1,)
        ind = pred_ans.argpartition(-10)[-10:]
        item_output.extend(predict_item_ids[ind])

    submission = pd.read_csv("sample_submission.csv")
    submission["item"] = lbe2.inverse_transform(item_output)
    submission.to_csv("output/" + args.submission + ".csv", index=False)


if __name__ == "__main__":
    main()
 