import os

import numpy as np
import pandas as pd

import scipy.sparse

USER = "user"
ITEM = "item"
RATING = "rating"


class MFDataset:
    
    
    def __init__(self, data_dir: str, rating_fname="train_ratings.csv") -> None:
        self.data_dir = data_dir
        self.rating_file = os.path.join(data_dir, rating_fname)
        
        self.rating_df = pd.read_csv(self.rating_file, usecols=[USER, ITEM])
        self.rating_df[RATING] = np.ones(len(self.rating_df.index))
        self.user_ids = np.sort(self.rating_df[USER].unique())
        self.item_ids = np.sort(self.rating_df[ITEM].unique())
        self.num_user = len(self.user_ids)
        self.num_item = len(self.item_ids)
        
        self.interaction_matrix = scipy.sparse.csr_matrix(self.get_interaction_matrix())
    
    def get_interaction_matrix(self):
        print(f"Make user-item interaction matrix based on given '{os.path.basename(self.rating_file)}'...")
        
        interaction_matrix = self.rating_df.pivot_table(RATING, USER, ITEM).fillna(0.)
        return interaction_matrix.to_numpy()
    
    def idx2item(self, idx_list):
        return self.item_ids[idx_list]
    
    def idx2user(self, idx_list):
        return self.user_ids[idx_list]
