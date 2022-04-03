import os

import numpy as np
import pandas as pd

import scipy.sparse
from tqdm import tqdm


class MFDataset:
    USER = "user"
    ITEM = "item"
    
    def __init__(self, data_dir: str, rating_fname="train_ratings.csv") -> None:
        self.data_dir = data_dir
        self.rating_file = os.path.join(data_dir, rating_fname)
        
        self.rating_df = pd.read_csv(self.rating_file, usecols=[self.USER, self.ITEM])
        self.user_ids = np.array(sorted(self.rating_df[self.USER].unique()))
        self.item_ids = np.array(sorted(self.rating_df[self.ITEM].unique()))
        self.num_user = len(self.user_ids)
        self.num_item = len(self.item_ids)
        
        self.interaction_matrix = scipy.sparse.csr_matrix(self.get_interaction_matrix())
    
    def get_interaction_matrix(self):
        print("Make user-item interaction matrix based on given file...")
        
        interation_df = pd.DataFrame(0., index=self.user_ids, columns=self.item_ids)
        for u_id, i_id in tqdm(self.rating_df.values):
            interation_df.at[u_id, i_id] = 1.0
            
        return interation_df
    
    def idx2item(self, idx_list):
        return self.item_ids[idx_list]
    
    def idx2user(self, idx_list):
        return self.user_ids[idx_list]
