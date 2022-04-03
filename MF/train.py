import argparse
from importlib import import_module
import multiprocessing
import os

import numpy as np
import pandas as pd

from dataset import MFDataset
from model import *


def main(args):
    # dataset
    dataset = MFDataset(args.data_dir)
    
    # model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        factors=args.n_factors,
        learning_rate=args.lr,
        regularization=args.regularization,
        iterations=args.epoch,
        num_threads=multiprocessing.cpu_count() // 2
    )
    
    # train
    print("Start training!!!")
    model.fit(dataset.interaction_matrix)
    
    # prediction (make top K recoomendation)
    print("\nStart prediction step...")
    recommends, scores = model.recommend(
        np.arange(dataset.num_user), 
        dataset.interaction_matrix,
        N=args.K)
    
    # writing sumbssion file
    submission = pd.DataFrame()
    submission["user"] = np.repeat(dataset.user_ids, args.K)
    submission["item"] = dataset.idx2item(recommends).reshape(-1, )
    submission.to_csv(args.output, index=False)
    print("Done!!!")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="~/input/data/train", help="input data directory")
    parser.add_argument("--output", type=str, default="./output/submission.csv")
    
    parser.add_argument("--K", type=int, default=10, help="K for the number of recommendation")
    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--n_factors", type=int, default=64)
    parser.add_argument("--model", type=str, default="BayesianPersonalizedRanking", help="model type (default: BayesianPersonalizedRanking)")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args)
    