#!/usr/bin/env python

"""
Script for running five random seed and performing hyperparameter optimization and training all 12 XGBoost models.
"""

# ---- DEPENDENCIES ---- #

import os
import sys
sys.path.append(os.path.join(".."))
import argparse
from src.train_xgboost_models import main as run_seed

# ---- MAIN FUNCTION ---- #

def main(out_path="../output/xgboost/results/", test_size=0.20, val_size=0.13):

    # create seed list
    seed_list = [60, 66, 31, 18, 97]

    # train models for each seed
    for seed in seed_list:
        run_seed(out_path=out_path, test_size=test_size, val_size=val_size, rand_seed=seed)

if __name__=="__main__":

    # --- ARGUMENT PARSER --- #

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_path', type=str, required=False, default="../output/xgboost/results/",
                        help='define path to output dir')
    
    parser.add_argument('--test_size', type=float, required=False, default=0.20,
                        help='define size of test split as float, e.g. 0.20')

    parser.add_argument('--val_size', type=float, required=False, default=0.13,
                        help='define size of val split as float, e.g. 0.25')
    
    args = parser.parse_args()

    # -- RUN MAIN FUNCTION --- #

    main(out_path = args.out_path,
         test_size=args.test_size,
         val_size=args.val_size)