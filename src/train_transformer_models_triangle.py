#!/usr/bin/env python

"""
Script for training triangle transformer model

"""

# ---- DEPENDENCIES ---- #
import os
import sys
sys.path.append(os.path.join(".."))
import argparse
from utils.train_transformer import train_and_evaluate_transformer, prepare_data
from utils.model_utils import prepare_data_xgboost_transformer

# ---- MAIN FUNCTION ---- #

def main(out_path, test_size, val_size, n_sweeps, n_epochs):

    # ---- DATA PREPARATION ---- #

    print("[INFO] Preparing data...")
    
    # make the four data splits
    triangle_data, autobiographical_data, chronic_data, first_episode_data = prepare_data_xgboost_transformer(seed=1, text_features_path="../data/features/text_features.csv")

    # prepare data loader objects for all splits
    train_dataloader_tri, eval_dataloader_tri, test_dataloader_tri = prepare_data(data=triangle_data,
                                                                                  test_size=test_size, 
                                                                                  val_size=val_size,
                                                                                  task_angle = True)
    
    train_dataloader_auto, eval_dataloader_auto, test_dataloader_auto = prepare_data(data=autobiographical_data,
                                                                                  test_size=test_size, 
                                                                                  val_size=val_size,
                                                                                  task_angle = True)

    print(test_dataloader_tri.dataset["ID"])
    print(eval_dataloader_tri.dataset["ID"])
    print(test_dataloader_auto.dataset["ID"])
    
    # make output if it does not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    # ---- TRAIN TRANSFORMER MODELS ---- #
    
    print("[INFO] Training and evaluating triangle data transformer model...")

    train_and_evaluate_transformer(train_dataloader=train_dataloader_tri,
                                   eval_dataloader=eval_dataloader_tri,
                                   test_dataloader=test_dataloader_tri,
                                   gen_test_dataloader=test_dataloader_auto,
                                   out_path=out_path, 
                                   out_filename_cr="triangle_transformer_classification_report.txt", 
                                   out_filename_cf="triangle_transformer_confusion_matrix.png", 
                                   test_size=test_size, 
                                   val_size=val_size,
                                   sweep_project_name="triangle_transformer",
                                   n_sweeps=n_sweeps,
                                   n_epochs=n_epochs)
    
    print(f"[INFO] Done! Model performance metrics have been saved in {out_path}...")

if __name__=="__main__":
    
    # --- ARGUMENT PARSER --- #

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_path', type=str, required=False, default="../output/transformers/results",
                        help='define path to output dir')
    
    parser.add_argument('--test_size', type=float, required=False, default=0.20,
                        help='define size of test split as float, e.g. 0.20')

    parser.add_argument('--val_size', type=float, required=False, default=0.13,
                        help='define size of val split as float, e.g. 0.25')
    
    parser.add_argument('--n_sweeps', type=int, required=False, default=20,
                    help='define number of hyperparameter optimization sweeps')
    
    parser.add_argument('--n_epochs', type=int, required=False, default=20,
                    help='define number of epochs for training the models using optimal hyperparameters')
    
    args = parser.parse_args()

    # -- RUN MAIN FUNCTION --- #

    main(out_path = args.out_path,
         test_size=args.test_size,
         val_size=args.val_size,
         n_sweeps=args.n_sweeps,
         n_epochs=args.n_epochs)